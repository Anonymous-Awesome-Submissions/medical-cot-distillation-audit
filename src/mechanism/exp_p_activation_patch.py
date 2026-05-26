#!/usr/bin/env python3
"""
Layer-level activation patching to localize the
hedging circuit in Qwen3-8B SFT.

Setup:
  - Vanilla model: predicts hedge token at position i
  - SFT model: predicts assertive token at same position
  - Question: which residual stream layer holds the divergence?

Method (layer-level activation patching):
  1. Find positions where vanilla and SFT diverge in hedge probability.
  2. For each position, cache vanilla's resid_post at every layer.
  3. Run SFT forward, but at layer L replace its resid_post with vanilla's.
  4. Measure: P_SFT-with-vanilla-resid-at-L(hedge) vs P_SFT(hedge) baseline.
  5. The layer that maximally restores hedge probability is the locus.

This is a single-direction patch (vanilla → SFT). Could also do reverse
(SFT → vanilla) for symmetry but single direction is enough to localize.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

VANILLA_PATH = "Qwen/Qwen3-8B"
SFT_PATH = "experiments/module1/qwen3_8b_sft_merged"
COT_DIR = "experiments/module1/q2_hallucination"
OUT_DIR = "experiments/module1/mechanism/circuit"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

HEDGE_WORDS = [
    " might", " may", " could", " possibly", " perhaps", " probably",
    " likely", " unlikely", " suggest", " suggests", " indicate", " indicates",
    " appears", " appear", " consistent", " typically", " generally",
]


def get_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def load_hooked(model_path, dtype=torch.bfloat16):
    print(f"Loading {model_path} ...", flush=True)
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-8B", hf_model=hf_model, tokenizer=tokenizer,
        device="cuda", dtype=dtype,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
    )
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positions", type=int, default=80,
                    help="Number of hedge-divergence positions to analyze")
    args = ap.parse_args()

    # ---- Load vanilla CoTs to get prompts ----
    print("Loading vanilla CoT samples ...", flush=True)
    samples = []
    with open(f"{COT_DIR}/vanilla_cot_medqa_test500.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:60]

    # ---- Load vanilla model ----
    van_model, tokenizer = load_hooked(VANILLA_PATH)
    hedge_ids = torch.tensor(get_token_ids(tokenizer, HEDGE_WORDS), device="cuda")

    # ---- Find hedge-divergence positions on vanilla side ----
    # We pick positions where vanilla's NEXT-token distribution puts substantial
    # mass on hedge tokens (>= 5%). These are the positions where hedging is
    # at stake, so they're informative.
    print("\n[Vanilla pass] Identify hedge-divergence positions ...", flush=True)
    candidate_positions = []  # (sample_idx, token_pos, prefix_tokens)
    for s_idx, s in enumerate(samples):
        text = s.get("text", "")[:6000]
        tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")
        if tokens.shape[1] > 200:
            tokens = tokens[:, :200]
        with torch.no_grad():
            logits = van_model(tokens, return_type="logits")
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        hedge_probs = probs[:, hedge_ids].sum(dim=-1).float().cpu().numpy()
        for pos in range(5, len(hedge_probs)):
            if hedge_probs[pos] >= 0.05:  # at least 5% hedge mass
                candidate_positions.append({
                    "sample_idx": s_idx,
                    "token_pos": int(pos),
                    "tokens": tokens[:, :pos + 1].clone(),
                    "vanilla_hedge_p": float(hedge_probs[pos]),
                })
        if len(candidate_positions) >= args.max_positions * 2:
            break
    # Top by vanilla hedge probability
    candidate_positions.sort(key=lambda x: -x["vanilla_hedge_p"])
    candidate_positions = candidate_positions[:args.max_positions]
    print(f"  {len(candidate_positions)} candidate positions (highest vanilla hedge prob)",
          flush=True)
    if not candidate_positions:
        print("No candidates. Aborting."); return

    # Cache vanilla resid_post at every layer for each candidate position
    print("\n[Vanilla pass] Caching residuals ...", flush=True)
    n_layers = van_model.cfg.n_layers
    van_residuals = {}  # (sample_idx, token_pos) -> dict layer -> tensor (1, d_model)
    for c in candidate_positions:
        with torch.no_grad():
            _, cache = van_model.run_with_cache(c["tokens"], return_type=None)
        # We want resid at the position, ALL layers
        per_layer = {}
        for L in range(n_layers):
            r = cache[f"blocks.{L}.hook_resid_post"][:, -1, :].clone()
            per_layer[L] = r.cpu()  # store on CPU to save GPU mem
        van_residuals[(c["sample_idx"], c["token_pos"])] = per_layer

    print(f"  Cached vanilla residuals for {len(van_residuals)} positions × "
          f"{n_layers} layers", flush=True)

    del van_model
    torch.cuda.empty_cache()

    # ---- Load SFT model ----
    sft_model, _ = load_hooked(SFT_PATH)

    # ---- Baseline: SFT hedge probability at each position ----
    print("\n[SFT pass] Baseline hedge probability ...", flush=True)
    sft_baseline_hedge = {}
    for c in candidate_positions:
        with torch.no_grad():
            logits = sft_model(c["tokens"], return_type="logits")
        probs = torch.softmax(logits[0, -1], dim=-1)
        sft_baseline_hedge[(c["sample_idx"], c["token_pos"])] = float(probs[hedge_ids].sum())

    # ---- Patch each layer ----
    # For each candidate position c and each layer L:
    #   Run SFT forward with hook that overrides resid_post[L] at the last token
    print(f"\n[Patching] {len(candidate_positions)} positions × {n_layers} layers ...",
          flush=True)
    patched_hedge = np.zeros((len(candidate_positions), n_layers))
    t0 = time.time()
    for c_idx, c in enumerate(candidate_positions):
        key = (c["sample_idx"], c["token_pos"])
        for L in range(n_layers):
            van_resid = van_residuals[key][L].to("cuda")  # (1, d_model)

            def hook_fn(act, hook):
                # act shape: (batch, seq_len, d_model). Override last position.
                act[:, -1, :] = van_resid
                return act

            with torch.no_grad():
                logits = sft_model.run_with_hooks(
                    c["tokens"],
                    fwd_hooks=[(f"blocks.{L}.hook_resid_post", hook_fn)],
                    return_type="logits",
                )
            probs = torch.softmax(logits[0, -1], dim=-1)
            patched_hedge[c_idx, L] = float(probs[hedge_ids].sum())
        if (c_idx + 1) % 5 == 0:
            print(f"  {c_idx+1}/{len(candidate_positions)}, "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    # ---- Aggregate: per-layer recovery (vs SFT baseline) ----
    sft_baseline = np.array([sft_baseline_hedge[(c["sample_idx"], c["token_pos"])]
                             for c in candidate_positions])
    van_baseline = np.array([c["vanilla_hedge_p"] for c in candidate_positions])
    print(f"\n=== Per-layer hedge recovery ===")
    print(f"  Avg vanilla hedge p (target): {van_baseline.mean():.4f}")
    print(f"  Avg SFT baseline hedge p    : {sft_baseline.mean():.4f}")
    print(f"  Layer  Patched hedge p   Recovery (vs SFT baseline)")
    recovery = patched_hedge.mean(axis=0) - sft_baseline.mean()
    for L in range(n_layers):
        bar = "#" * max(0, int(recovery[L] / 0.005))
        print(f"  {L:3d}    {patched_hedge[:, L].mean():.4f}        {recovery[L]:+.4f}  {bar}")

    # Save
    np.savez(f"{OUT_DIR}/activation_patch_layer.npz",
             patched_hedge=patched_hedge, sft_baseline=sft_baseline,
             vanilla_baseline=van_baseline,
             candidate_positions=[(c["sample_idx"], c["token_pos"]) for c in candidate_positions])
    summary = {
        "n_positions": len(candidate_positions),
        "n_layers": n_layers,
        "vanilla_hedge_avg": float(van_baseline.mean()),
        "sft_baseline_hedge_avg": float(sft_baseline.mean()),
        "per_layer_patched_hedge_avg": patched_hedge.mean(axis=0).tolist(),
        "per_layer_recovery": recovery.tolist(),
        "max_recovery_layer": int(np.argmax(recovery)),
        "max_recovery_value": float(np.max(recovery)),
    }
    with open(f"{OUT_DIR}/activation_patch_layer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/activation_patch_layer_summary.json")
    print(f"\nMax-recovery layer: {summary['max_recovery_layer']} "
          f"(restored {summary['max_recovery_value']:+.4f} hedge prob)")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), patched_hedge.mean(axis=0), "o-", label="Patched SFT")
    ax.axhline(sft_baseline.mean(), ls="--", color="C1", label="SFT baseline")
    ax.axhline(van_baseline.mean(), ls="--", color="C0", label="Vanilla baseline")
    ax.set_xlabel("Layer L (patched resid_post[L] from vanilla)")
    ax.set_ylabel("P(hedge token) at next position")
    ax.set_title("Activation patching: which layer recovers hedging?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/activation_patch_per_layer.png", dpi=150)
    plt.savefig(f"{OUT_DIR}/activation_patch_per_layer.pdf")
    plt.close(fig)
    print(f"Saved figure: {OUT_DIR}/activation_patch_per_layer.png/pdf")


if __name__ == "__main__":
    main()
