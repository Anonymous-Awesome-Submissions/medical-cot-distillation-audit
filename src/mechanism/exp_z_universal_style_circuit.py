#!/usr/bin/env python3
"""
Universal style-layer localization across model families and style features.

For a given (model_family, style_feature) pair, this script:
  1. Loads vanilla and SFT models for that family.
  2. Identifies vanilla-vs-SFT divergence positions for the style feature.
  3. Caches vanilla resid_post at every layer at those positions.
  4. Patches each layer in SFT and measures recovery toward vanilla style.

Style features supported:
  - hedge: probability of next token being a hedging marker
  - first_person: probability of next token being 1st-person pronoun (" I", " we", " my", " our", " us")
  - certainty: probability of next token being a certainty marker (" definitely", " certainly",
    " clearly", " obviously", " surely", " indeed", " confirmed", " established")

Usage:
  python exp_z_universal_style_circuit.py \
      --vanilla-path /path/to/vanilla \
      --sft-path /path/to/sft_or_merged \
      --tl-name "meta-llama/Llama-3.1-8B" \
      --style hedge \
      --out-tag llama_hedge \
      --max-positions 80
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

COT_DIR = "experiments/module1/q2_hallucination"
OUT_DIR_BASE = "experiments/module1/mechanism/style_universe"

STYLE_VOCABS = {
    "hedge": [
        " might", " may", " could", " possibly", " perhaps", " probably",
        " likely", " unlikely", " suggest", " suggests", " indicate",
        " indicates", " appears", " appear", " consistent", " typically",
        " generally",
    ],
    "first_person": [
        " I", " we", " my", " our", " us", " I'm", " I'll",
        " we'll", " we've", " I've",
    ],
    "certainty": [
        " definitely", " certainly", " clearly", " obviously", " surely",
        " indeed", " confirmed", " established", " precisely", " exactly",
        " absolutely", " conclusively",
    ],
}


def get_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(toks[0])
    return sorted(ids)


def load_hooked(model_path, tl_name, dtype=torch.bfloat16, lora_adapter=None):
    print(f"Loading {model_path} (TL name: {tl_name}) ...", flush=True)
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    if lora_adapter:
        from peft import PeftModel
        print(f"  Merging LoRA from {lora_adapter}", flush=True)
        hf_model = PeftModel.from_pretrained(hf_model, lora_adapter,
                                             torch_dtype=dtype)
        hf_model = hf_model.merge_and_unload()
    return HookedTransformer.from_pretrained(
        tl_name, hf_model=hf_model, tokenizer=tokenizer,
        device="cuda", dtype=dtype,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
    ), tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-path", required=True)
    ap.add_argument("--sft-path", required=True,
                    help="Either merged-SFT model path, or vanilla path "
                         "with --sft-lora to apply LoRA on the fly")
    ap.add_argument("--sft-lora", default=None,
                    help="If provided, --sft-path is the vanilla and we merge "
                         "this LoRA adapter into it for the SFT model")
    ap.add_argument("--tl-name", required=True,
                    help="HookedTransformer name like 'Qwen/Qwen3-8B', "
                         "'meta-llama/Llama-3.1-8B', 'mistralai/Mistral-7B-v0.1'")
    ap.add_argument("--style", choices=list(STYLE_VOCABS.keys()), required=True)
    ap.add_argument("--out-tag", required=True,
                    help="Output filename tag, e.g. 'llama_hedge'")
    ap.add_argument("--max-positions", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Min vanilla style prob to be a candidate position")
    args = ap.parse_args()

    out_dir = Path(OUT_DIR_BASE)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use existing vanilla CoT prefixes as text prompts (medical reasoning text).
    samples = []
    with open(f"{COT_DIR}/vanilla_cot_medqa_test500.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:80]

    style_words = STYLE_VOCABS[args.style]
    print(f"Style: {args.style}, words: {style_words}", flush=True)

    # ---- Load vanilla ----
    van_model, tokenizer = load_hooked(args.vanilla_path, args.tl_name)
    style_ids = torch.tensor(get_token_ids(tokenizer, style_words), device="cuda")
    print(f"  {len(style_ids)} style token IDs in {args.tl_name} tokenizer", flush=True)

    # ---- Find divergence positions ----
    candidate_positions = []
    for s_idx, s in enumerate(samples):
        text = s.get("text", "")[:6000]
        if not text:
            continue
        tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")
        if tokens.shape[1] > 200:
            tokens = tokens[:, :200]
        with torch.no_grad():
            logits = van_model(tokens, return_type="logits")
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        style_probs = probs[:, style_ids].sum(dim=-1).float().cpu().numpy()
        for pos in range(5, len(style_probs)):
            if style_probs[pos] >= args.threshold:
                candidate_positions.append({
                    "sample_idx": s_idx, "token_pos": int(pos),
                    "tokens": tokens[:, :pos + 1].clone(),
                    "vanilla_style_p": float(style_probs[pos]),
                })
        if len(candidate_positions) >= args.max_positions * 2:
            break
    candidate_positions.sort(key=lambda x: -x["vanilla_style_p"])
    candidate_positions = candidate_positions[:args.max_positions]
    print(f"  {len(candidate_positions)} candidate positions "
          f"(threshold={args.threshold})", flush=True)

    n_layers = van_model.cfg.n_layers
    van_residuals = {}
    for c in candidate_positions:
        with torch.no_grad():
            _, cache = van_model.run_with_cache(c["tokens"], return_type=None)
        per_layer = {L: cache[f"blocks.{L}.hook_resid_post"][:, -1, :].clone().cpu()
                     for L in range(n_layers)}
        van_residuals[(c["sample_idx"], c["token_pos"])] = per_layer
    print(f"  Cached vanilla residuals", flush=True)

    del van_model
    torch.cuda.empty_cache()

    # ---- Load SFT ----
    if args.sft_lora:
        sft_model, _ = load_hooked(args.sft_path, args.tl_name,
                                    lora_adapter=args.sft_lora)
    else:
        sft_model, _ = load_hooked(args.sft_path, args.tl_name)

    sft_baseline = []
    for c in candidate_positions:
        with torch.no_grad():
            logits = sft_model(c["tokens"], return_type="logits")
        probs = torch.softmax(logits[0, -1], dim=-1)
        sft_baseline.append(float(probs[style_ids].sum()))
    sft_baseline = np.array(sft_baseline)
    van_baseline = np.array([c["vanilla_style_p"] for c in candidate_positions])
    print(f"  Vanilla style p: {van_baseline.mean():.4f}", flush=True)
    print(f"  SFT     style p: {sft_baseline.mean():.4f}", flush=True)
    print(f"  Gap (van - sft): {van_baseline.mean()-sft_baseline.mean():+.4f}", flush=True)

    # ---- Patching ----
    print(f"\n[Patching] {len(candidate_positions)} pos × {n_layers} L ...",
          flush=True)
    patched_p = np.zeros((len(candidate_positions), n_layers))
    t0 = time.time()
    for c_idx, c in enumerate(candidate_positions):
        key = (c["sample_idx"], c["token_pos"])
        for L in range(n_layers):
            van_resid = van_residuals[key][L].to("cuda")
            def hook_fn(act, hook):
                act[:, -1, :] = van_resid
                return act
            with torch.no_grad():
                logits = sft_model.run_with_hooks(
                    c["tokens"],
                    fwd_hooks=[(f"blocks.{L}.hook_resid_post", hook_fn)],
                    return_type="logits",
                )
            probs = torch.softmax(logits[0, -1], dim=-1)
            patched_p[c_idx, L] = float(probs[style_ids].sum())
        if (c_idx + 1) % 5 == 0:
            print(f"  {c_idx+1}/{len(candidate_positions)}, "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    recovery = patched_p.mean(axis=0) - sft_baseline.mean()
    print(f"\n=== {args.out_tag} ({n_layers} layers) ===")
    print(f"  Layer  Patched   Recovery")
    for L in range(n_layers):
        bar = "#" * max(0, int(recovery[L] / 0.005))
        print(f"  {L:3d}    {patched_p[:, L].mean():.4f}    {recovery[L]:+.4f}  {bar}")
    max_L = int(np.argmax(recovery))
    rel_depth = max_L / max(n_layers - 1, 1)
    print(f"\nMax recovery layer: {max_L}/{n_layers-1} (Δ={recovery[max_L]:+.4f}), "
          f"relative depth={rel_depth:.3f}")

    np.savez(f"{out_dir}/{args.out_tag}_patch.npz",
             patched_style=patched_p, sft_baseline=sft_baseline,
             vanilla_baseline=van_baseline)
    summary = {
        "tag": args.out_tag,
        "tl_name": args.tl_name,
        "style": args.style,
        "n_positions": len(candidate_positions),
        "n_layers": n_layers,
        "vanilla_style_avg": float(van_baseline.mean()),
        "sft_baseline_avg": float(sft_baseline.mean()),
        "per_layer_recovery": recovery.tolist(),
        "max_recovery_layer": max_L,
        "max_recovery_value": float(recovery[max_L]),
        "relative_depth": rel_depth,
    }
    with open(f"{out_dir}/{args.out_tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir}/{args.out_tag}_summary.json")


if __name__ == "__main__":
    main()
