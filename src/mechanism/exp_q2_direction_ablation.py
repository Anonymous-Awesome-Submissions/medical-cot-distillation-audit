#!/usr/bin/env python3
"""
exp_q2 — FIXED minimal "0.94 hedge-channel" rigor (replaces exp_q's broken
full-resid mean-ablation). Uses standard DIRECTIONAL ablation (project out a
direction), the operation we use to ablate the band direction/subspace.

Pipeline (all on the VANILLA Qwen3-8B; HookedTransformer):
  - Find hedge-divergence positions (vanilla hedge-prob >= 5%) AND non-hedge
    positions (vanilla hedge-prob < 1%) in the vanilla CoTs.
  - Cache vanilla resid_post at all layers at both position sets.
  - HEDGE DIRECTION at each layer L: d_L = mean(resid_post[L] | hedge-positions)
    - mean(resid_post[L] | non-hedge-positions), unit-normalized. (Difference-of-
    means, the standard interpretable "this is the hedge feature" direction.)
  - DIRECTIONAL ABLATION at a set of layers S: at the last token, for L in S,
    resid_post[L] <- resid_post[L] - (resid_post[L] . d_L) d_L.  Measure the
    drop in hedge-token probability mass at hedge-positions.
  - Conditions:
      * band  = [31,32,33] (the ~0.94 peak)
      * early = [3,4,5]
      * all   = every layer
      * band + RANDOM direction (instead of the hedge direction), 5 seeds
  - SYMMETRIC NOISING patch (kept from exp_q): patch the SFT model's resid_post
    into the vanilla model at the band -> hedge drop (the noising complement of
    the existing denoising patch in activation_patch_layer.npz).
  - SPECIFICITY: for every intervention also report the absolute change (Delta prob)
    in certainty-marker and first-person-pronoun mass (NOT percentages -- baselines tiny).

Hypothesis (for the "localized hedge read-out" claim):
  ablating the HEDGE direction at the BAND should drop hedge-prob a lot, almost
  as much as ablating it at ALL layers, and MUCH more than (a) the early band,
  (b) a random direction at the band -- while barely touching cert/fp tokens.

Output: experiments/module1/mechanism/circuit_rigor/exp_q2.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

VANILLA_PATH = "Qwen/Qwen3-8B"
SFT_PATH = "experiments/module1/qwen3_8b_sft_merged"
COT_DIR = "experiments/module1/q2_hallucination"
OUT_DIR = Path("experiments/module1/mechanism/circuit_rigor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEDGE_WORDS = [" might", " may", " could", " possibly", " perhaps", " probably", " likely",
               " unlikely", " suggest", " suggests", " indicate", " indicates", " appears",
               " appear", " consistent", " typically", " generally"]
CERTAINTY_WORDS = [" definitely", " certainly", " clearly", " obviously", " surely", " indeed",
                   " confirmed", " established", " precisely", " exactly", " absolutely"]
FIRSTPERSON_WORDS = [" I", " we", " my", " our", " us", " let's", " Let's"]


def tok_ids(tokenizer, words):
    ids = set()
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


def load_hooked(model_path, dtype=torch.bfloat16):
    print(f"Loading {model_path} ...", flush=True)
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-8B", hf_model=hf, tokenizer=tokenizer, device="cuda", dtype=dtype,
        fold_ln=False, center_writing_weights=False, center_unembed=False)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hedge-pos", type=int, default=80)
    ap.add_argument("--max-nonhedge-pos", type=int, default=160)
    ap.add_argument("--band", type=int, nargs="+", default=[31, 32, 33])
    ap.add_argument("--early-band", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--n-random-dirs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    van_model, tokenizer = load_hooked(VANILLA_PATH)
    n_layers = van_model.cfg.n_layers
    d_model = van_model.cfg.d_model
    print(f"  n_layers={n_layers} d_model={d_model}", flush=True)
    HEDGE_IDS = torch.tensor(tok_ids(tokenizer, HEDGE_WORDS), device="cuda")
    CERT_IDS = torch.tensor(tok_ids(tokenizer, CERTAINTY_WORDS), device="cuda")
    FP_IDS = torch.tensor(tok_ids(tokenizer, FIRSTPERSON_WORDS), device="cuda")
    print(f"  token sets: hedge={len(HEDGE_IDS)} cert={len(CERT_IDS)} fp={len(FP_IDS)}", flush=True)

    # ---- collect hedge + non-hedge positions from vanilla CoTs ----
    samples = []
    with open(f"{COT_DIR}/vanilla_cot_medqa_test500.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    hedge_pos, nonhedge_pos = [], []
    for s_idx, s in enumerate(samples):
        text = s.get("text") or s.get("teacher_cot") or s.get("cot") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 220:
            ids = ids[:220]
        if len(ids) < 25:
            continue
        with torch.no_grad():
            logits = van_model(torch.tensor([ids], device="cuda"))[0]
            hp = torch.softmax(logits.float(), dim=-1)[:, HEDGE_IDS].sum(-1).cpu().numpy()
        for pos in range(20, len(hp) - 1):
            if hp[pos] >= 0.05 and len(hedge_pos) < args.max_hedge_pos * 4:
                hedge_pos.append({"s": s_idx, "pos": pos, "prefix": ids[:pos + 1], "hp": float(hp[pos])})
            elif hp[pos] < 0.01 and len(nonhedge_pos) < args.max_nonhedge_pos * 2:
                nonhedge_pos.append({"s": s_idx, "pos": pos, "prefix": ids[:pos + 1]})
        if len(hedge_pos) >= args.max_hedge_pos * 4 and len(nonhedge_pos) >= args.max_nonhedge_pos * 2:
            break
    hedge_pos.sort(key=lambda x: -x["hp"])
    hedge_pos = hedge_pos[:args.max_hedge_pos]
    random.shuffle(nonhedge_pos); nonhedge_pos = nonhedge_pos[:args.max_nonhedge_pos]
    print(f"  hedge positions: {len(hedge_pos)}  non-hedge positions: {len(nonhedge_pos)}", flush=True)
    if len(hedge_pos) < 10 or len(nonhedge_pos) < 10:
        print("Too few positions — abort."); return

    # ---- cache vanilla resid_post at all layers for both sets ----
    def cache_resids(positions):
        out = []  # list of {L: tensor[d]}
        for c in positions:
            with torch.no_grad():
                _, cache = van_model.run_with_cache(torch.tensor([c["prefix"]], device="cuda"),
                                                    names_filter=lambda n: n.endswith("hook_resid_post"))
            out.append({L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].float().clone() for L in range(n_layers)})
        return out
    print("\n[vanilla] caching resids at hedge + non-hedge positions ...", flush=True)
    t0 = time.time()
    hedge_resids = cache_resids(hedge_pos)
    nonhedge_resids = cache_resids(nonhedge_pos)
    print(f"  cached in {time.time()-t0:.0f}s", flush=True)

    # ---- hedge direction per layer (diff of means, unit-normalized) ----
    hedge_dir = {}
    for L in range(n_layers):
        mh = torch.stack([r[L] for r in hedge_resids]).mean(0)
        mn = torch.stack([r[L] for r in nonhedge_resids]).mean(0)
        d = mh - mn
        nrm = d.norm()
        hedge_dir[L] = (d / nrm) if nrm > 0 else torch.zeros_like(d)

    # ---- baselines (vanilla, at hedge positions) ----
    def mass(p, ids):
        return float(p[ids].sum())
    base = {"hedge": [], "cert": [], "fp": []}
    base_p_hedge_logit_mean = []
    print("\n[vanilla] baselines at hedge positions ...", flush=True)
    for c in hedge_pos:
        with torch.no_grad():
            p = torch.softmax(van_model(torch.tensor([c["prefix"]], device="cuda"))[0, -1, :].float(), dim=-1)
        base["hedge"].append(mass(p, HEDGE_IDS)); base["cert"].append(mass(p, CERT_IDS)); base["fp"].append(mass(p, FP_IDS))
    base = {k: np.array(v) for k, v in base.items()}
    print(f"  baseline hedge={base['hedge'].mean():.4f} cert={base['cert'].mean():.4f} fp={base['fp'].mean():.4f}", flush=True)

    # ---- directional-ablation runner ----
    def directional_ablate(layers, direction_per_layer):
        """direction_per_layer: dict L->unit tensor[d]. Returns mean hedge/cert/fp after ablation, + per-pos drop frac."""
        h_after, c_after, f_after = [], [], []
        for c in hedge_pos:
            def make_hook(L):
                d = direction_per_layer[L].to("cuda")
                def hook(resid, hook):
                    x = resid[:, -1, :].float()
                    proj = (x @ d).unsqueeze(-1) * d  # (batch,1)*d -> (batch,d)
                    resid[:, -1, :] = (x - proj).to(resid.dtype)
                    return resid
                return hook
            hooks = [(f"blocks.{L}.hook_resid_post", make_hook(L)) for L in layers]
            with torch.no_grad():
                p = torch.softmax(van_model.run_with_hooks(torch.tensor([c["prefix"]], device="cuda"),
                                                           fwd_hooks=hooks)[0, -1, :].float(), dim=-1)
            h_after.append(mass(p, HEDGE_IDS)); c_after.append(mass(p, CERT_IDS)); f_after.append(mass(p, FP_IDS))
        h_after = np.array(h_after)
        drop_frac = (base["hedge"] - h_after) / np.clip(base["hedge"], 1e-6, None)
        return {"layers": list(layers),
                "hedge_before": float(base["hedge"].mean()), "hedge_after": float(h_after.mean()),
                "hedge_drop_frac": float(drop_frac.mean()),
                "cert_before": float(base["cert"].mean()), "cert_after": float(np.mean(c_after)), "cert_delta": float(np.mean(c_after) - base["cert"].mean()),
                "fp_before": float(base["fp"].mean()), "fp_after": float(np.mean(f_after)), "fp_delta": float(np.mean(f_after) - base["fp"].mean())}

    results = {"n_hedge_pos": len(hedge_pos), "n_nonhedge_pos": len(nonhedge_pos),
               "baseline": {k: float(v.mean()) for k, v in base.items()},
               "band": list(args.band), "early_band": list(args.early_band)}

    print("\n=== Directional ablation of the HEDGE direction ===", flush=True)
    r_band = directional_ablate(args.band, hedge_dir); results["ablate_hedge_dir_band"] = r_band
    print(f"  band {args.band}: hedge {r_band['hedge_before']:.4f} -> {r_band['hedge_after']:.4f}  (drop {r_band['hedge_drop_frac']*100:.1f}%)  certΔ={r_band['cert_delta']:+.4f} fpΔ={r_band['fp_delta']:+.4f}", flush=True)
    r_early = directional_ablate(args.early_band, hedge_dir); results["ablate_hedge_dir_early"] = r_early
    print(f"  early {args.early_band}: hedge drop {r_early['hedge_drop_frac']*100:.1f}%", flush=True)
    r_all = directional_ablate(list(range(n_layers)), hedge_dir); results["ablate_hedge_dir_all"] = r_all
    print(f"  ALL layers: hedge drop {r_all['hedge_drop_frac']*100:.1f}%  (this is the ceiling)", flush=True)
    # late-half control: layers in [n_layers//2 .. n_layers-1] (to show band is better than 'any late layers')
    late_half = list(range(n_layers // 2, n_layers))
    r_latehalf = directional_ablate(late_half, hedge_dir); results["ablate_hedge_dir_late_half"] = r_latehalf
    print(f"  late-half {late_half[0]}..{late_half[-1]}: hedge drop {r_latehalf['hedge_drop_frac']*100:.1f}%", flush=True)

    print("\n=== Directional ablation of a RANDOM direction at the band (control) ===", flush=True)
    rng = np.random.default_rng(args.seed)
    rand_drops = []
    results["ablate_random_dir_band"] = []
    for ri in range(args.n_random_dirs):
        rd = {L: torch.tensor(rng.standard_normal(d_model).astype(np.float32)) for L in args.band}
        for L in args.band:
            rd[L] = rd[L] / rd[L].norm()
        rr = directional_ablate(args.band, rd)
        rand_drops.append(rr["hedge_drop_frac"]); results["ablate_random_dir_band"].append(rr)
        print(f"  random dir {ri}: hedge drop {rr['hedge_drop_frac']*100:.1f}%", flush=True)
    results["random_dir_band_drop_mean"] = float(np.mean(rand_drops)); results["random_dir_band_drop_std"] = float(np.std(rand_drops))

    # ---- symmetric NOISING patch: SFT resid into vanilla at band ----
    print("\n=== SYMMETRIC NOISING: patch SFT resid into vanilla at the band ===", flush=True)
    del van_model; torch.cuda.empty_cache()
    sft_model, _ = load_hooked(SFT_PATH)
    sft_band_resid = []
    for c in hedge_pos:
        with torch.no_grad():
            _, cache = sft_model.run_with_cache(torch.tensor([c["prefix"]], device="cuda"),
                                                names_filter=lambda n: n.endswith("hook_resid_post"))
        sft_band_resid.append({L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].clone() for L in args.band})
    del sft_model; torch.cuda.empty_cache()
    van_model, _ = load_hooked(VANILLA_PATH)
    h_after_noise = []
    for i, c in enumerate(hedge_pos):
        def make_hook(L, i=i):
            repl = sft_band_resid[i][L]
            def hook(resid, hook):
                resid[:, -1, :] = repl.to(resid.dtype)
                return resid
            return hook
        hooks = [(f"blocks.{L}.hook_resid_post", make_hook(L)) for L in args.band]
        with torch.no_grad():
            p = torch.softmax(van_model.run_with_hooks(torch.tensor([c["prefix"]], device="cuda"), fwd_hooks=hooks)[0, -1, :].float(), dim=-1)
        h_after_noise.append(mass(p, HEDGE_IDS))
    h_after_noise = np.array(h_after_noise)
    noise_drop = float(((base["hedge"] - h_after_noise) / np.clip(base["hedge"], 1e-6, None)).mean())
    results["noising_sft_into_vanilla_band"] = {"hedge_before": float(base["hedge"].mean()), "hedge_after": float(h_after_noise.mean()), "hedge_drop_frac": noise_drop}
    print(f"  noising: hedge {base['hedge'].mean():.4f} -> {h_after_noise.mean():.4f}  (drop {noise_drop*100:.1f}%)", flush=True)

    # ---- verdict ----
    print("\n=== exp_q2 VERDICT (minimal 0.94 hedge-channel rigor, FIXED) ===", flush=True)
    bd = r_band["hedge_drop_frac"]; ad = r_all["hedge_drop_frac"]; ed = r_early["hedge_drop_frac"]
    rd = results["random_dir_band_drop_mean"]; lhd = r_latehalf["hedge_drop_frac"]
    cert_d = abs(r_band["cert_delta"]); fp_d = abs(r_band["fp_delta"])
    notes = []
    notes.append(f"ablate hedge-dir @ band: hedge drops {bd*100:.1f}%  (ceiling: ALL layers = {ad*100:.1f}%; late-half = {lhd*100:.1f}%; early = {ed*100:.1f}%)")
    notes.append(f"ablate RANDOM dir @ band: hedge drops {rd*100:.1f}%±{results['random_dir_band_drop_std']*100:.1f}")
    notes.append(f"specificity: band ablation Δcert={r_band['cert_delta']:+.4f} Δfp={r_band['fp_delta']:+.4f} (should be small)")
    notes.append(f"symmetric noising (SFT->vanilla band): hedge drops {noise_drop*100:.1f}%")
    sel_ok = bd > 3 * max(rd, 0.005) and bd > 1.5 * max(ed, 0.005)        # band >> random dir and >> early band
    eff_ok = bd > 0.30                                                     # band ablation kills >=30% of hedging
    cap_ok = ad > 0 and bd >= 0.5 * ad                                     # band captures >=half of the all-layer ceiling
    # hedge-specific: the ABSOLUTE drop in hedge mass dwarfs the absolute change in cert/fp mass
    abs_hedge_drop = r_band["hedge_before"] - r_band["hedge_after"]
    spec_ok = abs_hedge_drop > 5 * max(cert_d, fp_d, 1e-4)
    sym_ok = noise_drop > 0.15
    n_ok = sum([sel_ok, eff_ok, cap_ok, spec_ok, sym_ok])
    if n_ok >= 4:
        verdict = f"STRONG ({n_ok}/5): ablating the hedge direction at the ~0.94 band selectively and substantially suppresses hedge-token production, captures most of the all-layer ceiling, is hedge-specific, and the noising patch agrees. With cross-family r=0.88 + per-head negative, this is enough for the 'localized hedge read-out channel' claim."
    elif n_ok == 3:
        verdict = f"OK ({n_ok}/5): the localized-channel claim mostly holds; report effect sizes + the weak leg honestly."
    else:
        verdict = f"WEAK ({n_ok}/5): the ~0.94 band is not cleanly THE hedge read-out — scope the claim to 'the hedge direction is decodable/readable from the late-layer residual; per-layer patching peaks near depth 0.94' without an 'ablation kills it' claim, or investigate further."
    print(f"  {verdict}", flush=True)
    for n in notes:
        print(f"    - {n}", flush=True)
    results["verdict"] = verdict; results["verdict_notes"] = notes
    with open(OUT_DIR / "exp_q2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR}/exp_q2.json", flush=True)


if __name__ == "__main__":
    main()
