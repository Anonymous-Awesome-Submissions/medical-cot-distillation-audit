#!/usr/bin/env python3
"""R003 + R004: held-out hedge-direction ablation + alternative direction estimator.

R003 (B6): split hedge-divergence positions 50 train / 30 test (and matching split
of non-hedge positions). Fit difference-of-means hedge direction on TRAIN only,
evaluate ablation magnitude on TEST only. Goal: confirm 98.7% isn't an estimation
/ evaluation overlap artifact.

R004 (B7): also fit a SUPERVISED LINEAR CLASSIFIER (logistic regression) hedge vs
non-hedge on TRAIN, take its weight direction, evaluate ablation magnitude on TEST.
Compares to difference-of-means.

Output: experiments/module1/mechanism/circuit_rigor/exp_q2_heldout_altdir.json
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch

VANILLA_PATH = "Qwen/Qwen3-8B"
COT_DIR = "experiments/module1/q2_hallucination"
OUT_DIR = Path("experiments/module1/mechanism/circuit_rigor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEDGE_WORDS = [" might", " may", " could", " possibly", " perhaps", " probably", " likely",
               " unlikely", " suggest", " suggests", " indicate", " indicates", " appears",
               " appear", " consistent", " typically", " generally"]


def tok_ids(tokenizer, words):
    ids = set()
    for w in words:
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


def load_hooked():
    print(f"Loading {VANILLA_PATH} ...", flush=True)
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(VANILLA_PATH, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(VANILLA_PATH, torch_dtype=torch.bfloat16,
                                              trust_remote_code=True)
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-8B", hf_model=hf, tokenizer=tokenizer, device="cuda",
        dtype=torch.bfloat16, fold_ln=False, center_writing_weights=False, center_unembed=False)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hedge-train", type=int, default=50)
    ap.add_argument("--n-hedge-test", type=int, default=30)
    ap.add_argument("--n-nonhedge-train", type=int, default=100)
    ap.add_argument("--n-nonhedge-test", type=int, default=60)
    ap.add_argument("--band", type=int, nargs="+", default=[31, 32, 33])
    ap.add_argument("--seed", type=int, default=20260517)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    model, tokenizer = load_hooked()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    HEDGE_IDS = torch.tensor(tok_ids(tokenizer, HEDGE_WORDS), device="cuda")
    print(f"n_layers={n_layers} d_model={d_model} hedge_tok={len(HEDGE_IDS)}", flush=True)

    # ---- collect positions ----
    samples = []
    with open(f"{COT_DIR}/vanilla_cot_medqa_test500.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))

    n_hedge_need = args.n_hedge_train + args.n_hedge_test
    n_nonhedge_need = args.n_nonhedge_train + args.n_nonhedge_test
    hedge_pos, nonhedge_pos = [], []
    for s_idx, s in enumerate(samples):
        text = s.get("text") or s.get("cot") or ""
        if not text: continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 220: ids = ids[:220]
        if len(ids) < 25: continue
        with torch.no_grad():
            logits = model(torch.tensor([ids], device="cuda"))[0]
            hp = torch.softmax(logits.float(), -1)[:, HEDGE_IDS].sum(-1).cpu().numpy()
        for pos in range(20, len(hp) - 1):
            if hp[pos] >= 0.05 and len(hedge_pos) < n_hedge_need * 3:
                hedge_pos.append({"prefix": ids[:pos + 1], "hp": float(hp[pos])})
            elif hp[pos] < 0.01 and len(nonhedge_pos) < n_nonhedge_need * 2:
                nonhedge_pos.append({"prefix": ids[:pos + 1]})
        if len(hedge_pos) >= n_hedge_need * 3 and len(nonhedge_pos) >= n_nonhedge_need * 2:
            break

    hedge_pos.sort(key=lambda x: -x["hp"])
    hedge_pos = hedge_pos[:n_hedge_need]
    random.shuffle(nonhedge_pos); nonhedge_pos = nonhedge_pos[:n_nonhedge_need]
    print(f"hedge {len(hedge_pos)}  nonhedge {len(nonhedge_pos)}", flush=True)

    # Split
    random.shuffle(hedge_pos); random.shuffle(nonhedge_pos)
    hedge_train = hedge_pos[:args.n_hedge_train]
    hedge_test  = hedge_pos[args.n_hedge_train:args.n_hedge_train + args.n_hedge_test]
    nh_train    = nonhedge_pos[:args.n_nonhedge_train]
    nh_test     = nonhedge_pos[args.n_nonhedge_train:args.n_nonhedge_train + args.n_nonhedge_test]
    print(f"split: hedge {len(hedge_train)}/{len(hedge_test)}  nonhedge {len(nh_train)}/{len(nh_test)}", flush=True)

    # ---- cache resids at all band layers + last token ----
    def cache_band_resids(positions):
        out = []
        for c in positions:
            with torch.no_grad():
                _, cache = model.run_with_cache(torch.tensor([c["prefix"]], device="cuda"),
                                                names_filter=lambda n: n.endswith("hook_resid_post"))
            out.append({L: cache[f"blocks.{L}.hook_resid_post"][0, -1, :].float().clone()
                        for L in args.band})
        return out
    print("Caching resids at band layers ...", flush=True)
    t0 = time.time()
    rt_hedge_train = cache_band_resids(hedge_train)
    rt_hedge_test  = cache_band_resids(hedge_test)
    rt_nh_train    = cache_band_resids(nh_train)
    rt_nh_test     = cache_band_resids(nh_test)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    # ---- direction (i): difference-of-means, fit on TRAIN ----
    dir_diffmean = {}
    for L in args.band:
        mh = torch.stack([r[L] for r in rt_hedge_train]).mean(0)
        mn = torch.stack([r[L] for r in rt_nh_train]).mean(0)
        d = mh - mn
        dir_diffmean[L] = (d / d.norm()) if d.norm() > 0 else torch.zeros_like(d)

    # ---- direction (ii): supervised logistic regression weight, fit on TRAIN ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    dir_logreg = {}
    for L in args.band:
        X = torch.stack([r[L] for r in rt_hedge_train] + [r[L] for r in rt_nh_train]).cpu().numpy()
        y = np.array([1] * len(rt_hedge_train) + [0] * len(rt_nh_train))
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=args.seed).fit(Xs, y)
        # weight is in standardized space; project back: w * (1/scale)
        w = clf.coef_[0] / sc.scale_
        w = w / np.linalg.norm(w)
        dir_logreg[L] = torch.tensor(w, dtype=torch.float32)

    # ---- baseline hedge prob on TEST positions ----
    def hedge_mass(prefix):
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor([prefix], device="cuda"))[0, -1, :].float(), dim=-1)
        return float(p[HEDGE_IDS].sum())
    print("Baseline hedge mass on TEST ...", flush=True)
    base_test = np.array([hedge_mass(c["prefix"]) for c in hedge_test])
    print(f"  test baseline hedge mean: {base_test.mean():.4f}", flush=True)

    # ---- ablation runner ----
    def ablate(layers, direction_per_layer, positions):
        h_after = []
        for c in positions:
            def make_hook(L):
                d = direction_per_layer[L].to("cuda")
                def hook(resid, hook):
                    x = resid[:, -1, :].float()
                    proj = (x @ d).unsqueeze(-1) * d
                    resid[:, -1, :] = (x - proj).to(resid.dtype)
                    return resid
                return hook
            hooks = [(f"blocks.{L}.hook_resid_post", make_hook(L)) for L in layers]
            with torch.no_grad():
                p = torch.softmax(model.run_with_hooks(torch.tensor([c["prefix"]], device="cuda"),
                                                      fwd_hooks=hooks)[0, -1, :].float(), dim=-1)
            h_after.append(float(p[HEDGE_IDS].sum()))
        h_after = np.array(h_after)
        return float(h_after.mean()), float(((base_test - h_after) / np.clip(base_test, 1e-6, None)).mean())

    print("\nHeld-out ablation: difference-of-means direction (fit on TRAIN, eval on TEST)", flush=True)
    ha_dm, dr_dm = ablate(args.band, dir_diffmean, hedge_test)
    print(f"  hedge {base_test.mean():.4f} -> {ha_dm:.4f}  drop {dr_dm*100:.1f}%", flush=True)

    print("\nHeld-out ablation: supervised logistic-regression direction (fit on TRAIN, eval on TEST)", flush=True)
    ha_lr, dr_lr = ablate(args.band, dir_logreg, hedge_test)
    print(f"  hedge {base_test.mean():.4f} -> {ha_lr:.4f}  drop {dr_lr*100:.1f}%", flush=True)

    # ---- for full picture: also do random direction baseline on TEST ----
    print("\nRandom-direction control on TEST", flush=True)
    rng = np.random.default_rng(args.seed)
    rand_drops = []
    for ri in range(5):
        rd = {L: torch.tensor(rng.standard_normal(d_model).astype(np.float32)) for L in args.band}
        for L in args.band: rd[L] = rd[L] / rd[L].norm()
        _, dr = ablate(args.band, rd, hedge_test)
        rand_drops.append(dr)
    print(f"  random dir test drop: {np.mean(rand_drops)*100:.2f} ± {np.std(rand_drops)*100:.2f}%", flush=True)

    # ---- per-layer agreement: cosine between diff-of-means and logreg directions ----
    cos = {}
    for L in args.band:
        a = dir_diffmean[L].cpu().numpy(); b = dir_logreg[L].cpu().numpy()
        cos[L] = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    out = {
        "config": {
            "n_hedge_train": args.n_hedge_train, "n_hedge_test": args.n_hedge_test,
            "n_nonhedge_train": args.n_nonhedge_train, "n_nonhedge_test": args.n_nonhedge_test,
            "band": list(args.band), "seed": args.seed,
        },
        "baseline_hedge_test_mean": float(base_test.mean()),
        "ablate_diffmean_heldout": {"hedge_after": ha_dm, "drop_frac": dr_dm},
        "ablate_logreg_heldout":   {"hedge_after": ha_lr, "drop_frac": dr_lr},
        "random_dir_test_drop": {"mean": float(np.mean(rand_drops)), "std": float(np.std(rand_drops))},
        "cosine_diffmean_logreg_per_layer": cos,
        "original_full_set_drop": 0.987,
    }
    with open(OUT_DIR / "exp_q2_heldout_altdir.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'exp_q2_heldout_altdir.json'}")
    print(f"\nVerdict:")
    print(f"  Held-out diff-of-means drop: {dr_dm*100:.1f}%  (vs original full-set 98.7%)")
    print(f"  Held-out logreg drop:        {dr_lr*100:.1f}%")
    print(f"  Random baseline:             {np.mean(rand_drops)*100:.1f}%")
    if dr_dm > 0.85:
        print("  → Original 98.7% essentially holds out-of-sample.")
    elif dr_dm > 0.50:
        print("  → Original 98.7% was somewhat optimistic; held-out is still very large but report this number.")
    else:
        print("  → Original 98.7% was overfit; must report held-out as headline.")


if __name__ == "__main__":
    main()
