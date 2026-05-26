"""
Calibration and hedge-correctness robustness checks on EXISTING data (pure-CPU).

(1) Step-level mutual information between hedge presence and step correctness for
    vanilla / sft8b / weak_sft / teacher.
(2) Shuffle-floor: n=1000 permutations of (hedge, correct), build the null MI
    distribution; the claim requires observed vanilla MI > 3 x the 99th-pct floor.
(3) Bootstrap CI: n=1000 step-resamples, 95% CI on vanilla MI and sft8b MI;
    the "collapse" claim requires non-overlapping CIs AND the difference CI excludes 0.
(4) Per-item curve: step error rate vs hedge-density bin, vanilla vs sft8b — vanilla
    should be monotone (hedge tracks uncertainty), sft8b flat. Spearman rho test.
(5) ECE / Brier on the 64-chain self-consistency runs: confidence = majority-vote
    fraction, accuracy = is the majority answer correct. Compare vanilla vs SFT.

Output: experiments/module1/mechanism/cl3_robustness/cl3_robustness.json + verdict.
"""
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

GLM_DIR = "experiments/module1/q2_hallucination/audit_results_glm_styleblind"
CHAINS_VANILLA = "experiments/module1/test_chains_vanilla_8b_sc64"
CHAINS_SFT = "experiments/module1/test_chains_sft_8b_sc64"
OUT = Path("experiments/module1/mechanism/cl3_robustness")
OUT.mkdir(parents=True, exist_ok=True)

HEDGE_RE = re.compile("|".join([
    r"\bmight\b", r"\bmay\b", r"\bcould\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\bprobably\b", r"\blikely\b", r"\bunlikely\b", r"\bsuggest", r"\bindicate",
    r"\bappear", r"\bconsistent\s+with\b", r"\btypically\b", r"\bgenerally\b",
]), re.IGNORECASE)
ANS_RE = re.compile(r"answer\s+is\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE)


def mi_binary(x, y):
    """Exact MI (nats) of two binary arrays from the 2x2 joint."""
    x = np.asarray(x).astype(int); y = np.asarray(y).astype(int)
    n = len(x)
    if n == 0:
        return 0.0
    mi = 0.0
    for xv in (0, 1):
        for yv in (0, 1):
            pxy = np.mean((x == xv) & (y == yv))
            px = np.mean(x == xv); py = np.mean(y == yv)
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py))
    return mi


def hedge_density(s):
    s = s or ""
    nw = max(1, len(s.split()))
    return len(HEDGE_RE.findall(s)) / nw


def load_step_data():
    """Return {cond: list of (hedge_present:int, correct:int, hedge_dens:float)} from GLM style-blind audit."""
    out = {}
    for cond in ["vanilla", "sft8b", "weak_sft", "teacher"]:
        f = Path(GLM_DIR) / f"judgments_{cond}.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            j = r.get("judgment")
            if j not in ("error", "correct"):
                continue
            txt = r.get("step_text", "")
            rows.append((1 if HEDGE_RE.search(txt) else 0,
                         1 if j == "correct" else 0,
                         hedge_density(txt)))
        out[cond] = rows
    return out


def ece_brier_from_chains(chains_dir):
    """confidence = majority-vote fraction over 64 chains; accuracy = majority answer == gold."""
    confs, accs = [], []
    for f in sorted(glob.glob(f"{chains_dir}/*.jsonl")):
        if f.endswith(".lock"):
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            gold = d.get("correct_answer")
            sampled = d.get("sampled", [])
            if not gold or not sampled:
                continue
            votes = []
            for ch in sampled:
                m = ANS_RE.findall(ch.get("text", ""))
                if m:
                    votes.append(m[-1].upper())
            if not votes:
                continue
            from collections import Counter
            cnt = Counter(votes)
            maj, maj_n = cnt.most_common(1)[0]
            conf = maj_n / len(votes)
            confs.append(conf); accs.append(1 if maj == gold else 0)
    confs = np.array(confs); accs = np.array(accs)
    if len(confs) == 0:
        return None
    # ECE with 10 bins
    bins = np.linspace(0.25, 1.0001, 11)  # MCQ vote fraction in [0.25,1]
    ece = 0.0
    bininfo = []
    for i in range(len(bins) - 1):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if m.sum() == 0:
            continue
        avg_conf = confs[m].mean(); avg_acc = accs[m].mean()
        ece += (m.sum() / len(confs)) * abs(avg_conf - avg_acc)
        bininfo.append({"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "n": int(m.sum()),
                        "avg_conf": float(avg_conf), "avg_acc": float(avg_acc)})
    brier = float(np.mean((confs - accs) ** 2))
    return {"n_questions": int(len(confs)), "ece": float(ece), "brier": brier,
            "overall_conf": float(confs.mean()), "overall_acc": float(accs.mean()),
            "bins": bininfo}


def main():
    rng = np.random.default_rng(42)
    data = load_step_data()
    print("Loaded step data:", {k: len(v) for k, v in data.items()}, flush=True)

    result = {"step_level_mi": {}, "shuffle_floor": {}, "bootstrap_ci": {},
              "per_item_curve": {}, "ece_brier": {}}

    # ===== (1) step-level MI =====
    print("\n=== (1) step-level MI(hedge_present ; step_correct), nats ===", flush=True)
    for cond, rows in data.items():
        h = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows])
        mi = mi_binary(h, c)
        # lift: P(correct|hedge) - P(correct|no hedge), in pp
        lift = (c[h == 1].mean() - c[h == 0].mean()) * 100 if (h == 1).any() and (h == 0).any() else float("nan")
        result["step_level_mi"][cond] = {"n": len(rows), "p_hedge": float(h.mean()),
                                         "p_correct": float(c.mean()), "mi_nats": float(mi),
                                         "correct_lift_hedge_pp": float(lift)}
        print(f"  {cond:10s}: n={len(rows)} p_hedge={h.mean():.3f} p_correct={c.mean():.3f} "
              f"MI={mi:.5f} nats  correct_lift|hedge={lift:+.2f}pp", flush=True)
    if "vanilla" in result["step_level_mi"] and "sft8b" in result["step_level_mi"]:
        v = result["step_level_mi"]["vanilla"]["mi_nats"]; s = result["step_level_mi"]["sft8b"]["mi_nats"]
        result["step_level_mi"]["pct_drop_van_to_sft"] = (1 - s / v) * 100 if v > 0 else float("nan")
        print(f"  --> MI drop vanilla->sft8b = {(1 - s/v)*100:.1f}%", flush=True)

    # ===== (2) shuffle-floor =====
    print("\n=== (2) shuffle-floor (n=1000 permutations of (hedge,correct)) ===", flush=True)
    for cond in ["vanilla", "sft8b"]:
        if cond not in data:
            continue
        rows = data[cond]
        h = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows])
        obs = mi_binary(h, c)
        null = np.array([mi_binary(h, rng.permutation(c)) for _ in range(1000)])
        floor95 = float(np.percentile(null, 95)); floor99 = float(np.percentile(null, 99))
        ratio = obs / floor99 if floor99 > 0 else float("inf")
        result["shuffle_floor"][cond] = {"observed_mi": float(obs), "floor_p95": floor95,
                                         "floor_p99": floor99, "obs_over_floor99": ratio,
                                         "p_value_perm": float((null >= obs).mean())}
        print(f"  {cond:10s}: observed MI={obs:.5f}  shuffle p99={floor99:.5f}  "
              f"obs/floor99={ratio:.2f}  perm-p={(null >= obs).mean():.4f}", flush=True)

    # ===== (3) bootstrap CI =====
    print("\n=== (3) bootstrap CI on MI (n=1000 step-resamples) ===", flush=True)
    boot_mi = {}
    for cond in ["vanilla", "sft8b", "weak_sft", "teacher"]:
        if cond not in data:
            continue
        rows = data[cond]
        h = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows])
        n = len(rows)
        bs = np.array([mi_binary(h[idx], c[idx]) for idx in (rng.integers(0, n, n) for _ in range(1000))])
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        boot_mi[cond] = bs
        result["bootstrap_ci"][cond] = {"mi_mean": float(bs.mean()), "ci95": ci}
        print(f"  {cond:10s}: MI 95% CI = [{ci[0]:.5f}, {ci[1]:.5f}]", flush=True)
    if "vanilla" in boot_mi and "sft8b" in boot_mi:
        # paired-ish difference: resample steps independently, take difference of bootstrap means
        diff = boot_mi["vanilla"] - boot_mi["sft8b"]  # not paired but indicative
        dci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
        v_lo = result["bootstrap_ci"]["vanilla"]["ci95"][0]
        s_hi = result["bootstrap_ci"]["sft8b"]["ci95"][1]
        result["bootstrap_ci"]["van_minus_sft_diff_ci95"] = dci
        result["bootstrap_ci"]["ci_nonoverlap"] = bool(v_lo > s_hi)
        print(f"  van-sft diff 95% CI = [{dci[0]:.5f}, {dci[1]:.5f}]  (excludes 0: {dci[0] > 0})", flush=True)
        print(f"  vanilla CI lower ({v_lo:.5f}) > sft8b CI upper ({s_hi:.5f})? {v_lo > s_hi}", flush=True)

    # ===== (4) per-item curve: error rate vs hedge-density bin =====
    print("\n=== (4) step error rate vs hedge-density bin (vanilla vs sft8b) ===", flush=True)
    from scipy.stats import spearmanr
    for cond in ["vanilla", "sft8b"]:
        if cond not in data:
            continue
        rows = data[cond]
        dens = np.array([r[2] for r in rows]); err = np.array([1 - r[1] for r in rows])
        # bins: 0 (no hedge), then quartiles of positive density
        bins_out = []
        zero_mask = dens == 0
        if zero_mask.any():
            bins_out.append({"bin": "0", "n": int(zero_mask.sum()), "err_rate": float(err[zero_mask].mean())})
        pos = dens[dens > 0]
        if len(pos) > 8:
            qs = np.quantile(pos, [0, .25, .5, .75, 1.0])
            for i in range(4):
                m = (dens > qs[i] if i > 0 else dens > 0) & (dens <= qs[i + 1])
                if m.sum() > 0:
                    bins_out.append({"bin": f"({qs[i]:.4f},{qs[i+1]:.4f}]", "n": int(m.sum()),
                                     "err_rate": float(err[m].mean())})
        # spearman of (density, error) over all steps
        rho, p = spearmanr(dens, err)
        result["per_item_curve"][cond] = {"bins": bins_out, "spearman_rho": float(rho), "spearman_p": float(p)}
        print(f"  {cond:10s}: spearman(density, error) rho={rho:+.4f} p={p:.2e}", flush=True)
        for b in bins_out:
            print(f"    {b['bin']:>22s}: n={b['n']:5d}  err_rate={b['err_rate']*100:6.2f}%", flush=True)

    # ===== (5) ECE / Brier on 64-chain runs =====
    print("\n=== (5) ECE / Brier (self-consistency vote fraction as confidence) ===", flush=True)
    for tag, d in [("vanilla", CHAINS_VANILLA), ("sft", CHAINS_SFT)]:
        r = ece_brier_from_chains(d)
        result["ece_brier"][tag] = r
        if r:
            print(f"  {tag:8s}: N={r['n_questions']}  acc={r['overall_acc']*100:.1f}%  "
                  f"mean_conf={r['overall_conf']*100:.1f}%  ECE={r['ece']:.4f}  Brier={r['brier']:.4f}", flush=True)
        else:
            print(f"  {tag:8s}: NO DATA at {d}", flush=True)
    if result["ece_brier"].get("vanilla") and result["ece_brier"].get("sft"):
        de = result["ece_brier"]["sft"]["ece"] - result["ece_brier"]["vanilla"]["ece"]
        db = result["ece_brier"]["sft"]["brier"] - result["ece_brier"]["vanilla"]["brier"]
        result["ece_brier"]["delta_ece_sft_minus_vanilla"] = de
        result["ece_brier"]["delta_brier_sft_minus_vanilla"] = db
        print(f"  --> ΔECE (sft - vanilla) = {de:+.4f}   ΔBrier = {db:+.4f}  "
              f"({'WORSE post-SFT' if de > 0 else 'better post-SFT'})", flush=True)

    # ===== verdict =====
    print("\n=== calibration / hedge-MI robustness verdict ===", flush=True)
    notes = []
    sf_v = result["shuffle_floor"].get("vanilla", {})
    obs_over = sf_v.get("obs_over_floor99", 0)
    perm_p = sf_v.get("p_value_perm", 1.0)
    nonoverlap = result["bootstrap_ci"].get("ci_nonoverlap", False)
    diff_ci = result["bootstrap_ci"].get("van_minus_sft_diff_ci95", [0, 0])
    rho_v = result["per_item_curve"].get("vanilla", {}).get("spearman_rho", 0)
    rho_v_p = result["per_item_curve"].get("vanilla", {}).get("spearman_p", 1.0)
    rho_s = result["per_item_curve"].get("sft8b", {}).get("spearman_rho", 0)
    de = result["ece_brier"].get("delta_ece_sft_minus_vanilla", None)
    notes.append(f"vanilla MI / shuffle-floor99 = {obs_over:.2f} (need >3); perm-p = {perm_p:.4f}")
    notes.append(f"vanilla vs sft8b MI CIs non-overlapping = {nonoverlap}; diff CI = {diff_ci}")
    notes.append(f"per-item: vanilla spearman(density,error) rho={rho_v:+.3f} (p={rho_v_p:.2e}); sft8b rho={rho_s:+.3f}")
    notes.append(f"ΔECE (sft - vanilla) = {de}")
    mi_real = obs_over > 3 and perm_p < 0.05
    collapse_real = nonoverlap and diff_ci[0] > 0
    curve_ok = (rho_v_p < 0.05) and (abs(rho_v) > abs(rho_s))  # vanilla curve significant & steeper than sft
    ece_ok = (de is not None and de > 0)
    if mi_real and collapse_real and curve_ok and ece_ok:
        verdict = "GREEN: vanilla MI is above the permutation floor with non-overlapping bootstrap CIs, the per-item curve is steeper in vanilla, and ECE worsens after SFT."
    elif mi_real and collapse_real:
        verdict = "YELLOW: MI and its drop are present, but the per-item-curve and/or ECE corroboration is weak."
    elif not mi_real:
        verdict = "RED: vanilla MI is at the noise floor (not above the permutation floor), so the hedge-correctness coupling is unsupported here."
    else:
        verdict = "YELLOW/RED: the MI drop is not cleanly significant; treat it as suggestive and report a CI."
    print(f"  {verdict}", flush=True)
    for n in notes:
        print(f"    - {n}", flush=True)
    result["verdict"] = verdict
    result["verdict_notes"] = notes

    with open(OUT / "cl3_robustness.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {OUT}/cl3_robustness.json", flush=True)


if __name__ == "__main__":
    main()
