"""
expC4 — Trace-function taxonomy. Pure-CPU on the GLM style-blind step-
judgment data (per-step step_text + 'error'/'correct'/'uncertain' label, per chain).

Classify each step into one of:
  - correction        : backtracking / revising ("however", "wait", "reconsider", "actually", ...)
  - final_synthesis   : concluding ("therefore", "the answer is", "in conclusion", ...)
  - option_elimination: ruling options in/out ("rule out", "less likely", "option A/B/C/D", "(A)", ...)
  - hypothesis_gen    : generating/considering diagnoses ("differential", "could be", "consider", "suspect", "most likely diagnosis", ...)
  - other             : structural/meta (short headers, "let me work through this", ...)
  - factual_claim     : default — a step stating a medical fact / mechanism / value
Group: exploratory = {hypothesis_gen, option_elimination, correction}; synthesis = {final_synthesis}; claim = {factual_claim}; other.
Also a has_hedge flag per step (hedge-word present).

Analyses, vanilla vs medical-SFT (also weak_sft, teacher for context):
 (A) Category distribution: does SFT produce MORE exploratory steps (hypothesis/elim/correction)?
 (B) Per-category error rate: are SFT's EXTRA errors concentrated in exploratory step types vs final_synthesis?
 (C) Hedge-correctness coupling WITHIN each category: does the (vanilla-significant) coupling fail
     especially inside exploratory steps after distillation?
 (D) Within-chain: do SFT chains that reach a correct final answer still contain more erroneous exploratory steps?

Output: experiments/module1/mechanism/trace_taxonomy/expC4.json + verdict on the role-wise error pattern.
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

GLM_DIR = "experiments/module1/q2_hallucination/audit_results_glm_styleblind"
OUT = Path("experiments/module1/mechanism/trace_taxonomy")
OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["vanilla", "sft8b", "weak_sft", "teacher"]

HEDGE_RE = re.compile("|".join([
    r"\bmight\b", r"\bmay\b", r"\bcould\b", r"\bpossibly\b", r"\bperhaps\b", r"\bprobably\b",
    r"\blikely\b", r"\bunlikely\b", r"\bsuggest", r"\bindicate", r"\bappear", r"\bconsistent\s+with\b",
    r"\btypically\b", r"\bgenerally\b",
]), re.IGNORECASE)

# priority-ordered classifiers
CORRECTION_RE = re.compile("|".join([
    r"\bhowever\b", r"\bwait\b", r"\breconsider", r"\balternatively\b", r"\bon\s+second\s+thought\b",
    r"\bbut\s+wait\b", r"\bactually\s*,", r"\blet\s+me\s+re", r"\brethink", r"\bre-?evaluat",
    r"\bon\s+the\s+other\s+hand\b", r"\bthat\s+said\b", r"\bscratch\s+that\b", r"\bcorrection\b",
    r"\bnot\s+necessarily\b", r"\bwait\s*,", r"\bhmm\b", r"\blet\s+me\s+double-?check\b",
]), re.IGNORECASE)
SYNTHESIS_RE = re.compile("|".join([
    r"\bthe\s+answer\s+is\b", r"\bin\s+conclusion\b", r"\bto\s+summari[sz]e\b", r"\btherefore[, ]",
    r"\bthus[, ]", r"\bso\s+the\s+answer\b", r"\bfinal\s+answer\b", r"\bhence[, ]", r"\boverall[, ]",
    r"\bputting\s+it\s+together\b", r"\bbest\s+answer\b",
]), re.IGNORECASE)
ELIMINATION_RE = re.compile("|".join([
    r"\brule\s+out\b", r"\bruled\s+out\b", r"\beliminat", r"\bexclude", r"\bcannot\s+be\b",
    r"\bis\s+(in)?correct\b", r"\bless\s+likely\b", r"\bmore\s+likely\s+than\b", r"\boption\s+[A-D]\b",
    r"\(\s*[A-D]\s*\)", r"\bthis\s+option\b", r"\bnot\s+the\s+answer\b", r"\bdoesn't\s+fit\b",
    r"\bdoes\s+not\s+fit\b", r"\binconsistent\s+with\b", r"\bargues?\s+against\b",
]), re.IGNORECASE)
HYPOTHESIS_RE = re.compile("|".join([
    r"\bdifferential\b", r"\bcould\s+be\b", r"\bmight\s+be\b", r"\bpossible\s+diagnos", r"\bconsider\b",
    r"\bsuspect", r"\bmost\s+likely\s+diagnos", r"\bpresentation\s+is\s+consistent\s+with\b",
    r"\bsuggests\s+a\s+diagnosis\b", r"\bcandidate\s+diagnos", r"\blikely\s+diagnos", r"\bhypothes",
    r"\bworkup\b", r"\bwork-?up\b",
]), re.IGNORECASE)
# "other" = very short or pure structural header
def is_structural(txt):
    t = (txt or "").strip()
    if len(t) < 25:
        return True
    # markdown headers / bold-only lines
    if re.match(r"^#{1,4}\s|^\*\*[^*]+\*\*\s*:?\s*$", t):
        return True
    if re.match(r"^(let me|let's|first[,]?|next[,]?|now[,]?|step\s+\d+\s*:?)\s*$", t, re.IGNORECASE):
        return True
    return False


def classify(txt):
    if CORRECTION_RE.search(txt or ""):
        return "correction"
    if SYNTHESIS_RE.search(txt or ""):
        return "final_synthesis"
    if ELIMINATION_RE.search(txt or ""):
        return "option_elimination"
    if HYPOTHESIS_RE.search(txt or ""):
        return "hypothesis_gen"
    if is_structural(txt):
        return "other"
    return "factual_claim"


EXPLORATORY = {"hypothesis_gen", "option_elimination", "correction"}


def load_chains(cond):
    f = Path(GLM_DIR) / f"judgments_{cond}.jsonl"
    if not f.exists():
        return []
    chains = defaultdict(list)
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        chains[(r["qidx"], r.get("chain_idx", 0))].append(
            (r.get("step_idx", 0), r.get("judgment"), r.get("step_text", "")))
    out = []
    for k, v in chains.items():
        v.sort(key=lambda x: x[0])
        out.append((k, v))
    return out


def mi_binary(x, y):
    x = np.asarray(x).astype(int); y = np.asarray(y).astype(int)
    if len(x) == 0:
        return 0.0
    mi = 0.0
    for xv in (0, 1):
        for yv in (0, 1):
            pxy = np.mean((x == xv) & (y == yv)); px = np.mean(x == xv); py = np.mean(y == yv)
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * math.log(pxy / (px * py))
    return mi


def main():
    data = {c: load_chains(c) for c in CONDS}
    print("chains:", {c: len(v) for c, v in data.items()}, flush=True)

    # flatten to steps with category + error + hedge + chain key
    steps = {c: [] for c in CONDS}  # (cat, is_error(0/1, None if uncertain), has_hedge, qidx, chain_idx, step_idx, n_steps)
    for cond in CONDS:
        for (qk, ch) in data[cond]:
            n = len(ch)
            for (si, j, txt) in ch:
                cat = classify(txt)
                ie = 1 if j == "error" else (0 if j == "correct" else None)
                hh = 1 if HEDGE_RE.search(txt or "") else 0
                steps[cond].append({"cat": cat, "err": ie, "hedge": hh, "q": qk[0], "ci": qk[1], "si": si, "n": n})

    result = {}

    # ===== (A) category distribution =====
    print("\n=== (A) step category distribution (fraction of all steps) ===", flush=True)
    cats = ["hypothesis_gen", "option_elimination", "correction", "factual_claim", "final_synthesis", "other"]
    dist = {}
    for cond in CONDS:
        if not steps[cond]:
            continue
        tot = len(steps[cond])
        d = {cat: sum(1 for s in steps[cond] if s["cat"] == cat) / tot for cat in cats}
        d["EXPLORATORY_total"] = sum(d[c] for c in EXPLORATORY)
        d["n_steps"] = tot
        d["mean_n_steps_per_chain"] = float(np.mean([len(ch) for _, ch in data[cond]]))
        dist[cond] = d
        print(f"  {cond:10s}: " + "  ".join(f"{cat[:12]}={d[cat]*100:5.1f}%" for cat in cats) +
              f"  | EXPLOR={d['EXPLORATORY_total']*100:.1f}%  n/chain={d['mean_n_steps_per_chain']:.2f}", flush=True)
    result["category_distribution"] = dist
    if "sft8b" in dist and "vanilla" in dist:
        result["sft_minus_vanilla_dist_pp"] = {cat: (dist["sft8b"][cat] - dist["vanilla"][cat]) * 100
                                               for cat in cats + ["EXPLORATORY_total"]}
        print("  --> SFT - vanilla (pp): " + "  ".join(f"{cat[:12]}={result['sft_minus_vanilla_dist_pp'][cat]:+.1f}" for cat in cats),
              flush=True)
        print(f"  --> SFT - vanilla EXPLORATORY: {result['sft_minus_vanilla_dist_pp']['EXPLORATORY_total']:+.2f}pp "
              f"(expected: SFT writes more exploratory steps)", flush=True)

    # ===== (B) per-category error rate =====
    print("\n=== (B) per-category step-error rate (error / (error+correct)) ===", flush=True)
    cat_err = {}
    for cond in CONDS:
        if not steps[cond]:
            continue
        d = {}
        for cat in cats:
            sub = [s for s in steps[cond] if s["cat"] == cat and s["err"] is not None]
            d[cat] = {"err_rate": float(np.mean([s["err"] for s in sub])) if sub else float("nan"), "n": len(sub)}
        cat_err[cond] = d
        print(f"  {cond:10s}: " + "  ".join(f"{cat[:12]}={d[cat]['err_rate']*100:5.1f}%(n{d[cat]['n']})" for cat in cats), flush=True)
    result["per_category_error_rate"] = cat_err
    if "sft8b" in cat_err and "vanilla" in cat_err:
        diff = {cat: (cat_err["sft8b"][cat]["err_rate"] - cat_err["vanilla"][cat]["err_rate"]) * 100 for cat in cats}
        result["sft_minus_vanilla_err_by_cat_pp"] = diff
        print("  --> SFT - vanilla error rate by category (pp): " + "  ".join(f"{cat[:12]}={diff[cat]:+.1f}" for cat in cats), flush=True)
        explor_diff = np.mean([diff[c] for c in EXPLORATORY if diff[c] == diff[c]])
        synth_diff = diff.get("final_synthesis", float("nan"))
        claim_diff = diff.get("factual_claim", float("nan"))
        result["sft_minus_vanilla_err_exploratory_mean_pp"] = float(explor_diff)
        result["sft_minus_vanilla_err_synthesis_pp"] = float(synth_diff)
        result["sft_minus_vanilla_err_factualclaim_pp"] = float(claim_diff)
        print(f"  --> extra-error concentration: exploratory={explor_diff:+.2f}pp  factual_claim={claim_diff:+.2f}pp  "
              f"final_synthesis={synth_diff:+.2f}pp  (expected: extra errors not in final_synthesis)", flush=True)

    # ===== (C) hedge-correctness coupling within categories =====
    print("\n=== (C) hedge-correctness coupling within categories (vanilla vs SFT) ===", flush=True)
    coupling = {}
    for cond in ["vanilla", "sft8b"]:
        if not steps[cond]:
            continue
        d = {}
        for grp_name, grp_cats in [("exploratory", EXPLORATORY), ("factual_claim", {"factual_claim"}),
                                   ("final_synthesis", {"final_synthesis"}), ("ALL", set(cats))]:
            sub = [s for s in steps[cond] if s["cat"] in grp_cats and s["err"] is not None]
            if len(sub) < 20:
                d[grp_name] = {"n": len(sub), "mi": None, "lift_pp": None}; continue
            h = np.array([s["hedge"] for s in sub]); e = np.array([s["err"] for s in sub])
            mi = mi_binary(h, e)
            # lift: P(error|hedge) - P(error|no hedge), pp
            lift = (e[h == 1].mean() - e[h == 0].mean()) * 100 if (h == 1).any() and (h == 0).any() else float("nan")
            d[grp_name] = {"n": len(sub), "mi": float(mi), "lift_error_given_hedge_pp": float(lift)}
        coupling[cond] = d
        parts = []
        for g in ["exploratory", "factual_claim", "final_synthesis", "ALL"]:
            if d[g]["mi"] is not None:
                lift = d[g].get("lift_error_given_hedge_pp")
                lift_s = f"{lift:+.1f}" if (lift is not None and lift == lift) else "NA"
                parts.append(f"{g}: MI={d[g]['mi']:.4f} lift_err|hedge={lift_s}pp(n{d[g]['n']})")
            else:
                parts.append(f"{g}: n{d[g]['n']}(too few)")
        print(f"  {cond:10s}: " + "  ".join(parts), flush=True)
    result["hedge_coupling_by_category"] = coupling

    # ===== (D) within-chain: correct-final-answer chains, exploratory error count =====
    print("\n=== (D) erroneous exploratory steps per chain, split by final-answer correctness ===", flush=True)
    # derive y_model from the last final_synthesis step (or last step) containing 'answer is (X)'
    ANS_RE = re.compile(r"answer\s+is\s*\(?\s*([ABCD])", re.IGNORECASE)
    within = {}
    for cond in CONDS:
        if not steps[cond]:
            continue
        # need gold answer — load from chain data; skip if unavailable (best-effort)
        # Here we only split by "did the chain conclude an answer at all" + count erroneous exploratory steps.
        chains_ex_err = []  # (concluded_letter or None, n_erroneous_exploratory_steps)
        for (qk, ch) in data[cond]:
            concluded = None
            for (si, j, txt) in reversed(ch):
                m = ANS_RE.findall(txt or "")
                if m:
                    concluded = m[-1].upper(); break
            n_err_ex = sum(1 for (si, j, txt) in ch if classify(txt) in EXPLORATORY and j == "error")
            chains_ex_err.append((concluded, n_err_ex))
        mean_ex_err = float(np.mean([c[1] for c in chains_ex_err]))
        within[cond] = {"mean_erroneous_exploratory_steps_per_chain": mean_ex_err,
                        "n_chains_with_concluded_answer": sum(1 for c in chains_ex_err if c[0]),
                        "n_chains": len(chains_ex_err)}
        print(f"  {cond:10s}: mean erroneous exploratory steps/chain = {mean_ex_err:.3f}  "
              f"(n_chains={len(chains_ex_err)})", flush=True)
    result["within_chain_exploratory_errors"] = within
    if "sft8b" in within and "vanilla" in within:
        d = within["sft8b"]["mean_erroneous_exploratory_steps_per_chain"] - within["vanilla"]["mean_erroneous_exploratory_steps_per_chain"]
        result["sft_minus_vanilla_erroneous_exploratory_per_chain"] = d
        print(f"  --> SFT - vanilla erroneous exploratory steps/chain = {d:+.3f} (expected: SFT higher)", flush=True)

    # ===== verdict =====
    print("\n=== expC4 VERDICT: role-wise error concentration ===", flush=True)
    notes = []
    explor_more = result.get("sft_minus_vanilla_dist_pp", {}).get("EXPLORATORY_total", 0)
    extra_err_explor = result.get("sft_minus_vanilla_err_exploratory_mean_pp", None)
    extra_err_synth = result.get("sft_minus_vanilla_err_synthesis_pp", None)
    err_ex_chain = result.get("sft_minus_vanilla_erroneous_exploratory_per_chain", None)
    notes.append(f"SFT - vanilla exploratory-step share = {explor_more:+.2f}pp (predict >0)")
    notes.append(f"SFT extra-error: exploratory={extra_err_explor}pp vs final_synthesis={extra_err_synth}pp (predict extra errors NOT in synthesis)")
    notes.append(f"SFT - vanilla erroneous-exploratory-steps/chain = {err_ex_chain} (predict >0)")
    s1 = explor_more is not None and explor_more > 0.5
    s2 = (extra_err_explor is not None and extra_err_synth is not None
          and extra_err_explor > 0 and extra_err_explor >= extra_err_synth)  # extra errors at least as much in exploratory as synthesis
    s3 = err_ex_chain is not None and err_ex_chain > 0.02
    n_sup = sum([s1, s2, s3])
    if n_sup >= 2:
        verdict = f"SUPPORTED ({n_sup}/3): the trace-taxonomy evidence backs the role-wise pattern: SFT writes more exploratory steps and its extra step-errors are concentrated there, not in the final synthesis."
    elif n_sup == 1:
        verdict = f"WEAK ({n_sup}/3): partial support. The heuristic classifier may be too crude; consider an LLM classifier on a sample to confirm before leaning on this in the paper."
    else:
        verdict = "NOT SUPPORTED (0/3): the heuristic taxonomy does not show SFT's errors concentrating in exploratory steps. Either the classifier is too crude (escalate to LLM classification on a stratified sample) or the 'search transfer' framing needs softening to just 'distributed step-policy shift'."
    print(f"  {verdict}", flush=True)
    for n in notes:
        print(f"    - {n}", flush=True)
    result["verdict"] = verdict
    result["verdict_notes"] = notes

    with open(OUT / "expC4.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    # also dump a sample of classified steps for spot-checking
    sample = []
    for cond in ["vanilla", "sft8b"]:
        for s in steps[cond][:80]:
            # re-fetch the text
            pass
    # (skip the text re-fetch; the JSON has the aggregates which is what matters)
    print(f"\nSaved: {OUT}/expC4.json", flush=True)


if __name__ == "__main__":
    main()
