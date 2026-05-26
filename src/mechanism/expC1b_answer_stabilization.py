"""
expC1b — Within-CoT answer-stabilization / pre-commitment test.

The answer is not decodable at the pre-CoT token. The hypothesis tested here:
the model commits to its answer *progressively as it writes the CoT*, and the
distilled model commits *earlier* within the CoT.

Method (no probing/steering — pure behavioral, robust):
  For each MedQA test question, for model in {vanilla, medical-SFT}:
    1. Greedy-generate the full CoT.
    2. Parse into steps (split on "Step N:" markers, else on blank lines).
    3. For k = 0, 1, ..., min(n_steps, MAX_K):
         truncate the CoT at the end of step k, append the answer-prompt
         '\n\nThe answer is (', force-decode the next single token, record the
         letter (A/B/C/D, or None).
    4. Per question:
         - stabilization_step = smallest k such that the forced answer at k
           equals the forced answer at every k' in [k, n_used] (and is a valid letter).
         - agree_with_final[k] = (forced answer at k == the model's actual final greedy answer).
    5. Aggregate:
         - mean stabilization_step (and as a fraction of n_steps), vanilla vs SFT.
         - agree-with-final curve vs k (and vs k/n_steps), vanilla vs SFT.
         - stratify by difficulty (vanilla per-chain accuracy bin, from the 64-chain run).
         - prediction under "SFT commits earlier": SFT mean stabilization fraction < vanilla's,
           and SFT's agree-with-final curve rises earlier.

Output: experiments/module1/mechanism/answer_stabilization/expC1b_{tag}.json
A separate compare step joins the two tags.

Run array 0=vanilla, 1=sft.
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OUT_DIR = "experiments/module1/mechanism/answer_stabilization"
CHAINS_VANILLA = "experiments/module1/test_chains_vanilla_8b_sc64"

PROMPT_TMPL = """You are a medical expert. Answer the following clinical question by reasoning step by step.

Number each step as "Step 1:", "Step 2:", etc. For each step, state the specific medical fact or relationship you are using.

IMPORTANT: You MUST end your response with exactly "The answer is (X)." where X is A, B, C, or D.

Question: {question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}"""

ANS_RE = re.compile(r"answer\s+is\s*\(?\s*([ABCD])\s*\)?", re.IGNORECASE)
STEP_SPLIT_RE = re.compile(r"(?=\bStep\s+\d+\s*:)", re.IGNORECASE)
LETTERS = ["A", "B", "C", "D"]


def extract_answer(text):
    ms = ANS_RE.findall(text or "")
    return ms[-1].upper() if ms else None


def parse_steps(cot, max_steps=12):
    """Split a CoT into a list of step strings. Prefer 'Step N:' markers; else blank lines."""
    cot = cot.strip()
    parts = [p.strip() for p in STEP_SPLIT_RE.split(cot) if p.strip()]
    if len(parts) >= 2:
        return parts[:max_steps]
    # fallback: blank-line split
    paras = [p.strip() for p in re.split(r"\n\s*\n", cot) if len(p.strip()) > 15]
    return paras[:max_steps] if paras else [cot]


def load_questions(chains_dir, n):
    import glob
    rows = []
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
            q = d.get("question"); opts = d.get("options", {}); gold = d.get("correct_answer")
            sampled = d.get("sampled", [])
            if not (q and opts and gold):
                continue
            nc = nt = 0
            for ch in sampled:
                a = extract_answer(ch.get("text", ""))
                if a is not None:
                    nt += 1; nc += (a == gold)
            rows.append({"question_idx": d.get("question_idx"), "question": q, "options": opts,
                         "gold": gold, "vanilla_perchain_acc": nc / nt if nt else 0.0})
            if len(rows) >= n:
                return rows
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-tag", required=True, choices=["vanilla", "sft"])
    ap.add_argument("--n-questions", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-k", type=int, default=10, help="max truncation step index to probe")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {args.model_tag}: {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    qs = load_questions(CHAINS_VANILLA, args.n_questions)
    print(f"  {len(qs)} questions", flush=True)

    def build_prompt(q):
        msgs = [{"role": "user", "content": PROMPT_TMPL.format(
            question=q["question"], **{k: q["options"].get(k, "") for k in LETTERS})}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return PROMPT_TMPL.format(question=q["question"], **{k: q["options"].get(k, "") for k in LETTERS})

    # answer-prompt suffix (appended after a truncated CoT to force-decode the letter)
    ANS_SUFFIX = '\n\nThe answer is ('
    # token ids for "A","B","C","D" as they'd appear after "("
    letter_ids = {}
    for L in LETTERS:
        # try a few encodings; pick the single-token id
        for variant in [L, " " + L]:
            tids = tok.encode(variant, add_special_tokens=False)
            if len(tids) == 1:
                letter_ids[L] = tids[0]; break
    if len(letter_ids) < 4:
        # fallback: take the first token of each
        for L in LETTERS:
            letter_ids[L] = tok.encode(L, add_special_tokens=False)[0]
    letter_id_arr = torch.tensor([letter_ids[L] for L in LETTERS], device="cuda")

    records = []
    t0 = time.time()
    for i, q in enumerate(qs):
        prompt = build_prompt(q)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        # 1. greedy-generate full CoT
        with torch.no_grad():
            gen = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
        cot = tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        final_ans = extract_answer(cot)
        steps = parse_steps(cot)
        n_steps = len(steps)
        kmax = min(n_steps, args.max_k)
        # 2. for k = 0..kmax: truncate at end of step k, force-decode A/B/C/D logits
        # k=0: empty CoT (just the question). k>=1: cumulative steps 1..k.
        cumulative_cots = [""] + ["\n\n".join(steps[:k]) for k in range(1, kmax + 1)]
        forced = []          # forced[k] = top letter
        probs_by_k = []      # probs_by_k[k] = [p_A, p_B, p_C, p_D]
        gold_idx = LETTERS.index(q["gold"])
        model_idx = LETTERS.index(final_ans) if final_ans in LETTERS else None
        for k, partial_cot in enumerate(cumulative_cots):
            text_k = prompt + (partial_cot + ANS_SUFFIX if partial_cot else ANS_SUFFIX.lstrip())
            ids_k = tok(text_k, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                logits = model(ids_k).logits[0, -1, :].float()
            lett_logits = logits[letter_id_arr]
            p = torch.softmax(lett_logits, dim=-1).cpu().numpy()
            probs_by_k.append([float(x) for x in p])
            forced.append(LETTERS[int(lett_logits.argmax())])
        probs_arr = np.array(probs_by_k)  # (n_k, 4)
        # per-k derived metrics
        ent_by_k = [-float(np.sum(pr * np.log(pr + 1e-12))) for pr in probs_arr]   # entropy over A/B/C/D
        sorted_p = -np.sort(-probs_arr, axis=1)
        margin_top_by_k = [float(sp[0] - sp[1]) for sp in sorted_p]                 # top vs 2nd
        p_model_by_k = [float(pr[model_idx]) if model_idx is not None else float("nan") for pr in probs_arr]
        p_gold_by_k = [float(pr[gold_idx]) for pr in probs_arr]
        # margin of y_model over its best alternative
        margin_model_by_k = []
        for pr in probs_arr:
            if model_idx is None:
                margin_model_by_k.append(float("nan")); continue
            alt = max(pr[j] for j in range(4) if j != model_idx)
            margin_model_by_k.append(float(pr[model_idx] - alt))
        # 3. lock-point tau with margin threshold m: earliest k s.t. top==y_model for all k'>=k AND margin_top>=m at k
        lock_by_m = {}
        for m in (0.0, 0.5, 1.0):  # NOTE m here is on probability scale (margin_top in [0,1]); 0.5/1.0 are aggressive — also report logit-scale below
            tau = kmax
            for k in range(len(forced)):
                if (final_ans in LETTERS and all(forced[j] == final_ans for j in range(k, len(forced)))
                        and margin_top_by_k[k] >= m):
                    tau = k; break
            lock_by_m[str(m)] = {"tau": tau, "tau_norm": tau / max(1, n_steps)}
        # also a logit-margin lock (margin_top in logit units >= 1.0): recompute from logits — approximate via prob ratio
        # (skip; the prob-margin sweep above suffices for the gate)
        agree_final = [int(forced[k] == final_ans) if final_ans else 0 for k in range(len(forced))]
        agree_gold = [int(forced[k] == q["gold"]) for k in range(len(forced))]
        # threshold-free: normalized-position AUC of p_model and margin_model (mean over k, since k is the position)
        auc_p_model = float(np.nanmean(p_model_by_k)) if model_idx is not None else float("nan")
        auc_margin_model = float(np.nanmean(margin_model_by_k)) if model_idx is not None else float("nan")
        entropy_slope = float(np.polyfit(range(len(ent_by_k)), ent_by_k, 1)[0]) if len(ent_by_k) > 1 else 0.0
        records.append({
            "question_idx": q["question_idx"], "gold": q["gold"], "final_ans": final_ans,
            "n_steps": n_steps, "kmax": kmax, "forced_by_k": forced,
            "probs_by_k": probs_by_k, "entropy_by_k": ent_by_k,
            "margin_top_by_k": margin_top_by_k, "p_model_by_k": p_model_by_k,
            "p_gold_by_k": p_gold_by_k, "margin_model_by_k": margin_model_by_k,
            "lock_by_m": lock_by_m,
            "agree_with_final_by_k": agree_final, "agree_with_gold_by_k": agree_gold,
            "auc_p_model": auc_p_model, "auc_margin_model": auc_margin_model, "entropy_slope": entropy_slope,
            "vanilla_perchain_acc": q["vanilla_perchain_acc"],
            "final_correct": int(final_ans == q["gold"]) if final_ans else 0,
            "y_wrong": int(final_ans != q["gold"]) if final_ans else None,
        })
        if (i + 1) % 25 == 0:
            mean_lock = np.mean([r["lock_by_m"]["0.5"]["tau_norm"] for r in records])
            mean_p_k0 = np.mean([r["p_model_by_k"][0] for r in records if r["p_model_by_k"][0] == r["p_model_by_k"][0]])
            print(f"  {i+1}/{len(qs)}  {(i+1)/max(time.time()-t0,1):.2f} q/s  "
                  f"mean_lock_frac(m=0.5)={mean_lock:.3f}  mean_p_model(k=0)={mean_p_k0:.3f}", flush=True)

    # aggregate
    valid = [r for r in records if r["final_ans"]]
    def lock_frac(rs, m):
        return float(np.mean([r["lock_by_m"][str(m)]["tau_norm"] for r in rs])) if rs else float("nan")
    summary = {
        "model_tag": args.model_tag, "n_questions": len(qs), "n_valid": len(valid),
        "final_accuracy": float(np.mean([r["final_correct"] for r in valid])) if valid else float("nan"),
        "mean_n_steps": float(np.mean([r["n_steps"] for r in valid])) if valid else float("nan"),
        "mean_lock_frac_by_m": {str(m): lock_frac(valid, m) for m in (0.0, 0.5, 1.0)},
        "mean_lock_step_by_m": {str(m): float(np.mean([r["lock_by_m"][str(m)]["tau"] for r in valid])) if valid else float("nan") for m in (0.0, 0.5, 1.0)},
        "auc_p_model_mean": float(np.nanmean([r["auc_p_model"] for r in valid])) if valid else float("nan"),
        "auc_margin_model_mean": float(np.nanmean([r["auc_margin_model"] for r in valid])) if valid else float("nan"),
        "entropy_slope_mean": float(np.mean([r["entropy_slope"] for r in valid])) if valid else float("nan"),
        "p_k0_model_mean": float(np.nanmean([r["p_model_by_k"][0] for r in valid])) if valid else float("nan"),
        "k0_top_eq_final_rate": float(np.mean([r["agree_with_final_by_k"][0] for r in valid])) if valid else float("nan"),
    }
    # agree-with-final curve vs normalized position (10 bins)
    for label, key in [("agree_final", "agree_with_final_by_k"), ("agree_gold", "agree_with_gold_by_k")]:
        fb = defaultdict(list)
        for r in valid:
            L = len(r[key])
            for k in range(L):
                fb[min(9, int((k / max(1, r["n_steps"])) * 10))].append(r[key][k])
        summary[f"{label}_curve_frac"] = {str(b): float(np.mean(v)) for b, v in sorted(fb.items())}
    # p_model and margin_top trajectories vs normalized position
    for label, key in [("p_model", "p_model_by_k"), ("margin_top", "margin_top_by_k"), ("entropy", "entropy_by_k")]:
        fb = defaultdict(list)
        for r in valid:
            L = len(r[key])
            for k in range(L):
                v = r[key][k]
                if v == v:  # not nan
                    fb[min(9, int((k / max(1, r["n_steps"])) * 10))].append(v)
        summary[f"{label}_traj_frac"] = {str(b): float(np.mean(v)) for b, v in sorted(fb.items())}
    # difficulty strata + wrong-vs-right split (lock_frac at m=0.5)
    bins = [(0, .25), (.25, .5), (.5, .75), (.75, 1.001)]
    summary["by_difficulty"] = {}
    for lo, hi in bins:
        sub = [r for r in valid if lo <= r["vanilla_perchain_acc"] < hi]
        summary["by_difficulty"][f"{lo:.2f}-{hi:.2f}"] = {
            "n": len(sub), "lock_frac_m0.5": lock_frac(sub, 0.5), "lock_frac_m0.0": lock_frac(sub, 0.0)}
    wrong = [r for r in valid if r.get("y_wrong") == 1]
    right = [r for r in valid if r.get("y_wrong") == 0]
    summary["lock_frac_m0.5_wrong_chains"] = lock_frac(wrong, 0.5)
    summary["lock_frac_m0.5_right_chains"] = lock_frac(right, 0.5)
    summary["n_wrong_chains"] = len(wrong); summary["n_right_chains"] = len(right)

    with open(out_dir / f"expC1b_{args.model_tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(out_dir / f"expC1b_{args.model_tag}_records.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\n=== {args.model_tag} ===")
    print(f"  final_acc={summary['final_accuracy']*100:.1f}%  mean_n_steps={summary['mean_n_steps']:.2f}  n_valid={len(valid)}")
    print(f"  lock_frac (m=0): {summary['mean_lock_frac_by_m']['0.0']:.3f}  (m=0.5): {summary['mean_lock_frac_by_m']['0.5']:.3f}  (m=1.0): {summary['mean_lock_frac_by_m']['1.0']:.3f}  (lower=earlier commitment)")
    print(f"  AUC_p_model={summary['auc_p_model_mean']:.3f}  AUC_margin_model={summary['auc_margin_model_mean']:.3f}  entropy_slope={summary['entropy_slope_mean']:.4f}")
    print(f"  k=0: p(y_model)={summary['p_k0_model_mean']:.3f}  top==final rate={summary['k0_top_eq_final_rate']:.3f}  (k=0 should be near-chance per expC1)")
    print(f"  lock_frac m=0.5: wrong-chains={summary['lock_frac_m0.5_wrong_chains']:.3f} (n={len(wrong)})  right-chains={summary['lock_frac_m0.5_right_chains']:.3f} (n={len(right)})")
    pmt = summary["p_model_traj_frac"]
    pmt_str = [round(pmt.get(str(b), float("nan")), 2) for b in range(10)]
    print(f"  p_model trajectory (10 bins): {pmt_str}")
    print(f"  Saved: {out_dir}/expC1b_{args.model_tag}.json")


if __name__ == "__main__":
    main()
