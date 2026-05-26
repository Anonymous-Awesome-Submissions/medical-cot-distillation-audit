#!/usr/bin/env python3
"""Answer-blind Kimi-K2.6 step-factuality audit for MedQA.

This is a targeted control for the possibility that showing the gold answer
to the judge over-penalizes reasonable differential-diagnosis steps that do
not point to the correct option. It reuses existing CoTs and changes only the
judge prompt: the answer options are shown, but the correct answer is not.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from scipy.stats import wilcoxon

SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
JUDGE_MODEL = "Pro/moonshotai/Kimi-K2.6"
VALID = {"correct", "error", "uncertain"}

STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]\s*(.+?)(?=(?:\n\s*(?:Step\s+)?\d+[\.\:\)])|$)",
    re.DOTALL | re.IGNORECASE,
)

ANSWER_BLIND_PROMPT = """You are a senior physician evaluating one step from a medical AI model's reasoning.

## Question
{question}

## Answer options
{options}

## ONE reasoning step from the AI's chain-of-thought
"{step_text}"

## Critical instructions

The correct answer is intentionally NOT provided.

Evaluate ONLY whether this step contains a medically false factual claim, given the case and answer options.

Do NOT penalize a step merely because it considers an alternative diagnosis, mechanism, or option. In a differential diagnosis, a hypothesis can be reasonable even if it is not ultimately selected.

Do NOT judge whether the step points to the final correct option. Judge only factual correctness of the local medical content.

IGNORE:
- The tone or register (committal vs hedged: "X is" vs "X may be")
- Grammatical confidence (assertive vs uncertain phrasing)
- Whether the writer used hedging words ("might", "may", "could", "possibly")
- Stylistic certainty markers ("clearly", "definitely", "obviously")
- Whether the step sounds confident or uncertain

Use "uncertain" if the step cannot be judged without knowing the final answer or without additional clinical information.

Respond exactly:
JUDGMENT: <correct|error|uncertain>
EXPLANATION: <one sentence focusing on factual content, not tone or final-answer alignment>"""


def parse_steps(text, max_steps=12):
    if not text:
        return []
    matches = STEP_RE.findall(text)
    steps = []
    for _, body in matches:
        body = body.strip()
        body = re.sub(r"\s*(?:The answer is\s*\(?[A-E]\)?\.?\s*)?$", "", body)
        if body and len(body) > 20:
            steps.append(body[:800])
    if not steps:
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        steps = paras[:max_steps]
    return steps[:max_steps]


def classify_role(step):
    s = step.lower()
    if any(k in s for k in ["however", "wait", "reconsider", "on second thought", "but ", "although"]):
        return "correction"
    if any(k in s for k in ["the answer is", "in conclusion", "therefore", "thus,", "conclusion"]):
        return "final_synthesis"
    if any(k in s for k in ["rule out", "rules out", "less likely", "option a", "option b", "option c", "option d", "(a)", "(b)", "(c)", "(d)"]):
        return "option_elimination"
    if any(k in s for k in ["differential", "could be", "consider", "most likely diagnosis", "likely diagnosis", "possible diagnosis", "suggests", "points to", "consistent with"]):
        return "hypothesis_gen"
    if len(s) < 25 or any(k in s for k in ["let me work through", "step-by-step", "key clinical findings"]):
        return "other"
    return "factual_claim"


def parse_judgment(text):
    m = re.search(r"JUDGMENT:\s*(\w+)", text or "", re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID:
        j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    return j, (m.group(1).strip() if m else "")


def make_call_kwargs(model, timeout):
    kw = dict(temperature=0.0, max_tokens=220, timeout=timeout)
    if ("GLM-" in model) or ("Kimi-K2." in model):
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
    return kw


def call_judge(client, model, prompt, extra_kw, max_retries=6):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **extra_kw,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "limit" in err.lower():
                time.sleep(min(2 ** attempt * 2.0, 60))
            elif "503" in err or "502" in err or "timeout" in err.lower() or "overload" in err.lower():
                time.sleep(min(2 ** attempt * 1.5, 45))
            else:
                time.sleep(min(2 ** attempt, 15))
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def build_tasks(input_path, tag, n_questions, chain_field, done):
    tasks = []
    with open(input_path) as f:
        for qi, line in enumerate(f):
            if n_questions is not None and qi >= n_questions:
                break
            if not line.strip():
                continue
            e = json.loads(line)
            question = e.get("question") or e.get("sft_input", "")
            options = e.get("options") or {}
            if not question or not options:
                continue
            opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
            text = e.get(chain_field, "") or e.get("text", "") or e.get("generated_cot", "") or e.get("teacher_cot", "")
            for sidx, step in enumerate(parse_steps(text)):
                kid = f"{qi}|0|{sidx}"
                if kid in done:
                    continue
                prompt = ANSWER_BLIND_PROMPT.format(question=question, options=opt_str, step_text=step)
                tasks.append({
                    "qidx": qi,
                    "question_idx": e.get("question_idx", qi),
                    "chain_idx": 0,
                    "step_idx": sidx,
                    "model_tag": tag,
                    "step_text": step,
                    "role": classify_role(step),
                    "prompt": prompt,
                })
    return tasks


def summarize_file(path):
    counts = Counter()
    by_q = {}
    by_role = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            j = r.get("judgment", "uncertain")
            counts[j] += 1
            q = int(r["qidx"])
            role = r.get("role", "unknown")
            by_q.setdefault(q, Counter())[j] += 1
            by_role.setdefault(role, Counter())[j] += 1

    def rate(c):
        den = c["correct"] + c["error"]
        return c["error"] / den if den else float("nan")

    return {
        "counts": dict(counts),
        "rate": rate(counts),
        "uncertain_rate": counts["uncertain"] / max(sum(counts.values()), 1),
        "by_q": by_q,
        "by_role": by_role,
    }


def paired_stats(base_by_q, sft_by_q, n_boot=5000, seed=13):
    qs = sorted(set(base_by_q) & set(sft_by_q))
    diffs = []
    for q in qs:
        b = base_by_q[q]
        s = sft_by_q[q]
        bden = b["correct"] + b["error"]
        sden = s["correct"] + s["error"]
        if bden and sden:
            diffs.append(s["error"] / sden - b["error"] / bden)
    diffs = np.array(diffs, dtype=float)
    mean = float(np.mean(diffs)) if len(diffs) else float("nan")
    rng = np.random.default_rng(seed)
    ci = (float("nan"), float("nan"))
    if len(diffs):
        boot = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n_boot)]
        ci = tuple(np.percentile(boot, [2.5, 97.5]).tolist())
    try:
        p = float(wilcoxon(diffs, zero_method="wilcox").pvalue)
    except Exception:
        p = float("nan")
    return {"n": int(len(diffs)), "mean": mean, "ci": ci, "wilcoxon_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--chain-field", default="text")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key and not args.dry_run:
        sys.exit("SILICONFLOW_API_KEY not set")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"judgments_{args.model_tag}.jsonl"

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(f"{d['qidx']}|{d['chain_idx']}|{d['step_idx']}")
                except Exception:
                    pass

    tasks = build_tasks(args.input, args.model_tag, args.n_questions, args.chain_field, done)
    print(f"[{args.model_tag}] resume={len(done)} tasks={len(tasks)} output={out_path}", flush=True)
    if args.dry_run:
        roles = Counter(t["role"] for t in tasks)
        print(f"dry-run roles={dict(roles)}", flush=True)
        if tasks:
            print(tasks[0]["prompt"][:1200], flush=True)
        return
    if not tasks:
        return

    client = OpenAI(api_key=key, base_url=SILICONFLOW_BASE)
    extra_kw = make_call_kwargs(args.judge_model, args.timeout)
    counts = Counter()
    lock = threading.Lock()
    t0 = time.time()

    def worker(task):
        reply = call_judge(client, args.judge_model, task["prompt"], extra_kw, max_retries=args.max_retries)
        j, exp = parse_judgment(reply)
        return task, j, exp

    with open(out_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            completed = 0
            for fut in as_completed(futures):
                task, j, exp = fut.result()
                counts[j] += 1
                row = {
                    "qidx": task["qidx"],
                    "question_idx": task["question_idx"],
                    "chain_idx": task["chain_idx"],
                    "step_idx": task["step_idx"],
                    "model_tag": task["model_tag"],
                    "judge_model": args.judge_model,
                    "answer_blind": True,
                    "role": task["role"],
                    "step_text": task["step_text"][:500],
                    "judgment": j,
                    "explanation": exp[:300],
                }
                with lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                completed += 1
                if completed % 200 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1)
                    eta = (len(tasks) - completed) / max(rate, 1e-3)
                    den = counts["correct"] + counts["error"]
                    er = counts["error"] / den * 100 if den else 0.0
                    print(
                        f"  {completed}/{len(tasks)} {rate:.2f}/s el={elapsed:.0f}s eta={eta:.0f}s "
                        f"err={er:.1f}% unc={counts['uncertain']}",
                        flush=True,
                    )

    den = counts["correct"] + counts["error"]
    print(f"[{args.model_tag}] done counts={dict(counts)}", flush=True)
    if den:
        print(f"[{args.model_tag}] error/(correct+error)={counts['error']}/{den}={counts['error']/den*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
