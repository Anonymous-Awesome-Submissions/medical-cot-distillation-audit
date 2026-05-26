"""
Math step audit via GLM-4-32B. Same style-blind prompt structure as medical
but adapted for math: ask judge if the step's COMPUTATION/REASONING is
correct, ignoring tone.
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

from openai import OpenAI

JUDGE_MODEL_DEFAULT = "THUDM/GLM-4-32B-0414"
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"

STYLE_BLIND_MATH_PROMPT = """You are a math tutor evaluating one step of a student's reasoning.

## Problem
{question}

## Correct numeric answer
{correct_answer}

## ONE step from the student's solution
"{step_text}"

## Critical instructions

You MUST evaluate ONLY whether this step contains a COMPUTATIONAL or LOGICAL ERROR.

A step has an ERROR if:
- An arithmetic computation is wrong (e.g., 12 × 3 = 35)
- An algebraic manipulation is incorrect
- A logical inference does not follow
- A formula is misapplied

A step is CORRECT if:
- All math is accurate
- The reasoning is valid (regardless of whether it leads to the gold answer)
- The step is purely descriptive ("Let me think about this") without errors

IGNORE:
- The tone or register (committal vs hedged)
- Whether the writer used "let me try", "perhaps", "I think"
- Stylistic certainty markers
- Whether the step sounds confident or uncertain

ONLY judge: "If I IGNORE the tone, is the math/logic in this step actually wrong?"

Respond exactly:
JUDGMENT: <correct|error|uncertain>
EXPLANATION: <one sentence>"""

VALID = {"correct", "error", "uncertain"}

STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]\s*(.+?)(?=(?:\n\s*(?:Step\s+)?\d+[\.\:\)])|$)",
    re.DOTALL | re.IGNORECASE,
)


def parse_steps(text, max_steps=12):
    if not text: return []
    matches = STEP_RE.findall(text)
    steps = []
    for idx, body in matches:
        body = body.strip()
        body = re.sub(r"\s*(?:The answer is\s*\$?[-+]?[\d,.]+\.?\s*)?$", "", body)
        if body and len(body) > 20:
            steps.append(body[:800])
    if not steps:
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        steps = paras[:max_steps]
    return steps[:max_steps]


def parse_judgment(text):
    m = re.search(r"JUDGMENT:\s*(\w+)", text, re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID: j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return j, (m.group(1).strip() if m else "")


def build_prompt(question, gold_num, step_text):
    return STYLE_BLIND_MATH_PROMPT.format(
        question=question, correct_answer=gold_num, step_text=step_text,
    )


_lock = threading.Lock()


def judge_one(client, model, prompt, extra_kw=None, max_retries=4):
    extra_kw = extra_kw or {}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=200, timeout=60, **extra_kw,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                time.sleep(min(2 ** attempt * 1.5, 30))
            else:
                time.sleep(min(2 ** attempt, 8))
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key: sys.exit("SILICONFLOW_API_KEY not set")
    client = OpenAI(api_key=key, base_url=SILICONFLOW_BASE)
    # GLM-* and Kimi-K2.x are hybrid reasoning models -> force NON-thinking.
    extra_kw = {}
    if ("GLM-" in args.judge_model) or ("Kimi-K2." in args.judge_model):
        extra_kw["extra_body"] = {"thinking": {"type": "disabled"}}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"judgments_{args.model_tag}.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(f"{d['qidx']}|{d['chain_idx']}|{d['step_idx']}")
                except: pass

    entries = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line: entries.append(json.loads(line))
            if args.n_questions and len(entries) >= args.n_questions: break

    tasks = []
    for qi, e in enumerate(entries):
        question = e.get("question", "")
        gold_num = e.get("gold_numeric") or e.get("options", {}).get("correct", "")
        text = e.get("text", "")
        if not question or not text or not gold_num: continue
        steps = parse_steps(text)
        for sidx, step in enumerate(steps):
            key_id = f"{qi}|0|{sidx}"
            if key_id in done: continue
            tasks.append((qi, 0, sidx, step, build_prompt(question, gold_num, step)))
    print(f"Tasks: {len(tasks)}", flush=True)
    if not tasks: return

    fout = open(out_path, "a")
    t0 = time.time(); completed = 0
    counts = Counter()

    def worker(task):
        qi, cid, sidx, step, prompt = task
        reply = judge_one(client, args.judge_model, prompt, extra_kw)
        j, exp = parse_judgment(reply)
        return qi, cid, sidx, step, j, exp

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            try: qi, cid, sidx, step, j, exp = fut.result()
            except: continue
            counts[j] += 1
            with _lock:
                fout.write(json.dumps({
                    "qidx": qi, "chain_idx": cid, "step_idx": sidx,
                    "model_tag": args.model_tag,
                    "step_text": step[:500],
                    "judgment": j, "explanation": exp[:300],
                }) + "\n"); fout.flush()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                el = time.time() - t0
                rate = completed / max(el, 1)
                err = counts["error"] / max(completed, 1) * 100
                print(f"  {completed}/{len(tasks)} {rate:.1f}/s err={err:.1f}%", flush=True)

    fout.close()
    nc = counts["correct"] + counts["error"]
    print(f"\n=== {args.model_tag} math audit done ===")
    print(f"  Counts: {dict(counts)}")
    if nc:
        print(f"  Error rate (def-judged): {counts['error']/nc*100:.2f}%")


if __name__ == "__main__":
    main()
