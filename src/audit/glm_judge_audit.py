#!/usr/bin/env python3
"""
Step-level factuality audit via GLM-4-32B (cross-family from teacher
DeepSeek and student Qwen). Concurrent OpenAI-compatible client against
SiliconFlow. Output format is byte-identical to the previous DeepSeek-V3
audit so downstream analysis code Just Works.

Usage:
    export SILICONFLOW_API_KEY=<YOUR_SILICONFLOW_API_KEY>
    python glm_judge_audit.py \
        --input /path/to/cots.jsonl \
        --output-dir /path/to/audit_results_glm \
        --model-tag vanilla --n-questions 500 --concurrency 30
"""
import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

JUDGE_MODEL_DEFAULT = "THUDM/GLM-4-32B-0414"
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"

STEP_JUDGE_PROMPT = """You are a senior physician evaluating a medical AI model's reasoning.

## Question
{question}

## Options
{options}

## Correct Answer
({correct_answer}) {correct_text}

## ONE reasoning step from the AI's chain-of-thought
"{step_text}"

## Task
Judge whether this specific step contains a FACTUAL MEDICAL ERROR. Consider only:
- Is a stated clinical fact, mechanism, drug action, lab value, or causal claim WRONG or FABRICATED?

Do NOT flag:
- Irrelevance (not wrong, just not useful)
- Incomplete reasoning
- Speculation clearly marked ("possibly", "may indicate")
- Opinions or reasoning jumps that don't assert a fact

Respond in this exact format:
JUDGMENT: <correct|error|uncertain>
EXPLANATION: <one sentence>"""

VALID_JUDGMENTS = {"correct", "error", "uncertain"}

STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]\s*(.+?)(?=(?:\n\s*(?:Step\s+)?\d+[\.\:\)])|$)",
    re.DOTALL | re.IGNORECASE,
)


def parse_steps(text, max_steps=12):
    if not text:
        return []
    matches = STEP_RE.findall(text)
    steps = []
    for idx, body in matches:
        body = body.strip()
        body = re.sub(r"\s*(?:The answer is\s*\([A-E]\)\.?\s*)?$", "", body)
        if body and len(body) > 20:
            steps.append(body[:800])
    if not steps:
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        steps = paras[:max_steps]
    return steps[:max_steps]


def parse_judgment(text):
    m = re.search(r"JUDGMENT:\s*(\w+)", text, re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID_JUDGMENTS:
        j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    exp = m.group(1).strip() if m else ""
    return j, exp


def build_prompt(question, options, gold_letter, step_text):
    opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    correct_text = options.get(gold_letter, "")
    return STEP_JUDGE_PROMPT.format(
        question=question, options=opt_str,
        correct_answer=gold_letter, correct_text=correct_text,
        step_text=step_text,
    )


_lock = threading.Lock()


def judge_one(client, model, prompt, max_retries=4):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=200,
                timeout=60,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                time.sleep(min(2 ** attempt * 1.5, 30))
            else:
                time.sleep(min(2 ** attempt, 10))
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--n-chains-per-q", type=int, default=1)
    ap.add_argument("--chain-field", default="auto")
    ap.add_argument("--concurrency", type=int, default=30)
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        sys.exit("SILICONFLOW_API_KEY not set")

    client = OpenAI(api_key=key, base_url=SILICONFLOW_BASE)
    print(f"Judge: {args.judge_model}, concurrency={args.concurrency}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"judgments_{args.model_tag}.jsonl")
    done_path = os.path.join(args.output_dir, f"done_{args.model_tag}.txt")

    # Resume: load done step keys
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(f"{d['qidx']}|{d['chain_idx']}|{d['step_idx']}")
                except Exception:
                    pass
    print(f"Resume: {len(done)} step judgements already in {out_path}", flush=True)

    # Load input
    entries = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if args.n_questions and len(entries) >= args.n_questions:
                break
    print(f"Loaded {len(entries)} questions", flush=True)

    # Build all (qidx, chain_idx, step_idx, prompt, step_text) tasks
    tasks = []
    for qi, e in enumerate(entries):
        question = e.get("question") or e.get("sft_input", "")
        options = e.get("options") or {}
        gold = (e.get("correct_answer") or e.get("answer_idx") or "").strip()
        if not gold or not options:
            continue
        chain_texts = []
        if args.chain_field == "auto":
            if "sft_target" in e and e["sft_target"]:
                chain_texts.append(e["sft_target"])
            elif "sampled" in e and e["sampled"]:
                samples = e["sampled"][:args.n_chains_per_q]
                chain_texts.extend([s.get("text", "") for s in samples if s.get("text")])
            elif "text" in e:
                chain_texts.append(e["text"])
        else:
            chain_texts.append(e.get(args.chain_field, ""))
        for cid, text in enumerate(chain_texts):
            steps = parse_steps(text)
            for sidx, step in enumerate(steps):
                key_id = f"{qi}|{cid}|{sidx}"
                if key_id in done:
                    continue
                prompt = build_prompt(question, options, gold, step)
                tasks.append((qi, cid, sidx, step, prompt))
    print(f"Total step tasks to judge: {len(tasks)}", flush=True)
    if not tasks:
        return

    fout = open(out_path, "a")
    t0 = time.time()
    completed = 0
    counts = {"correct": 0, "error": 0, "uncertain": 0}

    def worker(task):
        qi, cid, sidx, step, prompt = task
        reply = judge_one(client, args.judge_model, prompt)
        j, exp = parse_judgment(reply)
        return qi, cid, sidx, step, j, exp

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                qi, cid, sidx, step, j, exp = fut.result()
            except Exception as e:
                print(f"  Worker exception: {e}", flush=True)
                continue
            counts[j] = counts.get(j, 0) + 1
            with _lock:
                fout.write(json.dumps({
                    "qidx": qi, "chain_idx": cid, "step_idx": sidx,
                    "model_tag": args.model_tag,
                    "step_text": step[:500],
                    "judgment": j,
                    "explanation": exp[:300],
                }) + "\n")
                fout.flush()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1)
                eta = (len(tasks) - completed) / max(rate, 0.001)
                err_rate = counts["error"] / max(completed, 1) * 100
                print(f"  {completed}/{len(tasks)}, rate={rate:.1f}/s, "
                      f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s, "
                      f"error_rate={err_rate:.1f}%", flush=True)

    fout.close()
    print(f"\n=== Audit done ({args.model_tag}) ===")
    print(f"  Total steps: {completed}")
    print(f"  Counts: {counts}")
    if completed:
        print(f"  Error rate: {counts['error']/completed*100:.2f}%")
        print(f"  Uncertain rate: {counts['uncertain']/completed*100:.2f}%")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
