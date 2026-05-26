#!/usr/bin/env python3
"""
Step-level factuality audit.

For each (question, CoT chain) pair we split the chain into reasoning steps,
then ask DeepSeek-V3 whether each step contains a factual medical error.

Usage:
    export DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
    python step_factuality_judge.py \
        --input /path/to/chains.jsonl \
        --output-dir /path/to/out \
        --model-tag teacher \
        --n-questions 500

Output: one jsonl line per step judgement, with
    {qidx, model_tag, step_idx, step_text, judgment: "correct|error|uncertain",
     explanation}
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


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
EXPLANATION: <one sentence>
"""


VALID_JUDGMENTS = {"correct", "error", "uncertain"}


def call_deepseek(prompt, api_key, max_retries=5):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 150,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                              headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                wait = min(2 ** attempt * 2, 60)
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    API error {r.status_code}: {r.text[:200]}", flush=True)
                time.sleep(2)
        except Exception as e:
            print(f"    Request error: {e}", flush=True)
            time.sleep(2)
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def parse_judgment(text):
    m = re.search(r"JUDGMENT:\s*(\w+)", text, re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID_JUDGMENTS:
        j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    exp = m.group(1).strip() if m else ""
    return j, exp


# ─── Step parsing ──────────────────────────────────────────────────────
STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]\s*(.+?)(?=(?:\n\s*(?:Step\s+)?\d+[\.\:\)])|$)",
    re.DOTALL | re.IGNORECASE,
)


def parse_steps(text, max_steps=12):
    """Parse a CoT into list of step strings. Falls back to sentence split."""
    if not text:
        return []
    matches = STEP_RE.findall(text)
    steps = []
    for idx, body in matches:
        body = body.strip()
        # Trim final "The answer is (X)." if stuck on last step
        body = re.sub(r"\s*(?:The answer is\s*\([A-E]\)\.?\s*)?$", "", body)
        if body and len(body) > 20:
            steps.append(body[:800])
    # Fall back: paragraph split
    if not steps:
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        steps = paras[:max_steps]
    return steps[:max_steps]


def build_prompt(question, options, gold_letter, step_text):
    opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    correct_text = options.get(gold_letter, "")
    return STEP_JUDGE_PROMPT.format(
        question=question,
        options=opt_str,
        correct_answer=gold_letter,
        correct_text=correct_text,
        step_text=step_text,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="jsonl with per-question chain data. "
                         "Expected fields: question, options, correct_answer, "
                         "and one of {chains, sampled, sft_target, text}")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True,
                    help="label attached to every judgement row")
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--n-chains-per-q", type=int, default=1,
                    help="how many CoTs per question to audit "
                         "(1 = just the first / majority)")
    ap.add_argument("--chain-field", default="auto",
                    help="which field contains the CoT text. "
                         "'auto' = try ['sft_target','sampled[0].text','text']")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"judgments_{args.model_tag}.jsonl")
    done_path = os.path.join(args.output_dir, f"done_{args.model_tag}.txt")

    # Resume: track completed (qidx, chain_idx) pairs
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)

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
    print(f"Loaded {len(entries)} questions from {args.input}")

    # Open outputs
    fout = open(out_path, "a")
    fdone = open(done_path, "a")

    total_calls = 0
    t_start = time.time()

    for qi, e in enumerate(entries):
        question = e.get("question") or e.get("sft_input", "")
        options = e.get("options") or {}
        if not options and "sft_input" in e:
            # sft_input already contains Question + Options; try to parse options
            sft_in = e["sft_input"]
            opt_match = re.findall(r"\(?([A-E])\)?\s*[\.:\)]?\s*(.+)", sft_in)
            options = dict(opt_match[-5:]) if opt_match else {}
        gold = (e.get("correct_answer") or e.get("answer_idx") or "").strip()
        if not gold or not options:
            continue

        # Pick the chain text(s) to judge
        chain_texts = []
        if args.chain_field == "auto":
            if "sft_target" in e and e["sft_target"]:
                chain_texts.append(e["sft_target"])
            elif "sampled" in e and e["sampled"]:
                # Take up to n chains, prefer the majority answer's chains
                samples = e["sampled"][:args.n_chains_per_q]
                chain_texts.extend([s.get("text", "") for s in samples if s.get("text")])
            elif "text" in e:
                chain_texts.append(e["text"])
        else:
            chain_texts.append(e.get(args.chain_field, ""))

        for cid, text in enumerate(chain_texts):
            key_id = f"{qi}|{cid}"
            if key_id in done:
                continue
            steps = parse_steps(text)
            if not steps:
                fdone.write(key_id + "\n")
                fdone.flush()
                continue

            for sidx, step in enumerate(steps):
                prompt = build_prompt(question, options, gold, step)
                reply = call_deepseek(prompt, key)
                judgment, explanation = parse_judgment(reply)
                fout.write(json.dumps({
                    "qidx": qi,
                    "chain_idx": cid,
                    "step_idx": sidx,
                    "model_tag": args.model_tag,
                    "step_text": step[:500],
                    "judgment": judgment,
                    "explanation": explanation[:300],
                }) + "\n")
                fout.flush()
                total_calls += 1

            fdone.write(key_id + "\n")
            fdone.flush()

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = total_calls / max(elapsed, 1)
            print(f"  [{qi+1}/{len(entries)}] calls={total_calls} "
                  f"rate={rate:.1f}/s elapsed={elapsed:.0f}s",
                  flush=True)

    fout.close()
    fdone.close()
    print(f"\nDone: {total_calls} judgments in {time.time()-t_start:.0f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
