#!/usr/bin/env python3
"""
Generate DeepSeek-V3 teacher CoT on MedQA test questions so we can compare
teacher vs student vs vanilla at the step level on the SAME questions.

Output: jsonl with fields {question_idx, question, options, correct_answer,
teacher_answer, teacher_cot}.
"""
import argparse
import json
import os
import re
import sys
import time

import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


COT_PROMPT = """You are an expert physician taking the USMLE. Work through the following question as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Why some answer options fit and others don't
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is (X)." where X is A, B, C, or D.

Question: {question}

Options:
{options}"""


ANSWER_RE = re.compile(r"[Tt]he\s+answer\s+is\s*\(?([A-E])\)?")


def call_deepseek(prompt, api_key, max_retries=5):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
                              headers=headers, json=payload, timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                time.sleep(min(2 ** attempt * 2, 60))
            else:
                print(f"    API error {r.status_code}: {r.text[:200]}", flush=True)
                time.sleep(2)
        except Exception as e:
            print(f"    Request error: {e}", flush=True)
            time.sleep(2)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")

    # Resume: track done qidx
    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done_ids.add(d["question_idx"])
        print(f"Resuming: {len(done_ids)} questions already done")

    # Load questions
    questions = []
    with open(args.test_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    # Sample first N (deterministic)
    import random
    random.seed(args.seed)
    idxs = list(range(len(questions)))
    random.shuffle(idxs)
    selected = idxs[: args.n_questions]

    fout = open(args.output, "a")
    t_start = time.time()
    n_done = 0

    for qi in selected:
        if qi in done_ids:
            continue
        q = questions[qi]
        question = q["question"]
        options = q.get("options") or {}
        gold = (q.get("answer_idx") or q.get("correct_answer") or "").strip()
        opt_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        prompt = COT_PROMPT.format(question=question, options=opt_str)

        cot = call_deepseek(prompt, key)
        ans_match = ANSWER_RE.search(cot or "")
        teacher_ans = ans_match.group(1).upper() if ans_match else ""

        fout.write(json.dumps({
            "question_idx": qi,
            "question": question,
            "options": options,
            "correct_answer": gold,
            "teacher_answer": teacher_ans,
            "teacher_cot": cot,
        }) + "\n")
        fout.flush()
        n_done += 1

        if n_done % 25 == 0:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1)
            eta = (args.n_questions - len(done_ids) - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{args.n_questions - len(done_ids)}] "
                  f"rate={rate:.1f}/s  ETA={eta/60:.0f}min", flush=True)

    fout.close()
    print(f"\nDone: {n_done} teacher CoTs in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
