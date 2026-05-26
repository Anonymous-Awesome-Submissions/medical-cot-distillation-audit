#!/usr/bin/env python3
"""Generate DeepSeek teacher CoTs for an explicit list of MedQA question ids."""

import argparse
import json
import os
import sys
import time

from generate_teacher_cot import ANSWER_RE, COT_PROMPT, call_deepseek


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--qid-file", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")

    questions = []
    with open(args.test_data) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    qids = []
    with open(args.qid_file) as f:
        for line in f:
            line = line.strip()
            if line:
                qids.append(int(line))

    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["question_idx"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fout = open(args.output, "a")
    t_start = time.time()
    n_new = 0
    total_missing = sum(1 for qid in qids if qid not in done_ids)

    print(
        f"Shard input qids={len(qids)} existing={len(done_ids)} "
        f"to_generate={total_missing}",
        flush=True,
    )

    for qid in qids:
        if qid in done_ids:
            continue
        q = questions[qid]
        options = q.get("options") or {}
        gold = (q.get("answer_idx") or q.get("correct_answer") or "").strip()
        opt_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        prompt = COT_PROMPT.format(question=q["question"], options=opt_str)

        cot = call_deepseek(prompt, key)
        ans_match = ANSWER_RE.search(cot or "")
        teacher_ans = ans_match.group(1).upper() if ans_match else ""

        fout.write(
            json.dumps(
                {
                    "question_idx": qid,
                    "question": q["question"],
                    "options": options,
                    "correct_answer": gold,
                    "teacher_answer": teacher_ans,
                    "teacher_cot": cot,
                }
            )
            + "\n"
        )
        fout.flush()
        n_new += 1

        if n_new % 10 == 0 or n_new == total_missing:
            elapsed = time.time() - t_start
            rate = n_new / max(elapsed, 1)
            remaining = total_missing - n_new
            eta = remaining / max(rate, 1e-6)
            print(
                f"  [{n_new}/{total_missing}] rate={rate:.2f}/s "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

    fout.close()
    print(f"Done: wrote {n_new} new CoTs in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
