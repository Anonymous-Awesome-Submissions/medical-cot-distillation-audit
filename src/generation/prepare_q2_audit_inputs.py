#!/usr/bin/env python3
"""
Prepare Q2 step-factuality audit inputs:
 - teacher  : already exists (q2_hallucination/teacher_cot_medqa_test500.jsonl)
 - vanilla  : take the FIRST of the 64 sampled chains on the same 500 qidx
 - SFT-8B   : take the FIRST of the 64 sampled chains on the same 500 qidx

Output: vanilla_cot_medqa_test500.jsonl and sft8b_cot_medqa_test500.jsonl
with fields matching what step_factuality_judge.py expects.
"""
import glob
import json
import os


BASE = "experiments/module1"
Q2_DIR = f"{BASE}/q2_hallucination"
os.makedirs(Q2_DIR, exist_ok=True)


def load_chains(dirname, n_chains=1):
    """Return {qidx: {question, options, correct_answer, cot_text}}."""
    out = {}
    for fp in sorted(glob.glob(f"{BASE}/{dirname}/gpu_shard_*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    q = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qidx = q.get("question_idx")
                if qidx is None:
                    continue
                samples = q.get("sampled", [])
                if not samples:
                    continue
                # Use the first chain as a representative CoT
                first = samples[0]
                out[qidx] = {
                    "question_idx": qidx,
                    "question": q.get("question", ""),
                    "options": q.get("options", {}),
                    "correct_answer": (q.get("correct_answer") or "").strip(),
                    "text": first.get("text", ""),
                    "answer": first.get("answer", ""),
                }
    return out


def main():
    # Load teacher qidxs (the 500 sampled)
    teacher_file = f"{Q2_DIR}/teacher_cot_medqa_test500.jsonl"
    teacher_qidxs = set()
    with open(teacher_file) as f:
        for line in f:
            if line.strip():
                teacher_qidxs.add(json.loads(line)["question_idx"])
    print(f"Teacher qidxs: {len(teacher_qidxs)}")

    # Vanilla
    print("Loading vanilla chains ...")
    van = load_chains("test_chains_vanilla_8b_sc64")
    n_vanilla = 0
    with open(f"{Q2_DIR}/vanilla_cot_medqa_test500.jsonl", "w") as f:
        for qi in sorted(teacher_qidxs):
            if qi in van:
                f.write(json.dumps(van[qi]) + "\n")
                n_vanilla += 1
    print(f"  Vanilla entries written: {n_vanilla}")

    # SFT-8B
    print("Loading SFT-8B chains ...")
    sft = load_chains("test_chains_qwen3_8b_sc64")
    n_sft = 0
    with open(f"{Q2_DIR}/sft8b_cot_medqa_test500.jsonl", "w") as f:
        for qi in sorted(teacher_qidxs):
            if qi in sft:
                f.write(json.dumps(sft[qi]) + "\n")
                n_sft += 1
    print(f"  SFT-8B entries written: {n_sft}")


if __name__ == "__main__":
    main()
