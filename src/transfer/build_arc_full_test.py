#!/usr/bin/env python3
"""Build the FULL ARC-Challenge test set (1172q), reusing the existing 500 verbatim
at idx 0..499 and appending the complement at idx 500+. Also emits the complement-only
file so we generate/audit only the new questions and reuse the 500 already done."""
import json
from pathlib import Path
from datasets import load_dataset

AF = Path("experiments/module1/arc_full")

LETTERS = "ABCDE"

def norm(rec):
    """HF row -> {question, options{A..}, correct_answer letter}. Map numeric labels to letters."""
    labels = rec["choices"]["label"]
    texts = rec["choices"]["text"]
    # position-based letter mapping (handles numeric '1','2' or letter labels)
    lab2let = {lab: LETTERS[i] for i, lab in enumerate(labels)}
    options = {LETTERS[i]: t for i, t in enumerate(texts)}
    gold = lab2let.get(rec["answerKey"], rec["answerKey"])
    return {"question": rec["question"], "options": options, "correct_answer": gold}


def main():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    existing = [json.loads(l) for l in open(AF / "test_problems.jsonl")]
    ex_q = {e["question"] for e in existing}
    assert len(existing) == 500

    complement = []
    for r in ds:
        if r["question"] in ex_q:
            continue
        complement.append(norm(r))

    # full = existing 500 (verbatim, idx 0..499) + complement (idx 500..)
    full = []
    for e in existing:
        full.append({**e})  # keeps its question_idx (0..499)
    for j, c in enumerate(complement):
        full.append({**c, "question_idx": 500 + j})

    # sanity: idx 0..499 unchanged
    for i in range(500):
        assert full[i]["question_idx"] == i

    with open(AF / "full_test_problems.jsonl", "w") as f:
        for r in full:
            f.write(json.dumps(r) + "\n")
    with open(AF / "remaining_test_problems.jsonl", "w") as f:
        for j, c in enumerate(complement):
            f.write(json.dumps({**c, "question_idx": 500 + j}) + "\n")

    print(f"HF test={len(ds)}  existing=500  complement={len(complement)}  full={len(full)}")
    print(f"wrote full_test_problems.jsonl ({len(full)}) and remaining_test_problems.jsonl ({len(complement)})")


if __name__ == "__main__":
    main()
