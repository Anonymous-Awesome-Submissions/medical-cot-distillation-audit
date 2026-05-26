#!/usr/bin/env python3
"""Build full-1273-question CoT audit input files (one representative chain per
question = the FIRST of the 64 sampled chains, exactly as prepare_q2_audit_inputs.py
did for the 500-question version). Output sorted by question_idx so the audit's
enumerate-position == question_idx.

Output: experiments/module1/q2_hallucination_full/{tag}_cot_medqa_full.jsonl
"""
import glob, json, os
from pathlib import Path

BASE = Path("experiments/module1")
OUT = BASE / "q2_hallucination_full"
OUT.mkdir(parents=True, exist_ok=True)

SRC = {
    "vanilla":  BASE / "test_chains_vanilla_8b_sc64",
    "sft8b":    BASE / "test_chains_sft_8b_plain_t09",
    "weak_sft": BASE / "test_chains_weak_teacher_8b",
}

for tag, d in SRC.items():
    rows = {}
    for fp in sorted(glob.glob(str(d / "gpu_shard_*.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            qi = q.get("question_idx")
            samples = q.get("sampled") or []
            if qi is None or not samples:
                continue
            first = samples[0]
            rows[qi] = {
                "question_idx": qi,
                "question": q.get("question", ""),
                "options": q.get("options", {}),
                "correct_answer": (q.get("correct_answer") or "").strip(),
                "text": first.get("text", ""),
                "answer": first.get("answer", ""),
            }
    op = OUT / f"{tag}_cot_medqa_full.jsonl"
    with open(op, "w") as f:
        for qi in sorted(rows):
            f.write(json.dumps(rows[qi]) + "\n")
    print(f"{tag}: wrote {len(rows)} CoTs -> {op}")
