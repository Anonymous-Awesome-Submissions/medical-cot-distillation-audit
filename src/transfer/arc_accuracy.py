#!/usr/bin/env python3
"""Answer accuracy for ARC base vs distilled test CoTs (extract final option letter)."""
import json, re
from pathlib import Path

AF = Path("experiments/module1/arc_full")

# robust: last "answer is (X)" / "answer is X" / "answer: X" near the end
ANS_RE = re.compile(r"answer\s*(?:is|:)?\s*\(?\s*([A-E])\b", re.IGNORECASE)


def extract(text):
    tail = text[-400:] if len(text) > 400 else text
    ms = ANS_RE.findall(tail)
    if ms:
        return ms[-1].upper()
    ms = ANS_RE.findall(text)
    return ms[-1].upper() if ms else None


def acc(fname):
    n = c = miss = 0
    for line in open(AF / fname):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        pred = extract(r.get("text", ""))
        if pred is None:
            miss += 1
            continue
        if pred == (r.get("correct_answer") or "").strip().upper():
            c += 1
    return n, c, miss


for tag, fn in [("base", "base_test_cots.jsonl"), ("distilled", "distilled_test_cots.jsonl")]:
    n, c, miss = acc(fn)
    print(f"{tag:9s}: acc={100*c/max(n,1):.2f}%  ({c}/{n}; unparsed={miss})")
