#!/usr/bin/env python3
"""Kimi step audit under model-agnostic segmentation rules.

The primary audit segments numbered/markdown traces by explicit step markers and
falls back to paragraphs. This script deliberately ignores those model-specific
markers and applies the same segmentation rule to every trace. It is a direct
robustness check for the concern that "per-step" error rates are induced by
different formatting styles rather than different trace reliability.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
VALID = {"correct", "error", "uncertain"}

STYLE_BLIND_PROMPT = """You are a senior physician evaluating a medical AI model's reasoning.

## Question
{question}

## Options
{options}

## Correct Answer
({correct_answer}) {correct_text}

## ONE reasoning segment from the AI's chain-of-thought
"{step_text}"

## Critical instructions

You MUST evaluate ONLY whether this segment contains a FACTUAL or LOGICAL ERROR (a wrong medical fact, a wrong mechanism, an inference that does not follow, a misapplied rule). IGNORE tone, hedging, confidence, and stylistic register entirely. A purely descriptive or organizational segment ("Let me consider the options") with no factual claim is CORRECT.

Respond exactly:
JUDGMENT: <correct|error|uncertain>
EXPLANATION: <one sentence focusing on FACTUAL content, not tone>"""


def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n\s*[-_*]{3,}\s*\n", "\n\n", text)
    text = re.sub(r"^\s{0,3}#{1,5}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:Step\s*)?\d+[\.\):]\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_chunks(text: str, min_chars: int = 45, max_chars: int = 650, max_chunks: int = 28) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    raw = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", para)
        raw.extend(p.strip() for p in pieces if len(p.strip()) > 10)
    chunks, cur = [], ""
    for piece in raw:
        if not cur:
            cur = piece
        elif len(cur) + 1 + len(piece) <= max_chars:
            cur = cur + " " + piece
        else:
            if len(cur) >= min_chars:
                chunks.append(cur[:900])
            cur = piece
        if len(chunks) >= max_chunks:
            break
    if len(chunks) < max_chunks and len(cur) >= min_chars:
        chunks.append(cur[:900])
    return chunks[:max_chunks]


def word_windows(text: str, window_words: int = 90, max_chunks: int = 28) -> list[str]:
    text = clean_text(text)
    words = text.split()
    chunks = []
    for i in range(0, len(words), window_words):
        chunk = " ".join(words[i:i + window_words]).strip()
        if len(chunk) >= 45:
            chunks.append(chunk[:900])
        if len(chunks) >= max_chunks:
            break
    return chunks


def parse_judgment(text: str) -> tuple[str, str]:
    m = re.search(r"JUDGMENT:\s*(\w+)", text or "", re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID:
        j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    return j, (m.group(1).strip() if m else "")


def call_judge(client: OpenAI, model: str, prompt: str, max_retries: int = 5) -> str:
    extra = {}
    if ("GLM-" in model) or ("Kimi-K2." in model):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
                timeout=120,
                **extra,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err or "limit" in err:
                time.sleep(min(2 ** attempt * 2.0, 60))
            elif "timeout" in err or "overload" in err or "502" in err or "503" in err:
                time.sleep(min(2 ** attempt * 1.5, 45))
            else:
                time.sleep(min(2 ** attempt, 20))
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--segmentation", choices=["sentence", "window"], default="sentence")
    ap.add_argument("--judge-model", default="Pro/moonshotai/Kimi-K2.6")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key and not args.dry_run:
        sys.exit("SILICONFLOW_API_KEY not set")
    client = OpenAI(api_key=key or "EMPTY", base_url=SILICONFLOW_BASE)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judgments_{args.model_tag}_{args.segmentation}.jsonl"

    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(f"{d['qidx']}|{d['segment_idx']}")
                except Exception:
                    pass

    tasks = []
    with Path(args.input).open() as f:
        for qi, line in enumerate(f):
            if args.n_questions and qi >= args.n_questions:
                break
            if not line.strip():
                continue
            e = json.loads(line)
            text = e.get("text", "")
            segments = sentence_chunks(text) if args.segmentation == "sentence" else word_windows(text)
            opt_str = "\n".join(f"({k}) {v}" for k, v in sorted((e.get("options") or {}).items()))
            for sidx, seg in enumerate(segments):
                key_id = f"{qi}|{sidx}"
                if key_id in done:
                    continue
                prompt = STYLE_BLIND_PROMPT.format(
                    question=(e.get("question") or "")[:3000],
                    options=opt_str,
                    correct_answer=(e.get("correct_answer") or "").strip(),
                    correct_text=(e.get("options") or {}).get((e.get("correct_answer") or "").strip(), ""),
                    step_text=seg,
                )
                tasks.append((qi, sidx, seg, prompt))

    print(f"[uniform-seg] tag={args.model_tag} mode={args.segmentation} done={len(done)} tasks={len(tasks)}")
    if args.dry_run:
        for t in tasks[:5]:
            print(json.dumps({"qidx": t[0], "segment_idx": t[1], "segment": t[2][:200]}, ensure_ascii=False))
        return
    if not tasks:
        return

    lock = threading.Lock()
    counts = {"correct": 0, "error": 0, "uncertain": 0}
    fout = out_path.open("a")
    t0 = time.time()
    completed = 0

    def worker(task):
        qi, sidx, seg, prompt = task
        reply = call_judge(client, args.judge_model, prompt)
        j, exp = parse_judgment(reply)
        return qi, sidx, seg, j, exp

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            qi, sidx, seg, j, exp = fut.result()
            rec = {
                "qidx": qi,
                "segment_idx": sidx,
                "model_tag": args.model_tag,
                "segmentation": args.segmentation,
                "judge_model": args.judge_model,
                "step_text": seg[:700],
                "judgment": j,
                "explanation": exp[:300],
            }
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            counts[j] = counts.get(j, 0) + 1
            completed += 1
            if completed % 200 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1)
                eta = (len(tasks) - completed) / max(rate, 1e-6)
                committed = counts["correct"] + counts["error"]
                err = counts["error"] / max(committed, 1) * 100
                print(f"  {completed}/{len(tasks)} {rate:.2f}/s eta={eta/60:.1f}m err={err:.1f}% counts={counts}", flush=True)
    fout.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
