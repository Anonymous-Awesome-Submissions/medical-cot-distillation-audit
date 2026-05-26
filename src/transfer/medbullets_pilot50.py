#!/usr/bin/env python3
"""MedBullets5 50-question pilot for teacher/base/distilled trace reliability."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "module1" / "medbullets_pilot50"
BASE_DIR = ROOT / "experiments" / "module1" / "test_chains_vanilla_8b_medbullets"
DIST_DIR = ROOT / "experiments" / "module1" / "test_chains_qwen3_8b_medbullets_sc64"
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
KIMI_MODEL = "Pro/moonshotai/Kimi-K2.6"
DEEPSEEK_MODEL = "deepseek-chat"


ANSWER_RE = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:|final\s+answer\s*:)\s*\(?([A-J])\)?",
    re.IGNORECASE,
)
LOOSE_ANSWER_RE = re.compile(r"\(([A-J])\)")
VALID = {"correct", "error", "uncertain"}


TEACHER_PROMPT = """You are an expert physician answering a medical board-style multiple-choice question. Reason step by step, numbering your steps, and end with exactly "The answer is (X)." where X is the option letter.

Question:
{question}

Options:
{options}"""


STYLE_BLIND_MEDICAL_PROMPT = """You are a senior physician evaluating one segment of a medical AI model's reasoning.

## Question
{question}

## Options
{options}

## Correct answer
({correct_answer}) {correct_text}

## One reasoning segment from the model
"{step_text}"

## Critical instructions

Evaluate ONLY whether the segment contains a factual or logical error: a wrong medical fact, a wrong mechanism, a misread clinical finding, a wrong exclusion of a diagnosis, or an inference that does not follow.

IGNORE:
- The tone or register
- Whether the wording is assertive or hedged
- Whether the model sounds confident or uncertain
- Whether the segment comes from a numbered list or prose

Do NOT mark a segment as wrong merely because it considers an alternative diagnosis. Mark ERROR only if the underlying medical claim is wrong or the inference is unsupported. A purely organizational segment with no factual claim is CORRECT. If the segment cannot be judged from the case and standard medical knowledge, use UNCERTAIN.

Respond exactly:
JUDGMENT: <correct|error|uncertain>
EXPLANATION: <one sentence focusing on factual/logical content, not tone>"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chain_rows(path_or_dir: Path) -> dict[int, dict[str, Any]]:
    files = [path_or_dir] if path_or_dir.is_file() else sorted(path_or_dir.glob("gpu_shard_*.jsonl"))
    by_idx: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for path in files:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qidx = int(row["question_idx"])
                if qidx in by_idx:
                    duplicates += 1
                    continue
                by_idx[qidx] = row
    if duplicates:
        print(f"Deduplicated {duplicates} repeated question_idx rows from {path_or_dir}", flush=True)
    return by_idx


def parse_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    if m:
        return m.group(1).upper()
    matches = LOOSE_ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else ""


def first_chain(row: dict[str, Any]) -> tuple[str, str]:
    sampled = row.get("sampled") or []
    if not sampled:
        return "", ""
    text = sampled[0].get("text", "") or ""
    answer = (sampled[0].get("answer") or "").strip().upper()
    if not answer:
        answer = parse_answer(text)
    return text, answer


def build(args: argparse.Namespace) -> None:
    base = load_chain_rows(BASE_DIR)
    distilled = load_chain_rows(DIST_DIR)
    common = sorted(set(base) & set(distilled))[: args.n]
    if len(common) < args.n:
        raise SystemExit(f"Only {len(common)} common MedBullets5 questions found")

    problems = []
    cond_rows = {"base": [], "distilled": []}
    for qidx in common:
        b = base[qidx]
        d = distilled[qidx]
        if b["question"] != d["question"]:
            raise SystemExit(f"Question mismatch at qidx={qidx}")
        gold = (b.get("correct_answer") or b.get("answer_idx") or "").strip().upper()
        problem = {
            "question_idx": qidx,
            "question": b["question"],
            "options": b["options"],
            "correct_answer": gold,
        }
        problems.append(problem)
        text, answer = first_chain(b)
        cond_rows["base"].append({**problem, "answer": answer, "text": text})
        text, answer = first_chain(d)
        cond_rows["distilled"].append({**problem, "answer": answer, "text": text})

    write_jsonl(OUT_DIR / "problems.jsonl", problems)
    for tag, rows in cond_rows.items():
        write_jsonl(OUT_DIR / f"{tag}_cots.jsonl", rows)
    print(f"Built {len(common)} MedBullets5 pilot questions in {OUT_DIR}")


def call_deepseek(client: OpenAI, model: str, prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1600,
                timeout=180,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "rate" in err or "limit" in err:
                time.sleep(min(2 ** attempt * 2.0, 60))
            else:
                time.sleep(min(2 ** attempt, 20))
    return ""


def generate_teacher(args: argparse.Namespace) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    rows = read_jsonl(OUT_DIR / "problems.jsonl")
    out_path = OUT_DIR / "teacher_cots.jsonl"
    done = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            done[int(row["question_idx"])] = row
    todo = [row for row in rows if int(row["question_idx"]) not in done]
    print(f"Teacher generation: {len(done)} done, {len(todo)} to generate")
    if not todo:
        return

    lock = threading.Lock()
    with out_path.open("a") as fout:
        t0 = time.time()

        def worker(row: dict[str, Any]) -> dict[str, Any]:
            opts = "\n".join(f"{k}. {v}" for k, v in sorted(row["options"].items()))
            text = call_deepseek(client, args.model, TEACHER_PROMPT.format(
                question=row["question"],
                options=opts,
            ))
            return {**row, "answer": parse_answer(text), "text": text, "teacher_model": args.model}

        completed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(worker, row) for row in todo]
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1.0)
                    eta = (len(todo) - completed) / max(rate, 1e-6)
                    print(f"  {completed}/{len(todo)}  {rate:.2f}/s  eta={eta/60:.1f}m", flush=True)


def clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^\s{0,3}#{1,5}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:Step\s*)?\d+[\.\):]\s*", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\s*(?:The answer is\s*\([A-J]\)\.?\s*)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_chunks(text: str, min_chars: int = 45, max_chars: int = 650, max_chunks: int = 18) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        split = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", para)
        pieces.extend(p.strip() for p in split if len(p.strip()) > 10)
    chunks = []
    cur = ""
    for piece in pieces:
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


def parse_judgment(text: str) -> tuple[str, str]:
    m = re.search(r"JUDGMENT:\s*(\w+)", text or "", re.IGNORECASE)
    judgment = m.group(1).lower() if m else "uncertain"
    if judgment not in VALID:
        judgment = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    return judgment, (m.group(1).strip() if m else "")


def call_judge(client: OpenAI, model: str, prompt: str, max_retries: int = 5) -> str:
    extra: dict[str, Any] = {}
    if "Kimi-K2." in model or "GLM-" in model:
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
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "rate" in err or "limit" in err:
                time.sleep(min(2 ** attempt * 2.0, 60))
            elif "timeout" in err or "overload" in err or "502" in err or "503" in err:
                time.sleep(min(2 ** attempt * 1.5, 45))
            else:
                time.sleep(min(2 ** attempt, 20))
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def audit(args: argparse.Namespace) -> None:
    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        raise SystemExit("SILICONFLOW_API_KEY not set")
    in_path = OUT_DIR / f"{args.tag}_cots.jsonl"
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")
    client = OpenAI(api_key=key, base_url=SILICONFLOW_BASE)
    rows = read_jsonl(in_path)

    out_dir = OUT_DIR / "audit_kimi_k26"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judgments_{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            done.add(f"{row['question_idx']}|{row['segment_idx']}")

    tasks = []
    for row in rows:
        opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(row["options"].items()))
        gold = row["correct_answer"]
        for seg_idx, segment in enumerate(sentence_chunks(row.get("text", ""))):
            key_id = f"{row['question_idx']}|{seg_idx}"
            if key_id in done:
                continue
            prompt = STYLE_BLIND_MEDICAL_PROMPT.format(
                question=row["question"][:5000],
                options=opt_str,
                correct_answer=gold,
                correct_text=row["options"].get(gold, ""),
                step_text=segment,
            )
            tasks.append((row, seg_idx, segment, prompt))

    print(f"Audit {args.tag}: {len(done)} done, {len(tasks)} to judge")
    if not tasks:
        return

    counts = Counter()
    lock = threading.Lock()
    with out_path.open("a") as fout:
        t0 = time.time()

        def worker(task: tuple[dict[str, Any], int, str, str]) -> dict[str, Any]:
            row, seg_idx, segment, prompt = task
            reply = call_judge(client, args.judge_model, prompt)
            judgment, explanation = parse_judgment(reply)
            return {
                "question_idx": row["question_idx"],
                "segment_idx": seg_idx,
                "model_tag": args.tag,
                "judge_model": args.judge_model,
                "segmentation": "uniform_sentence_chunks",
                "step_text": segment[:700],
                "judgment": judgment,
                "explanation": explanation[:300],
            }

        completed = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(worker, task) for task in tasks]
            for fut in as_completed(futures):
                rec = fut.result()
                counts[rec["judgment"]] += 1
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1.0)
                    eta = (len(tasks) - completed) / max(rate, 1e-6)
                    committed = counts["correct"] + counts["error"]
                    err = 100.0 * counts["error"] / max(committed, 1)
                    print(
                        f"  {completed}/{len(tasks)} {rate:.2f}/s eta={eta/60:.1f}m "
                        f"err={err:.1f}% counts={dict(counts)}",
                        flush=True,
                    )


def summarize(_: argparse.Namespace) -> None:
    out_dir = OUT_DIR / "audit_kimi_k26"
    summary: dict[str, Any] = {}
    for tag in ["teacher", "base", "distilled"]:
        path = out_dir / f"judgments_{tag}.jsonl"
        if not path.exists():
            print(f"Missing {path}")
            continue
        rows = read_jsonl(path)
        counts = Counter(row.get("judgment", "uncertain") for row in rows)
        committed = counts["correct"] + counts["error"]
        rate = counts["error"] / committed if committed else float("nan")
        by_q = defaultdict(Counter)
        for row in rows:
            by_q[int(row["question_idx"])][row.get("judgment", "uncertain")] += 1
        q_rates = []
        for c in by_q.values():
            den = c["correct"] + c["error"]
            if den:
                q_rates.append(c["error"] / den)
        mean_q = sum(q_rates) / len(q_rates) if q_rates else float("nan")
        summary[tag] = {
            "n_segments": len(rows),
            "correct": counts["correct"],
            "error": counts["error"],
            "uncertain": counts["uncertain"],
            "committed_error_rate": rate,
            "mean_question_error_rate": mean_q,
            "n_questions_with_committed": len(q_rates),
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if {"teacher", "base", "distilled"} <= set(summary):
        t = summary["teacher"]["committed_error_rate"]
        b = summary["base"]["committed_error_rate"]
        d = summary["distilled"]["committed_error_rate"]
        print()
        print(f"ordering teacher < base < distilled: {t < b < d}")
        print(f"rates: teacher={100*t:.1f}%  base={100*b:.1f}%  distilled={100*d:.1f}%")

    print("\nanswer accuracy on the same 50 first chains:")
    for tag in ["teacher", "base", "distilled"]:
        path = OUT_DIR / f"{tag}_cots.jsonl"
        if not path.exists():
            continue
        total = correct = missing = 0
        for row in read_jsonl(path):
            total += 1
            ans = (row.get("answer") or "").strip().upper()
            if not ans:
                missing += 1
            if ans == (row.get("correct_answer") or "").strip().upper():
                correct += 1
        print(f"  {tag:9s}: {100*correct/max(total,1):.1f}% ({correct}/{total}), missing={missing}")

    if {"teacher", "base", "distilled"} <= set(summary):
        print("\npaired question-level deltas:")
        rng = random.Random(42)
        by_tag = {}
        for tag in ["teacher", "base", "distilled"]:
            by_q = defaultdict(Counter)
            for row in read_jsonl(out_dir / f"judgments_{tag}.jsonl"):
                by_q[int(row["question_idx"])][row.get("judgment", "uncertain")] += 1
            by_tag[tag] = by_q
        common = sorted(set(by_tag["teacher"]) & set(by_tag["base"]) & set(by_tag["distilled"]))
        for a, b in [("base", "teacher"), ("distilled", "base"), ("distilled", "teacher")]:
            diffs = []
            for qidx in common:
                ca = by_tag[a][qidx]
                cb = by_tag[b][qidx]
                da = ca["correct"] + ca["error"]
                db = cb["correct"] + cb["error"]
                if da and db:
                    diffs.append(ca["error"] / da - cb["error"] / db)
            if not diffs:
                continue
            boots = []
            for _ in range(10000):
                sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
                boots.append(sum(sample) / len(sample))
            boots.sort()
            lo = boots[int(0.025 * len(boots))]
            hi = boots[int(0.975 * len(boots))]
            worse = sum(1 for d in diffs if d > 0)
            better = sum(1 for d in diffs if d < 0)
            tie = len(diffs) - worse - better
            print(
                f"  {a} - {b}: {100*sum(diffs)/len(diffs):+.2f} pp, "
                f"95% boot CI [{100*lo:+.2f}, {100*hi:+.2f}], "
                f"q worse/better/tie={worse}/{better}/{tie}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build")
    p.add_argument("--n", type=int, default=50)
    p.set_defaults(func=build)

    p = sub.add_parser("teacher")
    p.add_argument("--model", default=DEEPSEEK_MODEL)
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(func=generate_teacher)

    p = sub.add_parser("audit")
    p.add_argument("--tag", choices=["teacher", "base", "distilled"], required=True)
    p.add_argument("--judge-model", default=KIMI_MODEL)
    p.add_argument("--concurrency", type=int, default=16)
    p.set_defaults(func=audit)

    p = sub.add_parser("summarize")
    p.set_defaults(func=summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
