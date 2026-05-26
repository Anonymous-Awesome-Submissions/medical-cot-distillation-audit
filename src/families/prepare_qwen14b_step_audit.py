#!/usr/bin/env python3
"""Prepare paired Qwen3-14B MedQA CoTs for step-level audit.

The script samples the same question ids for vanilla and SFT Qwen3-14B,
extracts the first SC@64 chain for each question, and writes compact JSONL
files that can be passed directly to ``mega_judge_audit.py``.
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("experiments/module1")


def load_chain_dir(path: Path) -> dict[int, dict]:
    rows = {}
    for shard in sorted(path.glob("gpu_shard_*.jsonl")):
        if shard.name.endswith(".lock"):
            continue
        with shard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[int(row["question_idx"])] = row
    return rows


def first_chain(row: dict) -> tuple[str, str | None]:
    sampled = row.get("sampled") or []
    if sampled:
        return sampled[0].get("text", ""), sampled[0].get("answer")
    return row.get("text", ""), row.get("answer")


def write_audit_input(rows: dict[int, dict], qids: list[int], out_path: Path) -> None:
    with out_path.open("w") as w:
        for qid in qids:
            row = rows[qid]
            text, answer = first_chain(row)
            w.write(
                json.dumps(
                    {
                        "question_idx": qid,
                        "question": row.get("question"),
                        "options": row.get("options"),
                        "correct_answer": row.get("correct_answer"),
                        "text": text,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--name", default="pilot50", help="suffix used in output filenames")
    ap.add_argument(
        "--include-qids-file",
        type=Path,
        default=None,
        help="Optional newline-separated question ids to place first in the sample, preserving order.",
    )
    ap.add_argument(
        "--vanilla-dir",
        type=Path,
        default=DEFAULT_ROOT / "test_chains_vanilla_14b_sc64",
    )
    ap.add_argument(
        "--sft-dir",
        type=Path,
        default=DEFAULT_ROOT / "test_chains_sft_14b_sc64",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/module1/qwen14b_step_pilot50"),
    )
    args = ap.parse_args()

    vanilla = load_chain_dir(args.vanilla_dir)
    sft = load_chain_dir(args.sft_dir)
    shared = sorted(set(vanilla) & set(sft))
    if args.n_questions > len(shared):
        raise SystemExit(f"Requested {args.n_questions} questions, but only {len(shared)} paired ids exist")

    include_qids = []
    if args.include_qids_file:
        include_qids = [
            int(line.strip())
            for line in args.include_qids_file.read_text().splitlines()
            if line.strip()
        ]
        missing = sorted(set(include_qids) - set(shared))
        if missing:
            raise SystemExit(f"Included qids are not paired in both dirs: {missing[:10]}")
        if len(include_qids) > args.n_questions:
            raise SystemExit(
                f"Included {len(include_qids)} qids, but requested only {args.n_questions} questions"
            )

    rng = random.Random(args.seed)
    remaining_pool = [qid for qid in shared if qid not in set(include_qids)]
    remaining = sorted(rng.sample(remaining_pool, args.n_questions - len(include_qids)))
    qids = include_qids + remaining

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vanilla_out = args.output_dir / f"vanilla14b_medqa_{args.name}.jsonl"
    sft_out = args.output_dir / f"sft14b_medqa_{args.name}.jsonl"
    write_audit_input(vanilla, qids, vanilla_out)
    write_audit_input(sft, qids, sft_out)

    qid_path = args.output_dir / f"sample_qids_{args.name}.txt"
    qid_path.write_text("\n".join(map(str, qids)) + "\n")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Qwen3-14B paired MedQA step-error pilot input",
        "n_questions": args.n_questions,
        "seed": args.seed,
        "sample_name": args.name,
        "sample_qids_file": str(qid_path),
        "included_qids_file": str(args.include_qids_file) if args.include_qids_file else None,
        "included_qids_count": len(include_qids),
        "vanilla_source_dir": str(args.vanilla_dir),
        "sft_source_dir": str(args.sft_dir),
        "chain_selection": "first chain in sampled[0], matching the main MedQA audit convention",
        "vanilla_output": str(vanilla_out),
        "sft_output": str(sft_out),
        "shared_questions_available": len(shared),
    }
    manifest_path = args.output_dir / f"manifest_{args.name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"prepared {args.n_questions} paired questions")
    print(f"vanilla: {vanilla_out}")
    print(f"sft:     {sft_out}")
    print(f"qids:    {qid_path}")
    print(f"manifest:{manifest_path}")


if __name__ == "__main__":
    main()
