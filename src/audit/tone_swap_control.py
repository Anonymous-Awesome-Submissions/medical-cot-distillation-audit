#!/usr/bin/env python3
"""Large-scale tone-swap control for the MedQA step audit.

For each already-audited step, rewrite the step into the opposite surface style
while preserving every factual/logical claim, verify preservation, then re-judge
the rewritten step with the same style-blind medical prompt.

The intended main run is:
  base/vanilla steps  -> fluent prose style
  distilled/sft steps -> numbered-outline style

If the distilled-minus-base error gap persists after this two-way style swap,
the main result is unlikely to be only a judge reaction to prose vs markdown
surface form.
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

## ONE reasoning step from the AI's chain-of-thought
"{step_text}"

## Critical instructions

You MUST evaluate ONLY whether the FACTUAL CONTENT is wrong.

IGNORE:
- The tone or register (committal vs hedged: "X is" vs "X may be")
- Grammatical confidence (assertive vs uncertain phrasing)
- Whether the writer used hedging words ("might", "may", "could", "possibly")
- Stylistic certainty markers ("clearly", "definitely", "obviously")
- Whether the step sounds confident or uncertain

ONLY judge: "If I IGNORE the tone and read the underlying claim, is the medical fact WRONG?"

Examples:
  "Cipro is contraindicated in pregnancy" -> correct (factually true)
  "Cipro might be considered for pregnancy UTI" -> ERROR (the underlying claim -- that cipro is appropriate -- is wrong, regardless of hedge)
  "Doxycycline is generally avoided" -> correct (factually true)
  "Doxycycline may help in pregnancy UTI" -> ERROR (factually wrong regardless of hedge)

Respond exactly:
JUDGMENT: <correct|error|uncertain>
EXPLANATION: <one sentence focusing on FACTUAL content, not tone>"""

REWRITE_PROMPTS = {
    "prose": """Rewrite the following single reasoning step in fluent prose style, like one compact paragraph from a polished explanation.

Rules:
- Do not use markdown, bullets, headings, numbering, or "Step N" language.
- Keep EVERY factual claim, medical fact, causal mechanism, comparison, and logical inference exactly as stated.
- Do not add, remove, correct, weaken, strengthen, or reinterpret any content, even if it seems medically wrong.
- Preserve the answer option letters and medical terms if they appear.
- Output only the rewritten step.

Original step:
\"\"\"{step_text}\"\"\"""",
    "outline": """Rewrite the following single reasoning step in a concise numbered-outline style, like a markdown step from a structured solution.

Rules:
- Use a short step heading or numbered line, and bullets only if helpful.
- Keep EVERY factual claim, medical fact, causal mechanism, comparison, and logical inference exactly as stated.
- Do not add, remove, correct, weaken, strengthen, or reinterpret any content, even if it seems medically wrong.
- Preserve the answer option letters and medical terms if they appear.
- Output only the rewritten step.

Original step:
\"\"\"{step_text}\"\"\"""",
    "neutral": """Rewrite the following single reasoning step in a flat neutral style.

Rules:
- Remove markdown, decorative wording, and emphatic phrasing.
- Keep EVERY factual claim, medical fact, causal mechanism, comparison, and logical inference exactly as stated.
- Do not add, remove, correct, weaken, strengthen, or reinterpret any content, even if it seems medically wrong.
- Preserve the answer option letters and medical terms if they appear.
- Output only the rewritten step.

Original step:
\"\"\"{step_text}\"\"\"""",
}

VERIFY_PROMPT = """Compare the original reasoning step with the rewritten reasoning step.

Question: Did the rewritten step preserve the factual/logical content of the original?

Mark CHANGED if the rewrite added, removed, corrected, contradicted, weakened, strengthened, or reinterpreted any medical fact, causal mechanism, comparison, answer option, or logical inference. Stylistic changes alone are fine.

Original:
\"\"\"{original}\"\"\"

Rewritten:
\"\"\"{rewritten}\"\"\"

Respond exactly:
VERDICT: <preserved|changed|uncertain>
EXPLANATION: <one sentence>"""


def call_model(client: OpenAI, model: str, prompt: str, max_tokens: int, max_retries: int = 5) -> str:
    extra = {}
    if ("GLM-" in model) or ("Kimi-K2." in model):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
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
    return ""


def parse_judgment(text: str) -> tuple[str, str]:
    m = re.search(r"JUDGMENT:\s*(\w+)", text or "", re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID:
        j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    return j, (m.group(1).strip() if m else "")


def parse_verdict(text: str) -> tuple[str, str]:
    m = re.search(r"VERDICT:\s*(\w+)", text or "", re.IGNORECASE)
    v = m.group(1).lower() if m else "uncertain"
    if v not in {"preserved", "changed", "uncertain"}:
        v = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    return v, (m.group(1).strip() if m else "")


def load_context(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            rows[i] = {
                "question": r.get("question", ""),
                "options": r.get("options", {}) or {},
                "correct_answer": (r.get("correct_answer") or "").strip(),
            }
    return rows


def load_judgments(path: Path, context: dict[int, dict], limit: int | None, n_questions: int | None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qidx = int(r["qidx"])
            if n_questions is not None and qidx >= n_questions:
                continue
            if qidx not in context:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--target-style", choices=sorted(REWRITE_PROMPTS), required=True)
    ap.add_argument("--rewrite-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--verify-model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--judge-model", default="Pro/moonshotai/Kimi-K2.6")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-questions", type=int, default=None,
                    help="Use only questions with qidx < N; preferred for paired subset controls.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key and not args.dry_run:
        sys.exit("SILICONFLOW_API_KEY not set")
    client = OpenAI(api_key=key or "EMPTY", base_url=SILICONFLOW_BASE)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tone_swap_{args.model_tag}_{args.target_style}.jsonl"

    context = load_context(Path(args.cot))
    rows = load_judgments(Path(args.judgments), context, args.limit, args.n_questions)
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("rewrite_failed"):
                        continue
                    done.add(f"{d['qidx']}|{d['chain_idx']}|{d['step_idx']}|{d['target_style']}")
                except Exception:
                    pass
    tasks = [r for r in rows if f"{r['qidx']}|{r.get('chain_idx', 0)}|{r['step_idx']}|{args.target_style}" not in done]
    print(f"[tone-swap] tag={args.model_tag} target={args.target_style} rows={len(rows)} done={len(done)} tasks={len(tasks)}")
    if args.dry_run:
        for r in tasks[:3]:
            print(json.dumps({"qidx": r["qidx"], "step_idx": r["step_idx"], "judgment": r.get("judgment"),
                              "step_text": r.get("step_text", "")[:160]}, ensure_ascii=False))
        return
    if not tasks:
        return

    lock = threading.Lock()
    counts = {"correct": 0, "error": 0, "uncertain": 0, "preserved": 0, "changed": 0, "verify_uncertain": 0,
              "rewrite_failed": 0}

    def worker(r: dict) -> dict:
        step = (r.get("step_text") or "").strip()
        rewrite_prompt = REWRITE_PROMPTS[args.target_style].format(step_text=step[:1200])
        rewritten = call_model(client, args.rewrite_model, rewrite_prompt, max_tokens=600).strip().strip('"').strip()
        if not rewritten:
            return {**r, "target_style": args.target_style, "rewrite_failed": True}

        verify_text = call_model(
            client, args.verify_model,
            VERIFY_PROMPT.format(original=step[:1200], rewritten=rewritten[:1200]),
            max_tokens=220,
        )
        verdict, verdict_exp = parse_verdict(verify_text)

        ctx = context[int(r["qidx"])]
        opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(ctx["options"].items()))
        judge_prompt = STYLE_BLIND_PROMPT.format(
            question=ctx["question"][:3000],
            options=opt_str,
            correct_answer=ctx["correct_answer"],
            correct_text=ctx["options"].get(ctx["correct_answer"], ""),
            step_text=rewritten[:1200],
        )
        judge_text = call_model(client, args.judge_model, judge_prompt, max_tokens=240)
        j, exp = parse_judgment(judge_text)
        return {
            "qidx": r["qidx"],
            "chain_idx": r.get("chain_idx", 0),
            "step_idx": r["step_idx"],
            "model_tag": args.model_tag,
            "target_style": args.target_style,
            "orig_judgment": r.get("judgment"),
            "orig_step_text": step[:700],
            "rewritten_step_text": rewritten[:900],
            "rewrite_model": args.rewrite_model,
            "verify_model": args.verify_model,
            "preservation_verdict": verdict,
            "preservation_explanation": verdict_exp[:300],
            "judge_model": args.judge_model,
            "judgment": j,
            "explanation": exp[:300],
        }

    fout = out_path.open("a")
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, r) for r in tasks]
        for fut in as_completed(futures):
            rec = fut.result()
            completed += 1
            if rec.get("rewrite_failed"):
                counts["rewrite_failed"] += 1
                if completed % 100 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1)
                    eta = (len(tasks) - completed) / max(rate, 1e-6)
                    committed = counts["correct"] + counts["error"]
                    err = counts["error"] / max(committed, 1) * 100
                    print(f"  {completed}/{len(tasks)} {rate:.2f}/s eta={eta/60:.1f}m err={err:.1f}% counts={counts}", flush=True)
                continue
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            j = rec.get("judgment")
            if j in counts:
                counts[j] += 1
            v = rec.get("preservation_verdict")
            if v == "preserved":
                counts["preserved"] += 1
            elif v == "changed":
                counts["changed"] += 1
            elif v == "uncertain":
                counts["verify_uncertain"] += 1
            if completed % 100 == 0 or completed == len(tasks):
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
