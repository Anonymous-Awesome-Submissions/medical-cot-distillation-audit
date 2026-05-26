"""
Generate DeepSeek-V3 teacher CoTs on GSM8K (via SiliconFlow API).

We use ~3000 GSM8K train questions to keep cost low (~¥3-5 for 3K * 800 tokens
with concurrency=20). Output JSONL has fields matching our medical SFT data
format so existing train_lora_layerband.py works.
"""
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

PROMPT = """You are a math tutor working through a problem. Reason step by step. Show your reasoning naturally — when you're uncertain about an approach, say so; when you're confident, state directly.

End with exactly: "The answer is N." where N is the final numeric answer.

Problem: {question}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=3000)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Question index range start (inclusive)")
    ap.add_argument("--end-idx", type=int, default=None,
                    help="Question index range end (exclusive); None = n-questions")
    ap.add_argument("--output", required=True)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--source", default="gsm8k")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="GSM8K split to generate on")
    ap.add_argument("--keep-all", action="store_true",
                    help="write generated CoTs even when the final numeric answer is not verified")
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key: sys.exit("SILICONFLOW_API_KEY not set")
    client = OpenAI(api_key=key, base_url="https://api.siliconflow.cn/v1")

    # Load GSM8K questions
    if args.source == "gsm8k":
        # GSM8K available via huggingface datasets, or from local
        # For simplicity let's check if we have it locally
        from datasets import load_dataset
        try:
            ds = load_dataset("gsm8k", "main", split=args.split)
        except Exception:
            print("Falling back to local cache or download...")
            ds = load_dataset("gsm8k", "main", split=args.split)
        questions_all = [(d["question"], d["answer"]) for d in ds][:args.n_questions]
    else:
        sys.exit(f"Unknown source: {args.source}")

    # Apply index range
    end = args.end_idx if args.end_idx is not None else len(questions_all)
    questions = [(i, q, a) for i, (q, a) in enumerate(questions_all)
                 if args.start_idx <= i < end]
    print(f"  Total: {len(questions_all)}, range [{args.start_idx}, {end}), "
          f"this job: {len(questions)} questions", flush=True)

    # Resume support
    done_ids = set()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("question_idx"))
                except: pass
    print(f"  {len(done_ids)} already done, generating remaining", flush=True)

    tasks = []
    for (i, q, a) in questions:
        if i in done_ids: continue
        tasks.append((i, q, a))

    fout = open(args.output, "a")
    _lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}
    t0 = time.time()

    def call_teacher(task):
        idx, q, gold_answer = task
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.2",
                    messages=[{"role": "user", "content": PROMPT.format(question=q)}],
                    temperature=0.7, max_tokens=1500, timeout=60,
                )
                cot = resp.choices[0].message.content or ""
                # Extract gold numeric answer for verification (GSM8K format: "answer #### N")
                m = re.search(r"####\s*([-+]?\d+\.?\d*)", gold_answer)
                gold_num = m.group(1).strip().rstrip('.').replace(',', '') if m else ""
                # Extract teacher's answer
                m2 = re.search(r"[Tt]he\s+answer\s+is\s+\$?([-+]?[\d,]+\.?\d*)", cot)
                teacher_num = m2.group(1).strip().rstrip('.').replace(',', '') if m2 else ""
                # Numeric comparison (handle both "72" and "72.0")
                ok = False
                if teacher_num and gold_num:
                    try:
                        ok = abs(float(teacher_num) - float(gold_num)) < 0.01
                    except ValueError:
                        ok = teacher_num == gold_num
                return idx, q, gold_num, cot, teacher_num, ok
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    time.sleep(min(2 ** attempt * 2, 30))
                else:
                    time.sleep(min(2 ** attempt, 8))
        return idx, q, "", "", "", False

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_teacher, t) for t in tasks]
        completed = 0
        for fut in as_completed(futures):
            try:
                idx, q, gold_num, cot, teacher_num, ok = fut.result()
            except Exception as e:
                continue
            if cot and (ok or args.keep_all):
                # SFT data format: sft_input, sft_target
                sft_input = f"Solve this math problem step by step. Show your reasoning naturally.\n\nProblem: {q}"
                row = {
                    "question_idx": idx, "question": q,
                    "gold_numeric": gold_num, "teacher_answer": teacher_num,
                    "teacher_answer_correct": bool(ok),
                    "sft_input": sft_input, "sft_target": cot,
                    "text": cot,
                }
                with _lock:
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                counts["ok"] += 1
            else:
                counts["fail"] += 1
            completed += 1
            if completed % 50 == 0:
                el = time.time() - t0
                rate = completed / max(el, 1)
                eta = (len(tasks) - completed) / max(rate, 0.001)
                print(f"  {completed}/{len(tasks)} ok={counts['ok']} fail={counts['fail']} "
                      f"rate={rate:.1f}/s elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    fout.close()
    print(f"\nDone. ok={counts['ok']} fail={counts['fail']}")


if __name__ == "__main__":
    main()
