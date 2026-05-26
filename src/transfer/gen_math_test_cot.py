"""
Generate math test CoTs from a Qwen3-8B model (vanilla or math-SFT) on
GSM8K test set. Uses vLLM. Output format compatible with step audit pipeline.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

PROMPT = """You are a math tutor. Solve this problem step by step. Show your reasoning naturally.

End with exactly: "The answer is N." where N is the numeric answer.

Problem: {question}"""

ANSWER_RE = re.compile(r"[Tt]he\s+answer\s+is\s+\$?([-+]?[\d,]+\.?\d*)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--start", type=int, default=0,
                    help="start index into GSM8K test (for extending an existing prefix)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load GSM8K test
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = [(d["question"], d["answer"]) for d in ds][args.start:args.start + args.n_questions]
    print(f"  {len(questions)} test questions", flush=True)

    # Resume support
    done_ids = set()
    if Path(args.output).exists():
        for line in open(args.output):
            try: done_ids.add(json.loads(line)["question_idx"])
            except: pass

    items = [(i, q, a) for i, (q, a) in enumerate(questions) if i not in done_ids]
    if not items:
        print("Nothing to do."); return

    # Load model with vLLM
    from vllm import LLM, SamplingParams
    print(f"Loading model {args.model_path} ...", flush=True)
    llm = LLM(model=args.model_path, trust_remote_code=True,
              max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=False)
    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = []
    for i, q, a in items:
        msgs = [{"role": "user", "content": PROMPT.format(question=q)}]
        try:
            r = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except (TypeError, ValueError):
            r = PROMPT.format(question=q)
        prompts.append(r)

    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, n=1)
    print(f"Generating {len(prompts)} CoTs ...", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Extract gold answers and write
    with open(args.output, "a") as f:
        for (i, q, gold), out in zip(items, outputs):
            text = out.outputs[0].text.strip()
            # Gold from GSM8K format: "...#### N"
            m = re.search(r"####\s*([-+]?\d+\.?\d*)", gold)
            gold_num = m.group(1).strip() if m else ""
            # Student answer
            m2 = ANSWER_RE.search(text)
            stu_num = m2.group(1).strip().replace(',', '') if m2 else ""
            row = {
                "question_idx": i, "question": q,
                "options": {"correct": gold_num},  # use options to fit step audit format
                "correct_answer": "correct",  # for step audit compatibility
                "gold_numeric": gold_num,
                "student_numeric": stu_num,
                "text": text,
                "answer": "correct" if stu_num and gold_num and abs(float(stu_num) - float(gold_num)) < 0.01 else "wrong",
            }
            f.write(json.dumps(row) + "\n")

    # Quick accuracy
    rows = [json.loads(l) for l in open(args.output)]
    n_correct = sum(1 for r in rows if r["answer"] == "correct")
    print(f"\nAccuracy: {n_correct}/{len(rows)} = {n_correct/max(len(rows),1)*100:.2f}%")


if __name__ == "__main__":
    main()
