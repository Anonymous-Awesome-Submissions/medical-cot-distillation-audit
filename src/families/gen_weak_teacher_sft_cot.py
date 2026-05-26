#!/usr/bin/env python3
"""
Generate weak-teacher SFT (Qwen3-8B trained from Qwen3-14B vanilla teacher)
CoTs on the same 500 MedQA test questions used for vanilla / strong-teacher /
teacher audits. Output format identical so existing analysis pipelines work.
"""
import argparse
import json
import re
import time
from pathlib import Path

WEAK_TEACHER_MERGED = "experiments/module1/qwen3_8b_weak_teacher_merged"
COT_DIR = "experiments/module1/q2_hallucination"
OUT_PATH = f"{COT_DIR}/weak_sft_cot_medqa_test500.jsonl"

PROMPT = """You are an expert physician taking the USMLE. Work through the following question as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Why some answer options fit and others don't
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is (X)." where X is A, B, C, or D.

Question: {question}

Options:
{options}"""

ANSWER_RE = re.compile(r"[Tt]he\s+answer\s+is\s*\(?([A-E])\)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Load source vanilla CoTs to get exact same questions/options
    vanilla_path = f"{COT_DIR}/vanilla_cot_medqa_test500.jsonl"
    print(f"Loading source questions from {vanilla_path}", flush=True)
    items = []
    with open(vanilla_path) as f:
        for line in f:
            items.append(json.loads(line))
    items = items[:args.n_questions]
    print(f"  {len(items)} questions", flush=True)

    # Resume-friendly: collect done question_idx
    done_ids = set()
    if Path(OUT_PATH).exists():
        with open(OUT_PATH) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_idx"])
                except Exception:
                    pass
    items = [it for it in items if it["question_idx"] not in done_ids]
    print(f"  {len(items)} remaining after resume ({len(done_ids)} done)", flush=True)
    if not items:
        print("Nothing to do.")
        return

    # vLLM batched generation
    print(f"Loading vLLM model: {WEAK_TEACHER_MERGED}", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=WEAK_TEACHER_MERGED, trust_remote_code=True,
        max_model_len=4096, gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts using chat template (Qwen3 with thinking disabled — match SFT setup)
    prompts = []
    for it in items:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(it["options"].items()))
        user = PROMPT.format(question=it["question"], options=opts)
        msgs = [{"role": "user", "content": user}]
        try:
            r = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            r = user
        prompts.append(r)

    print(f"Generating {len(prompts)} CoTs ...", flush=True)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, n=1)
    t0 = time.time()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"  Generation done in {time.time()-t0:.1f}s", flush=True)

    # Append to output
    with open(OUT_PATH, "a") as f:
        for it, out in zip(items, outputs):
            text = out.outputs[0].text.strip()
            m = ANSWER_RE.search(text)
            answer = m.group(1).upper() if m else ""
            row = {
                "question_idx": it["question_idx"],
                "question": it["question"],
                "options": it["options"],
                "correct_answer": it["correct_answer"],
                "text": text,
                "answer": answer,
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(items)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
