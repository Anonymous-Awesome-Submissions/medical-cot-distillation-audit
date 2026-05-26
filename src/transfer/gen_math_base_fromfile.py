#!/usr/bin/env python3
"""Generate base/student CoTs on a MATH problem file (jsonl with question/gold_numeric).
vLLM, one T=0.7 chain per problem. Output matches the math step-audit format.
Same prompt as the DeepSeek teacher gen for a fair teacher-vs-base comparison."""
import argparse, json
from pathlib import Path

PROMPT = """Solve the following mathematics problem. Reason step by step, numbering your steps, and end with a line "The final answer is X." where X is the answer.

Problem: {question}"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=2000)
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.input)]
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": PROMPT.format(question=r["question"])}]
        try: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception: p = PROMPT.format(question=r["question"])
        prompts.append(p)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, n=1)
    outs = llm.generate(prompts, sp)
    with open(args.output, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"question_idx": r["question_idx"], "question": r["question"],
                "gold_numeric": r["gold_numeric"], "type": r.get("type"), "level": r.get("level"),
                "text": o.outputs[0].text.strip()}) + "\n")
    print(f"wrote {len(rows)} base CoTs -> {args.output}", flush=True)

if __name__ == "__main__":
    main()
