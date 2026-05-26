"""Eval cross-family Layer-Band SFT — generate CoTs + measure acc/hedge."""
import os, sys, argparse
import torch
import json, re, time
from pathlib import Path

PROJ_ROOT = "."
COT_DIR = "experiments/module1/q2_hallucination"
OUT_BASE = "experiments/module1/mechanism/crossfam_eval"

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

HEDGING_MARKERS = [
    r"\bmight\b", r"\bmay\b", r"\bcould\b", r"\bpossibly\b",
    r"\bperhaps\b", r"\bprobably\b", r"\blikely\b", r"\bunlikely\b",
    r"\bsuggest", r"\bindicate", r"\bappear", r"\bconsistent\s+with\b",
    r"\btypically\b", r"\bgenerally\b",
]


def hedge_density(text):
    if not text: return 0.0
    n = len(text.split())
    if n < 5: return 0.0
    tl = text.lower()
    cnt = sum(len(re.findall(p, tl)) for p in HEDGING_MARKERS)
    return 100.0 * cnt / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-adapter", default=None,
                    help="LoRA adapter to merge; omit for base/un-distilled generation")
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--n-questions", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(OUT_BASE) / args.arm_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cot = out_dir / "generated_cots.jsonl"

    done_ids = set()
    if out_cot.exists():
        for line in open(out_cot):
            try: done_ids.add(json.loads(line)["question_idx"])
            except: pass

    items = [json.loads(l) for l in open(f"{COT_DIR}/vanilla_cot_medqa_test500.jsonl")][:args.n_questions]
    items = [it for it in items if it["question_idx"] not in done_ids]
    print(f"  {len(items)} questions to gen ({len(done_ids)} done)", flush=True)

    if items:
        if args.lora_adapter:
            merged_path = out_dir / "merged_model"
            if not (merged_path / "config.json").exists():
                print(f"Merging LoRA into {args.base_model} ...", flush=True)
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from peft import PeftModel
                tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
                base = AutoModelForCausalLM.from_pretrained(
                    args.base_model, torch_dtype=torch.bfloat16,
                    trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True,
                )
                merged = PeftModel.from_pretrained(base, args.lora_adapter, torch_dtype=torch.bfloat16)
                merged = merged.merge_and_unload()
                merged_path.mkdir(parents=True, exist_ok=True)
                merged.save_pretrained(merged_path, safe_serialization=True, max_shard_size="5GB")
                tok.save_pretrained(merged_path)
                del merged, base
                torch.cuda.empty_cache()
            model_path = str(merged_path)
        else:
            # Base/un-distilled: load the stock model directly, no LoRA merge.
            print(f"No LoRA adapter; generating from base model {args.base_model}", flush=True)
            model_path = args.base_model

        from vllm import LLM, SamplingParams
        llm = LLM(model=model_path, trust_remote_code=True,
                  max_model_len=4096, gpu_memory_utilization=0.85,
                  enforce_eager=False)
        tokenizer = llm.get_tokenizer()

        prompts = []
        for it in items:
            opts = "\n".join(f"{k}. {v}" for k, v in sorted(it["options"].items()))
            user = PROMPT.format(question=it["question"], options=opts)
            msgs = [{"role": "user", "content": user}]
            try:
                r = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except (TypeError, ValueError):
                r = user
            prompts.append(r)

        sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1500, n=1)
        t0 = time.time()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        print(f"  Generation done in {time.time()-t0:.1f}s", flush=True)

        with open(out_cot, "a") as f:
            for it, out in zip(items, outputs):
                text = out.outputs[0].text.strip()
                m = ANSWER_RE.search(text)
                ans = m.group(1).upper() if m else ""
                f.write(json.dumps({
                    "question_idx": it["question_idx"], "question": it["question"],
                    "options": it["options"], "correct_answer": it["correct_answer"],
                    "text": text, "answer": ans,
                }) + "\n")

    rows = [json.loads(l) for l in open(out_cot)]
    n_correct = sum(1 for r in rows if r["answer"] == r["correct_answer"])
    accuracy = n_correct / max(len(rows), 1)
    hedges = [hedge_density(r["text"]) for r in rows]
    import numpy as np
    mean_h = float(np.mean(hedges))

    summary = {
        "arm_tag": args.arm_tag, "n_questions": len(rows),
        "accuracy": accuracy, "mean_hedge_density": mean_h,
    }
    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.arm_tag} ===")
    print(f"  Accuracy: {accuracy*100:.2f}% ({n_correct}/{len(rows)})")
    print(f"  Hedge density: {mean_h:.3f}")


if __name__ == "__main__":
    main()
