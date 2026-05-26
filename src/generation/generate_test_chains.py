"""
Generate test chains using the SFT (distilled) student model.

Generates K=64 reasoning chains on MedQA test set using the
LoRA-SFT'd model, then formats them for scoring with the
step-level audit.

Usage (GPU):
    python generate_test_chains.py --adapter-dir experiments/module1/lora_adapter

Output: experiments/module1/test_chains/chunk_XX.json (38 chunks)
"""

import argparse
import json
import math
import os
import re
import sys
import time

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)

from src.utils.answer_extract import extract_answer

# ============================================================
# Config
# ============================================================

MODEL_PATH = "Qwen/Qwen3-8B"

TEST_DATA = "data/MedQA-USMLE/4_options/phrases_no_exclude_test.jsonl"

# Inference prompt — must match SFT training input exactly (train-test consistency).
COT_PROMPT = """You are an expert physician taking the USMLE. Work through the following question as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Why some answer options fit and others don't
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is (X)." where X is A, B, C, or D.

Question: {question}

Options:
{options}"""


def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def _find_next_unfinished(output_dir, num_shards, total_q, completed_indices):
    """Find the next chunk that has unfinished questions. Returns chunk ID or None."""
    import glob as _glob
    for chunk_id in range(num_shards):
        shard_size = math.ceil(total_q / num_shards)
        start = chunk_id * shard_size
        end = min(start + shard_size, total_q)
        chunk_indices = set(range(start, end))
        remaining = chunk_indices - completed_indices
        if remaining:
            # Also check if another worker is actively writing this chunk
            shard_file = os.path.join(output_dir, f"gpu_shard_{chunk_id:02d}.jsonl")
            lock_file = shard_file + ".lock"
            try:
                # Try to create lock file (atomic on NFS)
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                print(f"   Auto-next: claiming chunk {chunk_id} "
                      f"({len(remaining)} questions remaining)")
                return chunk_id
            except FileExistsError:
                # Another worker has this chunk, skip
                continue
    print("   Auto-next: all chunks completed or claimed!")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Base model path (default: Qwen3-8B)")
    parser.add_argument("--test-data", type=str, default=TEST_DATA,
                        help="Path to test data JSONL (default: MedQA-USMLE)")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="Path to LoRA adapter directory (omit for vanilla generation)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: experiments/module1/test_chains)")
    parser.add_argument("--k", type=int, default=64,
                        help="Number of chains per question")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--n-shards", type=int, default=38,
                        help="Number of output shards")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--gpu-shard-id", type=int, default=0,
                        help="GPU shard ID (0-indexed)")
    parser.add_argument("--gpu-num-shards", type=int, default=1,
                        help="Total number of GPU shards (1 = single GPU)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Questions per generation batch (controls peak RAM)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max model length")
    parser.add_argument("--auto-next", action="store_true",
                        help="After finishing assigned chunk, auto-pick next unfinished chunk")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Pass enable_thinking=False to apply_chat_template (for Qwen3)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJ_ROOT, "experiments", "module1", "test_chains")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generate Test Chains (SFT Model)")
    print("=" * 60)

    # ---- Load test data ----
    print(f"\n1. Loading test data from {args.test_data} ...")
    questions = []
    with open(args.test_data) as f:
        for line in f:
            questions.append(json.loads(line))
    print(f"   Loaded {len(questions)} test questions")

    if args.max_questions > 0:
        questions = questions[:args.max_questions]
        print(f"   Limited to {len(questions)} questions")

    all_questions = questions
    total_q = len(all_questions)

    # ---- Load model ----
    use_lora = args.adapter_dir is not None
    model_path = args.model_path

    if use_lora:
        print(f"\n2. Loading vLLM with LoRA adapter from {args.adapter_dir} ...")
    else:
        print(f"\n2. Loading vLLM (vanilla, no adapter) ...")
    print(f"   Base model: {model_path}")

    from vllm import LLM, SamplingParams

    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest

        # Check if adapter exists
        adapter_config = os.path.join(args.adapter_dir, "adapter_config.json")
        if not os.path.exists(adapter_config):
            print(f"   ERROR: No adapter found at {args.adapter_dir}")
            sys.exit(1)

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_lora=True,
            max_lora_rank=32,
        )
        lora_request = LoRARequest("sft_adapter", 1, args.adapter_dir)
    else:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    sampling_params = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        logprobs=1,
    )

    mode_str = "LoRA" if use_lora else "vanilla"
    print(f"   Model loaded ({mode_str}). K={args.k}, temp={args.temperature}")

    # ---- Tokenizer for prompt building ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---- Process chunks (with auto-next support) ----
    import glob as _glob
    chunks_to_process = [args.gpu_shard_id]

    while chunks_to_process:
        current_chunk = chunks_to_process.pop(0)

        print(f"\n{'='*60}")
        print(f"=== Chunk {current_chunk}/{args.gpu_num_shards} ===")
        print(f"{'='*60}")

        # Compute question range for this chunk
        shard_size = math.ceil(total_q / args.gpu_num_shards)
        start = current_chunk * shard_size
        end = min(start + shard_size, total_q)
        questions = all_questions[start:end]
        original_indices = list(range(start, end))
        print(f"   Questions [{start}..{end}) = {len(questions)} questions")

        # Build prompts for this chunk
        prompts = []
        for q in questions:
            options_text = format_options(q["options"])
            prompt_text = COT_PROMPT.format(
                question=q["question"],
                options=options_text,
            )
            messages = [{"role": "user", "content": prompt_text}]
            chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if args.disable_thinking:
                chat_kwargs["enable_thinking"] = False
            formatted = tokenizer.apply_chat_template(messages, **chat_kwargs)
            prompts.append(formatted)

        # Resume support: scan ALL shard files for completed questions
        shard_file = os.path.join(args.output_dir, f"gpu_shard_{current_chunk:02d}.jsonl")
        completed_indices = set()
        for existing_file in sorted(_glob.glob(os.path.join(args.output_dir, "gpu_shard_*.jsonl"))):
            with open(existing_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        completed_indices.add(json.loads(line)["question_idx"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        if completed_indices:
            print(f"   Resume: found {len(completed_indices)} completed questions across all shard files")
            remaining = [(q, idx, p) for q, idx, p in zip(questions, original_indices, prompts)
                         if idx not in completed_indices]
            questions = [r[0] for r in remaining]
            original_indices = [r[1] for r in remaining]
            prompts = [r[2] for r in remaining]
            print(f"   Remaining: {len(questions)} questions to generate")
            if not questions:
                print("   All questions already completed. Skipping.")
                # Auto-next: find next unfinished chunk
                if args.auto_next and not chunks_to_process:
                    next_chunk = _find_next_unfinished(
                        args.output_dir, args.gpu_num_shards, total_q, completed_indices
                    )
                    if next_chunk is not None:
                        chunks_to_process.append(next_chunk)
                continue

        # Generate in batches
        batch_size = args.batch_size
        n_batches = math.ceil(len(prompts) / batch_size)
        print(f"\n4. Generating {len(prompts)} × {args.k} chains "
              f"(batch_size={batch_size}, {n_batches} batches) ...")

        stats = {
            "total_questions": len(questions),
            "total_chains": 0,
            "correct_chains": 0,
            "wrong_chains": 0,
            "no_answer_chains": 0,
            "question_accuracies": [],
        }

        out_f = open(shard_file, "a")
        t_gen = time.time()
        total_written = 0

        for batch_idx in range(n_batches):
            b_start = batch_idx * batch_size
            b_end = min(b_start + batch_size, len(prompts))
            batch_prompts = prompts[b_start:b_end]

            outputs = llm.generate(batch_prompts, sampling_params,
                                   lora_request=lora_request)

            for qi_in_batch, output in enumerate(outputs):
                qi = b_start + qi_in_batch
                q = questions[qi]
                correct = q.get("answer_idx", "")
                sampled = []
                n_correct = 0

                for ci, completion in enumerate(output.outputs):
                    chain_text = completion.text.strip()
                    extracted = extract_answer(chain_text)
                    n_tokens = len(completion.token_ids)
                    cum_logprob = completion.cumulative_logprob if hasattr(completion, 'cumulative_logprob') else 0.0
                    mean_logprob = cum_logprob / max(n_tokens, 1)

                    stats["total_chains"] += 1
                    if not extracted:
                        stats["no_answer_chains"] += 1
                    elif extracted == correct:
                        stats["correct_chains"] += 1
                        n_correct += 1
                    else:
                        stats["wrong_chains"] += 1

                    sampled.append({
                        "text": chain_text,
                        "answer": extracted or "",
                        "n_tokens": n_tokens,
                        "mean_logprob": round(mean_logprob, 6),
                    })

                q_acc = n_correct / len(sampled) if sampled else 0.0
                stats["question_accuracies"].append(q_acc)

                result = {
                    "question_idx": original_indices[qi],
                    "question": q["question"],
                    "options": q["options"],
                    "correct_answer": correct,
                    "sampled": sampled,
                }

                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                total_written += 1

            del outputs

            elapsed = time.time() - t_gen
            print(f"   Batch {batch_idx+1}/{n_batches}: "
                  f"{total_written}/{len(original_indices)} questions done "
                  f"({elapsed:.0f}s elapsed)")

        gen_time = time.time() - t_gen
        out_f.close()

        # Print summary for this chunk
        per_chain_acc = stats["correct_chains"] / max(stats["total_chains"], 1)
        sc_estimate = sum(1 for accs in stats["question_accuracies"] if accs > 0.5) / max(len(stats["question_accuracies"]), 1)
        avg_q_acc = sum(stats["question_accuracies"]) / max(len(stats["question_accuracies"]), 1)

        print(f"\n   Chunk {current_chunk} done: {total_written} questions, "
              f"per-chain={100*per_chain_acc:.1f}%, SC≈{100*sc_estimate:.1f}%, "
              f"time={gen_time:.0f}s")

        # Auto-next: find next unfinished chunk
        if args.auto_next and not chunks_to_process:
            # Re-scan completed indices after this chunk
            all_completed = set()
            for existing_file in sorted(_glob.glob(os.path.join(args.output_dir, "gpu_shard_*.jsonl"))):
                with open(existing_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            all_completed.add(json.loads(line)["question_idx"])
                        except (json.JSONDecodeError, KeyError):
                            continue
            next_chunk = _find_next_unfinished(
                args.output_dir, args.gpu_num_shards, total_q, all_completed
            )
            if next_chunk is not None:
                chunks_to_process.append(next_chunk)

    print(f"\nAll assigned chunks complete.")
    print(f"Output: {args.output_dir}/ ({args.n_shards} shards)")


if __name__ == "__main__":
    main()
