"""
Generate SFT training data using DeepSeek-V3 as teacher (v2).

Key design choices:
- No answer hint by default: teacher reasons freely, filtered post-hoc by correctness
- Fallback: after MAX_FAILURES consecutive wrong answers, inject hint (HuatuoGPT-o1 style)
- Free-form reasoning format (no forced "Step 1/2/3"), consistent with train/test prompt
- Generate until 2 correct chains per question (not fixed N per question)
- Jaccard diversity filter to select the 2 most distinct correct chains
- max_tokens 1024 -> 2048

Usage:
    python generate_sft_data_v2.py --dataset medqa
    python generate_sft_data_v2.py --dataset medmcqa
    python generate_sft_data_v2.py --dataset medqa --shard-id 0 --num-shards 4
    python generate_sft_data_v2.py --resume
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def extract_answer(text: str) -> str:
    """Extract answer letter (A-D) from generated CoT text."""
    patterns = [
        r'[Tt]he\s+answer\s+is\s*\(?([A-D])\)?',
        r'[Cc]orrect\s+answer\s*(?:is|:)\s*\(?([A-D])\)?',
        r'[Aa]nswer\s*:\s*\(?([A-D])\)?',
        r'\b([A-D])\)\s*is\s+(?:the\s+)?correct',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\(?([A-D])\)?',
        r'\\boxed\{([A-D])\}',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    # Last resort: final standalone letter
    m = re.search(r'\b([A-D])\b(?=[^A-Za-z]*$)', text.strip())
    if m:
        return m.group(1).upper()
    return ""

# ============================================================
# Config
# ============================================================

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek-V3

TARGET_CORRECT = 2       # collect this many correct chains per question
MAX_FAILURES   = 5       # after this many consecutive wrong answers, switch to hint prompt
MAX_ATTEMPTS   = 20      # hard cap on total API calls per question

# ============================================================
# Prompt Templates
# ============================================================

# Primary prompt: no answer hint. Teacher reasons freely.
# Same format used at SFT training time and inference time (train-test consistency).
PROMPT_NO_HINT = """You are an expert physician taking the USMLE. Work through the following question as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Why some answer options fit and others don't
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is (X)." where X is A, B, C, or D.

Question: {question}

Options:
{options_text}"""

# Fallback prompt: a correctness-conditioned prompt that supplies the correct answer after MAX_FAILURES consecutive failures.
# Inspired by HuatuoGPT-o1's gen_prompt_w_label.
PROMPT_WITH_HINT = """You are an expert physician taking the USMLE. Work through the following question as a natural reasoning process.

[Internal note: the correct answer is ({correct_option}) {correct_answer} — but reason through it genuinely, as if working it out fresh. Make sure your reasoning clearly explains why the correct answer is right and why the other options are wrong.]

Think about:
- What the key clinical findings suggest
- Why the correct answer fits the clinical picture
- Why each of the other options can be ruled out

You MUST end your response with exactly "The answer is ({correct_option})."

Question: {question}

Options:
{options_text}"""

# SFT training input prompt (no answer, no hint — identical to PROMPT_NO_HINT).
# Stored separately for clarity; used when writing the sft_input field.
SFT_INPUT_PROMPT = PROMPT_NO_HINT


# ============================================================
# Helpers
# ============================================================

def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity between two strings."""
    def ngrams(s, n):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa, sb = ngrams(a.lower(), n), ngrams(b.lower(), n)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def select_diverse_chains(chains: list[str], target: int = 2,
                          sim_threshold: float = 0.6) -> list[str]:
    """
    Greedy diversity selection: pick `target` chains that are mutually
    dissimilar (Jaccard < sim_threshold).

    Falls back to first `target` if we cannot find enough distinct chains.
    """
    if len(chains) <= target:
        return chains

    selected = [chains[0]]
    for candidate in chains[1:]:
        if len(selected) >= target:
            break
        if all(jaccard_similarity(candidate, s) < sim_threshold
               for s in selected):
            selected.append(candidate)

    # If strict diversity gave fewer than target, pad with remaining
    if len(selected) < target:
        for c in chains:
            if c not in selected:
                selected.append(c)
                if len(selected) >= target:
                    break

    return selected[:target]


# ============================================================
# DeepSeek API
# ============================================================

def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY not set.")
    return key


def call_deepseek(prompt: str, api_key: str,
                  temperature: float = 0.7,
                  max_tokens: int = 2048,
                  max_retries: int = 3) -> str | None:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                print(f"    API failed: {e}")
    return None


# ============================================================
# Per-question generation loop
# ============================================================

def generate_for_question(q: dict, api_key: str,
                           temperature: float = 0.7,
                           max_tokens: int = 2048) -> dict:
    """
    Generate chains for one question until TARGET_CORRECT correct chains
    are collected, with hint fallback after MAX_FAILURES consecutive failures.

    Returns a dict with:
        question_idx, correct_opt, correct_chains, hint_chains,
        n_attempts, n_correct, n_wrong, n_hint_used
    """
    options_text  = format_options(q["options"])
    correct_opt   = q["answer_idx"]
    correct_ans   = q["answer_text"]

    correct_chains = []   # chains generated WITHOUT hint
    hint_chains    = []   # chains generated WITH hint (after fallback)
    consecutive_failures = 0
    using_hint = False
    n_attempts = 0
    n_wrong = 0

    while (len(correct_chains) + len(hint_chains) < TARGET_CORRECT
           and n_attempts < MAX_ATTEMPTS):

        # Switch to hint mode after MAX_FAILURES consecutive failures
        if consecutive_failures >= MAX_FAILURES and not using_hint:
            using_hint = True

        if using_hint:
            prompt = PROMPT_WITH_HINT.format(
                correct_option=correct_opt,
                correct_answer=correct_ans,
                question=q["question"],
                options_text=options_text,
            )
        else:
            prompt = PROMPT_NO_HINT.format(
                question=q["question"],
                options_text=options_text,
            )

        result = call_deepseek(prompt, api_key,
                               temperature=temperature,
                               max_tokens=max_tokens)
        n_attempts += 1

        if result is None:
            consecutive_failures += 1
            continue

        extracted = extract_answer(result.strip())
        if extracted == correct_opt:
            if using_hint:
                hint_chains.append(result.strip())
            else:
                correct_chains.append(result.strip())
            consecutive_failures = 0  # reset on success
        else:
            n_wrong += 1
            consecutive_failures += 1

    return {
        "question_idx":   q["question_idx"],
        "question":       q["question"],
        "options":        q["options"],
        "correct_opt":    correct_opt,
        "correct_chains": correct_chains,
        "hint_chains":    hint_chains,
        "n_attempts":     n_attempts,
        "n_correct":      len(correct_chains) + len(hint_chains),
        "n_wrong":        n_wrong,
        "n_hint_used":    len(hint_chains),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["medqa", "medmcqa"], default="medqa")
    parser.add_argument("--input-file", type=str, default="",
                        help="Override input file path")
    parser.add_argument("--output-file", type=str, default="",
                        help="Override output file path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--shard-id", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()

    # ---- Resolve paths ----
    exp_dir = os.path.join(PROJ_ROOT, "experiments", "module1")
    if args.dataset == "medqa":
        default_input  = os.path.join(exp_dir, "train_with_paths.jsonl")
        default_output = os.path.join(exp_dir, "sft_data_v2_medqa.jsonl")
    else:
        default_input  = os.path.join(exp_dir, "medmcqa_train.jsonl")
        default_output = os.path.join(exp_dir, "sft_data_v2_medmcqa.jsonl")

    input_file  = args.input_file  or default_input
    output_file = args.output_file or default_output

    if args.shard_id >= 0:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_shard_{args.shard_id:02d}{ext}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ---- API key ----
    api_key = get_api_key()

    print("=" * 60)
    print(f"SFT Data Generation v2 — {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Input:       {input_file}")
    print(f"  Output:      {output_file}")
    print(f"  Target/q:    {TARGET_CORRECT} correct chains")
    print(f"  Hint after:  {MAX_FAILURES} consecutive failures")
    print(f"  max_tokens:  {args.max_tokens}")
    print(f"  Workers:     {args.workers}")
    if args.shard_id >= 0:
        print(f"  Shard:       {args.shard_id}/{args.num_shards}")

    # ---- Load questions ----
    questions = []
    with open(input_file) as f:
        for i, line in enumerate(f):
            q = json.loads(line)
            # Normalize field names from raw MedQA/MedMCQA format
            if "answer_text" not in q and "answer" in q:
                q["answer_text"] = q["answer"]
            if "question_idx" not in q:
                q["question_idx"] = i
            questions.append(q)
    print(f"\nLoaded {len(questions)} questions")

    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    # Shard filtering
    if args.shard_id >= 0:
        questions = [q for i, q in enumerate(questions)
                     if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}: {len(questions)} questions")

    # Resume
    done_idx = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                done_idx.add(rec["question_idx"])
        print(f"Resuming: {len(done_idx)} already done")

    todo = [q for q in questions if q["question_idx"] not in done_idx]
    print(f"To process: {len(todo)} questions")
    if not todo:
        print("Nothing to do.")
        return

    # ---- Generate ----
    t_start = time.time()
    n_done = 0
    n_sft  = 0
    n_hint_total = 0
    n_no_data = 0
    BATCH = 50

    mode = "a" if args.resume and done_idx else "w"
    with open(output_file, mode) as fout:
        batch_start = 0
        while batch_start < len(todo):
            batch = todo[batch_start:batch_start + BATCH]
            results = [None] * len(batch)

            def process(pos, q):
                return pos, generate_for_question(
                    q, api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process, i, q): i
                        for i, q in enumerate(batch)}
                for fut in as_completed(futs):
                    pos, res = fut.result()
                    results[pos] = res

            for res in results:
                if res is None:
                    continue
                n_done += 1
                options_text = format_options(res["options"])

                # Pool all correct chains; prefer no-hint chains
                all_chains = res["correct_chains"] + res["hint_chains"]

                if not all_chains:
                    n_no_data += 1
                    continue

                # Select 2 most diverse chains
                selected = select_diverse_chains(all_chains, target=TARGET_CORRECT)

                for chain in selected:
                    hint_used = chain in res["hint_chains"]
                    sft_input = SFT_INPUT_PROMPT.format(
                        question=res["question"],
                        options_text=options_text,
                    )
                    record = {
                        "question_idx":  res["question_idx"],
                        "sft_input":     sft_input,
                        "sft_target":    chain,
                        "correct_answer": res["correct_opt"],
                        "hint_used":     hint_used,
                        "n_attempts":    res["n_attempts"],
                        "n_tokens":      len(chain.split()),
                        "teacher_model": DEEPSEEK_MODEL,
                        "dataset":       args.dataset,
                    }
                    fout.write(json.dumps(record) + "\n")
                    n_sft += 1
                    if hint_used:
                        n_hint_total += 1

                fout.flush()

            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta  = (len(todo) - n_done) / rate if rate > 0 else 0
            print(f"  [{n_done}/{len(todo)}] "
                  f"sft_examples={n_sft} "
                  f"hint_used={n_hint_total} "
                  f"no_data={n_no_data} "
                  f"rate={rate:.1f}q/s ETA={eta:.0f}s")

            batch_start += BATCH

    # ---- Summary ----
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {total:.0f}s")
    print(f"  Questions processed : {n_done}")
    print(f"  SFT examples written: {n_sft}  (~{n_sft/max(n_done,1):.1f}/q)")
    print(f"  Hint fallback used  : {n_hint_total} chains "
          f"({100*n_hint_total/max(n_sft,1):.1f}%)")
    print(f"  No data at all      : {n_no_data}")
    print(f"  Output: {output_file}")

    # Cost estimate (DeepSeek-V3: $0.27/M input, $1.10/M output)
    avg_attempts = (n_sft / max(n_done, 1)) * 1.5  # rough
    est_calls = n_done * avg_attempts
    est_in  = est_calls * 300   # ~300 input tokens/call
    est_out = est_calls * 600   # ~600 output tokens/call (longer chains now)
    est_cost = est_in * 0.27/1e6 + est_out * 1.10/1e6
    print(f"\nEst. cost: ${est_cost:.2f}  "
          f"(~{est_calls:.0f} API calls)")

    stats = {
        "dataset": args.dataset,
        "n_questions": n_done,
        "n_sft_examples": n_sft,
        "n_hint_used": n_hint_total,
        "n_no_data": n_no_data,
        "time_seconds": round(total, 1),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    stats_file = output_file.replace(".jsonl", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats: {stats_file}")


if __name__ == "__main__":
    main()
