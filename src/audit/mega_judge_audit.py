#!/usr/bin/env python3
"""Step-factuality audit with a configurable LARGE non-thinking judge model
(>=100B-class: moonshotai/Kimi-K2-Instruct-0905, Qwen/Qwen3-235B-A22B-Instruct-2507,
zai-org/GLM-4.6 with thinking disabled). Same STYLE-BLIND prompt and parse_steps as
glm_judge_audit_styleblind.py, so results are directly comparable to the GLM-4-32B run.

Usage:
  python mega_judge_audit.py --input <cot.jsonl> --output-dir <dir> --model-tag <tag> \
      --judge-model moonshotai/Kimi-K2-Instruct-0905 --chain-field text --concurrency 16
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"

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

VALID = {"correct", "error", "uncertain"}
STEP_RE = re.compile(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]\s*(.+?)(?=(?:\n\s*(?:Step\s+)?\d+[\.\:\)])|$)", re.DOTALL | re.IGNORECASE)


def parse_steps(text, max_steps=12):
    if not text:
        return []
    matches = STEP_RE.findall(text)
    steps = []
    for idx, body in matches:
        body = body.strip()
        body = re.sub(r"\s*(?:The answer is\s*\([A-E]\)\.?\s*)?$", "", body)
        if body and len(body) > 20:
            steps.append(body[:800])
    if not steps:
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        steps = paras[:max_steps]
    return steps[:max_steps]


def parse_judgment(text):
    m = re.search(r"JUDGMENT:\s*(\w+)", text, re.IGNORECASE)
    j = m.group(1).lower() if m else "uncertain"
    if j not in VALID:
        j = "uncertain"
    m = re.search(r"EXPLANATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return j, (m.group(1).strip() if m else "")


def build_prompt(question, options, gold_letter, step_text):
    opt_str = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return STYLE_BLIND_PROMPT.format(question=question, options=opt_str,
                                     correct_answer=gold_letter,
                                     correct_text=options.get(gold_letter, ""),
                                     step_text=step_text)


def make_call_kwargs(judge_model):
    kw = dict(temperature=0.0, max_tokens=220, timeout=120)
    # GLM-* and Kimi-K2.x are hybrid reasoning models -> force NON-thinking.
    # (Qwen3-...-Instruct-2507 and Kimi-K2-Instruct-0905 are already non-thinking; don't pass the flag.)
    m = judge_model
    if ("GLM-" in m) or ("Kimi-K2." in m):
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
    return kw


_lock = threading.Lock()


def judge_one(client, model, prompt, extra_kw, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], **extra_kw)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "limit" in err.lower():
                time.sleep(min(2 ** attempt * 2.0, 45))
            elif "503" in err or "502" in err or "timeout" in err.lower() or "overload" in err.lower():
                time.sleep(min(2 ** attempt * 1.5, 30))
            else:
                time.sleep(min(2 ** attempt, 10))
    return "JUDGMENT: uncertain\nEXPLANATION: API failure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--chain-field", default="text")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        sys.exit("SILICONFLOW_API_KEY not set")
    client = OpenAI(api_key=key, base_url=SILICONFLOW_BASE)
    extra_kw = make_call_kwargs(args.judge_model)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"judgments_{args.model_tag}.jsonl")

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                d = json.loads(line)
                done.add(f"{d['qidx']}|{d['chain_idx']}|{d['step_idx']}")
            except Exception:
                pass
    print(f"[{args.model_tag} | {args.judge_model}] resume: {len(done)} done", flush=True)

    entries = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
            if args.n_questions and len(entries) >= args.n_questions:
                break

    tasks = []
    for qi, e in enumerate(entries):
        question = e.get("question") or e.get("sft_input", "")
        options = e.get("options") or {}
        gold = (e.get("correct_answer") or "").strip()
        if not gold or not options:
            continue
        text = e.get(args.chain_field, "") or e.get("text", "") or e.get("teacher_cot", "") or e.get("generated_cot", "")
        for sidx, step in enumerate(parse_steps(text)):
            kid = f"{qi}|0|{sidx}"
            if kid in done:
                continue
            tasks.append((qi, 0, sidx, step, build_prompt(question, options, gold, step)))
    print(f"[{args.model_tag} | {args.judge_model}] tasks: {len(tasks)}  (entries={len(entries)})", flush=True)
    if not tasks:
        return

    fout = open(out_path, "a")
    t0 = time.time()
    counts = {"correct": 0, "error": 0, "uncertain": 0}

    def worker(task):
        qi, cid, sidx, step, prompt = task
        reply = judge_one(client, args.judge_model, prompt, extra_kw)
        j, exp = parse_judgment(reply)
        return qi, cid, sidx, step, j, exp

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                qi, cid, sidx, step, j, exp = fut.result()
            except Exception as e:
                print(f"  err: {e}", flush=True)
                continue
            counts[j] = counts.get(j, 0) + 1
            with _lock:
                fout.write(json.dumps({"qidx": qi, "chain_idx": cid, "step_idx": sidx,
                                       "model_tag": args.model_tag, "judge_model": args.judge_model,
                                       "step_text": step[:500], "judgment": j, "explanation": exp[:300]}) + "\n")
                fout.flush()
            completed += 1
            if completed % 200 == 0 or completed == len(tasks):
                el = time.time() - t0
                rate = completed / max(el, 1)
                eta = (len(tasks) - completed) / max(rate, 1e-3)
                nc = counts["correct"] + counts["error"]
                er = counts["error"] / max(nc, 1) * 100
                print(f"  {completed}/{len(tasks)} {rate:.2f}/s el={el:.0f}s eta={eta:.0f}s  err_rate={er:.1f}%  unc={counts['uncertain']}", flush=True)
    fout.close()
    nc = counts["correct"] + counts["error"]
    print(f"\n=== {args.model_tag} | {args.judge_model} done === counts={counts}", flush=True)
    if nc:
        print(f"  error/(correct+error) = {counts['error']}/{nc} = {counts['error']/nc*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
