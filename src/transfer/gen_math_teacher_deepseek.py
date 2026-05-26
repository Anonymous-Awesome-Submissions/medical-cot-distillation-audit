#!/usr/bin/env python3
"""DeepSeek-V3.2 teacher CoTs on Hendrycks MATH (official DeepSeek API, NON-thinking).
Model 'deepseek-chat' = V3.2 non-thinking mode (thinking is 'deepseek-reasoner').
Output matches the math step-audit format (question / gold_numeric / text)."""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

PROMPT = """Solve the following mathematics problem. Reason step by step, numbering your steps, and end with a line "The final answer is X." where X is the answer.

Problem: {question}"""

_lock = threading.Lock()

def gen_one(client, model, prompt, max_retries=4):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=2000, timeout=180)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            time.sleep(min(2 ** attempt * 2, 30) if ("429" in err or "rate" in err.lower()) else min(2 ** attempt, 10))
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="deepseek-chat")  # V3.2 non-thinking
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key: sys.exit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    rows = [json.loads(l) for l in open(args.input)]
    done = set()
    if os.path.exists(args.output):
        for l in open(args.output):
            try: done.add(json.loads(l)["question_idx"])
            except: pass
    todo = [r for r in rows if r["question_idx"] not in done]
    print(f"{len(todo)} to generate ({len(done)} done)", flush=True)

    fout = open(args.output, "a")
    t0 = time.time(); n = 0
    def worker(r):
        return r, gen_one(client, args.model, PROMPT.format(question=r["question"]))
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(worker, r) for r in todo]):
            r, text = fut.result()
            with _lock:
                fout.write(json.dumps({"question_idx": r["question_idx"], "question": r["question"],
                    "gold_numeric": r["gold_numeric"], "type": r.get("type"), "level": r.get("level"),
                    "text": text}) + "\n"); fout.flush()
            n += 1
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {n/max(time.time()-t0,1):.2f}/s", flush=True)
    fout.close()
    print(f"done: {args.output}", flush=True)

if __name__ == "__main__":
    main()
