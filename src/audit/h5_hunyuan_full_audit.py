"""
Plan F: Run Hunyuan-A13B on the FULL 14,587-step audit (4 conditions:
vanilla, weak_sft, sft8b, teacher) under the same style-blind prompt.
Tests whether the 200-step direction reversal vs GLM is sample noise
or a real systematic disagreement.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from glm_judge_audit_styleblind import (
    parse_steps, parse_judgment, build_prompt, judge_one,
    SILICONFLOW_BASE,
)

JUDGE_MODEL = "tencent/Hunyuan-A13B-Instruct"
COT_DIR = "experiments/module1/q2_hallucination"
OUT_DIR = "experiments/module1/q2_hallucination/audit_results_hunyuan_styleblind"


_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--chain-field", default="text")
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--n-questions", type=int, default=500)
    args = ap.parse_args()

    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key: sys.exit("SILICONFLOW_API_KEY not set")
    client = OpenAI(api_key=key, base_url=SILICONFLOW_BASE)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"judgments_{args.model_tag}.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(f"{d['qidx']}|{d['chain_idx']}|{d['step_idx']}")
                except: pass
    print(f"Resume: {len(done)} done", flush=True)

    entries = [json.loads(l) for l in open(args.input)][:args.n_questions]

    tasks = []
    for qi, e in enumerate(entries):
        question = e.get("question", "")
        options = e.get("options", {})
        gold = (e.get("correct_answer") or "").strip()
        if not gold or not options: continue
        text = e.get(args.chain_field, "") or e.get("text", "")
        if not text: continue
        steps = parse_steps(text)
        for sidx, step in enumerate(steps):
            key_id = f"{qi}|0|{sidx}"
            if key_id in done: continue
            prompt = build_prompt(question, options, gold, step)
            tasks.append((qi, 0, sidx, step, prompt))
    print(f"Tasks: {len(tasks)}", flush=True)
    if not tasks: return

    fout = open(out_path, "a")
    t0 = time.time(); completed = 0
    counts = Counter()

    def worker(task):
        qi, cid, sidx, step, prompt = task
        reply = judge_one(client, JUDGE_MODEL, prompt)
        j, exp = parse_judgment(reply)
        return qi, cid, sidx, step, j, exp

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            try: qi, cid, sidx, step, j, exp = fut.result()
            except Exception as e:
                print(f"  Err: {e}", flush=True); continue
            counts[j] += 1
            with _lock:
                fout.write(json.dumps({
                    "qidx": qi, "chain_idx": cid, "step_idx": sidx,
                    "model_tag": args.model_tag,
                    "step_text": step[:500],
                    "judgment": j, "explanation": exp[:300],
                }) + "\n")
                fout.flush()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                el = time.time() - t0
                rate = completed / max(el, 1)
                eta = (len(tasks) - completed) / max(rate, 0.001)
                err = counts["error"] / max(completed, 1) * 100
                print(f"  {completed}/{len(tasks)} {rate:.1f}/s "
                      f"el={el:.0f}s eta={eta:.0f}s err={err:.1f}%", flush=True)

    fout.close()
    nc = counts["correct"] + counts["error"]
    print(f"\n=== Hunyuan {args.model_tag} done ===")
    print(f"  Counts: {dict(counts)}")
    if nc:
        print(f"  Error rate (def-judged): {counts['error']/nc*100:.2f}%")


if __name__ == "__main__":
    main()
