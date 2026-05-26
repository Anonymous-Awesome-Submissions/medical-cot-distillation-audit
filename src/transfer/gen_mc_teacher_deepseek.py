#!/usr/bin/env python3
"""DeepSeek-V3.2 teacher CoTs on a multiple-choice problem file (deepseek-chat, NON-thinking)."""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
PROMPT = """Answer the following multiple-choice question. Reason step by step, numbering your steps, then end with a line "The answer is X." where X is the option letter.

Question: {question}
Options:
{options}"""
_lock = threading.Lock()
def gen_one(client, model, prompt, mr=4):
    for a in range(mr):
        try:
            r=client.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],temperature=0.7,max_tokens=1500,timeout=180)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            time.sleep(min(2**a*2,30) if ("429" in str(e) or "rate" in str(e).lower()) else min(2**a,10))
    return ""
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--model",default="deepseek-chat"); ap.add_argument("--concurrency",type=int,default=16); args=ap.parse_args()
    key=os.environ.get("DEEPSEEK_API_KEY",""); 
    if not key: sys.exit("DEEPSEEK_API_KEY not set")
    client=OpenAI(api_key=key,base_url="https://api.deepseek.com")
    rows=[json.loads(l) for l in open(args.input)]
    done=set()
    if os.path.exists(args.output):
        for l in open(args.output):
            try: done.add(json.loads(l)["question_idx"])
            except: pass
    todo=[r for r in rows if r["question_idx"] not in done]
    print(f"{len(todo)} to gen",flush=True)
    fout=open(args.output,"a"); n=0
    def w(r):
        opts="\n".join(f"{k}. {v}" for k,v in sorted(r["options"].items()))
        return r, gen_one(client,args.model,PROMPT.format(question=r["question"],options=opts))
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(w,r) for r in todo]):
            r,text=fut.result()
            with _lock: fout.write(json.dumps({"question_idx":r["question_idx"],"question":r["question"],"options":r["options"],"correct_answer":r["correct_answer"],"text":text})+"\n"); fout.flush()
            n+=1
    fout.close(); print("done",flush=True)
main()
