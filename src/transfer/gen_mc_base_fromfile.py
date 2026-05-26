#!/usr/bin/env python3
"""Base/student CoTs on an MC problem file (vLLM, one T=0.7 chain). Same prompt as teacher."""
import argparse, json
from pathlib import Path
PROMPT = """Answer the following multiple-choice question. Reason step by step, numbering your steps, then end with a line "The answer is X." where X is the option letter.

Question: {question}
Options:
{options}"""
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-path",required=True); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); ap.add_argument("--lora-adapter",default=None); ap.add_argument("--max-tokens",type=int,default=1500); args=ap.parse_args()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in open(args.input)]
    model_path=args.model_path
    if args.lora_adapter:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        from pathlib import Path as _P
        mp=_P(args.output).parent/("merged_"+_P(args.lora_adapter).name)
        if not (mp/"config.json").exists():
            tk=AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
            bm=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,trust_remote_code=True,device_map="auto",low_cpu_mem_usage=True)
            mm=PeftModel.from_pretrained(bm,args.lora_adapter,torch_dtype=torch.bfloat16).merge_and_unload()
            mp.mkdir(parents=True,exist_ok=True); mm.save_pretrained(mp,safe_serialization=True,max_shard_size="5GB"); tk.save_pretrained(mp)
            del mm,bm; torch.cuda.empty_cache()
        model_path=str(mp)
    from vllm import LLM, SamplingParams
    llm=LLM(model=model_path,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.85); tok=llm.get_tokenizer()
    prompts=[]
    for r in rows:
        opts="\n".join(f"{k}. {v}" for k,v in sorted(r["options"].items()))
        msgs=[{"role":"user","content":PROMPT.format(question=r["question"],options=opts)}]
        try: p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        except Exception: p=PROMPT.format(question=r["question"],options=opts)
        prompts.append(p)
    outs=llm.generate(prompts,SamplingParams(temperature=0.7,top_p=0.95,max_tokens=args.max_tokens,n=1))
    with open(args.output,"w") as f:
        for r,o in zip(rows,outs): f.write(json.dumps({"question_idx":r["question_idx"],"question":r["question"],"options":r["options"],"correct_answer":r["correct_answer"],"text":o.outputs[0].text.strip()})+"\n")
    print(f"wrote {len(rows)}",flush=True)
main()
