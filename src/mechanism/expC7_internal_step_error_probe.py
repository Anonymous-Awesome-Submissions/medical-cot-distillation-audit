#!/usr/bin/env python3
"""
Internal step-error probe: can a linear probe on the residual stream at a step
boundary predict the judge's "error" label? We compare base vs SFT vs weak-SFT,
testing whether step-correctness stays linearly decodable internally even as
verbal hedging stops tracking it.

Phase 1 (this script, GPU, one model-tag per run):
  - Load the model's generated CoTs + the GLM style-blind step-judgments.
  - Re-segment each CoT with the SAME rule the audit used (parse_steps); for step k,
    find step_texts[k] as a substring of the CoT, take the END char offset, map to a
    token index via offset_mapping. The residual_post at that token = "the model's
    state right after writing step k".
  - Forward pass with output_hidden_states; capture residual at every layer at the
    step-end positions. Join with the GLM label (error=1 / correct=0 / drop uncertain).
  - Save: {qidx, step_idx, label, resid[n_layers, d_model]} -> npz per tag.

Phase 2 (a separate CPU step): per tag, per layer, train a 5-fold-CV-by-question
  logistic-regression probe resid -> label; report AUROC vs layer. Compare vanilla
  vs SFT: does the probe AUROC stay high in SFT while the verbal hedge<->error
  Spearman (already known: vanilla rho=0.09 p<0.001, SFT rho=0.02 p=0.50) collapses?
  Where does the best-probe layer sit relative to the ~0.94 hedge band? Does it move
  AWAY from the hedge band post-SFT?

Usage (PHASE 1): python expC7_internal_step_error_probe.py --tag vanilla
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("experiments/module1/mechanism/internal_step_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# NOTE: the GLM audit (glm_judge_audit_styleblind.py) keys steps by ENUMERATE POSITION
# in the input file (qidx = position 0..499), NOT by the file's question_idx field.
# So we must join the CoTs by their position in the SAME input file the audit used.
Q2 = "experiments/module1/q2_hallucination"
LAYERBAND = "experiments/module1/mechanism/layerband_eval"

# tag -> (model_path, cots_jsonl, audit_jsonl)
VAR = {
    "vanilla":  ("Qwen/Qwen3-8B",
                 f"{Q2}/vanilla_cot_medqa_test500.jsonl", f"{Q2}/audit_results_glm_styleblind/judgments_vanilla.jsonl"),
    "sft8b":    ("experiments/module1/qwen3_8b_sft_merged",
                 f"{Q2}/sft8b_cot_medqa_test500.jsonl", f"{Q2}/audit_results_glm_styleblind/judgments_sft8b.jsonl"),
    "weak_sft": ("experiments/module1/qwen3_8b_weak_teacher_merged",
                 f"{Q2}/weak_sft_cot_medqa_test500.jsonl", f"{Q2}/audit_results_glm_styleblind/judgments_weak_sft.jsonl"),
    "hedgeband": (f"{LAYERBAND}/hedgeband/merged_model",
                 f"{LAYERBAND}/hedgeband/generated_cots.jsonl", f"{LAYERBAND}/audit_styleblind/judgments_hedgeband.jsonl"),
}

STEP_RE = re.compile(r"(?:^|\n)\s*(?:Step\s+)?(\d+)[\.\:\)]\s*(.+?)(?=(?:\n\s*(?:Step\s+)?\d+[\.\:\)])|$)", re.DOTALL | re.IGNORECASE)


def parse_steps(text, max_steps=12):
    """Replicates the audit's parse_steps EXACTLY so step_idx aligns with the judge labels."""
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


def cot_text(d):
    return d.get("text") or d.get("teacher_cot") or d.get("cot") or d.get("generated_cot") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(VAR.keys()))
    ap.add_argument("--max-questions", type=int, default=500)
    ap.add_argument("--max-cot-tokens", type=int, default=900)
    args = ap.parse_args()
    model_path, cots_f, audit_f = VAR[args.tag]
    print(f"=== expC7 PHASE 1: tag={args.tag} ===\n model={model_path}\n cots={cots_f}\n audit={audit_f}", flush=True)

    # ---- load audit labels: (qidx, step_idx) -> label ----
    labels = {}
    for line in open(audit_f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        j = r.get("judgment") or r.get("glm_styleblind_judgment")
        if j in ("error", "correct"):
            labels[(r.get("qidx"), r.get("step_idx"))] = 1 if j == "error" else 0
    print(f"  audit labels (error/correct, not uncertain): {len(labels)}", flush=True)

    # ---- load CoTs keyed by ENUMERATE POSITION (matches the audit's qidx) ----
    cots = {}
    pos = 0
    for line in open(cots_f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        cots[pos] = cot_text(d)  # qidx in the audit == position here
        pos += 1
        if pos >= args.max_questions:
            break
    print(f"  CoTs loaded by position: {len(cots)} (qidx==position; matches the audit's enumerate keying)", flush=True)

    # ---- load model ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else len(model.transformer.h)
    d_model = model.config.hidden_size
    print(f"  n_layers={n_layers} d_model={d_model}", flush=True)

    # ---- for each CoT: re-segment, find step-end token positions, forward, capture resid ----
    out_resid, out_q, out_si, out_label = [], [], [], []
    t0 = time.time()
    n_done = 0
    for qi, text in cots.items():
        if not text:
            continue
        steps = parse_steps(text)
        if not steps:
            continue
        # tokenize the CoT with offsets, truncated
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]; offs = enc["offset_mapping"]
        if len(ids) > args.max_cot_tokens:
            ids = ids[:args.max_cot_tokens]; offs = offs[:args.max_cot_tokens]
        if len(ids) < 5:
            continue
        # for each step, find its end char offset, map to a token index
        step_tok_end = []  # (step_idx, token_index)
        search_from = 0
        for k, st in enumerate(steps):
            if (qi, k) not in labels:
                continue
            # match the first ~min(len(st),200) chars of the step text to be robust to truncation
            probe_str = st[:200]
            pos = text.find(probe_str, search_from)
            if pos < 0:
                pos = text.find(probe_str)  # retry from start
            if pos < 0:
                continue
            end_char = pos + len(st)  # end of the full step text
            search_from = max(search_from, pos + len(probe_str))
            # token whose offset span ends at/after end_char (i.e., the last token of the step)
            ti = None
            for j, (a, b) in enumerate(offs):
                if b >= end_char:
                    ti = j; break
            if ti is None:
                ti = len(ids) - 1  # step runs past truncation -> use last token
            ti = min(ti, len(ids) - 1)
            step_tok_end.append((k, ti))
        if not step_tok_end:
            continue
        # forward pass, capture resid_post at every layer at those token positions
        with torch.no_grad():
            out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # (n_layers+1,) each (1, T, D); idx 1..n = post-block
        for (k, ti) in step_tok_end:
            r = np.stack([hs[L + 1][0, ti, :].float().cpu().numpy() for L in range(n_layers)])  # (n_layers, d)
            out_resid.append(r); out_q.append(qi); out_si.append(k); out_label.append(labels[(qi, k)])
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(cots)} CoTs  {n_done/max(time.time()-t0,1):.2f}/s  collected {len(out_resid)} step-resids", flush=True)

    if not out_resid:
        print("No step residuals collected — abort."); return
    R = np.stack(out_resid).astype(np.float32)  # (N, n_layers, d_model)
    Q = np.array(out_q); SI = np.array(out_si); Y = np.array(out_label)
    print(f"\n  collected {R.shape[0]} step-residuals; error rate = {Y.mean()*100:.1f}%; n_layers={R.shape[1]}", flush=True)
    np.savez(OUT_DIR / f"resid_{args.tag}.npz", resid=R, qidx=Q, step_idx=SI, label=Y, n_layers=n_layers, d_model=d_model)
    print(f"Saved: {OUT_DIR}/resid_{args.tag}.npz  ({R.nbytes/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
