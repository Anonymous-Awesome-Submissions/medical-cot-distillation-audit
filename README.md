Project Structure
.
├── README.md
├── requirements.txt
└── src/
    ├── generation/
    │   ├── generate_teacher_cot.py            # DeepSeek-V3 teacher CoT generation on MedQA test questions (DeepSeek API)
    │   ├── generate_teacher_cot_by_qids.py    # DeepSeek teacher CoTs for an explicit list of MedQA question ids (wraps generate_teacher_cot)
    │   ├── generate_sft_data_v2.py            # Build the distillation SFT corpus: free-form DeepSeek-V3 teacher CoTs on MedQA/MedMCQA train questions, filtered to correct chains, as (question, CoT) pairs
    │   ├── generate_test_chains.py            # Sample K=64 student CoTs per MedQA test question (self-consistency SC@64) with vLLM
    │   ├── build_full_cot_inputs.py           # Assemble the full 1273-question single-chain CoT audit inputs (first of the 64 sampled chains)
    │   └── prepare_q2_audit_inputs.py         # Assemble the 500-question vanilla/SFT/teacher CoT step-audit input files
    ├── training/
    │   ├── train_lora_qwen3.py                # LoRA SFT distillation recipe (Qwen3-8B/14B, Llama, Mistral) on the teacher CoT corpus
    │   └── merge_sft_adapter.py               # Merge a trained LoRA adapter into the base model weights
    ├── audit/
    │   ├── step_factuality_judge.py           # DeepSeek-V3 per-step factuality judge (segments CoT, labels each step correct/error/uncertain)
    │   ├── mega_judge_audit.py                # Primary Kimi-K2.6 style-blind per-step factuality audit (MedQA 1273q, MedMCQA, MedBullets, multichain, cross-student)
    │   ├── glm_judge_audit.py                 # GLM-4-32B per-step audit with the original (non-style-blind) prompt
    │   ├── glm_judge_audit_styleblind.py      # GLM-4-32B per-step audit with the style-blind prompt (shared parse_steps/prompt helpers)
    │   ├── h5_hunyuan_full_audit.py           # Hunyuan-A13B full style-blind per-step audit (reuses glm_judge_audit_styleblind helpers)
    │   ├── answer_blind_kimi_audit.py         # Kimi-K2.6 per-step audit with the correct answer hidden from the judge (answer-blind control)
    │   ├── arc_judge_audit.py                 # Kimi-K2.6 style-blind per-step audit for ARC-Challenge science CoTs
    │   ├── glm_judge_math.py                  # GLM/Kimi style-blind per-step audit adapted for math computation steps (GSM8K/MATH)
    │   ├── uniform_segmentation_audit.py      # Kimi audit under model-agnostic sentence/window segmentation rules (segmentation control)
    │   └── tone_swap_control.py               # Rewrite each audited step into the opposite surface style and re-audit (tone-swap control)
    ├── transfer/
    │   ├── medbullets_pilot50.py              # MedBullets5 clinical-vignette transfer: teacher/base/distilled CoT generation + audit prep
    │   ├── gen_math_test_cot.py               # Student (vanilla / math-SFT) CoT generation on GSM8K/MATH test set via vLLM
    │   ├── gen_math_teacher_deepseek.py       # DeepSeek-V3.2 teacher CoTs on Hendrycks MATH (DeepSeek API)
    │   ├── gen_math_base_fromfile.py          # Base/student CoTs on a MATH problem file (vLLM, one T=0.7 chain)
    │   ├── gen_gsm8k_teacher_cot.py           # DeepSeek-V3 teacher CoTs on GSM8K train questions (SiliconFlow API)
    │   ├── gen_mc_teacher_deepseek.py         # DeepSeek-V3.2 teacher CoTs on a generic multiple-choice problem file (DeepSeek API)
    │   ├── gen_mc_base_fromfile.py            # Base/student CoTs on a multiple-choice problem file (vLLM, one T=0.7 chain)
    │   ├── build_arc_full_test.py             # Build the full 1172-question ARC-Challenge test set (reuses the first 500, appends the complement)
    │   └── arc_accuracy.py                    # Compute answer accuracy for ARC base vs distilled CoTs (extract final option letter)
    ├── families/
    │   ├── prepare_qwen14b_step_audit.py      # Sample paired base/SFT Qwen3-14B/32B MedQA CoTs (n=500) for step-level audit
    │   ├── eval_crossfam.py                   # Generate base/distilled CoTs for cross-family students (Llama-3.1-8B, Mistral-7B) for audit
    │   └── gen_weak_teacher_sft_cot.py        # Generate weak-teacher student CoTs on the audit questions
    ├── mechanism/
    │   ├── expC4_trace_taxonomy.py            # Trace-function taxonomy: classify steps by reasoning role and compute per-role error rates
    │   ├── expC1b_answer_stabilization.py     # Within-CoT answer-stabilization / pre-commitment test (pre-CoT prob, lock step, mid-CoT min)
    │   ├── expC3_cl3_robustness.py            # Calibration (ECE / Brier from the 64-chain SC runs) plus hedge-vs-correctness MI checks, vanilla vs distilled
    │   ├── expC7_internal_step_error_probe.py # Capture residual-stream activations at step boundaries joined with judge labels
    │   ├── expC7_kimi_full.py                 # Residual capture against the Kimi-K2.6 full-audit step labels
    │   ├── exp_q2_direction_ablation.py       # Hedge-direction directional ablation (project out the hedge direction per layer band)
    │   ├── exp_q2_heldout_and_altdir.py       # Held-out hedge-direction ablation + alternative direction estimators + random-direction null
    │   ├── exp_p_activation_patch.py          # Layerwise activation patching to localize the hedging circuit (per-layer patch recovery)
    │   └── exp_z_universal_style_circuit.py   # Cross-family style-layer localization (per-model hedge patch-recovery runs)
    └── utils/
        └── answer_extract.py                  # Regex MCQ answer-letter extractor (A-D) shared by the CoT generation scripts
