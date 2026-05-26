"""
Generic LoRA SFT on v2 SFT data (free-form CoT).

Supports:
- Qwen3 (8B/14B): pass --disable-thinking to add empty <think></think> prefix
- Llama-3.1-8B, Mistral-7B, etc: standard chat template (no --disable-thinking)

Usage:
    python train_lora_qwen3.py --model-path ... --sft-data ... --output-dir ... --disable-thinking
"""

import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ_ROOT)

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# Tokenization
# ============================================================

def tokenize_for_sft(examples, tokenizer, max_length=3072, disable_thinking=False):
    """Tokenize with labels masked on input prompt tokens.

    If disable_thinking=True (Qwen3):
      [user turn] <|im_start|>assistant\n<think>\n\n</think>\n\n [CoT] <|im_end|>
      The generation_prompt includes the empty think block → gets masked.

    Otherwise (Llama, Mistral, etc):
      [user turn] <|im_start|>assistant\n [CoT] <|im_end|>
      Standard chat template.

    Only the CoT chain tokens are trained on.
    """
    input_ids_list, labels_list, attention_mask_list = [], [], []

    chat_kwargs = {}
    if disable_thinking:
        chat_kwargs["enable_thinking"] = False

    for i in range(len(examples["sft_input"])):
        messages = [
            {"role": "user",      "content": examples["sft_input"][i]},
            {"role": "assistant", "content": examples["sft_target"][i]},
        ]

        # Full conversation
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **chat_kwargs,
        )
        full_tokens = tokenizer(
            full_text, truncation=True, max_length=max_length,
            return_tensors=None, add_special_tokens=False,
        )

        # Generation prompt — marks where the CoT chain starts
        user_msgs = [{"role": "user", "content": examples["sft_input"][i]}]
        gen_prompt = tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
        gen_tokens = tokenizer(
            gen_prompt, truncation=True, max_length=max_length,
            return_tensors=None, add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        labels    = input_ids.copy()
        mask_len  = len(gen_tokens["input_ids"])

        for j in range(min(mask_len, len(labels))):
            labels[j] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(full_tokens["attention_mask"])

    return {
        "input_ids":      input_ids_list,
        "labels":         labels_list,
        "attention_mask": attention_mask_list,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",   type=str, required=True)
    parser.add_argument("--sft-data",     type=str, required=True)
    parser.add_argument("--output-dir",   type=str, required=True)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--batch-size",   type=int,   default=2)
    parser.add_argument("--grad-accum",   type=int,   default=8)
    parser.add_argument("--lora-rank",    type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length",   type=int,   default=3072)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-split",   type=float, default=0.02)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--bf16",         action="store_true", default=True)
    parser.add_argument("--logging-steps",type=int,   default=10)
    parser.add_argument("--save-steps",   type=int,   default=500)
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Pass enable_thinking=False to chat template (for Qwen3)")
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    print("=" * 60)
    print(f"LoRA SFT: {model_name}")
    if args.disable_thinking:
        print("  (thinking mode: DISABLED)")
    print("=" * 60)

    # ---- Load data ----
    print(f"\n1. Loading {args.sft_data} ...")
    records = []
    with open(args.sft_data) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"   {len(records)} SFT examples")

    dataset = Dataset.from_list(records)
    if args.eval_split > 0 and len(records) > 200:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = dataset, None
    print(f"   Train={len(train_ds)}, Eval={len(eval_ds) if eval_ds else 0}")

    # ---- Tokenizer ----
    print(f"\n2. Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sanity check for thinking mode
    if args.disable_thinking:
        test_msgs = [{"role": "user", "content": "test"}]
        test_out = tokenizer.apply_chat_template(
            test_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert "<think>" in test_out and "</think>" in test_out, \
            "Expected empty <think></think> block in generation prompt"
        print("   Thinking mode: disabled (empty <think></think> prefix confirmed)")
    else:
        print("   Thinking mode: N/A (standard chat template)")

    # ---- Model ----
    print(f"\n3. Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    print(f"   {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # ---- LoRA ----
    print(f"\n4. Applying LoRA (r={args.lora_rank}, α={args.lora_alpha}) ...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ---- Tokenize ----
    print(f"\n5. Tokenizing (max_length={args.max_length}) ...")
    t0 = time.time()

    def tokenize_fn(x):
        return tokenize_for_sft(x, tokenizer, args.max_length, args.disable_thinking)

    train_tok = train_ds.map(tokenize_fn, batched=True, batch_size=64,
                             remove_columns=train_ds.column_names)
    eval_tok  = eval_ds.map(tokenize_fn,  batched=True, batch_size=64,
                             remove_columns=eval_ds.column_names) if eval_ds else None
    print(f"   Done in {time.time()-t0:.1f}s")

    # ---- Training ----
    os.makedirs(args.output_dir, exist_ok=True)
    total_steps = (len(train_tok) // (args.batch_size * args.grad_accum)) * args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        eval_steps=None,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    print(f"\n6. Training ({total_steps} steps approx) ...")
    trainer.train()

    print(f"\n7. Saving adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
