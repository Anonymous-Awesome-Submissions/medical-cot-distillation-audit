#!/usr/bin/env python3
"""Merge a LoRA SFT adapter into the base model for inference/evaluation.

The merged model is the distilled student used for vLLM generation.
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-dir", type=str,
                        default="experiments/module1/lora_adapter_qwen3_8b_v2")
    parser.add_argument("--output-dir", type=str,
                        default="./models/qwen3-8b-sft-merged")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output_dir, "model.safetensors.index.json")):
        print(f"Merged model already exists at {args.output_dir}, skipping.")
        return

    print(f"Loading base model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU for merge
    )

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done! Merged model saved.")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")


if __name__ == "__main__":
    main()
