"""
scripts/train_sft_baseline.py
──────────────────────────────
Train an SFT baseline on UltraFeedback chosen responses for comparison with DMAPO.
Uses the same base model (Mistral-7B-Instruct-v0.2) and LoRA config.

Usage:
    python scripts/train_sft_baseline.py --training-config configs/training.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--output-dir", default="outputs/sft_baseline")
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    torch_dtype = torch.bfloat16

    # ── Load UltraFeedback-binarized (chosen only → SFT) ────────────────────
    log.info("Loading HuggingFaceH4/ultrafeedback_binarized …")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    log.info("Loaded %d examples", len(ds))

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        log.info("Capped to %d samples", len(ds))

    # Format for SFT: just the chosen conversation as text
    def format_sft(example):
        messages = [
            {"role": "user", "content": example["chosen"][0]["content"]},
            {"role": "assistant", "content": example["chosen"][1]["content"]},
        ]
        return {"messages": messages}

    ds = ds.map(format_sft, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.02, seed=42)
    log.info("Train: %d  Val: %d", len(ds["train"]), len(ds["test"]))

    # ── Load model + tokenizer ───────────────────────────────────────────────
    log.info("Loading %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT uses right padding

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto",
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── SFT training config ──────────────────────────────────────────────────
    train_cfg = cfg.get("train", {})

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=train_cfg.get("epochs", 1),
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        max_seq_length=1024,
        report_to="none",
    )

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("Starting SFT training …")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    log.info("Saving model to %s …", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("SFT baseline training complete.")


if __name__ == "__main__":
    main()
