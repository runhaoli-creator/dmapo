"""
scripts/train_dpo_baseline.py
──────────────────────────────
Train a DPO baseline on UltraFeedback-binarized for comparison with DMAPO.
Uses the same base model (Mistral-7B-Instruct-v0.2) and LoRA config.

Usage:
    python scripts/train_dpo_baseline.py --training-config configs/training.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--output-dir", default="outputs/dpo_baseline")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (for quick tests)")
    args = parser.parse_args()

    with open(args.training_config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    torch_dtype = torch.bfloat16

    # ── Load UltraFeedback-binarized ─────────────────────────────────────────
    log.info("Loading HuggingFaceH4/ultrafeedback_binarized …")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    log.info("Loaded %d preference pairs", len(ds))

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        log.info("Capped to %d samples", len(ds))

    # Format for TRL DPOTrainer: needs 'prompt', 'chosen', 'rejected'
    def format_dpo(example):
        prompt = example["chosen"][0]["content"]  # user message
        chosen = example["chosen"][1]["content"]   # assistant chosen
        rejected = example["rejected"][1]["content"]  # assistant rejected
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    ds = ds.map(format_dpo, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=0.02, seed=42)
    log.info("Train: %d  Val: %d", len(ds["train"]), len(ds["test"]))

    # ── Load model + tokenizer ───────────────────────────────────────────────
    log.info("Loading %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
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

    # ── DPO training config ──────────────────────────────────────────────────
    dpo_beta = cfg.get("dpo", {}).get("beta", 0.1)
    train_cfg = cfg.get("train", {})

    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=dpo_beta,
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
        max_length=1024,
        report_to="none",
        remove_unused_columns=False,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("Starting DPO training (beta=%.2f) …", dpo_beta)
    trainer = DPOTrainer(
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
    log.info("DPO baseline training complete.")


if __name__ == "__main__":
    main()
