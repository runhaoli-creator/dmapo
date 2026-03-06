"""
scripts/train_dpo.py
──────────────────────
Stage 5b (optional) – Fine-tune policy model with DPO using TRL + LoRA.
Use this as a baseline comparison against KTO.

Usage:
    python scripts/train_dpo.py --config configs/training.yaml

    # Multi-GPU:
    accelerate launch scripts/train_dpo.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmapo.training.trainer import load_model_and_tokenizer, apply_lora, load_jsonl_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5b – DPO training (baseline)")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    trn = cfg["training"]
    dpo_cfg = cfg["dpo"]
    ds_cfg = cfg["dataset"]

    # Override output dir to keep DPO separate
    output_dir = trn["output_dir"].rstrip("/") + "_dpo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dpo_train_path = ds_cfg["dpo_train"]
    dpo_val_path = ds_cfg["dpo_val"]

    if not Path(dpo_train_path).exists():
        log.error("DPO train file not found: %s — run build_kto_dataset.py first.", dpo_train_path)
        sys.exit(1)

    train_ds = load_jsonl_dataset(dpo_train_path)
    eval_ds = load_jsonl_dataset(dpo_val_path) if Path(dpo_val_path).exists() else None

    log.info("DPO Train: %d  |  Eval: %s", len(train_ds), len(eval_ds) if eval_ds else "none")

    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    model = apply_lora(model, cfg["lora"])

    from trl import DPOConfig, DPOTrainer

    dpo_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=trn.get("num_train_epochs", 2),
        per_device_train_batch_size=trn.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=trn.get("gradient_accumulation_steps", 8),
        learning_rate=trn.get("learning_rate", 5e-5),
        lr_scheduler_type=trn.get("lr_scheduler_type", "cosine"),
        warmup_ratio=trn.get("warmup_ratio", 0.05),
        optim=trn.get("optim", "adamw_torch_fused"),
        bf16=trn.get("bf16", True),
        fp16=trn.get("fp16", False),
        logging_steps=trn.get("logging_steps", 10),
        save_steps=trn.get("save_steps", 200),
        eval_steps=trn.get("eval_steps", 200),
        eval_strategy=trn.get("evaluation_strategy", "steps"),
        report_to=trn.get("report_to", "none"),
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        max_length=dpo_cfg.get("max_length", 1024),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
        remove_unused_columns=False,
        dataloader_num_workers=trn.get("dataloader_num_workers", 2),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting DPO training …")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved → %s", output_dir)


if __name__ == "__main__":
    main()
