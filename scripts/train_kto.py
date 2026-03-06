"""
scripts/train_kto.py
──────────────────────
Stage 5a – Fine-tune policy model with KTO using TRL + LoRA.
Supports single-GPU and multi-GPU (via Accelerate).

Usage:
    python scripts/train_kto.py --config configs/training.yaml

    # Multi-GPU:
    accelerate launch scripts/train_kto.py --config configs/training.yaml
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
    parser = argparse.ArgumentParser(description="Stage 5a – KTO training")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    trn = cfg["training"]
    kto_cfg = cfg["kto"]
    ds_cfg = cfg["dataset"]
    output_dir = trn["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    kto_train_path = ds_cfg["kto_train"]
    kto_val_path = ds_cfg["kto_val"]

    if not Path(kto_train_path).exists():
        log.error("KTO train file not found: %s — run build_kto_dataset.py first.", kto_train_path)
        sys.exit(1)

    train_ds = load_jsonl_dataset(kto_train_path)
    eval_ds = load_jsonl_dataset(kto_val_path) if Path(kto_val_path).exists() else None

    log.info("Train: %d  |  Eval: %s", len(train_ds), len(eval_ds) if eval_ds else "none")

    # ── Model ──────────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    model = apply_lora(model, cfg["lora"])

    # ── KTO Trainer ────────────────────────────────────────────────────────────
    from trl import KTOConfig, KTOTrainer

    kto_args = KTOConfig(
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
        beta=kto_cfg.get("beta", 0.1),
        desirable_weight=kto_cfg.get("desirable_weight", 1.0),
        undesirable_weight=kto_cfg.get("undesirable_weight", 1.0),
        max_length=kto_cfg.get("max_length", 1024),
        remove_unused_columns=False,
        dataloader_num_workers=trn.get("dataloader_num_workers", 2),
    )

    trainer = KTOTrainer(
        model=model,
        args=kto_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info("Starting KTO training …")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved → %s", output_dir)


if __name__ == "__main__":
    main()
