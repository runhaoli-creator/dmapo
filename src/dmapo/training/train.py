"""
src/dmapo/training/train.py
────────────────────────────
Stage 6 – KTO (or DPO) fine-tuning of the policy model.

Uses TRL's KTOTrainer / DPOTrainer with PEFT LoRA adapters.
Fully configurable via configs/training.yaml.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
import orjson
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_preference_dataset(path: str, fmt: str, test_size: float, shuffle: bool, seed: int = 42):
    records = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))

    ds = Dataset.from_list(records)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


# ── LoRA config ────────────────────────────────────────────────────────────────

def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def run_training(train_cfg: dict[str, Any]) -> None:
    model_cfg = train_cfg["model"]
    lora_cfg = train_cfg["lora"]
    algo_cfg = train_cfg.get(train_cfg["training"]["algorithm"], {})
    trn = train_cfg["training"]
    ds_cfg = train_cfg["dataset"]

    model_name = model_cfg["name"]
    algorithm = trn.get("algorithm", "kto").lower()

    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
        )

    log.info("Loading base model: %s", model_name)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
    )
    model = get_peft_model(model, build_lora_config(lora_cfg))
    model.print_trainable_parameters()

    log.info("Loading dataset from %s …", ds_cfg["path"])
    train_ds, eval_ds = load_preference_dataset(
        path=ds_cfg["path"],
        fmt=algorithm,
        test_size=ds_cfg.get("test_size", 0.05),
        shuffle=ds_cfg.get("shuffle", True),
    )
    log.info("Train: %d  |  Eval: %d", len(train_ds), len(eval_ds))

    output_dir = trn.get("output_dir", "outputs/dmapo_policy")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
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
        evaluation_strategy=trn.get("evaluation_strategy", "steps"),
        dataloader_num_workers=trn.get("dataloader_num_workers", 4),
        report_to=trn.get("report_to", "none"),
        remove_unused_columns=False,
    )

    if algorithm == "kto":
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
            beta=algo_cfg.get("beta", 0.1),
            desirable_weight=algo_cfg.get("desirable_weight", 1.0),
            undesirable_weight=algo_cfg.get("undesirable_weight", 1.0),
            max_length=algo_cfg.get("max_length", 1024),
            max_prompt_length=algo_cfg.get("max_prompt_length", 512),
            remove_unused_columns=False,
        )
        trainer = KTOTrainer(
            model=model,
            args=kto_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

    elif algorithm == "dpo":
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
            beta=algo_cfg.get("beta", 0.1),
            loss_type=algo_cfg.get("loss_type", "sigmoid"),
            max_length=algo_cfg.get("max_length", 1024),
            max_prompt_length=algo_cfg.get("max_prompt_length", 512),
            remove_unused_columns=False,
        )
        trainer = DPOTrainer(
            model=model,
            args=dpo_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

    else:
        raise ValueError(f"Unknown training algorithm: {algorithm}")

    log.info("Starting %s training …", algorithm.upper())
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 – KTO/DPO training")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = yaml.safe_load(fh)

    run_training(cfg)


if __name__ == "__main__":
    main()
