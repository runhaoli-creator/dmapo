"""
src/dmapo/training/trainer.py
───────────────────────────────
Shared training utilities: model loading, LoRA setup, dataset loading.
Used by both train_kto.py and train_dpo.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_jsonl_dataset(path: str, shuffle: bool = True, seed: int = 42) -> Dataset:
    records = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    ds = Dataset.from_list(records)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    return ds


def load_model_and_tokenizer(model_cfg: dict) -> tuple:
    model_name = model_cfg["name"]
    torch_dtype = _DTYPE_MAP.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
    attn_impl = model_cfg.get("attn_implementation", "eager")

    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model: %s  dtype=%s", model_name, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    return model, tokenizer


def apply_lora(model: Any, lora_cfg: dict) -> Any:
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
