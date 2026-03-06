"""
src/dmapo/judges/scorer.py
───────────────────────────
Unified scoring engine: runs all judges + process critic over a batch of
(prompt, response) pairs using a single loaded model.

Produces full-trace output records with per-judge scores, critic result,
mean score, variance, final score, and confidence gate verdict.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .helpfulness import HelpfulnessJudge
from .factuality import FactualityJudge
from .conciseness import ConcisenessJudge
from .base_judge import BaseJudge
from ..critics.process_critic import (
    SYSTEM_PROMPT as CRITIC_SYS,
    build_user_prompt as critic_user_prompt,
    parse_critic_output,
)

log = logging.getLogger(__name__)


def _build_judge_instances(judges_cfg: list[dict], default_model: str) -> list[BaseJudge]:
    cls_map = {
        "helpfulness": HelpfulnessJudge,
        "factuality": FactualityJudge,
        "conciseness": ConcisenessJudge,
    }
    out = []
    for jcfg in judges_cfg:
        name = jcfg["name"]
        model = jcfg.get("model") or default_model
        score_min = jcfg.get("score_min", 1.0)
        score_max = jcfg.get("score_max", 10.0)
        if name not in cls_map:
            log.warning("Unknown judge '%s', skipping.", name)
            continue
        out.append(cls_map[name](model_name=model, score_min=score_min, score_max=score_max))
    return out


def _run_batch(
    model: Any,
    tokenizer: Any,
    system_prompts: list[str],
    user_prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    chats = [
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ]
        for sys_p, usr_p in zip(system_prompts, user_prompts)
    ]
    formatted = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in chats
    ]
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    input_len = enc["input_ids"].shape[1]
    return tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)


def score_records(
    records: list[dict],
    judges_cfg: list[dict],
    critic_cfg: dict,
    default_model: str,
    alpha: float,
    batch_size: int,
    device: str | None = None,
) -> list[dict]:
    """
    Score every record in `records`.
    Each record must have "prompt" and "response" fields.
    Returns enriched records with full scoring trace.
    """
    judges = _build_judge_instances(judges_cfg, default_model)

    log.info("Loading scoring model: %s", default_model)
    tokenizer = AutoTokenizer.from_pretrained(default_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = device if device else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        default_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()

    n = len(records)

    # ── Run each judge ────────────────────────────────────────────────────────
    from tqdm import tqdm

    for judge in judges:
        log.info("  Judge: %s", judge.name)
        for i in tqdm(range(0, n, batch_size), desc=judge.name, leave=False):
            batch = records[i : i + batch_size]
            sys_prompts = [judge.system_prompt] * len(batch)
            usr_prompts = [judge.user_prompt(r["prompt"], r["response"]) for r in batch]
            outputs = _run_batch(
                model, tokenizer, sys_prompts, usr_prompts,
                max_new_tokens=32,
            )
            for rec, raw in zip(batch, outputs):
                rec.setdefault("judge_scores", {})[judge.name] = {
                    "score": judge.parse_score(raw),
                    "reason": judge.parse_reason(raw),
                    "raw": raw[:200],
                }

    # ── Process critic ────────────────────────────────────────────────────────
    critic_model_name = critic_cfg.get("model") or default_model
    log.info("  Process critic: %s", critic_model_name)
    for i in tqdm(range(0, n, batch_size), desc="Critic", leave=False):
        batch = records[i : i + batch_size]
        sys_prompts = [CRITIC_SYS] * len(batch)
        usr_prompts = [critic_user_prompt(r["prompt"], r["response"]) for r in batch]
        outputs = _run_batch(
            model, tokenizer, sys_prompts, usr_prompts,
            max_new_tokens=critic_cfg.get("max_new_tokens", 16),
        )
        for rec, raw in zip(batch, outputs):
            rec["critic"] = parse_critic_output(raw)

    # ── Aggregate: mean, variance, final score ────────────────────────────────
    for rec in records:
        scores = [v["score"] for v in rec.get("judge_scores", {}).values()]
        if not scores:
            rec["mean_score"] = 5.0
            rec["score_variance"] = 0.0
            rec["final_score"] = 5.0
            continue

        mean_s = statistics.mean(scores)
        var_s = statistics.variance(scores) if len(scores) > 1 else 0.0
        penalty = rec.get("critic", {}).get("penalty", 0.0)
        final_s = mean_s - alpha * penalty * mean_s

        rec["mean_score"] = round(mean_s, 4)
        rec["score_variance"] = round(var_s, 4)
        rec["final_score"] = round(final_s, 4)

    return records
