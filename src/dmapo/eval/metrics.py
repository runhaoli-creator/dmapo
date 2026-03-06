"""
src/dmapo/eval/metrics.py
──────────────────────────
Compute all evaluation metrics from gated data and model outputs.
No generation required for statistical metrics; generation-based metrics
(perplexity, win-rate) are optional and gated by available model paths.
"""

from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import orjson

log = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


# ── Statistical metrics (no model needed) ─────────────────────────────────────

def label_distribution(gated_records: list[dict]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for rec in gated_records:
        label = rec.get("gate_label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    total = len(gated_records)
    return {
        "total": total,
        "counts": counts,
        "fractions": {k: round(v / total, 4) for k, v in counts.items()} if total else {},
    }


def abstention_rate(scored_records: list[dict], gated_records: list[dict]) -> dict[str, Any]:
    n_scored = len(scored_records)
    n_kept = len(gated_records)
    # Gated records have gate_label; scored records do not.
    # Count labels from the gated set and derive filtered-out count.
    label_counts: dict[str, int] = {}
    for r in gated_records:
        lbl = r.get("gate_label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    n_filtered_out = n_scored - n_kept
    return {
        "total_scored": n_scored,
        "filtered_out": n_filtered_out,
        "kept": n_kept,
        "kept_label_counts": label_counts,
        "abstention_rate": round(n_filtered_out / n_scored, 4) if n_scored else 0.0,
        "keep_rate": round(n_kept / n_scored, 4) if n_scored else 0.0,
    }


def judge_variance_stats(scored_records: list[dict]) -> dict[str, Any]:
    variances = [r.get("score_variance", 0.0) for r in scored_records if "score_variance" in r]
    if not variances:
        return {}
    return {
        "mean_variance": round(statistics.mean(variances), 4),
        "median_variance": round(statistics.median(variances), 4),
        "max_variance": round(max(variances), 4),
        "min_variance": round(min(variances), 4),
        "std_variance": round(statistics.stdev(variances) if len(variances) > 1 else 0.0, 4),
    }


def response_length_stats(gated_records: list[dict]) -> dict[str, Any]:
    lengths = [len(r.get("response", "").split()) for r in gated_records]
    if not lengths:
        return {}
    des_lengths = [len(r.get("response", "").split()) for r in gated_records if r.get("gate_label") == "desirable"]
    und_lengths = [len(r.get("response", "").split()) for r in gated_records if r.get("gate_label") == "undesirable"]

    def _stats(vals: list[int]) -> dict:
        if not vals:
            return {}
        return {
            "mean": round(statistics.mean(vals), 1),
            "median": round(statistics.median(vals), 1),
            "min": min(vals),
            "max": max(vals),
        }

    return {
        "all": _stats(lengths),
        "desirable": _stats(des_lengths),
        "undesirable": _stats(und_lengths),
    }


def final_score_stats(gated_records: list[dict]) -> dict[str, Any]:
    scores = [r.get("final_score", 0.0) for r in gated_records]
    des_scores = [r.get("final_score", 0.0) for r in gated_records if r.get("gate_label") == "desirable"]
    und_scores = [r.get("final_score", 0.0) for r in gated_records if r.get("gate_label") == "undesirable"]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        return {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n": len(vals),
        }

    return {
        "all": _stats(scores),
        "desirable": _stats(des_scores),
        "undesirable": _stats(und_scores),
    }


def per_judge_stats(scored_records: list[dict]) -> dict[str, Any]:
    judge_scores: dict[str, list[float]] = {}
    for rec in scored_records:
        for judge_name, jdata in rec.get("judge_scores", {}).items():
            score = jdata.get("score", jdata) if isinstance(jdata, dict) else float(jdata)
            judge_scores.setdefault(judge_name, []).append(float(score))
    result = {}
    for name, scores in judge_scores.items():
        result[name] = {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 4),
            "n": len(scores),
        }
    return result


# ── Generation-based metrics (require loaded models) ──────────────────────────

def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    import torch
    from tqdm import tqdm

    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        with torch.no_grad():
            out = model(**enc, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


def compute_win_rate(
    policy_model: Any,
    base_model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    def _mean_logprob(model: Any, prompt_texts: list[str]) -> list[float]:
        chats = [[{"role": "user", "content": p}] for p in prompt_texts]
        formatted = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        enc = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        gen_ids = out.sequences[:, enc["input_ids"].shape[1]:]
        stacked = torch.stack(out.scores, dim=1)
        results = []
        for b in range(gen_ids.shape[0]):
            ids = gen_ids[b]
            valid = ids != tokenizer.pad_token_id
            if not valid.any():
                results.append(float("-inf"))
                continue
            lp = torch.nn.functional.log_softmax(stacked[b, :ids.shape[0]][valid], dim=-1)
            results.append(lp[torch.arange(valid.sum()), ids[valid]].mean().item())
        return results

    wins = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="WinRate", leave=False):
        batch = prompts[i : i + batch_size]
        pol_lp = _mean_logprob(policy_model, batch)
        bas_lp = _mean_logprob(base_model, batch)
        wins += sum(p > b for p, b in zip(pol_lp, bas_lp))

    return {
        "win_rate": round(wins / len(prompts), 4) if prompts else 0.0,
        "wins": wins,
        "n_prompts": len(prompts),
    }
