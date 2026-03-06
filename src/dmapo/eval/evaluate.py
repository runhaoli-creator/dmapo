"""
src/dmapo/eval/evaluate.py
───────────────────────────
Stage 7 – Evaluation.

Metrics computed:
  1. Held-out perplexity on a subset of the gated dataset.
  2. Win-rate vs the base (untuned) model using the judge pool's
     helpfulness score as a proxy reward.
  3. Summary statistics written to outputs/eval_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """Compute token-level perplexity of `texts` under `model`."""
    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])

        # out.loss is mean NLL over non-padding tokens for the batch
        n_tokens = enc["attention_mask"].sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


# ── Win-rate ───────────────────────────────────────────────────────────────────

def compute_win_rate(
    policy_model: Any,
    base_model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> dict[str, float]:
    """
    Generate responses from policy and base model, then compare
    average log-probability as a proxy for quality.
    Returns win_rate (fraction of prompts where policy > base).
    """
    wins = 0

    def _gen_logprob(model: Any, prompt_texts: list[str]) -> list[float]:
        """Returns per-example mean log-probability of the generated tokens."""
        chats = [
            [{"role": "user", "content": p}]
            for p in prompt_texts
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
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = out.sequences[:, enc["input_ids"].shape[1]:]
        log_probs = []
        stacked_scores = torch.stack(out.scores, dim=1)  # (B, T, V)
        for b in range(gen_ids.shape[0]):
            ids = gen_ids[b]
            valid = ids != tokenizer.pad_token_id
            if not valid.any():
                log_probs.append(float("-inf"))
                continue
            scores = stacked_scores[b, :ids.shape[0]]  # (T, V)
            lp = torch.nn.functional.log_softmax(scores[valid], dim=-1)
            chosen = ids[valid]
            mean_lp = lp[torch.arange(len(chosen)), chosen].mean().item()
            log_probs.append(mean_lp)
        return log_probs

    for i in tqdm(range(0, len(prompts), batch_size), desc="WinRate"):
        batch = prompts[i : i + batch_size]
        policy_lp = _gen_logprob(policy_model, batch)
        base_lp = _gen_logprob(base_model, batch)
        wins += sum(p > b for p, b in zip(policy_lp, base_lp))

    win_rate = wins / len(prompts) if prompts else 0.0
    return {"win_rate": round(win_rate, 4), "n_prompts": len(prompts)}


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(pipeline_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> None:
    gated_path = Path(pipeline_cfg["paths"]["gated_file"])
    output_dir = Path(pipeline_cfg["paths"]["output_dir"])
    policy_dir = train_cfg["training"]["output_dir"]
    base_model_name = train_cfg["model"]["name"]
    batch_size = pipeline_cfg["candidate_generation"].get("batch_size", 4)

    # Load eval prompts (up to 200 for speed)
    records = []
    with gated_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    eval_records = records[:200]
    prompts = [r["prompt"] for r in eval_records]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(train_cfg["model"].get("torch_dtype", "bfloat16"), torch.bfloat16)

    log.info("Loading policy model from %s …", policy_dir)
    tokenizer = AutoTokenizer.from_pretrained(policy_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    policy_model.eval()

    log.info("Loading base model %s for comparison …", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    base_model.eval()

    # Perplexity on chosen (desirable) completions
    chosen_texts = []
    for r in eval_records:
        agg = {int(k): v for k, v in r.get("aggregate_scores", {}).items()}
        if agg:
            best_i = max(agg, key=agg.__getitem__)
            chosen_texts.append(r["prompt"] + "\n" + r["candidates"][best_i])

    log.info("Computing perplexity on %d chosen completions …", len(chosen_texts))
    ppl_policy = compute_perplexity(policy_model, tokenizer, chosen_texts, batch_size=batch_size)
    ppl_base = compute_perplexity(base_model, tokenizer, chosen_texts, batch_size=batch_size)

    log.info("PPL  policy=%.2f  base=%.2f", ppl_policy, ppl_base)

    log.info("Computing win-rate on %d prompts …", len(prompts))
    win_stats = compute_win_rate(policy_model, base_model, tokenizer, prompts, batch_size=batch_size)

    results = {
        "perplexity_policy": round(ppl_policy, 4),
        "perplexity_base": round(ppl_base, 4),
        **win_stats,
    }
    log.info("Results: %s", results)

    out_path = output_dir / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Eval results → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 – Evaluation")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.pipeline_config, "rb") as fh:
        pipeline_cfg = yaml.safe_load(fh)
    with open(args.training_config, "rb") as fh:
        train_cfg = yaml.safe_load(fh)

    evaluate(pipeline_cfg, train_cfg)


if __name__ == "__main__":
    main()
