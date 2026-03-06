"""
src/dmapo/judges/judge_pool.py
──────────────────────────────
Stage 3 – Multi-agent scoring.

Loads a pool of heterogeneous judges (helpfulness, factuality, conciseness),
scores every (prompt, candidate) pair, and writes data/processed/scored.jsonl.
The same underlying instruction-tuned model is used for all judges via
different system/user prompts, reusing a single loaded model instance.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import orjson
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_judge import BaseJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Built-in judge prompts ─────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an expert evaluator. "
    "After reading the instruction and the response, output a single integer score "
    "between 1 and 10 on a line by itself. "
    "Do NOT add explanation after the score."
)

_JUDGE_TEMPLATES: dict[str, str] = {
    "helpfulness": (
        "Instruction:\n{prompt}\n\nResponse:\n{response}\n\n"
        "Rate how HELPFUL this response is (1=not helpful, 10=extremely helpful).\nScore:"
    ),
    "factuality": (
        "Instruction:\n{prompt}\n\nResponse:\n{response}\n\n"
        "Rate how FACTUALLY ACCURATE this response is (1=many errors, 10=fully accurate).\nScore:"
    ),
    "conciseness": (
        "Instruction:\n{prompt}\n\nResponse:\n{response}\n\n"
        "Rate how CONCISE this response is (1=very verbose/padded, 10=perfectly concise).\nScore:"
    ),
}

_SCORE_RE = re.compile(r"\b([1-9]|10)\b")


class PromptJudge(BaseJudge):
    def __init__(
        self,
        name: str,
        cfg: dict[str, Any],
        tokenizer: Any,
        model: Any,
    ) -> None:
        super().__init__(cfg, tokenizer, model)
        self.name = name
        self._template = _JUDGE_TEMPLATES[name]

    def build_prompt(self, prompt: str, response: str) -> str:
        return self._template.format(prompt=prompt, response=response)

    def parse_score(self, output: str) -> float | None:
        # Take the LAST integer in [1,10] found in the output
        matches = _SCORE_RE.findall(output)
        if not matches:
            return None
        return self.clamp(float(matches[-1]))


# ── Inference helper ───────────────────────────────────────────────────────────

def _run_judge_batch(
    judge: PromptJudge,
    prompts: list[str],
    responses: list[str],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 16,
) -> list[float | None]:
    """Score a batch of (prompt, response) pairs with one judge."""
    chats = [
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": judge.build_prompt(p, r)},
        ]
        for p, r in zip(prompts, responses)
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
    decoded = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
    return [judge.parse_score(d) for d in decoded]


# ── Main scoring loop ──────────────────────────────────────────────────────────

def score_candidates(cfg: dict[str, Any]) -> None:
    judge_cfgs = cfg["judges"]
    batch_size = cfg["candidate_generation"].get("batch_size", 4)

    candidates_path = Path(cfg["paths"]["candidates_file"])
    out_path = Path(cfg["paths"]["scored_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # All judges share the same model (first judge's model setting)
    model_name = judge_cfgs[0]["model"]
    log.info("Loading judge model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    judge_names = [j["name"] for j in judge_cfgs]
    judges = [PromptJudge(name=n, cfg={}, tokenizer=tokenizer, model=model) for n in judge_names]

    records = []
    with candidates_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))

    log.info("Scoring %d examples with judges: %s …", len(records), judge_names)

    with out_path.open("wb") as fh:
        for judge in judges:
            log.info("  Judge: %s", judge.name)
            # Flatten all (prompt, candidate) pairs for this judge
            pairs: list[tuple[int, int, str, str]] = []
            for ex_idx, rec in enumerate(records):
                for cand_idx, cand in enumerate(rec["candidates"]):
                    pairs.append((ex_idx, cand_idx, rec["prompt"], cand))

            for i in tqdm(range(0, len(pairs), batch_size), desc=judge.name):
                batch = pairs[i : i + batch_size]
                ex_idxs, cand_idxs, prompts, responses = zip(*batch)
                scores = _run_judge_batch(judge, list(prompts), list(responses), tokenizer, model)

                for ex_i, cand_i, score in zip(ex_idxs, cand_idxs, scores):
                    if score is None:
                        score = 5.0  # neutral fallback
                    records[ex_i].setdefault("judge_scores", {}).setdefault(judge.name, {})[cand_i] = score

        for rec in records:
            fh.write(orjson.dumps(rec) + b"\n")

    log.info("Scored file written to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 – Multi-agent scoring")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = yaml.safe_load(fh)

    score_candidates(cfg)


if __name__ == "__main__":
    main()
