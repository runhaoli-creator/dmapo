"""
src/dmapo/data/generate_candidates.py
───────────────────────────────────────
Stage 2 – For each prompt, run a base LLM to produce N candidate responses.
Writes data/processed/candidates.jsonl.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import orjson
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def build_generation_config(gen_cfg: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=gen_cfg.get("max_new_tokens", 512),
        do_sample=True,
        temperature=gen_cfg.get("temperature", 0.8),
        top_p=gen_cfg.get("top_p", 0.95),
    )


def generate_candidates(cfg: dict[str, Any]) -> None:
    gen_cfg = cfg["candidate_generation"]
    model_name = gen_cfg["model"]
    num_candidates = gen_cfg.get("num_candidates", 4)
    batch_size = gen_cfg.get("batch_size", 4)

    prompts_path = Path(cfg["paths"]["raw_data_dir"]) / "prompts_combined.jsonl"
    out_path = Path(cfg["paths"]["candidates_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer / model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    gen_config = build_generation_config(gen_cfg)

    prompts = load_prompts(prompts_path)
    log.info("Generating %d candidates for %d prompts …", num_candidates, len(prompts))

    def _chat_format(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    with out_path.open("wb") as fh:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i : i + batch_size]
            formatted = [_chat_format(r["prompt"]) for r in batch]

            enc = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            # Generate num_candidates times
            all_responses: list[list[str]] = [[] for _ in batch]
            for _ in range(num_candidates):
                with torch.no_grad():
                    out = model.generate(**enc, generation_config=gen_config)
                input_len = enc["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(
                    out[:, input_len:], skip_special_tokens=True
                )
                for j, text in enumerate(decoded):
                    all_responses[j].append(text.strip())

            for record, responses in zip(batch, all_responses):
                entry = {
                    "source": record.get("source", ""),
                    "prompt": record["prompt"],
                    "candidates": responses,
                }
                fh.write(orjson.dumps(entry) + b"\n")

    log.info("Candidates written to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 – Generate candidates")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = yaml.safe_load(fh)

    generate_candidates(cfg)


if __name__ == "__main__":
    main()
