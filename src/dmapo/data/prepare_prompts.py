"""
src/dmapo/data/prepare_prompts.py
──────────────────────────────────
Stage 1 – Load and deduplicate prompts from one or more HuggingFace datasets
(UltraFeedback, HelpSteer2), then write them to data/raw/<source>.jsonl.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import orjson
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── per-source column mapping ──────────────────────────────────────────────────

_PROMPT_COLUMNS: dict[str, str] = {
    "openbmb/UltraFeedback": "instruction",
    "nvidia/HelpSteer2": "prompt",
}


def _extract_prompt(row: dict[str, Any], hf_dataset: str) -> str | None:
    col = _PROMPT_COLUMNS.get(hf_dataset)
    if col is None:
        # generic fallback: try common column names
        for candidate in ("instruction", "prompt", "question", "input"):
            if candidate in row:
                return str(row[candidate]).strip()
        return None
    return str(row[col]).strip() if row.get(col) else None


# ── main ───────────────────────────────────────────────────────────────────────

def prepare_prompts(cfg: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.get("seed", 42))
    all_prompts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source in cfg["prompt_sources"]:
        name = source["name"]
        hf_name = source["hf_dataset"]
        split = source.get("split", "train")
        max_prompts = source.get("max_prompts", 5000)

        log.info("Loading %s (split=%s, max=%d) …", hf_name, split, max_prompts)
        ds = load_dataset(hf_name, split=split, trust_remote_code=True)

        rows = list(ds)
        rng.shuffle(rows)

        collected = 0
        out_path = output_dir / f"{name}.jsonl"
        with out_path.open("wb") as fh:
            for row in tqdm(rows, desc=name, unit="row"):
                prompt = _extract_prompt(row, hf_name)
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                record = {"source": name, "prompt": prompt}
                fh.write(orjson.dumps(record) + b"\n")
                all_prompts.append(record)
                collected += 1
                if collected >= max_prompts:
                    break

        log.info("  → saved %d prompts to %s", collected, out_path)

    # Combined file
    combined_path = output_dir / "prompts_combined.jsonl"
    with combined_path.open("wb") as fh:
        for rec in all_prompts:
            fh.write(orjson.dumps(rec) + b"\n")
    log.info("Combined: %d prompts → %s", len(all_prompts), combined_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 – Prepare prompts")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = yaml.safe_load(fh)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    prepare_prompts(cfg, raw_dir)


if __name__ == "__main__":
    main()
