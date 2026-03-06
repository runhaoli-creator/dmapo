"""
src/dmapo/data/build_dataset.py
────────────────────────────────
Stage 5 – Dataset construction.

Reads the confidence-gated scored examples and converts them into a
KTO-format (or DPO-format) preference dataset.

KTO format  (one row per candidate):
  { "prompt": str, "completion": str, "label": bool }   # True=desirable

DPO format  (one row per prompt):
  { "prompt": str, "chosen": str, "rejected": str }

The top-scoring candidate (above desirable_quantile) is labelled desirable;
the bottom-scoring candidate (below undesirable_quantile) is labelled undesirable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def build_dataset(cfg: dict[str, Any]) -> None:
    dc_cfg = cfg["dataset_construction"]
    fmt = dc_cfg.get("format", "kto").lower()
    desirable_q = dc_cfg.get("desirable_quantile", 0.75)
    undesirable_q = dc_cfg.get("undesirable_quantile", 0.25)
    out_path = Path(dc_cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gated_path = Path(cfg["paths"]["gated_file"])

    records: list[dict[str, Any]] = []
    with gated_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))

    # Collect all aggregate scores to compute global quantile thresholds
    all_scores: list[float] = []
    for rec in records:
        all_scores.extend(rec.get("aggregate_scores", {}).values())

    if not all_scores:
        log.error("No scored records found. Exiting.")
        return

    q_high = float(np.quantile(all_scores, desirable_q))
    q_low = float(np.quantile(all_scores, undesirable_q))
    log.info("Score thresholds: desirable≥%.3f  undesirable≤%.3f", q_high, q_low)

    written = 0
    with out_path.open("wb") as fh:
        for rec in records:
            prompt = rec["prompt"]
            candidates: list[str] = rec["candidates"]
            agg: dict[str, float] = rec.get("aggregate_scores", {})

            # Convert str keys (from JSON) to int
            agg_int = {int(k): v for k, v in agg.items()}

            if fmt == "kto":
                for cand_i, cand_text in enumerate(candidates):
                    score = agg_int.get(cand_i)
                    if score is None:
                        continue
                    if score >= q_high:
                        entry = {"prompt": prompt, "completion": cand_text, "label": True}
                        fh.write(orjson.dumps(entry) + b"\n")
                        written += 1
                    elif score <= q_low:
                        entry = {"prompt": prompt, "completion": cand_text, "label": False}
                        fh.write(orjson.dumps(entry) + b"\n")
                        written += 1

            elif fmt == "dpo":
                if not agg_int:
                    continue
                best_i = max(agg_int, key=agg_int.__getitem__)
                worst_i = min(agg_int, key=agg_int.__getitem__)
                if best_i == worst_i:
                    continue
                entry = {
                    "prompt": prompt,
                    "chosen": candidates[best_i],
                    "rejected": candidates[worst_i],
                }
                fh.write(orjson.dumps(entry) + b"\n")
                written += 1

    log.info("Dataset written: %d rows → %s (format=%s)", written, out_path, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 – Build preference dataset")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = yaml.safe_load(fh)

    build_dataset(cfg)


if __name__ == "__main__":
    main()
