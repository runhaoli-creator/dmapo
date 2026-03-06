"""
src/dmapo/arbitration/confidence_gate.py
──────────────────────────────────────────
Stage 4 – Confidence gating.

For each example, compute the inter-judge score variance across all candidates.
If the variance exceeds `max_judge_variance`, the example is considered unreliable
and discarded. Also enforce a minimum score gap between the best and worst candidate
(min_score_gap) to ensure the preference signal is meaningful.

Input:  data/processed/scored.jsonl  (with aggregate_scores populated by process_critic)
Output: data/processed/gated.jsonl
"""

from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path
from typing import Any

import orjson
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def run_confidence_gate(cfg: dict[str, Any]) -> None:
    gate_cfg = cfg["confidence_gate"]
    max_variance = gate_cfg.get("max_judge_variance", 2.5)
    min_gap = gate_cfg.get("min_score_gap", 1.5)

    scored_path = Path(cfg["paths"]["scored_file"])
    out_path = Path(cfg["paths"]["gated_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = discarded_variance = discarded_gap = 0

    with scored_path.open("rb") as fh_in, out_path.open("wb") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            rec: dict[str, Any] = orjson.loads(line)
            total += 1

            agg: dict[str, float] = rec.get("aggregate_scores", {})
            scores = list(agg.values())

            # Need at least 2 candidates
            if len(scores) < 2:
                discarded_gap += 1
                continue

            # 1. Inter-judge variance gate
            # Compute variance across all per-judge raw scores for this example
            all_raw: list[float] = []
            for judge_dict in rec.get("judge_scores", {}).values():
                all_raw.extend(judge_dict.values())

            if len(all_raw) >= 2:
                var = statistics.variance(all_raw)
                if var > max_variance:
                    discarded_variance += 1
                    continue

            # 2. Score-gap gate
            best = max(scores)
            worst = min(scores)
            if (best - worst) < min_gap:
                discarded_gap += 1
                continue

            kept += 1
            fh_out.write(orjson.dumps(rec) + b"\n")

    log.info(
        "Confidence gate: total=%d  kept=%d  discarded(variance)=%d  discarded(gap)=%d",
        total, kept, discarded_variance, discarded_gap,
    )
    log.info("Gated dataset → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 – Confidence gating")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = yaml.safe_load(fh)

    run_confidence_gate(cfg)


if __name__ == "__main__":
    main()
