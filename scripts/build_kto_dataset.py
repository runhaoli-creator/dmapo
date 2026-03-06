"""
scripts/build_kto_dataset.py
──────────────────────────────
Stage 4 – Build KTO and DPO preference datasets from scored + gated candidates.

Usage:
    python scripts/build_kto_dataset.py --training-config configs/training.yaml \
                                         --arbitration-config configs/arbitration.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmapo.training.dataset_builder import build_kto, build_dpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 – Build KTO/DPO datasets")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--arbitration-config", default="configs/arbitration.yaml")
    args = parser.parse_args()

    with open(args.training_config) as fh:
        train_cfg = yaml.safe_load(fh)
    with open(args.arbitration_config) as fh:
        arb_cfg = yaml.safe_load(fh)

    ds_cfg = train_cfg["dataset"]
    min_gap = arb_cfg.get("min_score_gap", 1.5)

    gated_train = Path(arb_cfg["output"]["train"])
    gated_val = Path(arb_cfg["output"]["val"])

    for gated_path, kto_out, dpo_out in [
        (gated_train, Path(ds_cfg["kto_train"]), Path(ds_cfg["dpo_train"])),
        (gated_val,   Path(ds_cfg["kto_val"]),   Path(ds_cfg["dpo_val"])),
    ]:
        if not gated_path.exists():
            log.warning("Gated file not found: %s — skipping", gated_path)
            continue
        split = "train" if "train" in gated_path.name else "val"
        log.info("=== Building %s split ===", split)
        n_kto = build_kto(gated_path, kto_out)
        n_dpo = build_dpo(gated_path, dpo_out, min_score_gap=min_gap)
        log.info("  KTO rows: %d   DPO pairs: %d", n_kto, n_dpo)

    log.info("Dataset construction complete.")


if __name__ == "__main__":
    main()
