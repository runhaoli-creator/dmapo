"""
scripts/prepare_prompts.py
───────────────────────────
Stage 1 – Load UltraFeedback and HelpSteer2, normalise, split, and save JSONL.

Usage:
    python scripts/prepare_prompts.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import orjson
import yaml

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmapo.data.loader import load_and_normalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 – Prepare prompts")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max_samples for all datasets (useful for smoke-testing)",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    seed = cfg.get("seed", 42)
    rng = random.Random(seed)
    val_frac = cfg.get("val_fraction", 0.05)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfgs = cfg["datasets"]
    all_records: list[dict] = []

    # ── UltraFeedback ──────────────────────────────────────────────────────────
    uf_cfg = dataset_cfgs["ultrafeedback"]
    uf_records = load_and_normalise(
        hf_name=uf_cfg["hf_name"],
        split=uf_cfg.get("split", "train"),
        max_samples=args.max_samples or uf_cfg.get("max_samples", 10000),
        prompt_field=uf_cfg.get("prompt_field"),
        task_type_field=uf_cfg.get("task_type_field"),
        seed=seed,
    )
    write_jsonl(Path(cfg["ultrafeedback_out"]), uf_records)
    log.info("UltraFeedback: %d prompts → %s", len(uf_records), cfg["ultrafeedback_out"])
    all_records.extend(uf_records)

    # ── HelpSteer2 ─────────────────────────────────────────────────────────────
    hs_cfg = dataset_cfgs["helpsteer2"]
    hs_records = load_and_normalise(
        hf_name=hs_cfg["hf_name"],
        split=hs_cfg.get("split", "train"),
        max_samples=args.max_samples or hs_cfg.get("max_samples", 5000),
        prompt_field=hs_cfg.get("prompt_field"),
        task_type_field=hs_cfg.get("task_type_field"),
        seed=seed,
    )
    write_jsonl(Path(cfg["helpsteer2_out"]), hs_records)
    log.info("HelpSteer2: %d prompts → %s", len(hs_records), cfg["helpsteer2_out"])
    all_records.extend(hs_records)

    # ── Deduplicate & shuffle ──────────────────────────────────────────────────
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in all_records:
        if rec["prompt"] not in seen:
            seen.add(rec["prompt"])
            deduped.append(rec)
    rng.shuffle(deduped)
    log.info("Combined (deduped): %d prompts", len(deduped))

    # ── Train / val split ──────────────────────────────────────────────────────
    n_val = max(1, int(len(deduped) * val_frac))
    val_records = deduped[:n_val]
    train_records = deduped[n_val:]

    combined_path = Path(cfg["combined_out"])
    write_jsonl(combined_path, deduped)

    # Also write train/val splits alongside
    stem = combined_path.stem
    write_jsonl(combined_path.parent / f"{stem}_train.jsonl", train_records)
    write_jsonl(combined_path.parent / f"{stem}_val.jsonl", val_records)

    log.info("Split → train=%d  val=%d", len(train_records), len(val_records))

    # ── Print samples ──────────────────────────────────────────────────────────
    print("\n── Sample rows (first 3) ───────────────────────────────────")
    for rec in deduped[:3]:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    print(f"\nTotal prompts: {len(deduped)}  (train={len(train_records)}, val={len(val_records)})")


if __name__ == "__main__":
    main()
