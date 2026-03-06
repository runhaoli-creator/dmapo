"""
scripts/generate_candidates.py
────────────────────────────────
Stage 2 – Generate N candidate responses per prompt using a causal LM.

Usage:
    python scripts/generate_candidates.py --config configs/generation.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmapo.data.generator import generate_candidates, _load_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 – Generate candidates")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split to generate for",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap total prompts per split (useful for smoke-testing)",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="Zero-based shard index for this worker (default: 0)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards / parallel workers (default: 1)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="CUDA device index to pin this worker to",
    )
    args = parser.parse_args()

    # Pin to specific GPU before any CUDA calls
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model_name = cfg["model"]
    num_candidates = cfg.get("num_candidates", 4)
    max_new_tokens = cfg.get("max_new_tokens", 512)
    temperature = cfg.get("temperature", 0.8)
    top_p = cfg.get("top_p", 0.95)
    batch_size = cfg.get("batch_size", 4)
    resume = cfg.get("resume", True)

    prompts_file = Path(cfg["prompts_file"])
    all_prompts = _load_jsonl(prompts_file)
    if not all_prompts:
        log.error("No prompts found in %s — run prepare_prompts.py first.", prompts_file)
        sys.exit(1)

    # Load train/val split files if they exist alongside the combined file
    base = prompts_file.parent
    stem = prompts_file.stem
    train_path = base / f"{stem}_train.jsonl"
    val_path = base / f"{stem}_val.jsonl"

    if train_path.exists() and val_path.exists():
        train_prompts = _load_jsonl(train_path)
        val_prompts = _load_jsonl(val_path)
    else:
        # fallback: use all for train, empty val
        train_prompts = all_prompts
        val_prompts = []

    out_cfg = cfg.get("output", {})

    if args.max_samples:
        train_prompts = train_prompts[: args.max_samples]
        val_prompts = val_prompts[: args.max_samples]

    # Shard: each worker takes every Nth prompt starting at offset `shard`
    if args.num_shards > 1:
        train_prompts = train_prompts[args.shard :: args.num_shards]
        val_prompts   = val_prompts[args.shard :: args.num_shards]
        log.info("Shard %d/%d — train=%d  val=%d",
                 args.shard, args.num_shards, len(train_prompts), len(val_prompts))

    def shard_path(base_path: str) -> Path:
        p = Path(base_path)
        if args.num_shards > 1:
            return p.parent / f"{p.stem}_shard{args.shard:02d}{p.suffix}"
        return p

    splits_to_run = []
    if args.split in ("train", "both"):
        splits_to_run.append((train_prompts, shard_path(out_cfg.get("train", "data/processed/candidates_train.jsonl"))))
    if args.split in ("val", "both") and val_prompts:
        splits_to_run.append((val_prompts, shard_path(out_cfg.get("val", "data/processed/candidates_val.jsonl"))))

    for prompts, out_path in splits_to_run:
        log.info("=== Generating for: %s (%d prompts) ===", out_path.name, len(prompts))
        generate_candidates(
            prompts=prompts,
            model_name=model_name,
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
            out_path=out_path,
            resume=resume,
        )

    log.info("Candidate generation complete.")


if __name__ == "__main__":
    main()
