"""
scripts/score_candidates.py
─────────────────────────────
Stage 3 – Run all judges, process critic, and confidence gating.
Outputs scored + gated JSONL with full trace fields.

Usage:
    python scripts/score_candidates.py --judges-config configs/judges.yaml \
                                        --arbitration-config configs/arbitration.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import orjson
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dmapo.judges.scorer import score_records
from dmapo.arbitration.gating import apply_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        log.error("File not found: %s", path)
        sys.exit(1)
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec) + b"\n")


def process_split_records(
    records: list[dict],
    scored_path: Path,
    gated_path: Path,
    judges_cfg: dict,
    arb_cfg: dict,
    device: str | None = None,
) -> None:
    log.info("Scoring %d candidates", len(records))

    default_model = judges_cfg["default_model"]
    alpha = judges_cfg.get("alpha", 0.15)
    batch_size = judges_cfg.get("batch_size", 4)

    # Score
    records = score_records(
        records=records,
        judges_cfg=judges_cfg["judges"],
        critic_cfg=judges_cfg["critic"],
        default_model=default_model,
        alpha=alpha,
        batch_size=batch_size,
        device=device,
    )
    write_jsonl(scored_path, records)
    log.info("Scored %d records → %s", len(records), scored_path)

    # Gate
    records, gate_stats = apply_gate(
        records=records,
        variance_threshold=arb_cfg["variance_threshold"],
        desirable_threshold=arb_cfg["desirable_threshold"],
        undesirable_threshold=arb_cfg["undesirable_threshold"],
    )
    # Write only non-discarded records to gated file
    gated = [r for r in records if r["gate_label"] in ("desirable", "undesirable")]
    write_jsonl(gated_path, gated)

    log.info(
        "Gate stats: %s  → kept (des+undes)=%d → %s",
        gate_stats,
        len(gated),
        gated_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 – Score candidates")
    parser.add_argument("--judges-config", default="configs/judges.yaml")
    parser.add_argument("--arbitration-config", default="configs/arbitration.yaml")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--shard", type=int, default=0,
                        help="Zero-based shard index")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index to pin this worker to")
    args = parser.parse_args()

    # Pin GPU before any CUDA import
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.judges_config) as fh:
        judges_cfg = yaml.safe_load(fh)
    with open(args.arbitration_config) as fh:
        arb_cfg = yaml.safe_load(fh)

    in_cfg = judges_cfg["input"]
    out_judge = judges_cfg["output"]
    out_arb = arb_cfg["output"]

    def shard_path(base: str) -> Path:
        p = Path(base)
        if args.num_shards > 1:
            return p.parent / f"{p.stem}_shard{args.shard:02d}{p.suffix}"
        return p

    # Device for this worker
    device = "cuda:0" if args.gpu is not None else None

    splits = []
    if args.split in ("train", "both"):
        splits.append((
            Path(in_cfg["train"]),
            shard_path(out_judge["train"]),
            shard_path(out_arb["train"]),
        ))
    if args.split in ("val", "both"):
        splits.append((
            Path(in_cfg["val"]),
            shard_path(out_judge["val"]),
            shard_path(out_arb["val"]),
        ))

    for cand_path, scored_path, gated_path in splits:
        if not cand_path.exists():
            log.warning("Candidates file not found: %s — skipping", cand_path)
            continue
        log.info("=== Processing split: %s (shard %d/%d) ===", cand_path.stem, args.shard, args.num_shards)

        # Load and shard
        records = load_jsonl(cand_path)
        if args.num_shards > 1:
            records = records[args.shard :: args.num_shards]
            log.info("Shard %d/%d: %d records", args.shard, args.num_shards, len(records))

        process_split_records(records, scored_path, gated_path, judges_cfg, arb_cfg, device)

    log.info("Scoring complete.")


if __name__ == "__main__":
    main()
