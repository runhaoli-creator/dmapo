"""
src/dmapo/training/dataset_builder.py
───────────────────────────────────────
Reads gated, scored candidates and builds:
  - KTO dataset:  {prompt, completion, label}  where label = True/False
  - DPO dataset:  {prompt, chosen, rejected}   paired from same prompt
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

log = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec) + b"\n")


def build_kto(gated_path: Path, out_path: Path) -> int:
    """
    Build KTO-format dataset.
    Each candidate is one row; label=True for desirable, False for undesirable.
    """
    records = _load_jsonl(gated_path)
    kto_rows: list[dict] = []
    for rec in records:
        label_str = rec.get("gate_label", "discard")
        if label_str not in ("desirable", "undesirable"):
            continue
        kto_rows.append({
            "prompt": rec["prompt"],
            "completion": rec["response"],
            "label": label_str == "desirable",
            # trace fields
            "id": rec.get("id", ""),
            "candidate_id": rec.get("candidate_id", 0),
            "source_dataset": rec.get("source_dataset", ""),
            "final_score": rec.get("final_score", 0.0),
            "gate_label": label_str,
        })
    _write_jsonl(out_path, kto_rows)
    log.info("KTO dataset: %d rows → %s", len(kto_rows), out_path)
    return len(kto_rows)


def build_dpo(gated_path: Path, out_path: Path, min_score_gap: float = 1.5) -> int:
    """
    Build DPO-format dataset.
    For each unique prompt, pair the highest-scoring desirable with the
    lowest-scoring undesirable candidate (if gap >= min_score_gap).
    """
    records = _load_jsonl(gated_path)

    # Group by prompt
    by_prompt: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"desirable": [], "undesirable": []})
    for rec in records:
        label = rec.get("gate_label", "discard")
        if label in ("desirable", "undesirable"):
            by_prompt[rec["prompt"]][label].append(rec)

    dpo_rows: list[dict] = []
    for prompt, groups in by_prompt.items():
        des = sorted(groups["desirable"], key=lambda r: r.get("final_score", 0), reverse=True)
        und = sorted(groups["undesirable"], key=lambda r: r.get("final_score", 0))
        if not des or not und:
            continue
        chosen_rec = des[0]
        rejected_rec = und[0]
        gap = chosen_rec.get("final_score", 0) - rejected_rec.get("final_score", 0)
        if gap < min_score_gap:
            continue
        dpo_rows.append({
            "prompt": prompt,
            "chosen": chosen_rec["response"],
            "rejected": rejected_rec["response"],
            "chosen_score": chosen_rec.get("final_score"),
            "rejected_score": rejected_rec.get("final_score"),
            "score_gap": round(gap, 4),
            "id": chosen_rec.get("id", ""),
            "source_dataset": chosen_rec.get("source_dataset", ""),
        })

    _write_jsonl(out_path, dpo_rows)
    log.info("DPO dataset: %d pairs → %s", len(dpo_rows), out_path)
    return len(dpo_rows)
