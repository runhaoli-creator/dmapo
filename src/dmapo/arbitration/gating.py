"""
src/dmapo/arbitration/gating.py
────────────────────────────────
Confidence gating logic: given scored records, assign label or abstain.
"""
from __future__ import annotations

from typing import Any


def apply_gate(
    records: list[dict[str, Any]],
    variance_threshold: float,
    desirable_threshold: float,
    undesirable_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Adds a 'gate_label' field to each record:
        'desirable'   – final_score >= desirable_threshold and variance ok
        'undesirable' – final_score <= undesirable_threshold and variance ok
        'abstain'     – variance > variance_threshold
        'discard'     – score between thresholds (no useful signal)

    Returns a new list; input records are mutated in-place for convenience.
    """
    stats = {"desirable": 0, "undesirable": 0, "abstain": 0, "discard": 0}

    for rec in records:
        var = rec.get("score_variance", 0.0)
        score = rec.get("final_score", 5.0)

        if var > variance_threshold:
            label = "abstain"
        elif score >= desirable_threshold:
            label = "desirable"
        elif score <= undesirable_threshold:
            label = "undesirable"
        else:
            label = "discard"

        rec["gate_label"] = label
        stats[label] += 1

    return records, stats
