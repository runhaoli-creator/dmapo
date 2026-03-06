"""
src/dmapo/data/loader.py
─────────────────────────
Loads UltraFeedback and HelpSteer2 from HuggingFace and normalises them
into a shared schema:

    {
        "id":             str,
        "source_dataset": str,
        "prompt":         str,
        "task_type":      str | null,
    }
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from datasets import load_dataset

log = logging.getLogger(__name__)

# ── Field mappings per dataset ─────────────────────────────────────────────────

_FIELD_MAP: dict[str, dict[str, str | None]] = {
    "openbmb/UltraFeedback": {
        "prompt":    "instruction",
        "task_type": None,
    },
    "nvidia/HelpSteer2": {
        "prompt":    "prompt",
        "task_type": None,
    },
}

_FALLBACK_PROMPT_FIELDS = ("instruction", "prompt", "question", "input", "text")


def _make_id(source: str, index: int, prompt: str) -> str:
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return f"{source}_{index:06d}_{h}"


def _extract_field(row: dict[str, Any], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row and row[c]:
            return str(row[c]).strip()
    return None


def load_and_normalise(
    hf_name: str,
    split: str,
    max_samples: int,
    prompt_field: str | None = None,
    task_type_field: str | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Downloads a dataset, extracts prompts, and returns normalised records.
    """
    log.info("Loading %s (split=%s, max=%d) …", hf_name, split, max_samples)
    ds = load_dataset(hf_name, split=split, trust_remote_code=True)

    # Shuffle deterministically then cap
    ds = ds.shuffle(seed=seed)
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Resolve prompt field
    field_map = _FIELD_MAP.get(hf_name, {})
    p_field = prompt_field or field_map.get("prompt")
    t_field = task_type_field or field_map.get("task_type")

    fallbacks = [p_field] + list(_FALLBACK_PROMPT_FIELDS) if p_field else list(_FALLBACK_PROMPT_FIELDS)

    source_key = hf_name.split("/")[-1].lower().replace("-", "_")
    records: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(ds):
        prompt = _extract_field(row, fallbacks)  # type: ignore[arg-type]
        if not prompt:
            skipped += 1
            continue

        task_type = None
        if t_field and t_field in row:
            task_type = str(row[t_field]).strip() or None

        records.append(
            {
                "id": _make_id(source_key, idx, prompt),
                "source_dataset": hf_name,
                "prompt": prompt,
                "task_type": task_type,
            }
        )

    log.info("  → %d records (skipped %d empty)", len(records), skipped)
    return records
