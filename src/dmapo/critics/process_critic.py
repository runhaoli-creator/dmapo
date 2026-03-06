"""
src/dmapo/critics/process_critic.py
──────────────────────────────────────
Process critic: evaluates reasoning quality.
Returns a scalar penalty and a failure_type label.
"""
from __future__ import annotations

import re
import logging
from typing import Any

log = logging.getLogger(__name__)

_PENALTY_RE = re.compile(r"\b(yes|flawed|circular|unsupported|contradictory)\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a reasoning quality evaluator. "
    "Read the instruction and response carefully. "
    "Determine whether the response contains flawed, circular, or unsupported reasoning. "
    "Reply on the first line with YES or NO. "
    "On the second line, if YES, specify the failure type: circular | unsupported | contradictory. "
    "On the third line, give one short reason."
)


def build_user_prompt(prompt: str, response: str) -> str:
    return (
        f"Instruction:\n{prompt}\n\n"
        f"Response:\n{response}\n\n"
        "Is the reasoning flawed? (YES/NO, failure_type, reason):"
    )


def parse_critic_output(output: str) -> dict[str, Any]:
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]

    flawed = False
    failure_type = "none"
    reason = ""

    if lines:
        flawed = bool(re.match(r"^yes\b", lines[0], re.IGNORECASE))
    if len(lines) > 1 and flawed:
        for ft in ("circular", "unsupported", "contradictory"):
            if ft in lines[1].lower():
                failure_type = ft
                break
        else:
            failure_type = "unspecified"
    if len(lines) > 2:
        reason = lines[2][:200]
    elif len(lines) > 1:
        reason = lines[1][:200]

    return {
        "penalty": 1.0 if flawed else 0.0,
        "failure_type": failure_type,
        "reason": reason,
    }
