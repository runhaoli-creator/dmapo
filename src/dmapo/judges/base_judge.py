"""
src/dmapo/judges/base_judge.py
──────────────────────────────
Abstract base for all prompt-based judges.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod


_SCORE_RE = re.compile(r"\b(10|[1-9])(?:\.\d+)?\b")


class BaseJudge(ABC):
    name: str = "base"

    def __init__(self, model_name: str, score_min: float = 1.0, score_max: float = 10.0) -> None:
        self.model_name = model_name
        self.score_min = score_min
        self.score_max = score_max

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Static system prompt for this judge."""

    @abstractmethod
    def user_prompt(self, prompt: str, response: str) -> str:
        """Build the per-example user prompt."""

    def parse_score(self, output: str) -> float:
        matches = _SCORE_RE.findall(output)
        if not matches:
            return (self.score_min + self.score_max) / 2
        raw = float(matches[-1])
        return max(self.score_min, min(self.score_max, raw))

    def parse_reason(self, output: str) -> str:
        lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
        for line in lines:
            if not _SCORE_RE.fullmatch(line.strip()):
                return line[:200]
        return ""
