"""
src/dmapo/judges/helpfulness.py
"""
from __future__ import annotations
from .base_judge import BaseJudge


class HelpfulnessJudge(BaseJudge):
    name = "helpfulness"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert evaluator assessing the HELPFULNESS of AI responses. "
            "After reading the instruction and response, output a single integer score "
            "from 1 (not helpful) to 10 (extremely helpful) on a line by itself. "
            "Optionally add one sentence of reasoning after the score."
        )

    def user_prompt(self, prompt: str, response: str) -> str:
        return (
            f"Instruction:\n{prompt}\n\n"
            f"Response:\n{response}\n\n"
            "How HELPFUL is this response? Score (1-10):"
        )
