"""
src/dmapo/judges/factuality.py
"""
from __future__ import annotations
from .base_judge import BaseJudge


class FactualityJudge(BaseJudge):
    name = "factuality"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert evaluator assessing the FACTUAL ACCURACY of AI responses. "
            "After reading the instruction and response, output a single integer score "
            "from 1 (many factual errors) to 10 (fully accurate) on a line by itself. "
            "Optionally add one sentence of reasoning after the score."
        )

    def user_prompt(self, prompt: str, response: str) -> str:
        return (
            f"Instruction:\n{prompt}\n\n"
            f"Response:\n{response}\n\n"
            "How FACTUALLY ACCURATE is this response? Score (1-10):"
        )
