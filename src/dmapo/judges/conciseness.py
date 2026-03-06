"""
src/dmapo/judges/conciseness.py
"""
from __future__ import annotations
from .base_judge import BaseJudge


class ConcisenessJudge(BaseJudge):
    name = "conciseness"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert evaluator assessing the CONCISENESS of AI responses. "
            "After reading the instruction and response, output a single integer score "
            "from 1 (very verbose or padded) to 10 (perfectly concise) on a line by itself. "
            "Optionally add one sentence of reasoning after the score."
        )

    def user_prompt(self, prompt: str, response: str) -> str:
        return (
            f"Instruction:\n{prompt}\n\n"
            f"Response:\n{response}\n\n"
            "How CONCISE is this response? Score (1-10):"
        )
