"""
scripts/bench_judge.py
──────────────────────
Judge MT-Bench and AlpacaEval outputs using GPT-4o.

MT-Bench: Single-answer grading on a 1–10 scale per turn (standard protocol).
AlpacaEval: Pairwise comparison vs reference (GPT-4 Turbo outputs).

Usage:
    # MT-Bench judging
    python scripts/bench_judge.py --bench mt_bench \
        --input outputs/bench/dmapo_mt_bench.jsonl \
        --output outputs/bench/dmapo_mt_bench_judged.jsonl

    # AlpacaEval judging (pairwise vs reference)
    python scripts/bench_judge.py --bench alpaca_eval \
        --input outputs/bench/dmapo_alpaca_eval.jsonl \
        --output outputs/bench/dmapo_alpaca_eval_judged.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── MT-Bench single-answer grading prompt (official protocol) ─────────────────

MT_BENCH_SYSTEM = """You are a helpful assistant that evaluates the quality of AI assistant responses.
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

MT_BENCH_TURN1 = """[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

MT_BENCH_TURN2 = """[Context — Previous Turn]
[User Question]
{question_1}
[Assistant Answer]
{answer_1}

[Current Turn]
[User Question]
{question_2}

[The Start of Assistant's Answer]
{answer_2}
[The End of Assistant's Answer]"""


# ── AlpacaEval pairwise prompt ────────────────────────────────────────────────

ALPACA_EVAL_SYSTEM = """You are a helpful assistant that compares two AI assistant outputs and decides which one is better.
You will be given an instruction and two responses (A and B). Compare them on helpfulness, accuracy, relevance, and detail.
Output your verdict as exactly one of: "[[A]]", "[[B]]", or "[[tie]]"."""

ALPACA_EVAL_USER = """[Instruction]
{instruction}

[Response A]
{output_a}

[Response B]
{output_b}

Which response is better? Provide a brief explanation then your verdict in the format [[A]], [[B]], or [[tie]]."""


def call_gpt4(client: OpenAI, system: str, user: str, model: str = "gpt-4o",
              max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            log.warning("GPT-4 call failed (attempt %d): %s — retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    return ""


def extract_rating(text: str) -> float:
    """Extract [[N]] rating from judge output."""
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    if m:
        return float(m.group(1))
    # Fallback: look for "Rating: N"
    m = re.search(r"[Rr]ating[:\s]+(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else -1.0


def extract_winner(text: str) -> str:
    """Extract [[A]], [[B]], or [[tie]] from pairwise judge output."""
    m = re.search(r"\[\[(A|B|tie)\]\]", text, re.IGNORECASE)
    return m.group(1).upper() if m else "TIE"


def judge_mt_bench(client: OpenAI, records: list[dict]) -> list[dict]:
    judged = []
    for rec in tqdm(records, desc="Judging MT-Bench"):
        turns = rec["turns"]
        answers = rec["answers"]
        scores = []

        # Turn 1
        user_msg = MT_BENCH_TURN1.format(question=turns[0], answer=answers[0])
        judgment1 = call_gpt4(client, MT_BENCH_SYSTEM, user_msg)
        score1 = extract_rating(judgment1)
        scores.append(score1)

        # Turn 2 (if exists)
        if len(turns) > 1 and len(answers) > 1:
            user_msg = MT_BENCH_TURN2.format(
                question_1=turns[0], answer_1=answers[0],
                question_2=turns[1], answer_2=answers[1],
            )
            judgment2 = call_gpt4(client, MT_BENCH_SYSTEM, user_msg)
            score2 = extract_rating(judgment2)
            scores.append(score2)

        rec["scores"] = scores
        rec["avg_score"] = sum(s for s in scores if s > 0) / max(len([s for s in scores if s > 0]), 1)
        rec["judgments"] = [judgment1] + ([judgment2] if len(scores) > 1 else [])
        judged.append(rec)

    return judged


def judge_alpaca_eval(client: OpenAI, records: list[dict], reference: list[dict]) -> list[dict]:
    """Pairwise: model output (A) vs reference output (B)."""
    ref_map = {r["instruction"]: r.get("output", "") for r in reference}
    judged = []
    for rec in tqdm(records, desc="Judging AlpacaEval"):
        ref_output = ref_map.get(rec["instruction"], "")
        if not ref_output:
            continue

        user_msg = ALPACA_EVAL_USER.format(
            instruction=rec["instruction"],
            output_a=rec["output"],
            output_b=ref_output,
        )
        judgment = call_gpt4(client, ALPACA_EVAL_SYSTEM, user_msg)
        winner = extract_winner(judgment)

        rec["reference_output"] = ref_output
        rec["judgment"] = judgment
        rec["winner"] = winner
        judged.append(rec)

    return judged


def load_alpaca_eval_reference() -> list[dict]:
    """Load the official AlpacaEval reference outputs from local JSON."""
    import json as _json
    path = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "alpaca_eval" / "alpaca_eval.json"
    with open(path) as f:
        data = _json.load(f)
    return [{"instruction": r["instruction"], "output": r["output"],
             "dataset": r["dataset"]} for r in data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=["mt_bench", "alpaca_eval"])
    parser.add_argument("--input", required=True, help="Model output JSONL")
    parser.add_argument("--output", required=True, help="Judged output JSONL")
    parser.add_argument("--model", default="gpt-4o", help="Judge model")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        return
    client = OpenAI(api_key=api_key)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), args.input)

    if args.bench == "mt_bench":
        judged = judge_mt_bench(client, records)
        # Summary
        valid = [r for r in judged if r["avg_score"] > 0]
        avg = sum(r["avg_score"] for r in valid) / len(valid) if valid else 0
        log.info("MT-Bench average score: %.2f / 10 (%d questions)", avg, len(valid))

        # Per-category
        cats: dict[str, list[float]] = {}
        for r in valid:
            cats.setdefault(r["category"], []).append(r["avg_score"])
        for cat, scores in sorted(cats.items()):
            log.info("  %s: %.2f", cat, sum(scores) / len(scores))
    else:
        reference = load_alpaca_eval_reference()
        judged = judge_alpaca_eval(client, records, reference)
        wins = sum(1 for r in judged if r["winner"] == "A")
        ties = sum(1 for r in judged if r["winner"] == "TIE")
        losses = sum(1 for r in judged if r["winner"] == "B")
        total = len(judged)
        wr = wins / total if total else 0
        log.info("AlpacaEval win-rate: %.1f%% (%d wins, %d ties, %d losses / %d)",
                 wr * 100, wins, ties, losses, total)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
