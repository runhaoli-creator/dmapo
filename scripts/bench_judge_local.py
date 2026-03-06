"""
scripts/bench_judge_local.py
─────────────────────────────
Judge MT-Bench and AlpacaEval outputs using a local LLM (Qwen3-8B).

MT-Bench: Single-answer grading on a 1–10 scale per turn.
AlpacaEval: Pairwise comparison vs reference (text-davinci-003 outputs).

Usage:
    python scripts/bench_judge_local.py --bench mt_bench \
        --input outputs/bench/dmapo_mt_bench.jsonl \
        --output outputs/bench/dmapo_mt_bench_judged.jsonl \
        --gpu 0

    python scripts/bench_judge_local.py --bench alpaca_eval \
        --input outputs/bench/dmapo_alpaca_eval.jsonl \
        --output outputs/bench/dmapo_alpaca_eval_judged.jsonl \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

JUDGE_MODEL = "Qwen/Qwen3-8B"

# ── MT-Bench single-answer grading prompt ─────────────────────────────────────

MT_BENCH_SYSTEM = (
    "You are a helpful assistant that evaluates the quality of AI assistant responses.\n"
    "Please act as an impartial judge and evaluate the quality of the response provided "
    "by an AI assistant to the user question displayed below. Your evaluation should "
    "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
    "and level of detail of the response.\n"
    "Begin your evaluation by providing a short explanation. Be as objective as possible.\n"
    'After providing your explanation, you must rate the response on a scale of 1 to 10 '
    'by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'
)

MT_BENCH_TURN1 = (
    "[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n"
    "[The End of Assistant's Answer]"
)

MT_BENCH_TURN2 = (
    "[Context — Previous Turn]\n"
    "[User Question]\n{question_1}\n"
    "[Assistant Answer]\n{answer_1}\n\n"
    "[Current Turn]\n"
    "[User Question]\n{question_2}\n\n"
    "[The Start of Assistant's Answer]\n{answer_2}\n"
    "[The End of Assistant's Answer]"
)

# ── AlpacaEval pairwise prompt ────────────────────────────────────────────────

ALPACA_EVAL_SYSTEM = (
    "You are a helpful assistant that compares two AI assistant outputs and decides which "
    "one is better. You will be given an instruction and two responses (A and B). Compare "
    "them on helpfulness, accuracy, relevance, and detail.\n"
    'Output your verdict as exactly one of: "[[A]]", "[[B]]", or "[[tie]]".'
)

ALPACA_EVAL_USER = (
    "[Instruction]\n{instruction}\n\n"
    "[Response A]\n{output_a}\n\n"
    "[Response B]\n{output_b}\n\n"
    "Which response is better? Provide a brief explanation then your verdict "
    "in the format [[A]], [[B]], or [[tie]]."
)


def load_judge(model_name: str = JUDGE_MODEL, dtype=torch.bfloat16):
    log.info("Loading judge model %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def judge_call(model, tokenizer, system: str, user: str, max_new_tokens: int = 2048) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user + "\n/no_think"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    gen_ids = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def extract_rating(text: str) -> float:
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    if m:
        return float(m.group(1))
    m = re.search(r"[Rr]ating[:\s]+(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    # Last resort: find any standalone number 1-10
    m = re.search(r"\b(\d{1,2})\s*/\s*10\b", text)
    if m:
        return float(m.group(1))
    return -1.0


def extract_winner(text: str) -> str:
    m = re.search(r"\[\[(A|B|tie)\]\]", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback patterns
    text_lower = text.lower()
    if "response a is better" in text_lower or "response a wins" in text_lower:
        return "A"
    if "response b is better" in text_lower or "response b wins" in text_lower:
        return "B"
    return "TIE"


def judge_mt_bench(model, tokenizer, records: list[dict]) -> list[dict]:
    judged = []
    for rec in tqdm(records, desc="Judging MT-Bench"):
        turns = rec["turns"]
        answers = rec["answers"]
        scores = []

        # Turn 1
        user_msg = MT_BENCH_TURN1.format(question=turns[0], answer=answers[0])
        judgment1 = judge_call(model, tokenizer, MT_BENCH_SYSTEM, user_msg)
        score1 = extract_rating(judgment1)
        scores.append(score1)

        # Turn 2
        if len(turns) > 1 and len(answers) > 1:
            user_msg = MT_BENCH_TURN2.format(
                question_1=turns[0], answer_1=answers[0],
                question_2=turns[1], answer_2=answers[1],
            )
            judgment2 = judge_call(model, tokenizer, MT_BENCH_SYSTEM, user_msg)
            score2 = extract_rating(judgment2)
            scores.append(score2)

        valid_scores = [s for s in scores if s > 0]
        rec["scores"] = scores
        rec["avg_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else -1.0
        rec["judgments"] = [judgment1] + ([judgment2] if len(scores) > 1 else [])
        judged.append(rec)

    return judged


def judge_alpaca_eval(model, tokenizer, records: list[dict], reference: list[dict]) -> list[dict]:
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
        judgment = judge_call(model, tokenizer, ALPACA_EVAL_SYSTEM, user_msg)
        winner = extract_winner(judgment)

        rec["reference_output"] = ref_output
        rec["judgment"] = judgment
        rec["winner"] = winner
        judged.append(rec)

    return judged


def load_alpaca_eval_reference() -> list[dict]:
    # Try .json first (array format), then .jsonl
    json_path = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "alpaca_eval" / "alpaca_eval.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return [{"instruction": r["instruction"], "output": r["output"],
                 "dataset": r.get("dataset", "")} for r in data]
    jsonl_path = json_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        records = []
        with open(jsonl_path) as f:
            for line in f:
                records.append(json.loads(line))
        return records
    raise FileNotFoundError(f"Reference file not found: {json_path} or {jsonl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=["mt_bench", "alpaca_eval"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model, tokenizer = load_judge(args.judge_model)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), args.input)

    if args.bench == "mt_bench":
        judged = judge_mt_bench(model, tokenizer, records)
        valid = [r for r in judged if r["avg_score"] > 0]
        avg = sum(r["avg_score"] for r in valid) / len(valid) if valid else 0
        log.info("MT-Bench average: %.2f / 10  (%d questions)", avg, len(valid))
        cats: dict[str, list[float]] = {}
        for r in valid:
            cats.setdefault(r["category"], []).append(r["avg_score"])
        for cat, sc in sorted(cats.items()):
            log.info("  %-14s %.2f", cat, sum(sc) / len(sc))
    else:
        reference = load_alpaca_eval_reference()
        judged = judge_alpaca_eval(model, tokenizer, records, reference)
        wins = sum(1 for r in judged if r["winner"] == "A")
        ties = sum(1 for r in judged if r["winner"] in ("TIE",))
        losses = sum(1 for r in judged if r["winner"] == "B")
        total = len(judged)
        wr = wins / total * 100 if total else 0
        log.info("AlpacaEval win-rate: %.1f%% (%dW/%dT/%dL / %d)",
                 wr, wins, ties, losses, total)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
