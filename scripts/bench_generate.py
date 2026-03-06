"""
scripts/bench_generate.py
─────────────────────────
Generate responses for MT-Bench and AlpacaEval from a model (base, LoRA adapter, or merged).
Supports multi-turn (MT-Bench) and single-turn (AlpacaEval).

Usage:
    # Base model
    python scripts/bench_generate.py --model mistralai/Mistral-7B-Instruct-v0.2 \
        --bench mt_bench --output outputs/bench/base_mt_bench.jsonl

    # LoRA adapter
    python scripts/bench_generate.py --model mistralai/Mistral-7B-Instruct-v0.2 \
        --adapter outputs/dmapo_policy --bench mt_bench \
        --output outputs/bench/dmapo_mt_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_model(model_name: str, adapter_path: str | None, dtype=torch.bfloat16):
    """Load base model, optionally merge a LoRA adapter."""
    log.info("Loading tokenizer from %s", adapter_path or model_name)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading base model %s …", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
    )

    if adapter_path:
        log.info("Loading & merging LoRA from %s …", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def load_mt_bench(path: str = "data/benchmarks/mt_bench/question.jsonl") -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def load_alpaca_eval() -> list[dict]:
    """Load AlpacaEval dataset (805 instructions) from local JSON."""
    import json
    path = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "alpaca_eval" / "alpaca_eval.json"
    with open(path) as f:
        data = json.load(f)
    return [{"instruction": r["instruction"], "dataset": r["dataset"]} for r in data]


@torch.no_grad()
def generate_single(model, tokenizer, messages: list[dict], max_new_tokens: int = 2048) -> str:
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


def run_mt_bench(model, tokenizer, questions: list[dict], max_new_tokens: int = 2048) -> list[dict]:
    results = []
    for q in tqdm(questions, desc="MT-Bench"):
        turns = q["turns"]
        messages = []
        answers = []

        # Turn 1
        messages.append({"role": "user", "content": turns[0]})
        ans1 = generate_single(model, tokenizer, messages, max_new_tokens)
        answers.append(ans1)

        # Turn 2 (if exists)
        if len(turns) > 1:
            messages.append({"role": "assistant", "content": ans1})
            messages.append({"role": "user", "content": turns[1]})
            ans2 = generate_single(model, tokenizer, messages, max_new_tokens)
            answers.append(ans2)

        results.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "turns": turns,
            "answers": answers,
        })
    return results


def run_alpaca_eval(model, tokenizer, instructions: list[dict], max_new_tokens: int = 2048) -> list[dict]:
    results = []
    for item in tqdm(instructions, desc="AlpacaEval"):
        messages = [{"role": "user", "content": item["instruction"]}]
        output = generate_single(model, tokenizer, messages, max_new_tokens)
        results.append({
            "instruction": item["instruction"],
            "output": output,
            "dataset": item.get("dataset", ""),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name/path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--bench", required=True, choices=["mt_bench", "alpaca_eval"])
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=None, help="Pin to specific GPU")
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model, tokenizer = load_model(args.model, args.adapter)

    if args.bench == "mt_bench":
        questions = load_mt_bench()
        results = run_mt_bench(model, tokenizer, questions, args.max_new_tokens)
    else:
        instructions = load_alpaca_eval()
        results = run_alpaca_eval(model, tokenizer, instructions, args.max_new_tokens)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info("Saved %d results → %s", len(results), args.output)


if __name__ == "__main__":
    main()
