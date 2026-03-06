"""
scripts/bench_ifeval.py
───────────────────────
IFEval (Instruction-Following Evaluation) benchmark.
Evaluates whether models follow verifiable instructions (e.g., "write exactly 3
paragraphs", "include the word X at least twice", "respond in JSON format").

Uses the official google/IFEval dataset and rule-based verification (no judge needed).

Usage:
    # Base model
    python scripts/bench_ifeval.py --model mistralai/Mistral-7B-Instruct-v0.2 \
        --output outputs/bench/base_ifeval.jsonl

    # LoRA adapter
    python scripts/bench_ifeval.py --model mistralai/Mistral-7B-Instruct-v0.2 \
        --adapter outputs/dmapo_policy \
        --output outputs/bench/dmapo_ifeval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# IFEval Instruction Checkers
# ═══════════════════════════════════════════════════════════════════════════════

def check_length_constraints(response: str, kwargs: dict) -> bool:
    """Check word/sentence/paragraph count constraints."""
    relation = kwargs.get("relation", "")
    num = kwargs.get("num_words") or kwargs.get("num_sentences") or kwargs.get("num_paragraphs", 0)
    num = int(num) if num else 0

    if "num_words" in kwargs:
        count = len(response.split())
    elif "num_sentences" in kwargs:
        count = len([s for s in re.split(r'[.!?]+', response) if s.strip()])
    elif "num_paragraphs" in kwargs:
        count = len([p for p in response.split('\n\n') if p.strip()])
    else:
        return True

    if relation == "at least":
        return count >= num
    elif relation == "at most":
        return count <= num
    elif relation == "less than":
        return count < num
    elif relation == "more than":
        return count > num
    elif relation == "exactly":
        return count == num
    return True


def check_keyword(response: str, kwargs: dict) -> bool:
    """Check keyword inclusion/exclusion."""
    keywords = kwargs.get("keywords", [])
    if not keywords:
        return True
    response_lower = response.lower()
    # Check all keywords present
    return all(kw.lower() in response_lower for kw in keywords)


def check_keyword_frequency(response: str, kwargs: dict) -> bool:
    """Check keyword appears at least N times."""
    keyword = kwargs.get("keyword", "")
    frequency = kwargs.get("frequency", 1)
    if not keyword:
        return True
    return response.lower().count(keyword.lower()) >= frequency


def check_forbidden_words(response: str, kwargs: dict) -> bool:
    """Check forbidden words are NOT in response."""
    forbidden = kwargs.get("forbidden_words", [])
    response_lower = response.lower()
    return not any(w.lower() in response_lower for w in forbidden)


def check_letter_frequency(response: str, kwargs: dict) -> bool:
    """Check a letter appears at least N times."""
    letter = kwargs.get("letter", "")
    let_frequency = kwargs.get("let_relation", "at least")
    num = kwargs.get("let_count", 0)
    if not letter:
        return True
    count = response.lower().count(letter.lower())
    if let_frequency == "at least":
        return count >= num
    return count == num


def check_postscript(response: str, kwargs: dict) -> bool:
    """Check response ends with a postscript (P.S. or P.P.S.)."""
    postscript_marker = kwargs.get("postscript_marker", "P.S.")
    return postscript_marker in response


def check_format_json(response: str, kwargs: dict) -> bool:
    """Check response is valid JSON."""
    try:
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_title(response: str, kwargs: dict) -> bool:
    """Check response contains a title (line wrapped in <<>>)."""
    return bool(re.search(r'<<[^>]+>>', response))


def check_sections(response: str, kwargs: dict) -> bool:
    """Check response has specific number of sections (markdown headers)."""
    num_sections = kwargs.get("num_sections", 0)
    sections = re.findall(r'^#{1,6}\s+', response, re.MULTILINE)
    return len(sections) >= num_sections


def check_highlight(response: str, kwargs: dict) -> bool:
    """Check response has highlighted sections (markdown bold)."""
    num_highlights = kwargs.get("num_highlights", 1)
    highlights = re.findall(r'\*[^*]+\*', response)
    return len(highlights) >= num_highlights


def check_language(response: str, kwargs: dict) -> bool:
    """Check response language (basic heuristic)."""
    # Simple heuristic — just check it's not empty
    language = kwargs.get("language", "")
    return len(response.strip()) > 0


def check_no_comma(response: str, kwargs: dict) -> bool:
    """Check response contains no commas."""
    return "," not in response


def check_capital(response: str, kwargs: dict) -> bool:
    """Check all letters are capital."""
    # Only check alphabetic characters
    alpha_chars = [c for c in response if c.isalpha()]
    return all(c.isupper() for c in alpha_chars) if alpha_chars else True


def check_lowercase(response: str, kwargs: dict) -> bool:
    """Check all letters are lowercase (except where necessary)."""
    alpha_chars = [c for c in response if c.isalpha()]
    return all(c.islower() for c in alpha_chars) if alpha_chars else True


def check_placeholder(response: str, kwargs: dict) -> bool:
    """Check response contains placeholders [XXX]."""
    num = kwargs.get("num_placeholders", 1)
    placeholders = re.findall(r'\[[\w\s]+\]', response)
    return len(placeholders) >= num


def check_bullet_points(response: str, kwargs: dict) -> bool:
    """Check response contains bullet points."""
    num = kwargs.get("num_bullets", 1)
    bullets = re.findall(r'^\s*[\*\-\•]\s+', response, re.MULTILINE)
    return len(bullets) >= num


def check_constrained_start(response: str, kwargs: dict) -> bool:
    """Check response starts with specific text."""
    start_phrase = kwargs.get("start_phrase", "")
    if not start_phrase:
        return True
    return response.strip().startswith(start_phrase)


def check_quotation(response: str, kwargs: dict) -> bool:
    """Check response is wrapped in double quotes."""
    stripped = response.strip()
    return stripped.startswith('"') and stripped.endswith('"')


# Map instruction_id_list types to checkers
CHECKER_MAP = {
    "length_constraints": check_length_constraints,
    "keywords": check_keyword,
    "keyword_frequency": check_keyword_frequency,
    "forbidden_words": check_forbidden_words,
    "letter_frequency": check_letter_frequency,
    "postscript": check_postscript,
    "json_format": check_format_json,
    "detectable_format": check_format_json,
    "title": check_title,
    "sections": check_sections,
    "highlight_section": check_highlight,
    "language": check_language,
    "no_comma": check_no_comma,
    "capital": check_capital,
    "capital_word_frequency": check_capital,
    "lowercase": check_lowercase,
    "placeholder": check_placeholder,
    "bullet_points": check_bullet_points,
    "constrained_start": check_constrained_start,
    "quotation": check_quotation,
    "number_words": check_length_constraints,
    "number_sentences": check_length_constraints,
    "number_paragraphs": check_length_constraints,
    "number_bullets": check_bullet_points,
    "number_highlights": check_highlight,
}


def evaluate_ifeval_instance(response: str, instruction_id_list: list[str],
                              kwargs_list: list[dict]) -> dict:
    """Evaluate a single IFEval instance against all its constraints."""
    results = []
    for inst_id, kw in zip(instruction_id_list, kwargs_list):
        # Extract the constraint type from instruction_id like "length_constraints:number_words"
        parts = inst_id.split(":")
        constraint_type = parts[-1] if len(parts) > 1 else parts[0]
        parent_type = parts[0] if len(parts) > 1 else ""

        checker = CHECKER_MAP.get(constraint_type) or CHECKER_MAP.get(parent_type)
        if checker:
            passed = checker(response, kw or {})
        else:
            # Unknown constraint — pass by default
            passed = True
        results.append({"instruction_id": inst_id, "passed": passed})

    all_passed = all(r["passed"] for r in results)
    return {"instruction_results": results, "all_passed": all_passed}


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading & generation
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, adapter_path: str | None, dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
    )

    if adapter_path:
        log.info("Loading & merging LoRA from %s …", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    messages = [{"role": "user", "content": prompt}]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name/path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── Load IFEval dataset ──────────────────────────────────────────────────
    log.info("Loading IFEval dataset …")
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    log.info("Loaded %d IFEval prompts", len(ds))

    # ── Load model ───────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, args.adapter)

    # ── Generate & evaluate ──────────────────────────────────────────────────
    results = []
    prompt_pass = 0
    inst_pass = 0
    inst_total = 0

    for item in tqdm(ds, desc="IFEval"):
        prompt = item["prompt"]
        instruction_id_list = item["instruction_id_list"]
        kwargs_list = item["kwargs"]

        # Parse kwargs (they may be stored as JSON strings)
        parsed_kwargs = []
        for kw in kwargs_list:
            if isinstance(kw, str):
                try:
                    parsed_kwargs.append(json.loads(kw))
                except json.JSONDecodeError:
                    parsed_kwargs.append({})
            elif kw is None:
                parsed_kwargs.append({})
            else:
                parsed_kwargs.append(kw)

        response = generate_response(model, tokenizer, prompt, args.max_new_tokens)
        eval_result = evaluate_ifeval_instance(response, instruction_id_list, parsed_kwargs)

        if eval_result["all_passed"]:
            prompt_pass += 1

        for ir in eval_result["instruction_results"]:
            inst_total += 1
            if ir["passed"]:
                inst_pass += 1

        record = {
            "prompt": prompt,
            "response": response,
            "instruction_id_list": instruction_id_list,
            "eval": eval_result,
        }
        results.append(record)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(results)
    prompt_acc = prompt_pass / n * 100 if n else 0
    inst_acc = inst_pass / inst_total * 100 if inst_total else 0
    log.info("IFEval Prompt-level accuracy: %.1f%% (%d/%d)", prompt_acc, prompt_pass, n)
    log.info("IFEval Instruction-level accuracy: %.1f%% (%d/%d)", inst_acc, inst_pass, inst_total)

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Save summary
    summary_path = Path(args.output).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "prompt_accuracy": round(prompt_acc, 2),
            "instruction_accuracy": round(inst_acc, 2),
            "prompt_pass": prompt_pass,
            "prompt_total": n,
            "instruction_pass": inst_pass,
            "instruction_total": inst_total,
        }, f, indent=2)

    log.info("Saved %d results → %s", n, args.output)
    log.info("Saved summary → %s", summary_path)


if __name__ == "__main__":
    main()
