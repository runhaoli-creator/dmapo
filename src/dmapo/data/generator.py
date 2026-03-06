"""
src/dmapo/data/generator.py
────────────────────────────
Candidate generation: loads a causal LM and samples N responses per prompt.
Supports resume (skips already-generated prompt ids).

Speed strategy: tile each prompt num_candidates times so all candidates for a
batch are produced in a single model.generate() call instead of a sequential loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

log = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def _get_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    for rec in _load_jsonl(path):
        done.add(rec["id"])
    return done


def _chat_format(tokenizer: Any, prompt: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def generate_candidates(
    prompts: list[dict],
    model_name: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    out_path: Path,
    resume: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _get_done_ids(out_path) if resume else set()
    pending = [p for p in prompts if p["id"] not in done_ids]
    log.info(
        "Generating: total=%d  already_done=%d  pending=%d",
        len(prompts), len(done_ids), len(pending),
    )
    if not pending:
        log.info("Nothing to generate — all done.")
        return

    log.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # JIT-compile for ~20% speedup on Blackwell (sm_100 arch)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log.info("torch.compile enabled")
    except Exception as e:
        log.warning("torch.compile skipped: %s", e)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    with out_path.open("ab") as fh:
        for i in tqdm(range(0, len(pending), batch_size), desc="Generating"):
            batch = pending[i : i + batch_size]
            formatted = [_chat_format(tokenizer, r["prompt"]) for r in batch]

            # Tile each prompt num_candidates times → single generate() call
            tiled = []
            for text in formatted:
                tiled.extend([text] * num_candidates)

            enc = tokenizer(
                tiled,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(**enc, generation_config=gen_cfg)

            input_len = enc["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
            # decoded layout: [p0c0, p0c1, ..., p0cN, p1c0, p1c1, ..., p1cN, ...]

            for j, record in enumerate(batch):
                for cand_idx in range(num_candidates):
                    text = decoded[j * num_candidates + cand_idx].strip()
                    entry = {
                        "id": record["id"],
                        "source_dataset": record.get("source_dataset", ""),
                        "prompt": record["prompt"],
                        "task_type": record.get("task_type"),
                        "candidate_id": cand_idx,
                        "response": text,
                        "generator_model": model_name,
                    }
                    fh.write(orjson.dumps(entry) + b"\n")
            fh.flush()

    log.info("Candidates written to %s", out_path)
