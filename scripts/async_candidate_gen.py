"""Asynchronous batched candidate generation for the DMAPO on-policy pipeline.

Fans prompts out to an OpenAI-compatible vLLM inference endpoint (or the
DMAPO judge API when used as a dry-run mock) using bounded
``asyncio.Semaphore`` concurrency with exponential-backoff retries.

Intended replacement for the serial ``scripts/generate_candidates.py`` when
running against a remote or multi-GPU vLLM server.

Example::

    python scripts/async_candidate_gen.py \\
        --prompts data/processed/all_prompts.jsonl \\
        --output data/processed/candidates.jsonl \\
        --endpoint http://vllm:8000/v1/completions \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --k 4 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore


async def _one(
    client: "httpx.AsyncClient",
    endpoint: str,
    model: str,
    prompt: str,
    k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    sem: asyncio.Semaphore,
) -> list[str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "n": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    delay = 0.75
    async with sem:
        for attempt in range(retries + 1):
            try:
                r = await client.post(endpoint, json=payload, timeout=120.0)
                r.raise_for_status()
                body = r.json()
                return [choice["text"] for choice in body.get("choices", [])]
            except Exception as exc:
                if attempt == retries:
                    print(f"[warn] giving up on prompt after {retries + 1} attempts: {exc}",
                          file=sys.stderr)
                    return []
                await asyncio.sleep(delay)
                delay *= 2
    return []


async def _run(args: argparse.Namespace) -> None:
    if httpx is None:
        print("httpx is required: pip install httpx", file=sys.stderr)
        sys.exit(2)

    inp = Path(args.prompts)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    prompts: list[tuple[str, dict]] = []
    with inp.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("prompt") or obj.get("instruction") or obj.get("text")
            if not p:
                continue
            prompts.append((p, obj))

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            _one(client, args.endpoint, args.model, p, args.k, args.max_tokens,
                 args.temperature, args.top_p, args.retries, sem)
            for p, _ in prompts
        ]
        candidates_per_prompt = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    n_total = 0
    with out.open("w") as fh:
        for (prompt, meta), cands in zip(prompts, candidates_per_prompt):
            for j, c in enumerate(cands):
                n_total += 1
                record = {**meta, "prompt": prompt, "candidate_idx": j, "candidate": c}
                fh.write(json.dumps(record) + "\n")

    print(
        f"done: {len(prompts)} prompts × k={args.k} -> {n_total} candidates · "
        f"elapsed={elapsed:.1f}s · throughput={n_total / max(elapsed, 1e-6):.1f} cand/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
