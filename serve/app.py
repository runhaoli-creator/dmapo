"""FastAPI service exposing the DMAPO multi-judge scoring + gating pipeline.

Routes:

* ``GET  /health``              — liveness, judge pool readiness, cache backend
* ``GET  /metrics``              — request / accept / latency counters
* ``POST /v1/score``             — single (prompt, candidate) pair
* ``POST /v1/score_batch``       — batched scoring over up to 128 items

The judge pool is lazy-initialised on the first request so the module imports
cheaply under CI without torch / model weights.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from .cache import build_cache_from_env, make_key
from .schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    JudgeScore,
    MetricsResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger("dmapo.serve")
logging.basicConfig(level=os.getenv("DMAPO_LOG_LEVEL", "INFO"))


class _JudgePoolHandle:
    def __init__(self) -> None:
        self._pool: Any | None = None
        self._critic: Any | None = None
        self._requests_total = 0
        self._accepted_total = 0
        self._rejected_total = 0
        self._aggregate_scores: list[float] = []
        self._latencies_ms: list[float] = []

    def ensure_started(self) -> tuple[Any, Any]:
        if self._pool is not None:
            return self._pool, self._critic

        # Lazy import so CI never needs torch / transformers / model weights
        from dmapo.arbitration.confidence_gate import ConfidenceGate  # type: ignore
        from dmapo.critics.process_critic import ProcessCritic  # type: ignore
        from dmapo.judges.judge_pool import JudgePool  # type: ignore

        cfg_path = os.getenv("DMAPO_SERVE_CONFIG", "configs/judges.yaml")
        self._pool = JudgePool.from_config(cfg_path)
        self._critic = ProcessCritic.from_config(cfg_path)
        self._gate = ConfidenceGate.from_config(os.getenv("DMAPO_ARBITRATION_CONFIG", "configs/arbitration.yaml"))
        logger.info("Judge pool + process critic ready")
        return self._pool, self._critic

    def gate(self) -> Any:
        return self._gate

    def record(self, latency_ms: float, accepted: bool, aggregate: float) -> None:
        self._requests_total += 1
        if accepted:
            self._accepted_total += 1
        else:
            self._rejected_total += 1
        if len(self._aggregate_scores) >= 2048:
            self._aggregate_scores.pop(0)
        self._aggregate_scores.append(aggregate)
        if len(self._latencies_ms) >= 2048:
            self._latencies_ms.pop(0)
        bisect.insort(self._latencies_ms, latency_ms)

    def percentile(self, p: float) -> float:
        if not self._latencies_ms:
            return 0.0
        idx = min(len(self._latencies_ms) - 1, int(len(self._latencies_ms) * p))
        return self._latencies_ms[idx]

    def avg_aggregate(self) -> float:
        if not self._aggregate_scores:
            return 0.0
        return sum(self._aggregate_scores) / len(self._aggregate_scores)

    @property
    def counters(self) -> tuple[int, int, int]:
        return self._requests_total, self._accepted_total, self._rejected_total


handle = _JudgePoolHandle()
cache = build_cache_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("dmapo.serve starting (cache=%s)", cache.backend_name)
    yield
    logger.info("dmapo.serve shutting down")


app = FastAPI(
    title="DMAPO Judge API",
    description="Multi-agent scoring + confidence gating over (prompt, candidate) pairs.",
    version="0.1.0",
    lifespan=lifespan,
)


def _run_pipeline(pool: Any, critic: Any, gate: Any, req: ScoreRequest) -> ScoreResponse:
    requested = req.judges or ["helpfulness", "factuality", "conciseness"]
    scores_raw = pool.score(req.prompt, req.candidate, judge_names=requested)
    judges = [
        JudgeScore(judge=name, score=float(s.get("score", 0.0)), rationale=s.get("rationale"))
        for name, s in zip(requested, scores_raw)
    ]

    process_penalty = 0.0
    if req.run_process_critic:
        process_penalty = float(critic.penalty(req.prompt, req.candidate))

    aggregate = sum(j.score for j in judges) / max(1, len(judges)) - process_penalty
    accepted, reason = gate.decide(judges=[j.model_dump() for j in judges],
                                    process_penalty=process_penalty)

    return ScoreResponse(
        judges=judges,
        process_penalty=process_penalty,
        aggregate=aggregate,
        accepted=accepted,
        rejection_reason=None if accepted else reason,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ready = handle._pool is not None
    loaded: list[str] = []
    if ready:
        loaded = list(getattr(handle._pool, "judge_names", []))
    return HealthResponse(
        status="ok" if ready else "degraded",
        judge_pool_ready=ready,
        judges_loaded=loaded,
        cache_backend=cache.backend_name,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    total, acc, rej = handle.counters
    return MetricsResponse(
        requests_total=total,
        accepted_total=acc,
        rejected_total=rej,
        avg_aggregate_score=handle.avg_aggregate(),
        p95_latency_ms=handle.percentile(0.95),
    )


@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest) -> ScoreResponse:
    try:
        pool, critic = handle.ensure_started()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"judge pool unavailable: {exc}")

    key = make_key(req.prompt, req.candidate, list(req.judges or []), req.run_process_critic)
    cached = cache.get(key)
    if cached is not None:
        payload = json.loads(cached)
        return ScoreResponse(**payload)

    t0 = time.perf_counter()
    try:
        resp = _run_pipeline(pool, critic, handle.gate(), req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scoring failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    handle.record(latency_ms, resp.accepted, resp.aggregate)
    cache.set(key, resp.model_dump_json())
    return resp


@app.post("/v1/score_batch", response_model=BatchScoreResponse)
async def score_batch(req: BatchScoreRequest) -> BatchScoreResponse:
    try:
        pool, critic = handle.ensure_started()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"judge pool unavailable: {exc}")

    results: list[ScoreResponse] = []
    for item in req.items:
        key = make_key(item.prompt, item.candidate, list(item.judges or []), item.run_process_critic)
        cached = cache.get(key)
        if cached is not None:
            results.append(ScoreResponse(**json.loads(cached)))
            continue

        t0 = time.perf_counter()
        resp = _run_pipeline(pool, critic, handle.gate(), item)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        handle.record(latency_ms, resp.accepted, resp.aggregate)
        cache.set(key, resp.model_dump_json())
        results.append(resp)

    accepted = [r for r in results if r.accepted]
    accept_rate = len(accepted) / max(1, len(results))

    # Simple per-judge agreement proxy: 1 - (normalised stdev across judges)
    if accepted:
        per_item_spreads: list[float] = []
        for r in accepted:
            scores = [j.score for j in r.judges]
            if len(scores) >= 2:
                mean = sum(scores) / len(scores)
                spread = max(0.0, 1.0 - (max(scores) - min(scores)) / 10.0)
                per_item_spreads.append(spread)
        agreement = sum(per_item_spreads) / len(per_item_spreads) if per_item_spreads else 0.0
    else:
        agreement = 0.0

    return BatchScoreResponse(
        results=results,
        accept_rate=accept_rate,
        inter_judge_agreement=agreement,
    )
