"""Pydantic schemas for the DMAPO judge API.

Exposes the multi-agent scoring pipeline (helpfulness, factuality, conciseness
judges plus the process critic) over HTTP so downstream training jobs or
interactive debugging can reuse the same gating logic without re-instantiating
the judge pool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JudgeName = Literal["helpfulness", "factuality", "conciseness"]


class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    candidate: str = Field(..., min_length=1)
    judges: list[JudgeName] | None = Field(
        default=None,
        description="Subset of judges to run; defaults to all three.",
    )
    run_process_critic: bool = True


class JudgeScore(BaseModel):
    judge: JudgeName
    score: float = Field(..., ge=0.0, le=10.0)
    rationale: str | None = None


class ScoreResponse(BaseModel):
    judges: list[JudgeScore]
    process_penalty: float = 0.0
    aggregate: float
    accepted: bool
    rejection_reason: str | None = None


class BatchScoreRequest(BaseModel):
    items: list[ScoreRequest] = Field(..., min_length=1, max_length=128)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    accept_rate: float
    inter_judge_agreement: float = Field(
        ..., description="Fleiss-style agreement over the accepted subset.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    judge_pool_ready: bool
    judges_loaded: list[str]
    cache_backend: str


class MetricsResponse(BaseModel):
    requests_total: int
    accepted_total: int
    rejected_total: int
    avg_aggregate_score: float
    p95_latency_ms: float
