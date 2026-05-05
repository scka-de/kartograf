from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CorpusItem(BaseModel):
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_30d: list[float] | None = None
    region_id: str | None = None


class EvalCase(BaseModel):
    id: str
    source_path: str
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_30d: list[float] | None = None
    region_id: str | None = None
    assignment_distance: float | None = None


class Region(BaseModel):
    id: str
    label: str
    centroid: list[float]
    density: float
    heterogeneity: float
    member_count: int
    member_corpus_ids: list[str] = Field(default_factory=list)
    exemplar_corpus_ids: list[str] = Field(default_factory=list)
    eval_case_ids: list[str] = Field(default_factory=list)
    eval_density: float = 0.0


class Gap(BaseModel):
    region_id: str
    risk_score: float
    components: dict[str, float] = Field(default_factory=dict)
    rank: int


class GeneratedCase(BaseModel):
    id: str
    region_id: str
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)
    grounded_in_corpus_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    embedding_30d: list[float] | None = None
    validation_status: Literal["accepted", "rejected_redundant", "rejected_off_corpus"]
    validation_details: dict[str, Any] = Field(default_factory=dict)
    accepted: bool


class RedundantEval(BaseModel):
    case_id: str
    duplicate_of_case_id: str
    similarity: float


class Decision(BaseModel):
    step: int
    timestamp: datetime
    action: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    reason: str


class EvalRunResult(BaseModel):
    """Result of running `adk eval` as a subprocess. Never raises on target failure."""

    label: str
    command: list[str]
    exit_code: int
    pass_count: int | None = None
    fail_count: int | None = None
    pass_rate: float | None = None
    stdout: str
    stderr: str
    duration_seconds: float


class CoverageReport(BaseModel):
    audit_id: str
    target_agent: str
    corpus_source: str
    corpus_size: int
    noise_fraction: float
    eval_count: int
    coverage_score: float
    regions: list[Region]
    gaps: list[Gap]
    redundant_evals: list[RedundantEval] = Field(default_factory=list)
    generated_cases: list[GeneratedCase] = Field(default_factory=list)
    eval_run_before: EvalRunResult | None = None
    eval_run_after: EvalRunResult | None = None
    decisions: list[Decision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
