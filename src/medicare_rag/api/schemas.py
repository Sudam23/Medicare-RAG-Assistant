"""Request and response models.

Pydantic validates at the edge so the pipeline never sees a malformed value,
and the generated OpenAPI schema at ``/docs`` documents the service without a
second source of truth to keep in step.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from medicare_rag.config import DEFAULT_TOP_K, MAX_QUESTION_CHARS, MAX_TOP_K, MIN_QUESTION_CHARS


class SourceOut(BaseModel):
    number: int = Field(description="The [n] marker used in the answer text.")
    citation: str
    doc_id: str
    doc_title: str
    page: int = Field(description="The page number printed on the page, as a reader would find it.")
    section: str
    text: str
    url: str = Field(description="Deep link to that page of the published PDF.")
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    matched_terms: list[str] = []


class AnswerResponse(BaseModel):
    question: str
    answer: str
    strategy: str = Field(description="extractive | generated | refused")
    confidence: str = Field(description="high | medium | low")
    groundedness: float = Field(
        description="Fraction of answer sentences carrying a valid citation."
    )
    sources: list[SourceOut]
    notes: list[str] = []
    trace_id: str
    timings_ms: dict[str, float] = {}
    disclaimer: str


class AskRequest(BaseModel):
    question: str = Field(min_length=MIN_QUESTION_CHARS, max_length=MAX_QUESTION_CHARS)
    k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|lexical)$")
    doc_ids: list[str] | None = None


class FeedbackRequest(BaseModel):
    """Human-in-the-loop signal from the reader."""

    trace_id: str = Field(min_length=1, max_length=64)
    helpful: bool
    comment: str = Field(default="", max_length=500)


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    url: str
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    index_ready: bool
    llm_enabled: bool
    detail: str | None = None
