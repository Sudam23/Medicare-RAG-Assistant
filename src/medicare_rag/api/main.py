"""The HTTP service.

The expensive objects - the embedding model, the vector store, the BM25 index -
are built once during startup and held in module state. Building them per
request would add seconds to every answer; building them at import time would
break the test suite and make the container's health check lie about readiness.

One warm-up question runs at startup so the first real visitor does not pay for
the lazy model load.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from medicare_rag import __version__
from medicare_rag.api.schemas import (
    AnswerResponse,
    AskRequest,
    DocumentOut,
    FeedbackRequest,
    HealthResponse,
    SourceOut,
)
from medicare_rag.config import (
    DEFAULT_TOP_K,
    DISCLAIMER,
    EVAL_REPORT_PATH,
    MAX_TOP_K,
    MEDICARE_PUBLICATIONS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    WEB_DIR,
)
from medicare_rag.data.chunk import load_chunks
from medicare_rag.indexing.build import load_manifest
from medicare_rag.rag.answer import source_url
from medicare_rag.rag.graph import RagPipeline
from medicare_rag.rag.llm import build_llm
from medicare_rag.rag.retrieve import Retriever
from medicare_rag.rag.trace import TraceStore

logger = logging.getLogger("medicare_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

STATE: dict = {}

EXAMPLE_QUESTIONS = [
    "Does Medicare cover dental care?",
    "What is the Part B late enrollment penalty?",
    "Can I get hospice care at home?",
    "What does a Medigap policy not cover?",
    "I am turning 65 but still working, do I need to sign up?",
    "What is a Special Needs Plan?",
    "Does Medicare pay for a home health aide?",
    "What does Medigap Plan G cover?",
]
"""The chips offered in the UI.

Each one has to produce a real answer, not a refusal - a demo whose own suggested
question comes back "I could not find this" reads as broken even when the refusal
is correct behaviour. ``test_every_example_question_is_answerable`` enforces it.
The bare token "Plan G" used to be here and had to go: two characters carry too
little signal for either retriever, so it scored below the relevance floor and was
refused. That limitation is documented in docs/EVALUATION.md rather than tuned away.
"""


def warm_up(pipeline: RagPipeline) -> None:
    """Run one throwaway question so the first visitor does not pay for the load."""
    try:
        started = time.perf_counter()
        pipeline.run("what does medicare cover", k=2)
        logger.info("warm-up question took %.0f ms", (time.perf_counter() - started) * 1000)
    except Exception:  # pragma: no cover - warm-up must never block startup
        logger.exception("warm-up failed; serving anyway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        retriever = Retriever.load()
        llm = build_llm()
        pipeline = RagPipeline(retriever, llm=llm)
        warm_up(pipeline)
        STATE["pipeline"] = pipeline
        STATE["retriever"] = retriever
        STATE["traces"] = TraceStore()
        STATE["feedback"] = []
        STATE["manifest"] = load_manifest()
        STATE["chunks"] = load_chunks()
        logger.info(
            "ready: %d chunks, llm=%s",
            STATE["manifest"]["num_chunks"],
            getattr(llm, "name", "disabled (extractive answering)"),
        )
    except Exception as exc:  # pragma: no cover - surfaced through /api/health
        logger.exception("startup failed")
        STATE["error"] = str(exc)
    yield
    retriever = STATE.get("retriever")
    if retriever is not None:
        retriever.store.close()


app = FastAPI(
    title="Medicare RAG Assistant",
    version=__version__,
    description=(
        "Answers questions about Medicare using only the official CMS publications, "
        "with a citation on every claim. Demonstration project; not medical or insurance advice."
    ),
    lifespan=lifespan,
)

_requests: dict[str, deque] = defaultdict(deque)


#: Only these do real work. /api/health must stay free so a platform health
#: check can never be rate-limited into reporting the service as down.
RATE_LIMITED_PATHS = frozenset({"/api/ask", "/api/feedback"})


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """A small in-process limiter.

    One container, one counter - enough to stop a script from draining a free
    instance, and explicitly not a substitute for a real gateway in front of a
    multi-instance deployment.

    Only the endpoints that cost something are counted. Charging the cheap
    metadata calls at the same rate was a real bug: a single page load fires
    four of them, so a visitor who then clicked through the eight example chips
    could be refused service by the demo's own suggestions.
    """
    if request.url.path in RATE_LIMITED_PATHS:
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _requests[client]
        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit is {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s."
                },
            )
        window.append(now)
    return await call_next(request)


def get_pipeline() -> RagPipeline:
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail=STATE.get("error", "service is still starting"))
    return pipeline


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = "pipeline" in STATE
    return HealthResponse(
        status="ok" if ready else "starting",
        version=__version__,
        index_ready=ready,
        llm_enabled=bool(ready and STATE["pipeline"].llm is not None),
        detail=STATE.get("error"),
    )


def _answer_payload(question: str, k: int, mode: str, doc_ids: list[str] | None) -> AnswerResponse:
    pipeline = get_pipeline()
    answer, retrieval, trace = pipeline.run(question, k=k, mode=mode, doc_ids=doc_ids)
    STATE["traces"].add(trace)
    return AnswerResponse(
        question=question,
        answer=answer.text,
        strategy=answer.strategy,
        confidence=answer.confidence,
        groundedness=round(answer.groundedness, 3),
        sources=[SourceOut(**asdict(s)) for s in answer.sources],
        notes=answer.notes,
        trace_id=trace.trace_id,
        timings_ms=retrieval.timings_ms if retrieval else {},
        disclaimer=DISCLAIMER,
    )


@app.get("/api/ask", response_model=AnswerResponse)
def ask_get(
    q: str,
    k: int = DEFAULT_TOP_K,
    mode: str = "hybrid",
    docs: str | None = None,
) -> AnswerResponse:
    """Ask a question. ``docs`` is a comma-separated list of publication ids."""
    if mode not in ("hybrid", "dense", "lexical"):
        raise HTTPException(status_code=422, detail="mode must be hybrid, dense or lexical")
    k = max(1, min(k, MAX_TOP_K))
    doc_ids = [d for d in (docs or "").split(",") if d] or None
    if doc_ids:
        unknown = [d for d in doc_ids if d not in MEDICARE_PUBLICATIONS]
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown document ids: {unknown}")
    return _answer_payload(q, k, mode, doc_ids)


@app.post("/api/ask", response_model=AnswerResponse)
def ask_post(request: AskRequest) -> AnswerResponse:
    return _answer_payload(request.question, request.k, request.mode, request.doc_ids)


@app.get("/api/documents", response_model=list[DocumentOut])
def documents() -> list[DocumentOut]:
    chunks = STATE.get("chunks") or []
    out: list[DocumentOut] = []
    for doc_id, meta in MEDICARE_PUBLICATIONS.items():
        owned = [c for c in chunks if c.doc_id == doc_id]
        out.append(
            DocumentOut(
                doc_id=doc_id,
                title=meta["title"],
                url=source_url(doc_id, 1).split("#")[0],
                pages=len({c.pdf_page for c in owned}),
                chunks=len(owned),
            )
        )
    return out


@app.get("/api/examples")
def examples() -> dict:
    return {"examples": EXAMPLE_QUESTIONS}


@app.get("/api/stats")
def stats() -> dict:
    manifest = STATE.get("manifest")
    if manifest is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    pipeline = STATE.get("pipeline")
    return {
        **manifest,
        "llm_enabled": bool(pipeline and pipeline.llm is not None),
        "answer_strategy": "generated" if (pipeline and pipeline.llm) else "extractive",
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/evaluation")
def evaluation() -> dict:
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="no evaluation report; run `make eval`")
    import json

    return json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/api/traces")
def traces(limit: int = 20) -> dict:
    store: TraceStore = STATE.get("traces")
    if store is None:
        raise HTTPException(status_code=503, detail="service is still starting")
    return {"summary": store.summary(), "traces": [t.to_dict() for t in store.recent(limit)]}


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict:
    store: TraceStore = STATE.get("traces")
    if store is None:
        raise HTTPException(status_code=503, detail="service is still starting")
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="unknown trace id")
    return trace.to_dict()


@app.post("/api/feedback")
def feedback(request: FeedbackRequest) -> dict:
    """Record a reader's verdict on an answer.

    The human-in-the-loop half of the system. Kept in memory here because the
    demo has no database; the point is that every answer carries a trace id a
    human judgement can be attached to, which is what makes the quality of a
    deployed RAG system measurable after launch rather than only before it.
    """
    store: TraceStore = STATE.get("traces")
    if store is None or store.get(request.trace_id) is None:
        raise HTTPException(status_code=404, detail="unknown trace id")
    STATE["feedback"].append(
        {
            "trace_id": request.trace_id,
            "helpful": request.helpful,
            "comment": request.comment[:500],
            "at": time.time(),
        }
    )
    entries = STATE["feedback"]
    helpful = sum(1 for e in entries if e["helpful"])
    return {"recorded": True, "total": len(entries), "helpful": helpful}


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
