"""Per-request traces.

Every answer records what the pipeline actually did: which graph nodes ran, how
long each took, what was retrieved, whether the citation check passed, and what
the model cost. Without that, "the answer was wrong" is unanswerable - there is
no way to tell a retrieval miss from a generation failure from a guardrail
firing.

Traces live in a bounded in-memory ring. That is the right size for a single
demo container and deliberately not a substitute for real telemetry: a
production deployment would emit these as structured logs or spans to a
collector. The shape is the same either way.

Questions are redacted before they are stored, so an identifier that slipped
past the input guardrail is still not written down.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock

from medicare_rag.config import TRACE_RETENTION
from medicare_rag.rag.guardrails import redact


@dataclass(slots=True)
class Step:
    node: str
    duration_ms: float
    detail: dict = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    trace_id: str
    question: str
    started_at: float
    steps: list[Step] = field(default_factory=list)
    total_ms: float = 0.0
    strategy: str = ""
    groundedness: float = 0.0
    confidence: str = ""
    attempts: int = 0
    llm_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["steps"] = [asdict(step) for step in self.steps]
        return data


class TraceRecorder:
    """Builds one trace and times each node."""

    def __init__(self, question: str) -> None:
        self.trace = Trace(
            trace_id=uuid.uuid4().hex[:12],
            question=redact(question),
            started_at=time.time(),
        )
        self._started = time.perf_counter()

    def step(self, node: str, duration_ms: float, **detail: object) -> None:
        self.trace.steps.append(
            Step(node=node, duration_ms=round(duration_ms, 2), detail=dict(detail))
        )

    def finish(self, **fields: object) -> Trace:
        self.trace.total_ms = round((time.perf_counter() - self._started) * 1000, 2)
        for key, value in fields.items():
            setattr(self.trace, key, value)
        return self.trace


class TraceStore:
    """A thread-safe bounded store of recent traces."""

    def __init__(self, capacity: int = TRACE_RETENTION) -> None:
        self._traces: deque[Trace] = deque(maxlen=capacity)
        self._lock = Lock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._traces.appendleft(trace)

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            return next((t for t in self._traces if t.trace_id == trace_id), None)

    def recent(self, limit: int = 20) -> list[Trace]:
        with self._lock:
            return list(self._traces)[:limit]

    def summary(self) -> dict:
        """Aggregate health numbers - the view an operator would watch."""
        with self._lock:
            traces = list(self._traces)
        if not traces:
            return {"count": 0}
        latencies = sorted(t.total_ms for t in traces)
        strategies: dict[str, int] = {}
        for trace in traces:
            strategies[trace.strategy] = strategies.get(trace.strategy, 0) + 1
        return {
            "count": len(traces),
            "median_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[max(0, int(len(latencies) * 0.95) - 1)],
            "strategies": strategies,
            "mean_groundedness": round(sum(t.groundedness for t in traces) / len(traces), 3),
            "refusal_rate": round(strategies.get("refused", 0) / len(traces), 3),
            "total_input_tokens": sum(t.input_tokens for t in traces),
            "total_output_tokens": sum(t.output_tokens for t in traces),
        }
