"""Metrics for a retrieval-augmented system.

Four things are measured, because a RAG system can fail in four different
places and a single score would hide which one broke:

``recall@k``
    did retrieval put a chunk from a correct page in the top k? Everything
    downstream is capped by this.
``refusal accuracy``
    in-scope questions must be answered and out-of-scope questions must be
    refused. Reported separately, because a system that refuses everything
    scores perfectly on one half and uselessly on the other.
``groundedness``
    what fraction of the answer's sentences carry a valid citation. Computed
    deterministically from the text, with no second model grading the first.
``citation precision``
    of the sources the answer cited, how many were from a page the label says
    actually answers the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryOutcome:
    question_id: str
    question: str
    in_scope: bool
    retrieved_pages: list[tuple[str, int]] = field(default_factory=list)
    expected_pages: list[tuple[str, int]] = field(default_factory=list)
    hit_rank: int | None = None
    refused: bool = False
    strategy: str = ""
    groundedness: float = 0.0
    cited_pages: list[tuple[str, int]] = field(default_factory=list)
    latency_ms: float = 0.0
    top_score: float = 0.0


def recall_at(outcomes: list[QueryOutcome], k: int) -> float:
    scoped = [o for o in outcomes if o.in_scope]
    if not scoped:
        return 0.0
    return sum(1 for o in scoped if o.hit_rank is not None and o.hit_rank <= k) / len(scoped)


def mrr_at(outcomes: list[QueryOutcome], k: int = 10) -> float:
    scoped = [o for o in outcomes if o.in_scope]
    if not scoped:
        return 0.0
    total = sum(1.0 / o.hit_rank for o in scoped if o.hit_rank is not None and o.hit_rank <= k)
    return total / len(scoped)


def answer_rate(outcomes: list[QueryOutcome]) -> float:
    """Of in-scope questions, the fraction that got an answer rather than a refusal."""
    scoped = [o for o in outcomes if o.in_scope]
    if not scoped:
        return 0.0
    return sum(1 for o in scoped if not o.refused) / len(scoped)


def refusal_rate(outcomes: list[QueryOutcome]) -> float:
    """Of out-of-scope questions, the fraction correctly refused."""
    scoped = [o for o in outcomes if not o.in_scope]
    if not scoped:
        return 0.0
    return sum(1 for o in scoped if o.refused) / len(scoped)


def mean_groundedness(outcomes: list[QueryOutcome]) -> float:
    answered = [o for o in outcomes if o.in_scope and not o.refused]
    if not answered:
        return 0.0
    return sum(o.groundedness for o in answered) / len(answered)


def cited_correct_page_rate(outcomes: list[QueryOutcome]) -> float:
    """Of answered in-scope questions, how many cited at least one correct page.

    This is the metric that matches what a reader does. ``citation_precision``
    below divides by *every* source shown, so returning six sources for a
    question whose label names one page caps it near 0.17 however good the
    answer is - a low number there measures the width of the source list, not
    the quality of the citation.
    """
    answered = [o for o in outcomes if o.in_scope and not o.refused]
    if not answered:
        return 0.0
    return sum(1 for o in answered if set(o.cited_pages) & set(o.expected_pages)) / len(answered)


def citation_precision(outcomes: list[QueryOutcome]) -> float:
    """Of all sources cited on in-scope questions, how many were correct pages.

    Bounded above by (labelled pages / sources shown), so read it alongside
    ``cited_correct_page_rate`` rather than on its own.
    """
    cited = correct = 0
    for outcome in outcomes:
        if not outcome.in_scope or outcome.refused:
            continue
        expected = set(outcome.expected_pages)
        for page in outcome.cited_pages:
            cited += 1
            if page in expected:
                correct += 1
    return correct / cited if cited else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def summarise(outcomes: list[QueryOutcome], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    latencies = [o.latency_ms for o in outcomes]
    return {
        "num_questions": len(outcomes),
        "num_in_scope": sum(1 for o in outcomes if o.in_scope),
        "num_out_of_scope": sum(1 for o in outcomes if not o.in_scope),
        "recall_at": {str(k): round(recall_at(outcomes, k), 4) for k in ks},
        "mrr_at_10": round(mrr_at(outcomes, 10), 4),
        "answer_rate_in_scope": round(answer_rate(outcomes), 4),
        "refusal_rate_out_of_scope": round(refusal_rate(outcomes), 4),
        "mean_groundedness": round(mean_groundedness(outcomes), 4),
        "cited_correct_page_rate": round(cited_correct_page_rate(outcomes), 4),
        "citation_precision": round(citation_precision(outcomes), 4),
        "median_latency_ms": round(median(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2)
        if latencies
        else 0.0,
    }
