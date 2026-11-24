"""Run the evaluation and write ``docs/evaluation_report.json``.

Two layers are measured separately, because they fail for different reasons:

1. **Retrieval**, for each of the three modes, over the in-scope questions.
   This is the ceiling - the pipeline cannot answer from a page it never
   retrieved.
2. **The full pipeline**, over every question including the out-of-scope ones,
   which is where refusal behaviour and grounding are measured.

Every number in the README is produced here. None is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from medicare_rag import __version__
from medicare_rag.config import EVAL_REPORT_PATH, MIN_RELEVANCE_SCORE
from medicare_rag.evaluation.goldset import GoldQuestion, load_gold, verify_labels
from medicare_rag.evaluation.metrics import QueryOutcome, summarise
from medicare_rag.indexing.build import load_manifest
from medicare_rag.rag.graph import RagPipeline
from medicare_rag.rag.llm import build_llm
from medicare_rag.rag.retrieve import Retriever

MODES = ("lexical", "dense", "hybrid")
EVAL_K = 10


def _expected(question: GoldQuestion) -> list[tuple[str, int]]:
    return [(question.doc_id or "", page) for page in (question.pages or [])]


def evaluate_retrieval(
    retriever: Retriever, questions: list[GoldQuestion], mode: str
) -> list[QueryOutcome]:
    outcomes: list[QueryOutcome] = []
    for question in questions:
        if not question.in_scope:
            continue
        started = time.perf_counter()
        result = retriever.retrieve(question.question, k=EVAL_K, mode=mode)
        elapsed = (time.perf_counter() - started) * 1000

        expected = set(_expected(question))
        retrieved = [(c.chunk.doc_id, c.chunk.pdf_page) for c in result.chunks]
        hit_rank = next((i for i, page in enumerate(retrieved, start=1) if page in expected), None)

        outcomes.append(
            QueryOutcome(
                question_id=question.id,
                question=question.question,
                in_scope=True,
                retrieved_pages=retrieved,
                expected_pages=sorted(expected),
                hit_rank=hit_rank,
                latency_ms=elapsed,
                top_score=result.top_score,
            )
        )
    return outcomes


def evaluate_pipeline(pipeline: RagPipeline, questions: list[GoldQuestion]) -> list[QueryOutcome]:
    outcomes: list[QueryOutcome] = []
    for question in questions:
        started = time.perf_counter()
        answer, retrieval, _trace = pipeline.run(question.question, k=6)
        elapsed = (time.perf_counter() - started) * 1000

        expected = set(_expected(question))
        retrieved = (
            [(c.chunk.doc_id, c.chunk.pdf_page) for c in retrieval.chunks] if retrieval else []
        )
        hit_rank = next((i for i, page in enumerate(retrieved, start=1) if page in expected), None)

        outcomes.append(
            QueryOutcome(
                question_id=question.id,
                question=question.question,
                in_scope=question.in_scope,
                retrieved_pages=retrieved,
                expected_pages=sorted(expected),
                hit_rank=hit_rank,
                refused=answer.strategy == "refused",
                strategy=answer.strategy,
                groundedness=answer.groundedness,
                cited_pages=[(s.doc_id, _pdf_page_of(s, retrieval)) for s in answer.sources],
                latency_ms=elapsed,
                top_score=retrieval.top_score if retrieval else 0.0,
            )
        )
    return outcomes


def _pdf_page_of(source, retrieval) -> int:
    """Map a shown source back to its PDF page, which is what labels use."""
    if retrieval is None:
        return -1
    for chunk in retrieval.chunks:
        if chunk.chunk.doc_title == source.doc_title and chunk.chunk.citation_page == source.page:
            return chunk.chunk.pdf_page
    return -1


def run_evaluation(split: str = "test", output: Path = EVAL_REPORT_PATH) -> dict:
    questions = load_gold(split=split)
    if not questions:
        raise RuntimeError(f"no gold questions for split {split!r}")

    failures = verify_labels(questions)
    if failures:
        raise RuntimeError(f"gold labels failed verification: {failures}")
    print(f"[eval] {len(questions)} questions in split {split!r}; all labels verified")

    retriever = Retriever.load()
    llm = build_llm()
    pipeline = RagPipeline(retriever, llm=llm)

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_version": __version__,
        "split": split,
        "index": load_manifest(),
        "llm_enabled": llm is not None,
        "llm_model": getattr(llm, "name", None),
        "min_relevance_score": MIN_RELEVANCE_SCORE,
        "retrieval": {},
        "pipeline": {},
    }

    print(f"\n[eval] retrieval over {sum(1 for q in questions if q.in_scope)} in-scope questions")
    for mode in MODES:
        outcomes = evaluate_retrieval(retriever, questions, mode)
        stats = summarise(outcomes)
        report["retrieval"][mode] = stats
        r = stats["recall_at"]
        print(
            f"  {mode:<8} recall@1={r['1']:.3f} recall@3={r['3']:.3f} recall@5={r['5']:.3f} "
            f"recall@10={r['10']:.3f} mrr@10={stats['mrr_at_10']:.3f} ({stats['median_latency_ms']:.1f} ms)"
        )

    # Which questions does each arm find that the other misses - the evidence
    # for running both rather than picking one.
    dense = {
        o.question_id
        for o in evaluate_retrieval(retriever, questions, "dense")
        if o.hit_rank and o.hit_rank <= EVAL_K
    }
    lexical = {
        o.question_id
        for o in evaluate_retrieval(retriever, questions, "lexical")
        if o.hit_rank and o.hit_rank <= EVAL_K
    }
    in_scope_ids = {q.id for q in questions if q.in_scope}
    report["complementarity_at_10"] = {
        "k": EVAL_K,
        "both": len(dense & lexical),
        "only_dense": len(dense - lexical),
        "only_lexical": len(lexical - dense),
        "neither": len(in_scope_ids - dense - lexical),
        "total": len(in_scope_ids),
    }
    c = report["complementarity_at_10"]
    print(
        f"  complementarity@{EVAL_K}: both={c['both']} only_semantic={c['only_dense']} "
        f"only_keyword={c['only_lexical']} neither={c['neither']}"
    )

    print(f"\n[eval] full pipeline over all {len(questions)} questions")
    outcomes = evaluate_pipeline(pipeline, questions)
    stats = summarise(outcomes)
    report["pipeline"] = stats
    print(
        f"  answered in-scope      : {stats['answer_rate_in_scope']:.3f}\n"
        f"  refused out-of-scope   : {stats['refusal_rate_out_of_scope']:.3f}\n"
        f"  mean groundedness      : {stats['mean_groundedness']:.3f}\n"
        f"  cited a correct page   : {stats['cited_correct_page_rate']:.3f}\n"
        f"  citation precision     : {stats['citation_precision']:.3f}\n"
        f"  median latency         : {stats['median_latency_ms']:.1f} ms"
    )

    report["examples"] = {
        "answered": [
            {"question": o.question, "rank": o.hit_rank, "strategy": o.strategy}
            for o in outcomes
            if o.in_scope and not o.refused
        ][:5],
        "missed": [
            {
                "question": o.question,
                "expected": o.expected_pages,
                "top_score": round(o.top_score, 3),
            }
            for o in outcomes
            if o.in_scope and o.hit_rank is None
        ],
        "wrongly_answered_out_of_scope": [
            {"question": o.question, "top_score": round(o.top_score, 3)}
            for o in outcomes
            if not o.in_scope and not o.refused
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[eval] report written to {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output", type=Path, default=EVAL_REPORT_PATH)
    args = parser.parse_args()
    run_evaluation(split=args.split, output=args.output)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
