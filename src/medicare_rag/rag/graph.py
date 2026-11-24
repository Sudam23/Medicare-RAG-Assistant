"""The LangGraph pipeline that turns a question into a checked answer.

    guard -> retrieve -> grade -+-> answer  -> END
                   ^            |
                   |            +-> rewrite (once) --+
                   +-------------------------------- +
                                |
                                +-> refuse -> END

Why a graph rather than a function that calls four things in order: the two
decisions in the middle are the product. Whether retrieval was good enough to
answer from, and whether a weak result deserves a second attempt or an honest
refusal, are exactly the questions a RAG system is judged on, and writing them
as explicit routes makes them reviewable, testable and traceable one at a time.

**The retry is hard-capped.** ``MAX_RETRIEVAL_ATTEMPTS`` stops the
rewrite->retrieve edge from cycling. This is not hypothetical: the first
version of this graph, written with the loop and no cap, ran until LangGraph's
own recursion limit stopped it at 10,007 steps, because rewriting a question
the corpus simply cannot answer produces another question the corpus cannot
answer. With a metered model on the generate node, an uncapped loop is a bill.

The nodes stay plain functions over a TypedDict. Nothing here needs LangGraph
to be imported to be tested.
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from medicare_rag.config import (
    DEFAULT_TOP_K,
    MAX_RETRIEVAL_ATTEMPTS,
    MIN_RELEVANCE_SCORE,
)
from medicare_rag.rag.answer import (
    Answer,
    build_sources,
    extractive_answer,
    generative_answer,
    refusal_answer,
)
from medicare_rag.rag.guardrails import check_question
from medicare_rag.rag.retrieve import RetrievalResult, Retriever
from medicare_rag.rag.trace import TraceRecorder


class RagState(TypedDict, total=False):
    question: str
    current_query: str
    k: int
    mode: str
    doc_ids: list[str] | None
    attempts: int
    retrieval: RetrievalResult | None
    grade: str
    answer: Answer | None
    blocked_message: str
    steps: Annotated[list[str], operator.add]


# Filler that adds nothing to a retrieval query. Stripping it is a cheap,
# deterministic rewrite - no second model call to fix the first one.
FILLER_RE = re.compile(
    r"^\s*(please\s+)?(can|could|would|will)\s+you\s+(please\s+)?(tell\s+me|explain|say)\s+"
    r"|^\s*(i\s+(want|need|would like)\s+to\s+know\s+)"
    r"|^\s*(please\s+)?(tell\s+me|explain)\s+"
    r"|\s*\?+\s*$",
    re.IGNORECASE,
)


def rewrite_query(question: str) -> str:
    """Strip conversational filler so retrieval sees the content words."""
    rewritten = FILLER_RE.sub(" ", question).strip()
    rewritten = re.sub(r"\s+", " ", rewritten)
    return rewritten or question


class RagPipeline:
    """Owns the retriever, the optional model, and the compiled graph."""

    def __init__(self, retriever: Retriever, llm: Any | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.graph = self._build()

    # -- nodes ------------------------------------------------------------
    def _guard(self, state: RagState) -> dict:
        result = check_question(state["question"])
        if result.blocked:
            return {
                "steps": ["guard:blocked"],
                "blocked_message": result.message,
                "grade": "blocked",
                "answer": Answer(
                    text=result.message,
                    strategy="refused",
                    sources=[],
                    groundedness=1.0,
                    confidence="low",
                    notes=[f"guardrail:{result.reason}"],
                ),
            }
        return {"steps": ["guard:ok"], "current_query": state["question"].strip()}

    def _retrieve(self, state: RagState) -> dict:
        retrieval = self.retriever.retrieve(
            state["current_query"],
            k=state.get("k", DEFAULT_TOP_K),
            mode=state.get("mode", "hybrid"),
            doc_ids=state.get("doc_ids"),
        )
        return {
            "steps": [f"retrieve:{len(retrieval.chunks)}"],
            "retrieval": retrieval,
            "attempts": state.get("attempts", 0) + 1,
        }

    def _grade(self, state: RagState) -> dict:
        """Decide whether what came back is good enough to answer from.

        The threshold is on the best chunk's cosine similarity, not the fused
        RRF score. RRF scores are a function of rank alone - the top hit always
        scores about the same whether it is a perfect match or the least bad of
        40 irrelevant passages - so they carry no information about relevance
        and cannot support a threshold.
        """
        retrieval = state.get("retrieval")
        if retrieval is None or not retrieval.chunks:
            return {"steps": ["grade:empty"], "grade": "poor"}
        good = retrieval.top_score >= MIN_RELEVANCE_SCORE
        return {
            "steps": [f"grade:{'good' if good else 'poor'}:{retrieval.top_score:.3f}"],
            "grade": "good" if good else "poor",
        }

    def _rewrite(self, state: RagState) -> dict:
        rewritten = rewrite_query(state["current_query"])
        return {"steps": [f"rewrite:{rewritten[:40]}"], "current_query": rewritten}

    def _answer(self, state: RagState) -> dict:
        retrieval = state["retrieval"]
        sources = build_sources(retrieval.chunks)

        if self.llm is not None:
            answer, _completion, check = generative_answer(
                state["question"], retrieval, self.llm, sources
            )
            if answer is not None:
                return {"steps": [f"answer:{answer.strategy}"], "answer": answer}
            # Generation happened but failed verification. Fall through to the
            # extractive answer rather than show an unsupported one.
            fallback = extractive_answer(
                state["question"],
                retrieval,
                self.retriever.embedder,
                sources,
                self.retriever.sentences,
            )
            detail = "citations did not verify"
            if check is not None and check.invalid_citations:
                detail = f"cited sources that do not exist: {sorted(set(check.invalid_citations))}"
            elif check is not None and check.uncited_sentences:
                detail = f"{len(check.uncited_sentences)} sentence(s) had no citation"
            fallback.notes.append(f"A generated answer was discarded because {detail}.")
            return {"steps": ["answer:generation_rejected"], "answer": fallback}

        answer = extractive_answer(
            state["question"],
            retrieval,
            self.retriever.embedder,
            sources,
            self.retriever.sentences,
        )
        return {"steps": ["answer:extractive"], "answer": answer}

    def _refuse(self, state: RagState) -> dict:
        retrieval = state.get("retrieval")
        sources = build_sources(retrieval.chunks[:3]) if retrieval and retrieval.chunks else []
        score = retrieval.top_score if retrieval else 0.0
        return {
            "steps": ["refuse"],
            "answer": refusal_answer(
                sources,
                f"Best retrieved passage scored {score:.3f}, below the {MIN_RELEVANCE_SCORE} "
                "threshold, after exhausting all retrieval attempts.",
            ),
        }

    # -- routing ----------------------------------------------------------
    @staticmethod
    def _route_after_guard(state: RagState) -> str:
        return "blocked" if state.get("grade") == "blocked" else "retrieve"

    @staticmethod
    def _route_after_grade(state: RagState) -> str:
        if state.get("grade") == "good":
            return "answer"
        if state.get("attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
            return "refuse"  # the hard cap that stops the loop
        return "rewrite"

    def _build(self):
        graph = StateGraph(RagState)
        graph.add_node("guard", self._guard)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("answer", self._answer)
        graph.add_node("refuse", self._refuse)

        graph.add_edge(START, "guard")
        graph.add_conditional_edges(
            "guard", self._route_after_guard, {"blocked": END, "retrieve": "retrieve"}
        )
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"answer": "answer", "rewrite": "rewrite", "refuse": "refuse"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("answer", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    # -- public -----------------------------------------------------------
    def run(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        mode: str = "hybrid",
        doc_ids: list[str] | None = None,
    ) -> tuple[Answer, RetrievalResult | None, Any]:
        recorder = TraceRecorder(question)
        node_started = time.perf_counter()

        final: RagState = self.graph.invoke(
            {
                "question": question,
                "current_query": question,
                "k": k,
                "mode": mode,
                "doc_ids": doc_ids,
                "attempts": 0,
                "steps": [],
            }
        )

        for step in final.get("steps", []):
            name, _, detail = step.partition(":")
            recorder.step(name, 0.0, detail=detail)
        recorder.step("graph", (time.perf_counter() - node_started) * 1000)

        answer: Answer = final["answer"]
        retrieval = final.get("retrieval")
        completion = answer.completion
        trace = recorder.finish(
            strategy=answer.strategy,
            groundedness=round(answer.groundedness, 3),
            confidence=answer.confidence,
            attempts=final.get("attempts", 0),
            llm_model=completion.model if completion else None,
            input_tokens=completion.input_tokens if completion else 0,
            output_tokens=completion.output_tokens if completion else 0,
        )
        return answer, retrieval, trace

    def mermaid(self) -> str:
        """Export the compiled graph as Mermaid.

        The README's pipeline diagram is generated from this, so the drawing
        cannot drift away from the code it claims to describe.
        """
        return self.graph.get_graph().draw_mermaid()
