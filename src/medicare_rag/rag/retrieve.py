"""Hybrid retrieval over the Medicare chunk corpus.

Runs a vector search and a BM25 search over the same chunks and fuses their
rankings. Both arms are kept because they fail differently: the vector arm
cannot distinguish "Part A" from "Part B", and the keyword arm cannot connect
"nursing home" to "skilled nursing facility". The evaluation reports each arm
on its own so the choice is backed by a measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medicare_rag.config import (
    CANDIDATE_POOL,
    DEFAULT_TOP_K,
    LEXICAL_INDEX_PATH,
    MAX_TOP_K,
    QDRANT_PATH,
    RRF_K,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_LEXICAL,
)
from medicare_rag.data.chunk import Chunk, load_chunks
from medicare_rag.indexing.embedder import Embedder
from medicare_rag.indexing.store import VectorStore
from medicare_rag.rag.fusion import reciprocal_rank_fusion
from medicare_rag.rag.lexical import LexicalIndex

MODES = ("hybrid", "dense", "lexical")


@dataclass(slots=True)
class RetrievedChunk:
    """One chunk, with the evidence for why it was retrieved."""

    rank: int
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """A human-readable source reference, e.g. 'Medicare & You, p. 48'."""
        return f"{self.chunk.doc_title}, p. {self.chunk.citation_page}"


@dataclass(slots=True)
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]
    top_score: float


class Retriever:
    """Loads both indexes once and serves hybrid retrieval."""

    def __init__(
        self,
        chunks: list[Chunk],
        store: VectorStore,
        lexical: LexicalIndex,
        embedder: Embedder,
    ) -> None:
        self.chunks = chunks
        self.store = store
        self.lexical = lexical
        self.embedder = embedder
        self._position_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    @classmethod
    def load(
        cls,
        qdrant_path: Path = QDRANT_PATH,
        lexical_path: Path = LEXICAL_INDEX_PATH,
        *,
        warm_up: bool = True,
    ) -> Retriever:
        chunks = load_chunks()
        store = VectorStore(path=qdrant_path)
        lexical = LexicalIndex.load(lexical_path)
        if lexical.num_docs != len(chunks):
            raise RuntimeError(
                f"index/corpus mismatch: BM25 has {lexical.num_docs} docs, corpus has {len(chunks)}. "
                "Rebuild with `make index`."
            )
        embedder = Embedder()
        if warm_up:
            embedder.warm_up()
        return cls(chunks, store, lexical, embedder)

    # -- arms -------------------------------------------------------------
    def _dense(
        self, query: str, pool: int, doc_ids: list[str] | None
    ) -> tuple[list[int], dict[int, float]]:
        vector = self.embedder.embed_query(query)
        hits = self.store.search(vector, limit=pool, doc_ids=doc_ids)
        order: list[int] = []
        scores: dict[int, float] = {}
        for hit in hits:
            position = self._position_by_id.get(hit.chunk_id)
            if position is None:
                continue
            order.append(position)
            scores[position] = hit.score
        return order, scores

    def _lexical(
        self, query: str, pool: int, allowed: np.ndarray | None
    ) -> tuple[list[int], dict[int, float]]:
        raw, _ = self.lexical.score(query)
        if allowed is not None:
            raw = np.where(allowed, raw, 0.0)
        positive = int((raw > 0).sum())
        if positive == 0:
            return [], {}
        take = min(pool, positive)
        top = np.argpartition(raw, -take)[-take:]
        top = top[np.argsort(-raw[top])]
        return [int(i) for i in top], {int(i): float(raw[i]) for i in top}

    # -- public -----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        mode: str = "hybrid",
        doc_ids: list[str] | None = None,
        candidate_pool: int = CANDIDATE_POOL,
    ) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        k = max(1, min(k, MAX_TOP_K))
        timings: dict[str, float] = {}

        # A document filter has to narrow the search itself, not trim its
        # output, or asking for 6 chunks from one publication returns 2.
        allowed = None
        if doc_ids:
            wanted = set(doc_ids)
            allowed = np.array([chunk.doc_id in wanted for chunk in self.chunks], dtype=bool)

        dense_order: list[int] = []
        dense_scores: dict[int, float] = {}
        lexical_order: list[int] = []
        lexical_scores: dict[int, float] = {}

        if mode in ("hybrid", "dense"):
            started = time.perf_counter()
            dense_order, dense_scores = self._dense(query, candidate_pool, doc_ids)
            timings["dense"] = (time.perf_counter() - started) * 1000
        if mode in ("hybrid", "lexical"):
            started = time.perf_counter()
            lexical_order, lexical_scores = self._lexical(query, candidate_pool, allowed)
            timings["lexical"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        if mode == "dense":
            fused = [
                (position, 1.0 / (RRF_K + rank), {"dense": rank})
                for rank, position in enumerate(dense_order[:k], start=1)
            ]
        elif mode == "lexical":
            fused = [
                (position, 1.0 / (RRF_K + rank), {"lexical": rank})
                for rank, position in enumerate(lexical_order[:k], start=1)
            ]
        else:
            fused = reciprocal_rank_fusion(
                [
                    ("dense", dense_order, RRF_WEIGHT_DENSE),
                    ("lexical", lexical_order, RRF_WEIGHT_LEXICAL),
                ],
                k=RRF_K,
                top_n=k,
            )
        timings["fuse"] = (time.perf_counter() - started) * 1000

        results: list[RetrievedChunk] = []
        for rank, (position, score, provenance) in enumerate(fused, start=1):
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk=self.chunks[position],
                    score=round(score, 6),
                    dense_rank=provenance.get("dense"),
                    lexical_rank=provenance.get("lexical"),
                    dense_score=round(dense_scores[position], 4)
                    if position in dense_scores
                    else None,
                    lexical_score=round(lexical_scores[position], 3)
                    if position in lexical_scores
                    else None,
                    matched_terms=self.lexical.matched_terms(query, position),
                )
            )

        timings["total"] = round(sum(timings.values()), 2)
        for key in timings:
            timings[key] = round(timings[key], 2)

        # The relevance signal used for refusal is the cosine similarity of the
        # best chunk, not the fused score: RRF scores are rank artefacts and
        # mean nothing in absolute terms, so they cannot carry a threshold.
        top_score = max((c.dense_score for c in results if c.dense_score is not None), default=0.0)

        return RetrievalResult(
            query=query, mode=mode, chunks=results, timings_ms=timings, top_score=top_score
        )
