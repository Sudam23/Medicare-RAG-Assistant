"""Weighted Reciprocal Rank Fusion.

The two retrieval arms return scores that cannot be compared. A cosine
similarity is bounded in [-1, 1]; a BM25 score is unbounded and routinely
reaches 20. Adding them lets BM25 decide every ranking by accident, and
normalising them per query makes the scale depend on whichever documents
happened to match.

RRF sidesteps the problem by discarding the scores and keeping only the
positions: a document ranked ``r`` by an arm contributes ``weight / (k + r)``.
A document both arms rank moderately well can then beat one that a single arm
ranks first, which is the behaviour worth having - agreement between two
independent methods is stronger evidence than one loud opinion.

Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion outperforms Condorcet
and individual rank learning methods".
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[int], float]],
    k: int,
    top_n: int,
) -> list[tuple[int, float, dict[str, int]]]:
    """Fuse several ranked lists of document ids.

    Each entry is ``(arm_name, ordered_document_ids, weight)``. Returns
    ``(document_id, fused_score, {arm_name: rank})`` ordered best first, where
    ranks are 1-based and an arm that did not retrieve a document is absent.
    """
    scores: dict[int, float] = {}
    provenance: dict[int, dict[str, int]] = {}

    for arm, documents, weight in ranked_lists:
        for position, document_id in enumerate(documents, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + weight / (k + position)
            provenance.setdefault(document_id, {})[arm] = position

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(document_id, score, provenance[document_id]) for document_id, score in ordered[:top_n]]
