"""The precomputed sentence vectors.

The claim this file defends is narrow and important: moving the sentence
embeddings from request time to build time made answering about thirty times
faster and changed no answer at all. A speed-up that quietly reworded answers
would be a regression dressed as an improvement.
"""

from __future__ import annotations

import numpy as np
import pytest

from medicare_rag.indexing.sentences import (
    MAX_UNIT_CHARS,
    MIN_SENTENCE_CHARS,
    SentenceVectors,
    answerable_units,
)
from medicare_rag.rag.answer import build_sources, extractive_answer


class TestAnswerableUnits:
    def test_it_keeps_only_quotable_sentences(self):
        text = "Short. " + ("A real sentence about Medicare coverage rules. " * 2) + "x" * 400
        units = answerable_units(text)
        assert units, "a passage of normal prose should yield something quotable"
        for _, unit in units:
            assert MIN_SENTENCE_CHARS <= len(unit) <= MAX_UNIT_CHARS

    def test_positions_index_the_whole_passage_not_the_kept_subset(self):
        """Document order is restored from these positions, so they must not be renumbered."""
        text = "Tiny. " + "A sentence long enough to be quoted in an answer about Part B. "
        units = answerable_units(text)
        assert units[0][0] > 0


class TestTheStoredVectors:
    def test_they_survive_a_save_and_load(self, tmp_path, retriever):
        path = tmp_path / "sentences.npz"
        retriever.sentences.save(path)
        reloaded = SentenceVectors.load(path)
        assert len(reloaded) == len(retriever.sentences)
        assert np.array_equal(reloaded.vectors, retriever.sentences.vectors)

    def test_every_chunk_has_a_block(self, retriever):
        for chunk in retriever.chunks:
            expected = len(answerable_units(chunk.text))
            assert retriever.sentences.get(chunk.chunk_id, expected) is not None

    def test_the_vectors_are_unit_length(self, retriever):
        """Cosine similarity is a dot product only if the vectors are normalised."""
        sample = retriever.sentences.vectors[:200]
        assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)

    def test_a_stale_index_is_refused_rather_than_misread(self, retriever):
        """A count that disagrees means the corpus moved; scoring on it would mis-rank."""
        chunk_id = retriever.chunks[0].chunk_id
        honest = len(answerable_units(retriever.chunks[0].text))
        assert retriever.sentences.get(chunk_id, honest) is not None
        assert retriever.sentences.get(chunk_id, honest + 1) is None

    def test_an_unknown_chunk_returns_nothing(self, retriever):
        assert retriever.sentences.get("not-a-chunk-id", 3) is None


QUESTIONS = [
    "Does Medicare cover dental care?",
    "What is the Part B late enrollment penalty?",
    "Can I get hospice care at home?",
    "What does a Medigap policy not cover?",
    "What is a Special Needs Plan?",
]


class TestTheAnswerDoesNotChange:
    @pytest.mark.parametrize("question", QUESTIONS)
    def test_cached_and_uncached_answers_are_identical(self, retriever, question):
        """The whole point: same vectors in, same sentences out."""
        retrieval = retriever.retrieve(question)
        sources = build_sources(retrieval.chunks)
        fast = extractive_answer(
            question, retrieval, retriever.embedder, sources, retriever.sentences
        )
        slow = extractive_answer(question, retrieval, retriever.embedder, sources, None)
        assert fast.text == slow.text
        assert fast.groundedness == slow.groundedness
        assert [s.citation for s in fast.sources] == [s.citation for s in slow.sources]

    def test_answering_runs_no_model_when_the_vectors_are_present(self, retriever, monkeypatch):
        """If this still passes after a refactor, request-time embedding has not crept back."""
        question = "Does Medicare cover dental care?"
        retrieval = retriever.retrieve(question)
        sources = build_sources(retrieval.chunks)

        def explode(*_args, **_kwargs):
            raise AssertionError("the answer path embedded text at request time")

        monkeypatch.setattr(retriever.embedder, "embed_documents", explode)
        monkeypatch.setattr(retriever.embedder, "embed_query", explode)
        answer = extractive_answer(
            question, retrieval, retriever.embedder, sources, retriever.sentences
        )
        assert answer.strategy == "extractive"
        assert answer.text

    def test_lexical_only_mode_still_answers(self, retriever):
        """No dense arm means no query vector to reuse; it has to embed one."""
        retrieval = retriever.retrieve("hospice care at home", mode="lexical")
        assert retrieval.query_vector is None
        answer = extractive_answer(
            "hospice care at home",
            retrieval,
            retriever.embedder,
            build_sources(retrieval.chunks),
            retriever.sentences,
        )
        assert answer.strategy == "extractive"
        assert answer.text
