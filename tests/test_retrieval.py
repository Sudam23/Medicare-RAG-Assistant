"""Retrieval, fusion and the BM25 arm."""

from __future__ import annotations

import pytest

from medicare_rag.rag.fusion import reciprocal_rank_fusion
from medicare_rag.rag.lexical import LexicalIndex, tokenize
from medicare_rag.rag.retrieve import Retriever


class TestTokenisation:
    def test_plurals_are_stemmed(self) -> None:
        assert "deductible" in tokenize("deductibles")

    def test_dollar_amounts_survive(self) -> None:
        assert "$257" in tokenize("the $257 deductible")

    @pytest.mark.parametrize("word", ["not", "without", "with", "no", "if", "only"])
    def test_meaning_bearing_words_are_not_stopwords(self, word: str) -> None:
        """'covered' and 'not covered' must never reduce to the same query."""
        assert word in tokenize(f"is it {word} covered")

    def test_empty_question_words_are_dropped(self) -> None:
        assert "does" not in tokenize("does medicare cover this")


class TestBM25:
    def test_the_document_with_both_terms_wins(self) -> None:
        index = LexicalIndex.build(
            [
                "Medicare Part B covers doctor visits",
                "Hospice is comfort care",
                "The Part B deductible is $257",
            ]
        )
        scores, matched = index.score("Part B deductible")
        assert scores.argmax() == 2
        assert "deductible" in matched

    def test_an_unknown_term_scores_nothing(self) -> None:
        index = LexicalIndex.build(["Medicare covers doctor visits"])
        scores, matched = index.score("xylophone")
        assert matched == []
        assert scores.max() == 0.0


class TestFusion:
    def test_rank_one_in_both_arms_beats_rank_one_in_only_one(self) -> None:
        fused = reciprocal_rank_fusion(
            [("dense", [7, 1], 1.0), ("lexical", [7, 2], 1.0)], k=10, top_n=3
        )
        assert fused[0][0] == 7

    def test_agreement_beats_a_single_loud_opinion(self) -> None:
        """A doc both arms rank moderately can beat one a single arm ranks first."""
        fused = reciprocal_rank_fusion(
            [("dense", [1, 2, 3, 4, 9], 1.0), ("lexical", [5, 6, 7, 8, 9], 1.0)], k=1, top_n=10
        )
        ranks = {doc: i for i, (doc, _, _) in enumerate(fused)}
        assert ranks[9] < ranks[4]

    def test_provenance_records_each_arm_rank(self) -> None:
        fused = reciprocal_rank_fusion([("dense", [3], 1.0), ("lexical", [3], 1.0)], k=10, top_n=1)
        assert fused[0][2] == {"dense": 1, "lexical": 1}

    def test_weight_zero_removes_an_arm(self) -> None:
        fused = reciprocal_rank_fusion([("dense", [1], 1.0), ("lexical", [2], 0.0)], k=10, top_n=2)
        assert fused[0][0] == 1


class TestRetriever:
    def test_a_lay_question_finds_the_right_page(self, retriever: Retriever) -> None:
        result = retriever.retrieve("Does Medicare cover dental care?", k=5)
        assert any("dental" in c.chunk.text.lower() for c in result.chunks)

    def test_results_carry_the_evidence_for_their_rank(self, retriever: Retriever) -> None:
        result = retriever.retrieve("hospice care at home", k=3)
        top = result.chunks[0]
        assert top.dense_rank is not None or top.lexical_rank is not None
        assert top.citation.startswith(top.chunk.doc_title)

    def test_document_filter_narrows_the_search_not_the_output(self, retriever: Retriever) -> None:
        """Filtering after the cut would return fewer results than asked for."""
        result = retriever.retrieve("what is covered", k=6, doc_ids=["02154"])
        assert len(result.chunks) == 6
        assert {c.chunk.doc_id for c in result.chunks} == {"02154"}

    @pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
    def test_every_mode_returns_results(self, retriever: Retriever, mode: str) -> None:
        assert retriever.retrieve("Medicare costs", k=3, mode=mode).chunks

    def test_an_unknown_mode_is_rejected(self, retriever: Retriever) -> None:
        with pytest.raises(ValueError):
            retriever.retrieve("anything", mode="magic")

    def test_k_is_capped(self, retriever: Retriever) -> None:
        assert len(retriever.retrieve("Medicare", k=9999).chunks) <= 20

    def test_an_off_topic_question_scores_low(self, retriever: Retriever) -> None:
        """The signal the refusal threshold depends on."""
        on_topic = retriever.retrieve("Does Medicare cover dental care?", k=3)
        off_topic = retriever.retrieve("How do I bake sourdough bread?", k=3)
        assert off_topic.top_score < on_topic.top_score
