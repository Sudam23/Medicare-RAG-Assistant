"""The orchestration graph: routing, refusal, the retry cap and generation."""

from __future__ import annotations

from medicare_rag.config import MAX_RETRIEVAL_ATTEMPTS
from medicare_rag.rag.graph import RagPipeline, rewrite_query
from medicare_rag.rag.llm import StubLLM


class TestQueryRewrite:
    def test_conversational_filler_is_stripped(self) -> None:
        # "does" survives here and is dropped later by the BM25 stoplist; the
        # rewrite only removes the conversational wrapper and the question mark.
        assert (
            rewrite_query("Can you tell me does Medicare cover dental?")
            == "does Medicare cover dental"
        )
        assert rewrite_query("Please explain the Part B penalty?") == "the Part B penalty"
        assert rewrite_query("I want to know what hospice covers?") == "what hospice covers"

    def test_a_bare_question_is_left_alone(self) -> None:
        assert rewrite_query("Part B deductible") == "Part B deductible"

    def test_rewrite_never_returns_empty(self) -> None:
        assert rewrite_query("tell me") != ""


class TestRouting:
    def test_an_answerable_question_is_answered(self, pipeline: RagPipeline) -> None:
        answer, _, trace = pipeline.run("Does Medicare cover dental care?", k=4)
        assert answer.strategy == "extractive"
        assert answer.sources
        assert [s.node for s in trace.steps if s.node == "refuse"] == []

    def test_an_off_topic_question_is_refused(self, pipeline: RagPipeline) -> None:
        answer, _, _ = pipeline.run("How do I bake sourdough bread?", k=4)
        assert answer.strategy == "refused"

    def test_personal_data_is_blocked_before_retrieval_happens(self, pipeline: RagPipeline) -> None:
        """The question is logged and traced, so it must not be retrieved on at all."""
        answer, retrieval, trace = pipeline.run("my ssn is 123-45-6789 what is covered", k=4)
        assert answer.strategy == "refused"
        assert retrieval is None
        assert answer.sources == []
        assert [s.node for s in trace.steps] == ["guard", "graph"]

    def test_the_trace_never_stores_the_identifier(self, pipeline: RagPipeline) -> None:
        _, _, trace = pipeline.run("my ssn is 123-45-6789 what is covered", k=4)
        assert "123-45-6789" not in trace.question

    def test_the_retry_loop_is_capped(self, pipeline: RagPipeline) -> None:
        """Rewriting an unanswerable question produces another unanswerable one.

        Without the cap this ran until LangGraph's own recursion limit stopped
        it after 10,007 steps.
        """
        _, _, trace = pipeline.run("How do I bake sourdough bread?", k=4)
        retrievals = [s for s in trace.steps if s.node == "retrieve"]
        assert len(retrievals) == MAX_RETRIEVAL_ATTEMPTS
        assert trace.attempts == MAX_RETRIEVAL_ATTEMPTS


class TestExtractiveAnswers:
    def test_every_sentence_carries_a_citation(self, pipeline: RagPipeline) -> None:
        answer, _, _ = pipeline.run("What is the Part B late enrollment penalty?", k=4)
        assert answer.groundedness == 1.0

    def test_the_answer_stays_short(self, pipeline: RagPipeline) -> None:
        """Quoting whole bullet lists produced a 2,000-character 'answer'."""
        answer, _, _ = pipeline.run("Does Medicare cover dental care?", k=6)
        assert len(answer.text) <= 900

    def test_quoted_text_appears_in_a_source(self, pipeline: RagPipeline) -> None:
        """Extractive answers select; they never write."""
        answer, _, _ = pipeline.run("Can I get hospice care at home?", k=4)
        joined = " ".join(" ".join(s.text.split()) for s in answer.sources)
        first = answer.text.split(" [")[0]
        assert first in joined

    def test_sources_deep_link_to_the_right_pdf(self, pipeline: RagPipeline) -> None:
        answer, _, _ = pipeline.run("What is a Special Needs Plan?", k=3)
        for source in answer.sources:
            assert source.url.startswith("https://www.medicare.gov/publications/")
            assert "#page=" in source.url


class TestGeneration:
    def test_a_verified_generated_answer_is_used(self, retriever) -> None:
        llm = StubLLM("Medicare covers this in some cases [1].")
        answer, _, _ = RagPipeline(retriever, llm=llm).run("Does Medicare cover dental care?", k=3)
        assert answer.strategy == "generated"
        assert llm.calls

    def test_an_answer_citing_a_nonexistent_source_is_discarded(self, retriever) -> None:
        """The check that separates a RAG demo from something trustworthy."""
        llm = StubLLM("Medicare definitely covers this [99].")
        answer, _, _ = RagPipeline(retriever, llm=llm).run("Does Medicare cover dental care?", k=3)
        assert answer.strategy == "extractive"
        assert any("discarded" in note for note in answer.notes)

    def test_an_uncited_generated_answer_is_discarded(self, retriever) -> None:
        llm = StubLLM("Medicare definitely covers all dental work for everyone.")
        answer, _, _ = RagPipeline(retriever, llm=llm).run("Does Medicare cover dental care?", k=3)
        assert answer.strategy == "extractive"

    def test_the_model_can_refuse(self, retriever) -> None:
        llm = StubLLM("NOT_IN_SOURCES The sources do not mention this.")
        answer, _, _ = RagPipeline(retriever, llm=llm).run("Does Medicare cover dental care?", k=3)
        assert answer.strategy == "refused"

    def test_the_prompt_contains_the_sources_and_the_question(self, retriever) -> None:
        llm = StubLLM("Answer [1].")
        RagPipeline(retriever, llm=llm).run("Does Medicare cover dental care?", k=3)
        system, user = llm.calls[0]
        assert "ONLY the numbered sources" in system
        assert "Does Medicare cover dental care?" in user
        assert "[1]" in user
