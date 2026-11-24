"""Turning retrieved passages into an answer, with or without a model.

Two strategies, one output shape.

**Extractive** selects the sentences from the retrieved passages that best
match the question and quotes them verbatim. It cannot hallucinate, because it
never writes a word of its own - every sentence shown was printed in a CMS
publication. What it cannot do is synthesise: asked something that needs two
passages combined, it returns both and leaves the joining to the reader.

**Generative** sends the same passages to a model with instructions to answer
in plain English and cite every claim, then verifies those citations before
the answer is allowed out. If verification fails the answer is discarded and
the extractive answer is served instead - a failed generation degrades to the
honest path rather than to an unchecked one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medicare_rag.config import DISCLAIMER
from medicare_rag.indexing.embedder import Embedder
from medicare_rag.rag.guardrails import CitationCheck, check_citations, split_units
from medicare_rag.rag.llm import LLM, Completion
from medicare_rag.rag.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from medicare_rag.rag.retrieve import RetrievalResult, RetrievedChunk

NOT_IN_SOURCES = "NOT_IN_SOURCES"
MAX_EXTRACTIVE_SENTENCES = 5
MIN_SENTENCE_CHARS = 40
# A reader wants a short answer and the full passage underneath, not the
# passage pasted twice. Selection stops at whichever limit comes first.
MAX_EXTRACTIVE_CHARS = 700
MAX_UNIT_CHARS = 320
# Two sentences sharing this fraction of their words are treated as the same
# fact restated, and only the better-scoring one is quoted.
DUPLICATE_OVERLAP = 0.6
QUESTION_PENALTY = 0.15


def _overlap(a: set[str], b: set[str]) -> float:
    """Jaccard overlap between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(slots=True)
class Source:
    """One cited source, as shown to the reader."""

    number: int
    citation: str
    doc_id: str
    doc_title: str
    page: int
    section: str
    text: str
    url: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    matched_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Answer:
    text: str
    strategy: str  # "extractive" | "generated" | "refused"
    sources: list[Source]
    groundedness: float
    confidence: str  # "high" | "medium" | "low"
    disclaimer: str = DISCLAIMER
    citation_check: CitationCheck | None = None
    completion: Completion | None = None
    notes: list[str] = field(default_factory=list)


def source_url(doc_id: str, pdf_page: int) -> str:
    """A deep link to the exact page of the published PDF.

    The anchor needs the *file* page index, not the printed number, which is
    why the parser keeps both.
    """
    from medicare_rag.config import MEDICARE_PUBLICATIONS, PUBLICATION_BASE_URL

    slug = MEDICARE_PUBLICATIONS[doc_id]["slug"]
    return f"{PUBLICATION_BASE_URL}{slug}#page={pdf_page}"


def build_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            number=index,
            citation=item.citation,
            doc_id=item.chunk.doc_id,
            doc_title=item.chunk.doc_title,
            page=item.chunk.citation_page,
            section=item.chunk.section,
            text=item.chunk.text,
            url=source_url(item.chunk.doc_id, item.chunk.pdf_page),
            score=item.dense_score if item.dense_score is not None else item.score,
            dense_rank=item.dense_rank,
            lexical_rank=item.lexical_rank,
            matched_terms=item.matched_terms,
        )
        for index, item in enumerate(chunks, start=1)
    ]


def confidence_band(top_score: float) -> str:
    if top_score >= 0.80:
        return "high"
    if top_score >= 0.70:
        return "medium"
    return "low"


def extractive_answer(
    question: str,
    retrieval: RetrievalResult,
    embedder: Embedder,
    sources: list[Source],
) -> Answer:
    """Quote the sentences from the retrieved passages that answer the question.

    Sentences are ranked by similarity to the question, then re-ordered to
    follow the sources so the result reads in document order rather than in
    score order.
    """
    candidates: list[tuple[int, int, str]] = []  # (source number, position, sentence)
    for source in sources:
        for position, sentence in enumerate(split_units(source.text)):
            if MIN_SENTENCE_CHARS <= len(sentence) <= MAX_UNIT_CHARS:
                candidates.append((source.number, position, sentence))

    if not candidates:
        return Answer(
            text="The retrieved pages did not contain any readable text for this question.",
            strategy="refused",
            sources=sources,
            groundedness=0.0,
            confidence="low",
            notes=["no_extractable_sentences"],
        )

    question_vector = embedder.embed_query(question)
    sentence_vectors = embedder.embed_documents([sentence for _, _, sentence in candidates])
    similarities = sentence_vectors @ question_vector

    # A heading is the most similar thing on the page to the question, because
    # it *is* the question - "What's a Part B late enrollment penalty?" scores
    # higher than the sentence that answers it. Penalising question-form units
    # stops the answer quoting the question back at the reader.
    for position, (_, _, sentence) in enumerate(candidates):
        if sentence.rstrip().endswith("?"):
            similarities[position] -= QUESTION_PENALTY

    # Drop near-duplicates: the same rule is printed in several publications in
    # slightly different words, and quoting each copy pads the answer without
    # adding anything. Token overlap catches what a prefix comparison misses -
    # "Medicare-approved hospice care" vs "Medicare-certified hospice care".
    picked: list[tuple[int, int, str]] = []
    fingerprints: list[set[str]] = []
    budget = 0
    for index in np.argsort(-similarities):
        number, position, sentence = candidates[int(index)]
        tokens = {
            w
            for w in "".join(c if c.isalnum() else " " for c in sentence.lower()).split()
            if len(w) > 2
        }
        if any(_overlap(tokens, seen) >= DUPLICATE_OVERLAP for seen in fingerprints):
            continue
        if budget + len(sentence) > MAX_EXTRACTIVE_CHARS and picked:
            break
        fingerprints.append(tokens)
        picked.append((number, position, sentence))
        budget += len(sentence)
        if len(picked) >= MAX_EXTRACTIVE_SENTENCES:
            break

    # Re-order to follow the sources, so the answer reads in document order
    # rather than in score order.
    selected = sorted(picked, key=lambda item: (item[0], item[1]))
    text = " ".join(f"{sentence} [{number}]" for number, _, sentence in selected)
    check = check_citations(text, len(sources))
    return Answer(
        text=text,
        strategy="extractive",
        sources=sources,
        groundedness=check.groundedness,
        confidence=confidence_band(retrieval.top_score),
        citation_check=check,
        notes=["Quoted verbatim from the sources; no text was generated."],
    )


def generative_answer(
    question: str,
    retrieval: RetrievalResult,
    llm: LLM,
    sources: list[Source],
) -> tuple[Answer | None, Completion | None, CitationCheck | None]:
    """Generate an answer and verify it before returning it.

    Returns ``(None, completion, check)`` when the model refuses or when the
    answer fails citation verification, which tells the caller to fall back
    rather than to show it.
    """
    passages = [(source.citation, source.text) for source in sources]
    completion = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, passages))
    text = completion.text.strip()

    if text.startswith(NOT_IN_SOURCES):
        detail = (
            text[len(NOT_IN_SOURCES) :].strip(" :.-") or "The sources do not cover this question."
        )
        return (
            Answer(
                text=f"I could not answer this from the Medicare publications I have. {detail}",
                strategy="refused",
                sources=sources,
                groundedness=1.0,
                confidence="low",
                completion=completion,
                notes=["The model reported that the retrieved sources do not answer the question."],
            ),
            completion,
            None,
        )

    check = check_citations(text, len(sources))
    if not check.ok:
        return None, completion, check

    return (
        Answer(
            text=text,
            strategy="generated",
            sources=sources,
            groundedness=check.groundedness,
            confidence=confidence_band(retrieval.top_score),
            citation_check=check,
            completion=completion,
            notes=[
                f"Generated with {completion.model}, prompt {PROMPT_VERSION}, citations verified."
            ],
        ),
        completion,
        check,
    )


def refusal_answer(sources: list[Source], reason: str) -> Answer:
    return Answer(
        text=(
            "I could not find this in the Medicare publications I have, "
            "and I would rather tell you that than guess. Try rewording the "
            "question, or call 1-800-MEDICARE."
        ),
        strategy="refused",
        sources=sources,
        groundedness=1.0,
        confidence="low",
        notes=[reason],
    )
