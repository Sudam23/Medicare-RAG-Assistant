"""Input and output guardrails.

A question-answering service pointed at healthcare documents has three ways to
hurt someone, and each gets its own check.

1. **It can absorb personal data it has no business holding.** People paste
   their Medicare number into anything with a text box. The question is logged
   and traced, so an identifier in it would be persisted. Detected identifiers
   are rejected before anything is stored - the check runs on the raw question
   ahead of retrieval, logging and tracing.
2. **It can be talked out of its instructions.** The retrieved text is quoted
   into a model prompt, so both the question and the documents are untrusted
   input. Known override phrasings are rejected.
3. **It can answer confidently when it does not know.** That is handled in the
   graph by refusing on weak retrieval, not here.

These are regex checks, which catch the common cases and will miss a determined
attacker. They are a first layer, not a security boundary - the real protection
is that this service has no account access, no tools and no write path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from medicare_rag.config import MAX_QUESTION_CHARS, MIN_QUESTION_CHARS


class Verdict(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass(slots=True)
class GuardrailResult:
    verdict: Verdict
    reason: str = ""
    message: str = ""

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK


# A Medicare Beneficiary Identifier is 11 characters in a fixed pattern and
# deliberately excludes the letters S, L, O, I, B and Z to avoid digit
# confusion - which is what makes it recognisable rather than just alphanumeric.
# The card prints it in 4-3-4 groups ("1EG4-TE5-MK73"), which is the form
# people copy, so the separators have to be optional rather than absent.
MBI_RE = re.compile(
    r"\b[1-9][AC-HJ-KM-NP-RT-Y][AC-HJ-KM-NP-RT-Y0-9]\d"
    r"[- ]?[AC-HJ-KM-NP-RT-Y][AC-HJ-KM-NP-RT-Y0-9]\d"
    r"[- ]?[AC-HJ-KM-NP-RT-Y]{2}\d{2}\b",
    re.IGNORECASE,
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")

PII_CHECKS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("medicare_number", MBI_RE, "what looks like a Medicare number"),
    ("ssn", SSN_RE, "what looks like a Social Security number"),
    ("payment_card", CARD_RE, "what looks like a card or account number"),
    ("email", EMAIL_RE, "an email address"),
)

INJECTION_RE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|your\s+|the\s+)?(previous|prior|above|earlier|system)"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+"
    r"|act\s+as\s+(?:a|an)\s+"
    r"|reveal\s+(your|the)\s+(instructions|prompt|rules)",
    re.IGNORECASE,
)

PII_MESSAGE = (
    "For your safety this question was not processed, because it appears to "
    "contain {what}. Never share personal identifiers with a demo service. "
    "Please ask your question without any personal details."
)


def check_question(question: str) -> GuardrailResult:
    """Validate a question before it is retrieved on, logged or traced."""
    text = (question or "").strip()

    if len(text) < MIN_QUESTION_CHARS:
        return GuardrailResult(
            Verdict.BLOCK,
            "too_short",
            "Please enter a question of at least a few characters.",
        )
    if len(text) > MAX_QUESTION_CHARS:
        return GuardrailResult(
            Verdict.BLOCK,
            "too_long",
            f"Please shorten your question to {MAX_QUESTION_CHARS} characters or fewer.",
        )
    for reason, pattern, description in PII_CHECKS:
        if pattern.search(text):
            return GuardrailResult(Verdict.BLOCK, reason, PII_MESSAGE.format(what=description))
    if INJECTION_RE.search(text):
        return GuardrailResult(
            Verdict.BLOCK,
            "prompt_injection",
            "This question looks like an attempt to change the assistant's instructions. "
            "Please ask a question about Medicare.",
        )
    return GuardrailResult(Verdict.ALLOW)


def redact(text: str) -> str:
    """Mask anything identifier-shaped before it reaches a log or a trace."""
    for _, pattern, _ in PII_CHECKS:
        text = pattern.sub("[redacted]", text)
    return text


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
CITATION_RE = re.compile(r"\[(\d+)\]")


def split_sentences(answer: str) -> list[str]:
    """Split an answer into sentences for citation checking."""
    cleaned = " ".join(answer.split())
    if not cleaned:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]


def split_units(passage: str) -> list[str]:
    """Split a source passage into the smallest quotable units.

    Sentence splitting alone is not enough here. These documents are written as
    bulleted lists, and a run of bullets contains no sentence break a regex can
    see - "..., like: * An oral exam ... * A procedure ..." is one sentence by
    punctuation and six facts by meaning. Quoting it whole produced a
    three-hundred-word "answer" that was really the entire page.

    So the passage is split on line breaks first, which is where the bullets
    already are, and only then on sentence boundaries within each line.
    """
    units: list[str] = []
    for line in passage.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        # Strip the bullet glyph; it adds nothing once the text stands alone.
        line = re.sub(r"^[\u2022\u25aa\u25e6\u2716\-\u2013\u2014]\s*", "", line)
        units.extend(s for s in SENTENCE_SPLIT_RE.split(line) if s.strip())
    return [u.strip() for u in units if u.strip()]


@dataclass(slots=True)
class CitationCheck:
    """Whether a generated answer is actually supported by its sources."""

    total_sentences: int
    cited_sentences: int
    invalid_citations: list[int]
    uncited_sentences: list[str]

    @property
    def groundedness(self) -> float:
        """Fraction of sentences carrying at least one valid citation."""
        if self.total_sentences == 0:
            return 0.0
        return self.cited_sentences / self.total_sentences

    @property
    def ok(self) -> bool:
        return not self.invalid_citations and self.groundedness >= 0.999


def check_citations(answer: str, num_sources: int) -> CitationCheck:
    """Verify every sentence cites a source, and every citation exists.

    This is the check that makes the difference between a RAG demo and
    something trustworthy. A model asked to cite will happily write "[7]" when
    six sources were supplied, or state a figure with no marker at all. Both
    are caught here, deterministically, without a second model grading the
    first one.

    Sentences that make no factual claim - a closing pointer to 1-800-MEDICARE,
    a question back to the user - are exempt, otherwise a perfectly good answer
    fails on its last line.
    """
    sentences = split_sentences(answer)
    cited = 0
    invalid: list[int] = []
    uncited: list[str] = []

    for sentence in sentences:
        markers = [int(m) for m in CITATION_RE.findall(sentence)]
        out_of_range = [m for m in markers if m < 1 or m > num_sources]
        invalid.extend(out_of_range)
        if (markers and not out_of_range) or _is_exempt(sentence):
            cited += 1
        else:
            uncited.append(sentence)

    return CitationCheck(
        total_sentences=len(sentences),
        cited_sentences=cited,
        invalid_citations=invalid,
        uncited_sentences=uncited,
    )


EXEMPT_RE = re.compile(
    r"1-800-MEDICARE|medicare\.gov|^(this|that) (answer|information)|"
    r"^(i|we) (don'?t|do not|couldn'?t|could not)|"
    r"^(for|to) (more|further|your specific)|^check with|^contact ",
    re.IGNORECASE,
)


def _is_exempt(sentence: str) -> bool:
    """Sentences that point elsewhere rather than assert a fact about coverage."""
    return bool(EXEMPT_RE.search(sentence))
