"""BM25 over a NumPy inverted index.

Dense retrieval is good at meaning and bad at tokens. Medicare questions are
full of tokens that have to match exactly - "Part B", "Plan G", "$257",
"SEP", "IRMAA" - and an embedding happily places "Part A" and "Part B" close
together because they are nearly the same sentence. Keeping a keyword arm
alongside the vector arm is what stops that, and the evaluation reports both
arms separately so the claim can be checked rather than believed.

The index is a handful of NumPy arrays rather than a search engine: the corpus
is ~1,100 chunks, so an inverted index in memory is both simpler and faster
than running Elasticsearch next to the service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medicare_rag.config import BM25_B, BM25_K1, LEXICAL_INDEX_PATH

TOKEN_RE = re.compile(r"[a-z0-9$]+")

# Deliberately small. "Not", "with" and "without" carry real meaning in
# coverage rules - "covered" and "not covered" must not become the same
# query - so only genuinely empty words are dropped.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "at",
        "by",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "from",
        "my",
        "your",
        "you",
        "i",
        "we",
        "they",
        "he",
        "she",
        "do",
        "does",
        "did",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "how",
        "am",
    ]
)
# Words NOT in that list, on purpose: not, no, with, without, if, can, will,
# only, all, any, more, less. In a coverage rule those words are the rule -
# "covered" and "not covered" must never reduce to the same query.


def tokenize(text: str) -> list[str]:
    """Lowercase, keep digits and dollar signs, strip simple plurals."""
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        # s-stemmer: "deductibles" -> "deductible", but never "plans" -> "plan"
        # for two-letter words, and never touch something like "ss".
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith(("ss", "us", "is")):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


@dataclass(slots=True)
class LexicalIndex:
    """An inverted index with BM25 scoring."""

    vocabulary: dict[str, int]
    postings_term: np.ndarray  # term id per posting
    postings_doc: np.ndarray  # document id per posting
    postings_tf: np.ndarray  # term frequency per posting
    doc_lengths: np.ndarray
    idf: np.ndarray
    num_docs: int

    @classmethod
    def build(cls, documents: list[str]) -> LexicalIndex:
        vocabulary: dict[str, int] = {}
        terms: list[int] = []
        docs: list[int] = []
        freqs: list[int] = []
        lengths = np.zeros(len(documents), dtype=np.float32)

        for doc_id, text in enumerate(documents):
            tokens = tokenize(text)
            lengths[doc_id] = len(tokens)
            counts: dict[int, int] = {}
            for token in tokens:
                term_id = vocabulary.setdefault(token, len(vocabulary))
                counts[term_id] = counts.get(term_id, 0) + 1
            for term_id, count in counts.items():
                terms.append(term_id)
                docs.append(doc_id)
                freqs.append(count)

        postings_term = np.asarray(terms, dtype=np.int32)
        document_frequency = np.bincount(postings_term, minlength=len(vocabulary)).astype(
            np.float32
        )
        n = len(documents)
        # BM25's probabilistic idf, with the +1 that keeps it non-negative.
        idf = np.log(1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5)).astype(
            np.float32
        )

        return cls(
            vocabulary=vocabulary,
            postings_term=postings_term,
            postings_doc=np.asarray(docs, dtype=np.int32),
            postings_tf=np.asarray(freqs, dtype=np.float32),
            doc_lengths=lengths,
            idf=idf,
            num_docs=n,
        )

    # -- scoring ----------------------------------------------------------
    def score(self, query: str) -> tuple[np.ndarray, list[str]]:
        """Return a BM25 score per document and the query terms that matched."""
        scores = np.zeros(self.num_docs, dtype=np.float32)
        matched: list[str] = []
        average_length = float(self.doc_lengths.mean()) or 1.0

        for token in dict.fromkeys(tokenize(query)):
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            matched.append(token)
            selection = self.postings_term == term_id
            doc_ids = self.postings_doc[selection]
            tf = self.postings_tf[selection]
            lengths = self.doc_lengths[doc_ids]
            denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * lengths / average_length)
            scores[doc_ids] += self.idf[term_id] * (tf * (BM25_K1 + 1.0)) / denominator
        return scores, matched

    def matched_terms(self, query: str, doc_id: int) -> list[str]:
        """Which query terms actually occur in one document - shown as evidence."""
        present: list[str] = []
        for token in dict.fromkeys(tokenize(query)):
            term_id = self.vocabulary.get(token)
            if term_id is None:
                continue
            selection = (self.postings_term == term_id) & (self.postings_doc == doc_id)
            if bool(selection.any()):
                present.append(token)
        return present

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = LEXICAL_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vocabulary_terms=np.asarray(list(self.vocabulary.keys()), dtype=object),
            vocabulary_ids=np.asarray(list(self.vocabulary.values()), dtype=np.int32),
            postings_term=self.postings_term,
            postings_doc=self.postings_doc,
            postings_tf=self.postings_tf,
            doc_lengths=self.doc_lengths,
            idf=self.idf,
            num_docs=np.asarray([self.num_docs], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path = LEXICAL_INDEX_PATH) -> LexicalIndex:
        data = np.load(path, allow_pickle=True)
        vocabulary = {
            str(term): int(term_id)
            for term, term_id in zip(data["vocabulary_terms"], data["vocabulary_ids"], strict=True)
        }
        return cls(
            vocabulary=vocabulary,
            postings_term=data["postings_term"],
            postings_doc=data["postings_doc"],
            postings_tf=data["postings_tf"],
            doc_lengths=data["doc_lengths"],
            idf=data["idf"],
            num_docs=int(data["num_docs"][0]),
        )
