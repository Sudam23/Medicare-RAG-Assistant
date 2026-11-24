"""Sentence vectors, computed once when the index is built.

An extractive answer ranks every sentence of every retrieved passage against
the question, which needs one vector per sentence. Computing those per request
costs a model forward pass per sentence - 34 of them for a typical six-source
answer.

On a laptop that is 240 ms and invisible. On the free instance this is deployed
to, which gets roughly a tenth of a core, it was the entire response time:
27 seconds per answer, of which retrieval was 271 ms. The deployed demo was a
hundred times slower than the measured one, and the measurement was not wrong -
it was taken on hardware the demo does not run on.

A sentence's vector depends on the corpus, not on the question, so it can be
computed at build time and read back at request time. The question's own vector
already exists by then, because retrieval just computed it. An answer then
costs no forward passes at all - only a matrix multiply.

The stored vectors are the *same* vectors: the build applies the identical
splitting rule and the identical length window, so this changes how long an
answer takes and nothing about what it says. ``tests/test_sentences.py`` holds
that claim to the whole gold set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from medicare_rag.rag.guardrails import split_units

#: An extractive answer quotes whole sentences. Below this a fragment carries
#: no standalone meaning ("See page 12."); above it, the "sentence" is really
#: an unsplittable table row or address block, which reads as noise when quoted.
MIN_SENTENCE_CHARS = 40
MAX_UNIT_CHARS = 320


def answerable_units(text: str) -> list[tuple[int, str]]:
    """The quotable sentences of a passage, each with its position in it.

    The position is the index among *all* units, not among the kept ones, so
    that an answer can be re-ordered back into document order after being
    selected in score order.
    """
    return [
        (position, unit)
        for position, unit in enumerate(split_units(text))
        if MIN_SENTENCE_CHARS <= len(unit) <= MAX_UNIT_CHARS
    ]


class SentenceVectors:
    """Per-chunk sentence vectors, stored flat with an offset per chunk.

    One array and one offset table, rather than a dict of arrays: the whole
    corpus is a single contiguous block that loads in one read, and a chunk's
    sentences are a slice of it.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        offsets: np.ndarray,
        chunk_ids: list[str],
    ) -> None:
        self.vectors = vectors
        self.offsets = offsets
        self._position = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

    def __len__(self) -> int:
        return len(self._position)

    @property
    def num_sentences(self) -> int:
        return int(self.vectors.shape[0])

    @classmethod
    def build(cls, chunks, embedder) -> SentenceVectors:
        texts: list[str] = []
        offsets = [0]
        chunk_ids: list[str] = []
        for chunk in chunks:
            texts.extend(unit for _, unit in answerable_units(chunk.text))
            offsets.append(len(texts))
            chunk_ids.append(chunk.chunk_id)
        vectors = embedder.embed_documents(texts) if texts else np.zeros((0, 0), dtype=np.float32)
        return cls(
            np.asarray(vectors, dtype=np.float32),
            np.asarray(offsets, dtype=np.int64),
            chunk_ids,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            offsets=self.offsets,
            chunk_ids=np.array(list(self._position), dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> SentenceVectors:
        with np.load(path, allow_pickle=True) as data:
            return cls(
                data["vectors"],
                data["offsets"],
                [str(c) for c in data["chunk_ids"]],
            )

    def get(self, chunk_id: str, expected: int) -> np.ndarray | None:
        """The vectors for one chunk's quotable sentences, or None.

        ``expected`` is how many sentences the caller found in the passage it
        is holding. A mismatch means the index was built from a different
        corpus than the one being served, so the caller is told to embed the
        sentences itself rather than quietly scoring the wrong ones.
        """
        index = self._position.get(chunk_id)
        if index is None:
            return None
        block = self.vectors[self.offsets[index] : self.offsets[index + 1]]
        return block if block.shape[0] == expected else None
