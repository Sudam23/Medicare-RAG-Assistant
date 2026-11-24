"""Sentence embeddings on CPU via ONNX Runtime.

``fastembed`` runs the model through ONNX Runtime rather than PyTorch, which
keeps the container small enough for a free-tier host - there is no CUDA stack
and no torch wheel, and the whole model is about 130 MB.

The asymmetry between the two methods is the part that is easy to get wrong.
``bge`` models are trained with an instruction prefix on the *query* side only;
documents are embedded bare. Applying the prefix to both, or to neither, costs
retrieval accuracy silently - nothing errors, the numbers just get worse.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from fastembed import TextEmbedding

from medicare_rag.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, query_prefix


class Embedder:
    """Wraps the embedding model and enforces the query/document asymmetry."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        self._prefix = query_prefix(model_name)

    def warm_up(self) -> None:
        """Force the lazy model load so the first real request does not pay for it."""
        self.embed_query("warm up")

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(texts, batch_size=EMBEDDING_BATCH_SIZE)))
        return self._normalise(vectors.astype(np.float32))

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """Unit-normalise so a dot product is a cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        return vectors / norms

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Embed passages. No instruction prefix - see the module docstring."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query, with the model's instruction prefix applied."""
        return self._embed([self._prefix + text])[0]
