"""The vector store: Qdrant, running embedded.

Qdrant is normally a server. ``QdrantClient(path=...)`` runs the same engine
in-process against a directory on disk, which means the deployed service is a
single container with no second process to supervise and no network hop, while
the code is identical to what would talk to a Qdrant cluster. Swapping to a
hosted cluster is a change to one constructor argument.

Payload filters are the reason for using a real vector store rather than a
matrix of floats: "search only the Medigap guide" has to narrow the search
itself, not throw away results after the fact, or a filtered search returns
fewer hits than asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from medicare_rag.config import COLLECTION_NAME, EMBEDDING_DIM, QDRANT_PATH


@dataclass(slots=True)
class Hit:
    chunk_id: str
    score: float


class VectorStore:
    """A thin, typed wrapper over the Qdrant collection."""

    def __init__(self, path: Path = QDRANT_PATH, collection: str = COLLECTION_NAME) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.collection = collection
        self.client = QdrantClient(path=str(path))

    # -- build ------------------------------------------------------------
    def recreate(self, dim: int = EMBEDDING_DIM) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def upsert(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must be the same length")
        points = [
            models.PointStruct(id=index, vector=vector.tolist(), payload=payload)
            for index, (vector, payload) in enumerate(zip(vectors, payloads, strict=True))
        ]
        self.client.upsert(self.collection, points=points)

    # -- query ------------------------------------------------------------
    def search(
        self,
        vector: np.ndarray,
        limit: int,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        """Nearest neighbours, optionally restricted to certain publications."""
        query_filter = None
        if doc_ids:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
            )
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=query_filter,
            with_payload=["chunk_id"],
        )
        return [
            Hit(chunk_id=point.payload["chunk_id"], score=float(point.score))
            for point in response.points
        ]

    def count(self) -> int:
        return int(self.client.count(self.collection).count)

    def close(self) -> None:
        self.client.close()
