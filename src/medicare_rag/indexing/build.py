"""Build both indexes from the chunk corpus.

Produces three artifacts under ``data/index``:

``qdrant/``
    the vector collection, one point per chunk
``lexical_bm25.npz``
    the BM25 inverted index over the same chunks, in the same order
``manifest.json``
    what was built, from what, with what settings - so a served index can be
    identified rather than guessed at

The two indexes share the chunk ordering, which is what lets the fusion step
talk about "document 42" without a lookup table.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from medicare_rag.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_MANIFEST_PATH,
    LEXICAL_INDEX_PATH,
    QDRANT_PATH,
)
from medicare_rag.data.chunk import Chunk, load_chunks
from medicare_rag.indexing.embedder import Embedder
from medicare_rag.indexing.store import VectorStore
from medicare_rag.rag.lexical import LexicalIndex


def build_index(chunks: list[Chunk] | None = None) -> dict:
    chunks = chunks if chunks is not None else load_chunks()
    if not chunks:
        raise RuntimeError("no chunks to index; run `make data` first")
    started = time.perf_counter()

    print(f"[index] embedding {len(chunks):,} chunks with {EMBEDDING_MODEL}")
    embedder = Embedder()
    vectors = embedder.embed_documents(chunk.embed_text for chunk in chunks)

    # Qdrant's embedded mode owns the directory; rebuilding from scratch keeps
    # a rebuild deterministic instead of layering onto whatever was there.
    if QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
    store = VectorStore()
    store.recreate(dim=EMBEDDING_DIM)
    store.upsert(
        vectors,
        [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "printed_page": chunk.citation_page,
            }
            for chunk in chunks
        ],
    )
    indexed = store.count()
    store.close()
    print(f"[index] vector store holds {indexed:,} points")

    print("[index] building BM25 index")
    lexical = LexicalIndex.build([chunk.embed_text for chunk in chunks])
    lexical.save(LEXICAL_INDEX_PATH)

    elapsed = time.perf_counter() - started
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "num_chunks": len(chunks),
        "num_pages": len({(c.doc_id, c.pdf_page) for c in chunks}),
        "num_documents": len({c.doc_id for c in chunks}),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "vector_store": "qdrant-embedded",
        "collection": COLLECTION_NAME,
        "points_indexed": indexed,
        "chunking": {
            "target_chars": CHUNK_TARGET_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "bm25": {
            "vocabulary_terms": len(lexical.vocabulary),
            "postings": int(lexical.postings_term.size),
            "avg_doc_length": round(float(lexical.doc_lengths.mean()), 3),
        },
        "build_seconds": round(elapsed, 1),
    }
    INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[index] done in {elapsed:.1f}s -> {INDEX_MANIFEST_PATH}")
    return manifest


def load_manifest(path: Path = INDEX_MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    build_index()
