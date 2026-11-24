"""Central configuration.

Every tunable lives here so the ingestion pipeline, the retrieval service, the
API and the evaluation harness cannot drift apart. Paths resolve relative to
the repository root, which makes the project behave identically in a local
checkout and inside the Docker image.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = Path(os.environ.get("MEDICARE_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
WEB_DIR = Path(os.environ.get("MEDICARE_WEB_DIR", REPO_ROOT / "web"))

PAGES_PATH = PROCESSED_DIR / "pages.jsonl.gz"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl.gz"
CORPUS_STATS_PATH = PROCESSED_DIR / "corpus_stats.json"
QDRANT_PATH = INDEX_DIR / "qdrant"
LEXICAL_INDEX_PATH = INDEX_DIR / "lexical_bm25.npz"
SENTENCE_VECTORS_PATH = INDEX_DIR / "sentence_vectors.npz"
INDEX_MANIFEST_PATH = INDEX_DIR / "manifest.json"

GOLD_PATH = DATA_DIR / "gold" / "questions.jsonl"
EVAL_REPORT_PATH = REPO_ROOT / "docs" / "evaluation_report.json"

COLLECTION_NAME = "medicare_chunks"


# --------------------------------------------------------------------------
# Source documents
# --------------------------------------------------------------------------
# Every file is an official publication of the U.S. Centers for Medicare &
# Medicaid Services, downloaded from medicare.gov. Works of the U.S. federal
# government are in the public domain. The pairing of product number to slug
# was verified by fetching each URL; medicare.gov has no publications API.
MEDICARE_PUBLICATIONS: dict[str, dict[str, str]] = {
    "10050": {
        "slug": "10050-medicare-and-you.pdf",
        "title": "Medicare & You",
        "note": "The official annual handbook; the broadest source.",
    },
    "10116": {
        "slug": "10116-Your-Medicare-Benefits.pdf",
        "title": "Your Medicare Benefits",
        "note": "Item-by-item reference for what Part A and Part B cover.",
    },
    "02110": {
        "slug": "02110-medigap-guide-health-insurance.pdf",
        "title": "Choosing a Medigap Policy",
        "note": "Supplemental insurance, including the standardised plan letters.",
    },
    "10969": {
        "slug": "10969-medicare-and-home-health-care.pdf",
        "title": "Medicare and Home Health Care",
        "note": "",
    },
    "11036": {
        "slug": "11036-enrolling-medicare-part-a-part-b.pdf",
        "title": "Enrolling in Medicare Part A and Part B",
        "note": "Enrolment periods and late-enrolment penalties.",
    },
    "12026": {
        "slug": "12026-understanding-medicare-advantage-plans.pdf",
        "title": "Understanding Medicare Advantage Plans",
        "note": "",
    },
    "02154": {
        "slug": "02154-medicare-hospice-benefits.pdf",
        "title": "Medicare Hospice Benefits",
        "note": "",
    },
    "11389": {
        "slug": "11389-Medicare-Getting-Started.pdf",
        "title": "Medicare: Learn about Medicare basics",
        "note": "",
    },
    "11579": {
        "slug": "11579-medicare-costs.pdf",
        "title": "Medicare Costs",
        "note": "The current year's premium, deductible and coinsurance figures.",
    },
}

PUBLICATION_BASE_URL = "https://www.medicare.gov/publications/"
USER_AGENT = "medicare-rag-assistant/1.0 (open-source portfolio project)"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# A chunk is the unit that gets retrieved AND the unit that gets cited, so its
# size is a trade-off between the two. Too large and a citation points at a
# page of mixed topics; too small and a rule gets split from its exception.
CHUNK_TARGET_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
CHUNK_MIN_CHARS = 200


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("MEDICARE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256

QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def query_prefix(model_name: str = EMBEDDING_MODEL) -> str:
    """Return the instruction prefix a model expects on the query side only."""
    return QUERY_PREFIXES.get(model_name, "")


DEFAULT_TOP_K = 6
MAX_TOP_K = 20
CANDIDATE_POOL = 40

# Chosen by a grid search on the dev split only (scripts/tune_fusion.py,
# docs/tuning_report.json). The lexical weight is deliberately low: on this
# corpus the semantic arm does nearly all the work, and raising the keyword
# weight to 0.6 - enough to fix plan-letter queries like "Plan G" - costs five
# other questions to rescue two. See docs/EVALUATION.md.
RRF_K = int(os.environ.get("MEDICARE_RRF_K", "20"))
RRF_WEIGHT_DENSE = float(os.environ.get("MEDICARE_RRF_WEIGHT_DENSE", "1.0"))
RRF_WEIGHT_LEXICAL = float(os.environ.get("MEDICARE_RRF_WEIGHT_LEXICAL", "0.1"))

BM25_K1 = 1.5
BM25_B = 0.75


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------
# The graph refuses rather than guessing when the best retrieved chunk scores
# below this. Tuned on the dev split; see docs/EVALUATION.md.
MIN_RELEVANCE_SCORE = float(os.environ.get("MEDICARE_MIN_RELEVANCE", "0.62"))

# A question that survives one rewrite and still retrieves nothing relevant is
# refused. Without this cap the self-correction loop runs forever and, with a
# paid model behind it, bills forever.
MAX_RETRIEVAL_ATTEMPTS = int(os.environ.get("MEDICARE_MAX_ATTEMPTS", "2"))

ANSWER_MODEL = os.environ.get("MEDICARE_ANSWER_MODEL", "claude-haiku-5-5")
ANSWER_MAX_TOKENS = 700
ANSWER_TEMPERATURE = 0.0

# Set MEDICARE_LLM=off to force extractive answering even when a key is present.
LLM_ENABLED = os.environ.get("MEDICARE_LLM", "auto").lower() != "off"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


# --------------------------------------------------------------------------
# Guardrails and limits
# --------------------------------------------------------------------------
MAX_QUESTION_CHARS = 500
MIN_QUESTION_CHARS = 3
RATE_LIMIT_REQUESTS = int(os.environ.get("MEDICARE_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("MEDICARE_RATE_LIMIT_WINDOW", "60"))
TRACE_RETENTION = 200

DISCLAIMER = (
    "Demonstration project only. Answers are drawn from public Medicare "
    "publications and may be out of date or incomplete. This is not medical, "
    "legal or insurance advice. For decisions about your own coverage, call "
    "1-800-MEDICARE or visit medicare.gov."
)
