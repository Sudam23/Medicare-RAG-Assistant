"""Shared fixtures.

The retriever loads the real indexes once per session. They are small (1,130
chunks) so this costs under a second, and testing against the real corpus
catches things a synthetic fixture never would - a chunk whose text is empty,
a page whose printed number is missing, a citation that points at nothing.
"""

from __future__ import annotations

import pytest

from medicare_rag.rag.graph import RagPipeline
from medicare_rag.rag.llm import StubLLM
from medicare_rag.rag.retrieve import Retriever


@pytest.fixture(scope="session")
def retriever() -> Retriever:
    return Retriever.load()


@pytest.fixture(scope="session")
def pipeline(retriever: Retriever) -> RagPipeline:
    """The shipped configuration: no model, so answers are extractive."""
    return RagPipeline(retriever, llm=None)


@pytest.fixture()
def stub_llm() -> StubLLM:
    return StubLLM()
