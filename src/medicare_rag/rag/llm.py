"""The language model client, and the reason the service works without one.

The deployed demo has no API key. That is a deliberate constraint rather than
an oversight: a public endpoint wired to a metered model is an open invitation
to spend someone else's money, and a portfolio project should still answer
questions a year from now.

So generation is optional and detected at startup:

* ``ANTHROPIC_API_KEY`` present -> answers are generated and then checked for
  citation support before being shown;
* no key -> the service answers extractively instead, quoting the retrieved
  passages verbatim. That path cannot hallucinate at all, because it writes
  nothing; it only selects.

Both paths produce the same response shape with the same citations, and the
API says which one ran. ``StubLLM`` lets the generative path be tested without
a key and without a network call.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from medicare_rag.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_MODEL,
    ANSWER_TEMPERATURE,
    ANTHROPIC_API_KEY_ENV,
    LLM_ENABLED,
)


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Completion: ...


class AnthropicLLM:
    """Calls the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str = ANSWER_MODEL) -> None:
        import anthropic  # imported lazily so the package is optional at runtime

        self.name = model
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> Completion:
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=ANSWER_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(
            text=text.strip(),
            model=self.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class StubLLM:
    """A fixed-response model, for tests and for exercising the graph offline."""

    def __init__(self, response: str = "Medicare covers this in some cases [1].") -> None:
        self.name = "stub"
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(text=self.response, model="stub", latency_ms=0.0)


def build_llm() -> LLM | None:
    """Return a client if generation is both enabled and possible, else ``None``."""
    if not LLM_ENABLED:
        return None
    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV, "").strip()
    if not api_key:
        return None
    try:
        return AnthropicLLM(api_key)
    except Exception:  # pragma: no cover - a bad key must not stop the service
        return None
