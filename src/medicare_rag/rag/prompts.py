"""The prompt, kept in one place and versioned.

The prompt is a parameter of the system, not a detail of the code that calls
the model. Giving it a version string means a change to the wording shows up in
every trace and every evaluation report, so a drop in groundedness can be
attributed to a prompt edit rather than hunted for in the retriever.

Three instructions here do the real work:

* numbered sources and a required ``[n]`` marker on every factual sentence,
  because a citation that cannot be checked mechanically is decoration;
* an explicit, named way to say "not in the sources", because a model with no
  sanctioned refusal will invent something plausible instead;
* a ban on personalised advice, because the retrieved documents describe rules
  in general and a reader's own situation is never in the context.
"""

from __future__ import annotations

from collections.abc import Sequence

PROMPT_VERSION = "2026-11-24.v1"

SYSTEM_PROMPT = """You answer questions about Medicare using ONLY the numbered sources provided.

Rules you must follow:
1. Use only facts stated in the sources. Never add knowledge from anywhere else.
2. End every sentence that states a fact with its source marker, like [1] or [2][3].
3. If the sources do not contain the answer, reply with exactly: NOT_IN_SOURCES
   followed by one sentence saying what is missing. Do not guess.
4. Never tell the reader what they personally should do, what to buy, or what
   they qualify for. Describe what the rules say, not what they mean for them.
5. Write plainly, for someone who has never heard the jargon. Expand an
   abbreviation the first time you use it. Keep the whole answer under 120 words.
6. Quote dollar amounts, percentages and deadlines exactly as the sources give
   them, and name the year the source states.

You are a demonstration over public documents, not a Medicare representative."""


def format_sources(passages: Sequence[tuple[str, str]]) -> str:
    """Render numbered sources as ``[n] citation\\n text``."""
    blocks = []
    for index, (citation, text) in enumerate(passages, start=1):
        blocks.append(f"[{index}] {citation}\n{text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, passages: Sequence[tuple[str, str]]) -> str:
    return (
        f"Sources:\n\n{format_sources(passages)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, citing each fact."
    )
