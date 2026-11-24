"""The labelled question set.

Every in-scope question was written *from* a page of a publication, so the
label is correct by construction rather than by judgement, and each carries a
``must_contain`` string that is checked against that page's text - a label that
cannot be verified mechanically is not used. The first run of that check found
one wrong page out of seventy, which is the argument for having it.

Out-of-scope questions carry no page, because the correct behaviour is to
refuse. A retrieval system that is never measured on what it should *not*
answer will happily answer everything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medicare_rag.config import GOLD_PATH


@dataclass(slots=True)
class GoldQuestion:
    id: str
    question: str
    type: str  # "in_scope" | "out_of_scope"
    split: str  # "dev" | "test"
    doc_id: str | None = None
    pages: list[int] | None = None
    must_contain: list[str] | None = None

    @property
    def in_scope(self) -> bool:
        return self.type == "in_scope"


def load_gold(path: Path = GOLD_PATH, split: str | None = None) -> list[GoldQuestion]:
    questions = [
        GoldQuestion(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split:
        questions = [q for q in questions if q.split == split]
    return questions


def verify_labels(questions: list[GoldQuestion]) -> list[str]:
    """Return the ids of questions whose labelled page does not contain the key phrase."""
    from medicare_rag.data.parse import load_pages

    pages = {(page.doc_id, page.pdf_page): page for page in load_pages()}
    failures: list[str] = []
    for question in questions:
        if not question.in_scope:
            continue
        ok = False
        for page_number in question.pages or []:
            page = pages.get((question.doc_id, page_number))
            if page and any(k.lower() in page.text.lower() for k in question.must_contain or []):
                ok = True
        if not ok:
            failures.append(question.id)
    return failures
