"""Split pages into retrievable, citable chunks.

The chunk is both the unit that gets retrieved and the unit that gets cited,
which pins down its size from two directions. Too big and a citation sends the
reader to a page of unrelated topics; too small and a rule gets separated from
its exception - "Medicare covers X" landing in one chunk and "unless Y" in the
next is the failure mode that matters most here, because a confidently wrong
coverage answer is worse than no answer.

Splitting therefore happens on paragraph boundaries rather than at a fixed
character count, and consecutive chunks overlap so a rule split across the
boundary still appears whole in one of them.

Each chunk carries two texts:

``text``
    exactly what the reader is shown and what the answer is checked against.
``embed_text``
    the same text prefixed with its document, section and heading.

The prefix matters. In the sibling ICD-10 project, adding a parent heading to
each record lifted top-1 retrieval accuracy by 6.5 points, because a passage
that says only "You pay 20% of the Medicare-approved amount" is meaningless on
its own - the heading is what says *for what*. The same trick is applied here
and measured again in ``docs/EVALUATION.md`` rather than assumed to transfer.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from medicare_rag.config import (
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    CHUNKS_PATH,
)
from medicare_rag.data.parse import PageRecord, load_pages

# The en and em dashes are intentional: these PDFs really do use them as
# bullet glyphs, so matching only a hyphen would miss most bulleted lists.
BULLET_RE = re.compile(r"^\s*(?:[•▪◦\-–—]|\d+\.)\s+")  # noqa: RUF001


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    pdf_page: int
    printed_page: int | None
    section: str
    heading: str
    text: str
    embed_text: str

    @property
    def citation_page(self) -> int:
        return self.printed_page if self.printed_page is not None else self.pdf_page


def detect_heading(page_text: str) -> str:
    """The question or topic line a page opens with, if it has one.

    Medicare publications are written as a series of headed answers - "How does
    my other insurance work with Medicare?" - so the opening line is usually the
    single most useful retrieval signal on the page.
    """
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped or BULLET_RE.match(stripped):
            continue
        if len(stripped) > 120:
            return ""
        if stripped.endswith("?") or not stripped.endswith((".", ",", ";")):
            return stripped
        return ""
    return ""


def split_text(
    text: str, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """Split on paragraph boundaries into pieces of roughly ``target`` chars."""
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        # A single paragraph longer than the target is emitted on its own
        # rather than being cut mid-sentence.
        if size and size + len(paragraph) + 1 > target:
            chunks.append("\n".join(current))
            if overlap > 0:
                tail: list[str] = []
                kept = 0
                for previous in reversed(current):
                    if kept + len(previous) > overlap:
                        break
                    tail.insert(0, previous)
                    kept += len(previous) + 1
                current = tail
                size = kept
            else:
                current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 1

    if current:
        chunks.append("\n".join(current))

    # Fold a too-small trailing chunk back into its predecessor rather than
    # indexing a fragment that can never answer anything on its own.
    if len(chunks) > 1 and len(chunks[-1]) < CHUNK_MIN_CHARS:
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
        chunks.pop()
    return chunks


# Headings that label a layout slot rather than a topic. Prefixing a chunk with
# "Costs" adds no retrieval signal and, when the chunk already opens with that
# same word, actively dilutes the vector.
GENERIC_HEADINGS = frozenset(
    {"costs", "more information", "what it is", "note", "important", "example", "overview"}
)


def build_embed_text(chunk_text: str, doc_title: str, section: str, heading: str) -> str:
    """Prefix a chunk with the context needed to make it independently meaningful."""
    parts = [doc_title]
    if section:
        parts.append(section)
    useful = (
        heading
        and heading.strip().lower() not in GENERIC_HEADINGS
        and heading not in parts
        and not chunk_text.lstrip().startswith(heading)
    )
    if useful:
        parts.append(heading)
    return " | ".join(parts) + "\n" + chunk_text


def chunk_page(page: PageRecord) -> list[Chunk]:
    heading = detect_heading(page.text)
    pieces = split_text(page.text)
    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        if len(piece) < CHUNK_MIN_CHARS and len(pieces) == 1:
            continue  # a page with nothing but a stub line
        chunks.append(
            Chunk(
                chunk_id=f"{page.doc_id}-p{page.pdf_page}-c{index}",
                doc_id=page.doc_id,
                doc_title=page.doc_title,
                pdf_page=page.pdf_page,
                printed_page=page.printed_page,
                section=page.section,
                heading=heading,
                text=piece,
                embed_text=build_embed_text(piece, page.doc_title, page.section, heading),
            )
        )
    return chunks


def build_chunks(pages: list[PageRecord] | None = None, output: Path = CHUNKS_PATH) -> list[Chunk]:
    pages = pages if pages is not None else load_pages()
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page))

    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"[chunk] {len(chunks):,} chunks from {len(pages):,} pages -> {output}")
    return chunks


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle]


if __name__ == "__main__":  # pragma: no cover - manual entry point
    build_chunks()
