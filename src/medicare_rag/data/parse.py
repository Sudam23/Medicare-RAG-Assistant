"""Turn the publication PDFs into clean, page-anchored text records.

The page is the unit of citation, so the parser's job is to produce, for every
printed page, text a reader would recognise plus enough metadata to point back
at the exact page of the exact document.

Two details drive the design:

* **The printed page number is not the PDF page index.** "Your Medicare
  Benefits" prints "39" on the 41st page of the file, because of unnumbered
  front matter. A citation has to show the printed number - that is what a
  reader looks for - while the deep link needs the file index. Both are kept.
* **Every page repeats its section header and page number**, which would
  otherwise be embedded in each chunk and retrieved as if it were content.
  They are lifted into metadata and stripped from the body.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from medicare_rag.config import MEDICARE_PUBLICATIONS, PAGES_PATH, RAW_DIR

# "Section 2: Find out what Medicare covers" - the running header. The extra
# whitespace class matters: "Section 2:  Items & Services" has two spaces.
SECTION_RE = re.compile(r"^Section\s+(\d+)\s*:\s*(.+?)\s*$")
PAGE_NUMBER_RE = re.compile(r"^(\d{1,3})$")
# The en and em dashes are intentional: these PDFs really do use them as
# bullet glyphs, so matching only a hyphen would miss most bulleted lists.
BULLET_RE = re.compile(r"^\s*(?:[•▪◦\-–—]|\d+\.)\s+")  # noqa: RUF001
SENTENCE_END_RE = re.compile(r"[.!?:;]\s*$")


@dataclass(slots=True)
class PageRecord:
    """One printed page of one publication."""

    doc_id: str
    doc_title: str
    pdf_page: int  # 1-based index into the PDF file, for deep links
    printed_page: int | None  # the number shown on the page, for humans
    section: str
    text: str

    @property
    def citation_page(self) -> int:
        """The number to show a reader; falls back to the file index."""
        return self.printed_page if self.printed_page is not None else self.pdf_page


def _join_wrapped_lines(lines: list[str]) -> str:
    """Rebuild paragraphs from PDF line fragments.

    A PDF stores visual lines, so a sentence arrives split across several of
    them. Joining everything blindly would run bullet points together, so a
    line break is preserved only where it is meaningful: before a bullet, and
    after a line that already ended a sentence.
    """
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        starts_new = BULLET_RE.match(stripped) or not out
        previous_closed = bool(out) and bool(SENTENCE_END_RE.search(out[-1]))
        if starts_new or previous_closed:
            out.append(stripped)
        else:
            out[-1] = f"{out[-1]} {stripped}"
    return "\n".join(out)


def parse_page(
    raw_text: str, headers: frozenset[str] | set[str] = frozenset()
) -> tuple[str, int | None, str]:
    """Split one page into (section, printed page number, body text).

    ``headers`` are this document's running headers. They have to be dropped
    here, while the page is still a list of lines - once ``_join_wrapped_lines``
    has merged the header into the sentence that followed it, no later pass can
    separate them again.
    """
    lines = [line.rstrip() for line in raw_text.splitlines()]
    section = ""
    printed_page: int | None = None
    body: list[str] = []

    # The header block is the first few lines; a section title or a bare page
    # number appearing later in the page is real content, not furniture.
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index < 4:
            match = SECTION_RE.match(stripped)
            if match and not section:
                section = f"Section {match.group(1)}: {match.group(2)}"
                continue
            number = PAGE_NUMBER_RE.match(stripped)
            if number and printed_page is None:
                printed_page = int(number.group(1))
                continue
            if stripped in headers:
                continue
        body.append(line)

    return section, printed_page, _join_wrapped_lines(body)


# A running header that appears on most pages of a document is furniture, not
# content. "Section N:" is handled explicitly above, but publications use their
# own: the hospice booklet repeats "Welcome: Medicare Hospice Benefits" on
# every page, which then leaks into every chunk and gets quoted back as though
# it were an answer. Rather than keep a list of known headers, any short line
# repeated across most of a document's pages is treated as furniture.
RUNNING_HEADER_MIN_SHARE = 0.30
RUNNING_HEADER_MAX_CHARS = 70
RUNNING_HEADER_MIN_PAGES = 4


def find_running_headers(page_texts: list[str]) -> set[str]:
    """Short lines repeated across most pages of one document."""
    if len(page_texts) < RUNNING_HEADER_MIN_PAGES:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        # Only the first few lines can be a running header.
        for line in list(dict.fromkeys(text.splitlines()[:3])):
            stripped = line.strip()
            if stripped and len(stripped) <= RUNNING_HEADER_MAX_CHARS:
                counts[stripped] = counts.get(stripped, 0) + 1
    threshold = max(RUNNING_HEADER_MIN_PAGES, int(len(page_texts) * RUNNING_HEADER_MIN_SHARE))
    return {line for line, count in counts.items() if count >= threshold}


def strip_running_headers(text: str, headers: set[str]) -> str:
    """Remove running-header lines from the top of a page."""
    lines = text.splitlines()
    while lines and lines[0].strip() in headers:
        lines.pop(0)
    return "\n".join(lines)


def parse_publication(doc_id: str, pdf_path: Path) -> list[PageRecord]:
    """Parse one publication into page records, skipping pages with no prose."""
    meta = MEDICARE_PUBLICATIONS[doc_id]
    reader = PdfReader(str(pdf_path))
    records: list[PageRecord] = []
    current_section = ""

    raw_pages = [page.extract_text() or "" for page in reader.pages]
    headers = find_running_headers(raw_pages)

    for index, raw in enumerate(raw_pages, start=1):
        # parse_page runs FIRST so the section header is captured as metadata
        # before any header stripping can delete it; whatever furniture is left
        # at the top of the body is removed afterwards.
        section, printed_page, text = parse_page(raw, headers)
        # Pages inside a section often omit the running header; carry the last
        # one forward so every chunk can name the section it came from.
        if section:
            current_section = section
        if len(text) < 80:  # covers, blank pages, pure-image pages
            continue
        records.append(
            PageRecord(
                doc_id=doc_id,
                doc_title=meta["title"],
                pdf_page=index,
                printed_page=printed_page,
                section=current_section,
                text=text,
            )
        )
    return records


def build_pages(raw_dir: Path = RAW_DIR, output: Path = PAGES_PATH) -> list[PageRecord]:
    """Parse every publication and write one gzipped JSONL record per page."""
    records: list[PageRecord] = []
    for doc_id in MEDICARE_PUBLICATIONS:
        pdf_path = raw_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing {pdf_path}; run `make data` first")
        parsed = parse_publication(doc_id, pdf_path)
        print(
            f"[parse] {doc_id} {MEDICARE_PUBLICATIONS[doc_id]['title'][:40]:<42} {len(parsed):>4} pages"
        )
        records.extend(parsed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    print(f"[parse] {len(records):,} pages -> {output}")
    return records


def load_pages(path: Path = PAGES_PATH) -> list[PageRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [PageRecord(**json.loads(line)) for line in handle]


if __name__ == "__main__":  # pragma: no cover - manual entry point
    build_pages()
