"""Parsing and chunking."""

from __future__ import annotations

from medicare_rag.config import MEDICARE_PUBLICATIONS
from medicare_rag.data.chunk import (
    NAVIGATION_PAGE_RE,
    build_embed_text,
    chunk_page,
    detect_heading,
    is_navigation_page,
    load_chunks,
    split_text,
)
from medicare_rag.data.parse import PageRecord, find_running_headers, load_pages, parse_page


class TestPageParsing:
    def test_section_and_page_number_are_lifted_out_of_the_body(self) -> None:
        section, printed, text = parse_page(
            "Section 2: Find out what Medicare covers\n46\nHIV screenings are covered."
        )
        assert section == "Section 2: Find out what Medicare covers"
        assert printed == 46
        assert "Section 2" not in text
        assert text.startswith("HIV screenings")

    def test_a_page_number_later_in_the_page_is_content_not_furniture(self) -> None:
        _, printed, text = parse_page("Some real prose here that runs on.\n\n2026\nMore prose.")
        assert printed is None
        assert "2026" in text

    def test_wrapped_lines_are_rejoined_into_sentences(self) -> None:
        _, _, text = parse_page("A sentence that was split\nacross two PDF lines.")
        assert "split across two" in text

    def test_bullets_keep_their_own_lines(self) -> None:
        _, _, text = parse_page("Medicare covers:\n• One thing.\n• Another thing.")
        assert text.count("\n") >= 2

    def test_running_headers_are_detected(self) -> None:
        pages = ["Welcome: Hospice\nReal content here."] * 10
        assert "Welcome: Hospice" in find_running_headers(pages)

    def test_a_line_on_one_page_only_is_not_a_running_header(self) -> None:
        pages = ["Unique line\nbody"] + ["Other\nbody"] * 10
        assert "Unique line" not in find_running_headers(pages)

    def test_running_header_is_stripped_but_the_section_survives(self) -> None:
        section, _, text = parse_page(
            "Section 2: Items & Services\nWelcome: Booklet\nReal content.",
            headers={"Welcome: Booklet"},
        )
        assert section == "Section 2: Items & Services"
        assert "Welcome: Booklet" not in text


class TestChunking:
    def test_split_respects_the_target_size(self) -> None:
        text = "\n".join(
            f"Paragraph number {i} with enough words to matter here." for i in range(40)
        )
        chunks = split_text(text, target=400, overlap=0)
        assert len(chunks) > 1
        assert all(len(c) < 700 for c in chunks)

    def test_consecutive_chunks_overlap(self) -> None:
        text = "\n".join(
            f"Sentence {i} about Medicare coverage rules and costs." for i in range(30)
        )
        chunks = split_text(text, target=300, overlap=120)
        assert len(chunks) > 2
        assert any(line in chunks[1] for line in chunks[0].split("\n"))

    def test_heading_detection_picks_the_question_line(self) -> None:
        assert (
            detect_heading("How does my other insurance work?\nIt depends.")
            == "How does my other insurance work?"
        )

    def test_generic_headings_are_not_used_as_context(self) -> None:
        """Prefixing a chunk with 'Costs' adds no retrieval signal."""
        embedded = build_embed_text("You pay 20%.", "Your Medicare Benefits", "Section 2", "Costs")
        assert "Costs" not in embedded.splitlines()[0]

    def test_a_real_heading_is_used_as_context(self) -> None:
        embedded = build_embed_text(
            "It depends.", "Medicare & You", "Section 1", "How does my insurance work?"
        )
        assert "How does my insurance work?" in embedded.splitlines()[0]


class TestCorpus:
    def test_every_chunk_can_be_cited(self) -> None:
        for chunk in load_chunks():
            assert chunk.doc_id in MEDICARE_PUBLICATIONS
            assert chunk.citation_page >= 1
            assert chunk.text.strip()

    def test_printed_page_differs_from_pdf_index_for_many_pages(self) -> None:
        """Front matter shifts the numbering; citing the file index would be wrong."""
        pages = load_pages()
        mismatched = sum(
            1 for p in pages if p.printed_page is not None and p.printed_page != p.pdf_page
        )
        assert mismatched > 100

    def test_no_chunk_leaks_a_running_header(self) -> None:
        assert not [c for c in load_chunks() if c.text.startswith("Welcome: Medicare Hospice")]


class TestNavigationPages:
    """Contents listings and back-of-book indexes must never become chunks.

    They are the densest keyword pages in the whole corpus, so they rank well
    and then get quoted as answers. "What is Part A?" once returned the line
    "5 Medicare has different parts 5 Medicare Part A (Hospital Insurance) 6",
    read straight off a contents page.
    """

    def test_a_contents_page_produces_no_chunks(self) -> None:
        page = PageRecord(
            doc_id="11036",
            doc_title="Enrolling in Medicare Part A and Part B",
            pdf_page=3,
            printed_page=None,
            section="",
            text=(
                "Contents 5 Section 1-The Medicare Program 5 What's Medicare?\n"
                "5 Medicare has different parts 5 Medicare Part A (Hospital Insurance) "
                "6 Medicare Part B (Medical Insurance) 9 Can I get Part A and Part B?"
            ),
        )
        assert is_navigation_page(page)
        assert chunk_page(page) == []

    def test_an_index_page_produces_no_chunks(self) -> None:
        page = PageRecord(
            doc_id="10050",
            doc_title="Medicare & You",
            pdf_page=4,
            printed_page=None,
            section="",
            text=(
                "Index of topics Note: The bold page numbers have the most details. "
                "Assignment 11, 32-48, 51, 53, 57-58, 62, 63-64, 69, 119 B Balance exams 45"
            ),
        )
        assert is_navigation_page(page)
        assert chunk_page(page) == []

    def test_real_content_that_looks_numeric_is_kept(self) -> None:
        """The rejected digit-density heuristic flagged all three of these."""
        # Long enough to clear CHUNK_MIN_CHARS, so that an empty result would
        # mean the navigation filter rejected them and nothing else.
        for text in (
            "Medigap plans may cover: Medigap plans A B C D F* G* K L M N "
            "Medicare Part A coinsurance and hospital costs up to an additional "
            "365 days after Medicare benefits are used up 100% 100% 100% 100% "
            "100% 100% 100% 100% 100% 100% Medicare Part B coinsurance or "
            "copayment 100% 100% 100% 100% 50% 75% 100% 100%",
            "If your yearly income in 2024 was: File individual tax return "
            "$106,000 or less $212,000 or less you pay $185.00 each month. "
            "Above $106,000 up to $133,000 above $212,000 up to $266,000 you "
            "pay $259.00 each month. Above $133,000 up to $167,000 you pay "
            "$370.00 each month in 2026 for your Part B premium.",
            "Alabama State Health Insurance Assistance Program (SHIP) "
            "1-800-243-5463 Alaska Medicare Information Office 1-800-478-6065 "
            "Arizona State Health Insurance Assistance Program 1-800-432-4040 "
            "Arkansas Senior Health Insurance Information Program 1-800-224-6330 "
            "California Health Insurance Counseling Program 1-800-434-0222",
        ):
            page = PageRecord(
                doc_id="10050",
                doc_title="Medicare & You",
                pdf_page=80,
                printed_page=80,
                section="",
                text=text,
            )
            assert not is_navigation_page(page), text[:40]
            assert chunk_page(page), text[:40]

    def test_the_built_index_contains_no_contents_pages(self) -> None:
        """The shipped index, not a synthetic page."""
        leaked = [c for c in load_chunks() if NAVIGATION_PAGE_RE.match(c.text.strip())]
        assert not leaked, [f"{c.doc_id} p{c.pdf_page}" for c in leaked[:5]]
