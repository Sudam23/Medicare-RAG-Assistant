"""Parsing and chunking."""

from __future__ import annotations

from medicare_rag.config import MEDICARE_PUBLICATIONS
from medicare_rag.data.chunk import build_embed_text, detect_heading, load_chunks, split_text
from medicare_rag.data.parse import find_running_headers, load_pages, parse_page


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
