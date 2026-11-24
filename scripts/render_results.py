#!/usr/bin/env python3
"""Write the measured numbers into the README and the docs.

Every accuracy figure in this repository comes from ``docs/evaluation_report.json``,
which ``make eval`` produces by running the real pipeline over the held-out
questions. Nothing is typed in by hand, so a number in the README cannot quietly
disagree with the number the evaluation actually measured.

The README carries marked regions:

    <!-- BEGIN:pipeline-diagram --> ... <!-- END:pipeline-diagram -->
    <!-- BEGIN:retrieval-table --> ... <!-- END:retrieval-table -->
    <!-- BEGIN:pipeline-table -->  ... <!-- END:pipeline-table -->
    <!-- BEGIN:complementarity --> ... <!-- END:complementarity -->
    <!-- BEGIN:index-facts -->     ... <!-- END:index-facts -->

    python scripts/render_results.py          # rewrite them
    python scripts/render_results.py --check  # fail if they are stale (CI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "evaluation_report.json"
DIAGRAM = ROOT / "docs" / "pipeline_readme.md"
TARGETS = (ROOT / "README.md", ROOT / "docs" / "EVALUATION.md", ROOT / "docs" / "ARCHITECTURE.md")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def retrieval_table(report: dict) -> str:
    rows = report["retrieval"]
    header = (
        "| Retrieval strategy | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 | median latency |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    names = {
        "lexical": "Keyword only (BM25)",
        "dense": "Meaning only (embeddings)",
        "hybrid": "Hybrid (shipped)",
    }
    best = max(rows, key=lambda key: rows[key]["mrr_at_10"])
    body = ""
    for key in ("lexical", "dense", "hybrid"):
        metrics = rows[key]
        label = names[key] + (" **&larr; best MRR**" if key == best else "")
        body += (
            f"| {label} "
            f"| {pct(metrics['recall_at']['1'])} "
            f"| {pct(metrics['recall_at']['3'])} "
            f"| {pct(metrics['recall_at']['5'])} "
            f"| {pct(metrics['recall_at']['10'])} "
            f"| {metrics['mrr_at_10']:.3f} "
            f"| {metrics['median_latency_ms']:.1f} ms |\n"
        )
    n = rows["hybrid"]["num_questions"]
    return header + body + f"\n{n} held-out questions, scored once. Higher is better everywhere.\n"


def pipeline_table(report: dict) -> str:
    p = report["pipeline"]
    rows = [
        (
            "Answered an in-scope question",
            pct(p["answer_rate_in_scope"]),
            f"{p['num_in_scope']} questions the booklets do cover",
        ),
        (
            "Refused an out-of-scope question",
            pct(p["refusal_rate_out_of_scope"]),
            f"{p['num_out_of_scope']} questions they do not cover",
        ),
        (
            "Groundedness",
            pct(p["mean_groundedness"]),
            "every sentence traced back to a cited page",
        ),
        (
            "Cited the correct page",
            pct(p["cited_correct_page_rate"]),
            "the gold page was among the citations shown",
        ),
        ("Median response time", f"{p['median_latency_ms']:.0f} ms", "end to end, on a laptop CPU"),
        ("95th percentile", f"{p['p95_latency_ms']:.0f} ms", "the slow ones"),
    ]
    out = "| What is measured | Result | Over what |\n| --- | --- | --- |\n"
    for name, value, note in rows:
        out += f"| {name} | **{value}** | {note} |\n"
    return out


def complementarity(report: dict) -> str:
    c = report["complementarity_at_10"]
    return (
        f"| Found by | Questions (of {c['total']}) |\n| --- | --- |\n"
        f"| Both keyword and meaning search | {c['both']} |\n"
        f"| Meaning search only | {c['only_dense']} |\n"
        f"| Keyword search only | {c['only_lexical']} |\n"
        f"| Neither | {c['neither']} |\n"
    )


def system_diagram(report: dict) -> str:
    """The whole-system picture, with its figures taken from the real index.

    Drawn as Mermaid rather than ASCII art so GitHub renders it as an actual
    diagram, and generated rather than hand-drawn so the passage and term counts
    cannot drift away from the index that is actually shipped.
    """
    idx = report["index"]
    pipe = report["pipeline"]
    return f"""**Step 1 \u2014 building the index.** This runs once, on a laptop, in about two
minutes. Nothing here happens while somebody is waiting.

```mermaid
flowchart LR
    PDF["<b>9 official CMS booklets</b><br/>396 pages of PDF<br/>from medicare.gov"]
    PARSE["<b>Read every page</b><br/>recover the page number printed on it<br/>strip repeated headers,<br/>contents and index pages"]
    CHUNK["<b>Cut into passages</b><br/>~{idx["chunking"]["target_chars"]} characters,<br/>split at paragraph breaks<br/><b>{idx["num_chunks"]:,} passages</b> from {idx["num_pages"]} pages"]
    EMB["<b>Passage &rarr; meaning</b><br/>{idx["embedding_model"]}<br/>{idx["embedding_dim"]} numbers per passage"]
    LEX["<b>Passage &rarr; words</b><br/>BM25 over<br/>{idx["bm25"]["vocabulary_terms"]:,} distinct terms"]
    SENT["<b>Sentence &rarr; meaning</b><br/>every quotable sentence,<br/>embedded once here so that<br/>answering embeds nothing"]
    VDB[("<b>Meaning index</b><br/>Qdrant, on disk")]
    KDB[("<b>Word index</b><br/>BM25, on disk")]
    SDB[("<b>Sentence index</b><br/>{idx["sentence_vectors"]["count"]:,} vectors, on disk")]

    PDF --> PARSE --> CHUNK
    CHUNK --> EMB --> VDB
    CHUNK --> LEX --> KDB
    CHUNK --> SENT --> SDB

    classDef build fill:#dbe7fb,stroke:#2f5fa8,color:#1a1a1a
    classDef store fill:#ffe9a8,stroke:#9a7b25,stroke-width:2px,color:#1a1a1a
    class PDF,PARSE,CHUNK,EMB,LEX,SENT build
    class VDB,KDB,SDB store
```

**Step 2 \u2014 answering a question.** This runs every time somebody asks something,
and takes about {pipe["median_latency_ms"]:.0f} ms. It reads the indexes built in step 1 and writes nothing.

```mermaid
flowchart LR
    ASK["<b>A question</b><br/><i>\u201cDoes Medicare<br/>cover dental care?\u201d</i>"]
    API["<b>FastAPI</b><br/>/api/ask"]
    BRAIN["<b>LangGraph state machine</b><br/>check the question, search,<br/>judge what came back,<br/>answer or refuse<br/><i>(drawn in full below)</i>"]
    VDB2[("<b>Meaning index</b><br/>Qdrant")]
    KDB2[("<b>Word index</b><br/>BM25")]
    SDB2[("<b>Sentence index</b><br/>vectors from build time")]
    FUSE["<b>Merge the two lists</b><br/>weighted reciprocal<br/>rank fusion<br/><i>ranks merged, not scores</i>"]
    PICK["<b>Pick the sentences</b><br/>score each sentence of each<br/>passage against the question<br/><i>no model runs here</i>"]
    OUT["<b>Answer</b><br/>sentences quoted from<br/>the passages, each carrying<br/>its booklet and page"]

    ASK --> API --> BRAIN
    BRAIN -- "the question,<br/>as meaning" --> VDB2
    BRAIN -- "the question,<br/>as words" --> KDB2
    VDB2 -- "closest passages" --> FUSE
    KDB2 -- "best word matches" --> FUSE
    FUSE --> PICK
    SDB2 -- "sentence vectors" --> PICK
    PICK --> OUT

    classDef serve fill:#d8f0dd,stroke:#1e7e34,color:#1a1a1a
    classDef store fill:#ffe9a8,stroke:#9a7b25,stroke-width:2px,color:#1a1a1a
    class ASK,API,BRAIN,FUSE,PICK,OUT serve
    class VDB2,KDB2,SDB2 store
```"""


def index_facts(report: dict) -> str:
    idx = report["index"]
    return (
        f"- **{idx['num_documents']} official Medicare booklets**, "
        f"{idx['num_pages']} pages, split into **{idx['num_chunks']:,} searchable passages**\n"
        f"- Embedding model `{idx['embedding_model']}` ({idx['embedding_dim']} dimensions), "
        f"stored in {idx['vector_store']}\n"
        f"- Keyword index: {idx['bm25']['vocabulary_terms']:,} terms, "
        f"{idx['bm25']['postings']:,} postings\n"
        f"- Sentence index: {idx['sentence_vectors']['count']:,} vectors, so that composing "
        f"an answer runs the model zero times\n"
        f"- Whole index builds from the PDFs in {idx['build_seconds']:.0f} seconds on a laptop CPU\n"
    )


def blocks() -> dict[str, str]:
    report = json.loads(REPORT.read_text())
    return {
        "pipeline-diagram": DIAGRAM.read_text().strip(),
        "system-diagram": system_diagram(report),
        "retrieval-table": retrieval_table(report).strip(),
        "pipeline-table": pipeline_table(report).strip(),
        "complementarity": complementarity(report).strip(),
        "index-facts": index_facts(report).strip(),
    }


def apply(text: str, generated: dict[str, str]) -> tuple[str, list[str]]:
    touched: list[str] = []
    for name, body in generated.items():
        # The body is matched lazily and may be empty, so a freshly written file
        # with the two markers on consecutive lines fills in correctly.
        pattern = re.compile(
            rf"(<!-- BEGIN:{name} -->\n)(.*?)(<!-- END:{name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            continue
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(3), text)
        touched.append(name)
    return text, touched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any file is stale")
    args = parser.parse_args()

    if not REPORT.exists():
        print(f"missing {REPORT.relative_to(ROOT)} - run `make eval` first", file=sys.stderr)
        return 2

    generated = blocks()
    stale: list[str] = []
    for path in TARGETS:
        if not path.exists():
            continue
        before = path.read_text()
        after, touched = apply(before, generated)
        if not touched:
            continue
        name = path.relative_to(ROOT)
        if after == before:
            print(f"  up to date  {name}  ({', '.join(touched)})")
        elif args.check:
            stale.append(str(name))
            print(f"  STALE       {name}  ({', '.join(touched)})")
        else:
            path.write_text(after)
            print(f"  rewrote     {name}  ({', '.join(touched)})")

    if stale:
        print(
            "\nThese files disagree with docs/evaluation_report.json."
            "\nRun `make docs` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
