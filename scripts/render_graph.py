#!/usr/bin/env python3
"""Generate the pipeline diagram from the compiled LangGraph.

A diagram drawn by hand starts out right and drifts. This one is exported from
``RagPipeline`` itself, so if a node or an edge changes in the code, the picture
in the README changes with it on the next ``make docs``.

Writes two files:
  docs/pipeline.mmd       the raw Mermaid, straight from LangGraph
  docs/pipeline_readme.md the tidied version that the README embeds

The tidy pass only renames node ids to readable labels and drops LangGraph's
styling lines; it never adds or removes an edge, so the shape stays the shape
the code compiled.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from medicare_rag.rag.graph import RagPipeline  # noqa: E402


def _chunk_count() -> str:
    """Read the live index size, so the label cannot drift from the index."""
    manifest = ROOT / "data" / "index" / "manifest.json"
    try:
        return f"{json.loads(manifest.read_text())['num_chunks']:,} chunks"
    except (OSError, KeyError, ValueError):
        return "the indexed chunks"


LABELS = {
    "guard": "guard<br/>block unsafe or off-topic input",
    "retrieve": f"retrieve<br/>hybrid search over {_chunk_count()}",
    "grade": "grade<br/>is the best passage good enough?",
    "rewrite": "rewrite<br/>rephrase and try once more",
    "answer": "answer<br/>write it with citations",
    "refuse": "refuse<br/>say 'I do not know' instead of guessing",
}


class _StubRetriever:
    """The graph is wired at construction; nothing is retrieved to draw it."""

    def retrieve(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        raise NotImplementedError


def tidy(mermaid: str) -> str:
    """Make LangGraph's export readable without changing its edges."""
    lines: list[str] = []
    for line in mermaid.splitlines():
        stripped = line.strip()
        if stripped.startswith(("classDef", "class ", "style ")):
            continue
        if stripped.startswith("%%"):
            continue
        lines.append(line.rstrip())
    out = "\n".join(line for line in lines if line.strip())
    for node, label in LABELS.items():
        out = re.sub(rf"\b{node}\(" + r"([^)]*)\)", f"{node}({label})", out)
    # The classDef lines are gone, so a dangling ":::first" would make Mermaid
    # fail with "No such class".
    out = re.sub(r":::\w+", "", out)
    out = out.replace("<p>__start__</p>", "a question arrives")
    out = out.replace("<p>__end__</p>", "a response goes back")
    out = out.replace("__start__", "start").replace("__end__", "done")
    return out


def main() -> int:
    pipeline = RagPipeline(_StubRetriever())
    raw = pipeline.mermaid()

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "pipeline.mmd").write_text(raw + "\n")

    pretty = tidy(raw)
    (docs / "pipeline_readme.md").write_text("```mermaid\n" + pretty + "\n```\n")

    edges = len(re.findall(r"-->", raw))
    print(f"docs/pipeline.mmd        {len(raw.splitlines())} lines, {edges} edges")
    print("docs/pipeline_readme.md  written")
    print()
    print(pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
