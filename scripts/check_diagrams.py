#!/usr/bin/env python3
"""Render every Mermaid diagram in the docs, in a real browser.

A Mermaid block with a syntax error does not fail loudly: GitHub just shows a
broken grey box where the diagram should be. Since these diagrams are generated,
a change to the generator could break every one of them at once and nothing in
the test suite would notice.

So this loads the same Mermaid library GitHub uses, renders each block, and
fails if any of them does not produce an SVG.

    python scripts/check_diagrams.py
    python scripts/check_diagrams.py --shots out/    # keep the rendered PNGs

Exit code is 0 only if every diagram renders.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2e_cdp import Browser, ChromeProcess

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "README.md", ROOT / "docs" / "ARCHITECTURE.md", ROOT / "docs" / "EVALUATION.md"]
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diagram check</title></head>
<body><div id="out"></div>
<script type="module">
import mermaid from "%s";
mermaid.initialize({startOnLoad: false, securityLevel: "loose"});
const diagrams = %s;
const results = [];
for (const d of diagrams) {
  try {
    await mermaid.parse(d.text);
    const {svg} = await mermaid.render("d" + d.index, d.text);
    const holder = document.createElement("div");
    holder.id = "render-" + d.index;
    holder.innerHTML = svg;
    document.getElementById("out").appendChild(holder);
    const el = holder.querySelector("svg");
    results.push({...d, ok: !!el, width: el ? el.getBoundingClientRect().width : 0, error: null});
  } catch (err) {
    results.push({...d, ok: false, width: 0, error: String(err && err.message || err)});
  }
}
window.__results = results;
window.__done = true;
</script></body></html>
"""


def collect() -> list[dict]:
    blocks: list[dict] = []
    for path in SOURCES:
        if not path.exists():
            continue
        for match in re.finditer(r"```mermaid\n(.*?)```", path.read_text(), re.DOTALL):
            blocks.append(
                {
                    "index": len(blocks),
                    "source": str(path.relative_to(ROOT)),
                    "text": match.group(1).rstrip(),
                }
            )
    return blocks


async def run(blocks: list[dict], shots: Path | None) -> int:
    html = PAGE % (MERMAID, json.dumps(blocks))
    with tempfile.TemporaryDirectory(prefix="mermaid-check-") as tmp:
        page = Path(tmp) / "check.html"
        page.write_text(html)
        async with await Browser.attach(port=9223) as browser:
            await browser.goto(page.as_uri())
            await browser.wait_for("window.__done === true", 120, "mermaid to finish")
            results = await browser.js("window.__results")

            failures = []
            for r in results:
                head = r["text"].splitlines()[0][:48]
                if r["ok"] and r["width"] > 50:
                    print(f"  [PASS] {r['source']}  {head!r}  -> {r['width']:.0f}px wide")
                else:
                    failures.append(r)
                    print(f"  [FAIL] {r['source']}  {head!r}")
                    if r["error"]:
                        print(f"         {r['error']}")

            if shots is not None and not failures:
                shots.mkdir(parents=True, exist_ok=True)
                await browser.screenshot(shots / "diagrams.png")
                print(f"\n  screenshot -> {shots / 'diagrams.png'}")

            print(f"\n  {len(results) - len(failures)} rendered, {len(failures)} failed")
            return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, default=None, help="directory for a screenshot")
    args = parser.parse_args()

    blocks = collect()
    if not blocks:
        print("no mermaid diagrams found", file=sys.stderr)
        return 2
    print(f"Checking {len(blocks)} Mermaid diagram(s)")
    with ChromeProcess(port=9223):
        return asyncio.run(run(blocks, args.shots))


if __name__ == "__main__":
    raise SystemExit(main())
