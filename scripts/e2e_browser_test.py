#!/usr/bin/env python3
"""End-to-end browser test: drives a real Chrome against a running instance.

The unit tests check the pieces and the API tests check the HTTP surface. This
checks the thing a person actually touches - that a citation chip jumps to the
right source card, that switching to keyword-only really re-runs the search,
that the guardrail refusal reaches the screen, that the page does not scroll
sideways on a phone. None of that is visible to a request-level test.

It is deliberately not part of `make test`: it needs a browser and a running
server, so it is a deployment check rather than a unit check.

    python scripts/e2e_browser_test.py                        # the live demo
    python scripts/e2e_browser_test.py --base-url http://127.0.0.1:8000
    python scripts/e2e_browser_test.py --shots out/           # keep screenshots

Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2e_cdp import Browser, ChromeProcess

DEFAULT_BASE_URL = "https://medicare-rag-assistant.onrender.com"

ANSWER = (
    "(() => {const e = document.querySelector('#answer'); return e ? e.textContent.trim() : '';})()"
)
STRATEGY = "(() => {const e = document.querySelector('#badge-strategy'); return e ? e.textContent.trim() : '';})()"
CONFIDENCE = "(() => {const e = document.querySelector('#badge-confidence'); return e ? e.textContent.trim() : '';})()"
GROUNDED = "(() => {const e = document.querySelector('#badge-grounded'); return e ? e.textContent.trim() : '';})()"
SOURCE_COUNT = "document.querySelectorAll('#sources .source').length"
SOURCE_TITLES = (
    "Array.from(document.querySelectorAll('#sources .source-title')).map(e => e.textContent.trim())"
)
CITES = "Array.from(document.querySelectorAll('#answer .cite')).map(e => e.textContent.trim())"
NOTES = (
    "Array.from(document.querySelectorAll('#notes li')).map(e => e.textContent.trim()).join(' | ')"
)
TRACE_STEPS = "Array.from(document.querySelectorAll('#trace .step')).map(e => e.textContent.trim())"


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, label: str, condition: object, detail: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failures.append(label)
        print(f"  [{mark}] {label}" + (f"  -> {detail}" if detail else ""), flush=True)

    def section(self, title: str) -> None:
        print(f"\n--- {title} ---", flush=True)

    def summary(self) -> int:
        print("\n" + "=" * 68)
        print(f"  {self.passed} passed, {len(self.failures)} failed")
        for failure in self.failures:
            print("   FAILED:", failure)
        return 1 if self.failures else 0


async def mark_answer(browser: Browser) -> None:
    """Drop a sentinel into the answer panel.

    The app re-renders by replacing innerHTML, so the sentinel disappearing is
    proof that *this* request produced a fresh render. Waiting for "an answer
    exists" instead would pass instantly against the previous answer, and every
    assertion after a mode switch would silently read stale DOM.
    """
    await browser.js(
        """(() => {const a = document.querySelector('#answer');
        const s = document.createElement('span'); s.id = '__sentinel'; a.appendChild(s);})()"""
    )


async def settle(browser: Browser, label: str, timeout: float = 180) -> None:
    await browser.wait_for(
        "(() => !document.querySelector('#__sentinel') && "
        "(document.querySelector('#answer').textContent.trim().length > 0 "
        "|| !!document.querySelector('#status.error')))()",
        timeout,
        label,
    )
    await asyncio.sleep(0.4)


async def ask(browser: Browser, question: str) -> None:
    await mark_answer(browser)
    await browser.type_text("#question", question)
    await browser.press_enter()
    await settle(browser, f"answer for {question!r}")


async def select(browser: Browser, selector: str, value: str, label: str) -> None:
    await mark_answer(browser)
    await browser.js(
        f"""(() => {{const s = document.querySelector({selector!r});
        s.value = {value!r}; s.dispatchEvent(new Event('change', {{bubbles: true}}));}})()"""
    )
    await browser.click("#ask-btn")
    await settle(browser, label)


async def run(base_url: str, shots: Path | None) -> int:
    report = Report()

    async with await Browser.attach() as browser:

        async def shot(name: str) -> None:
            if shots is not None:
                await browser.screenshot(shots / name)

        report.section("1. the page loads and bootstraps from the API")
        await browser.goto(base_url)
        await browser.wait_for(
            "document.querySelectorAll('.chip').length > 0", 240, "example chips"
        )
        await browser.wait_for(
            "document.querySelector('#docs').options.length > 1", 120, "publication list"
        )
        report.check("page loads", True, str(await browser.js("document.title")))
        report.check(
            "all 9 publications offered in the filter",
            await browser.js("document.querySelector('#docs').options.length") == 10,
            "1 'All' + 9 booklets",
        )
        chips = await browser.js(
            "Array.from(document.querySelectorAll('.chip')).map(e => e.textContent.trim())"
        )
        report.check("example questions offered", len(chips) >= 5, f"{len(chips)} chips")
        report.check(
            "the disclaimer is visible without scrolling into the answer",
            bool(await browser.js("document.querySelector('#disclaimer').textContent.trim()")),
        )

        report.section("2. a real question, typed with real keystrokes")
        await ask(browser, "Does Medicare cover dental care?")
        answer = str(await browser.js(ANSWER))
        report.check("an answer is rendered", len(answer) > 100, f"{len(answer)} chars")
        report.check(
            "the answer is quoted, not generated",
            "quoted from sources" in str(await browser.js(STRATEGY)),
            str(await browser.js(STRATEGY)),
        )
        report.check(
            "it is not a wall of text",
            len(answer) < 1500,
            f"{len(answer)} chars",
        )
        cites = await browser.js(CITES)
        report.check("every claim carries a citation marker", len(cites) >= 2, str(cites))
        count = await browser.js(SOURCE_COUNT)
        report.check("source cards rendered", count >= 3, f"{count} sources")
        report.check(
            "groundedness is shown to the reader",
            "cited" in str(await browser.js(GROUNDED)),
            str(await browser.js(GROUNDED)),
        )
        await shot("01_answer.png")

        report.section("3. a citation actually resolves to its source")
        targets = await browser.js(
            "Array.from(document.querySelectorAll('#answer .cite')).map(e => e.getAttribute('href'))"
        )
        ids = await browser.js(
            "Array.from(document.querySelectorAll('#sources .source')).map(e => '#' + e.id)"
        )
        report.check(
            "every citation link points at a source card that exists",
            all(t in ids for t in targets),
            f"{targets[:4]} -> {ids}",
        )
        pdf_links = await browser.js(
            "Array.from(document.querySelectorAll('#sources a')).map(e => e.href)"
        )
        report.check(
            "each source links to the official CMS PDF on a page",
            pdf_links
            and all("medicare.gov/publications" in u and "#page=" in u for u in pdf_links),
            str(pdf_links[0]) if pdf_links else "none",
        )

        report.section("4. refusing is visible, not hidden")
        await ask(browser, "What is the best stock to buy?")
        report.check(
            "an out-of-scope question is refused",
            "refused" in str(await browser.js(STRATEGY)).lower(),
            str(await browser.js(STRATEGY)),
        )
        report.check(
            "and the reason is shown",
            "threshold" in str(await browser.js(NOTES)).lower(),
            str(await browser.js(NOTES))[:80],
        )
        await shot("02_refusal.png")

        report.section("5. the guardrail stops a Medicare number")
        await ask(browser, "my MBI is 1EG4-TE5-MK73 what does Part A cover")
        report.check(
            "a question carrying an MBI is refused",
            "refused" in str(await browser.js(STRATEGY)).lower(),
            str(await browser.js(STRATEGY)),
        )
        report.check(
            "the user is told why",
            "medicare number" in str(await browser.js(ANSWER)).lower(),
            str(await browser.js(ANSWER))[:70],
        )
        page_text = str(await browser.js("document.body.textContent"))
        report.check(
            "the identifier is not echoed anywhere on the page",
            "1EG4-TE5-MK73" not in page_text.upper(),
        )
        report.check(
            "and it never reached retrieval",
            await browser.js(SOURCE_COUNT) == 0,
            f"{await browser.js(SOURCE_COUNT)} sources shown",
        )
        await shot("03_guardrail.png")

        report.section("6. the retrieval mode switch really re-runs the search")
        await ask(browser, "What is a guaranteed issue right?")
        hybrid_titles = await browser.js(SOURCE_TITLES)
        await select(browser, "#mode", "lexical", "keyword-only results")
        keyword_titles = await browser.js(SOURCE_TITLES)
        report.check(
            "keyword-only returns a different set of sources",
            keyword_titles and keyword_titles != hybrid_titles,
            f"{hybrid_titles[:2]!s} vs {keyword_titles[:2]!s}",
        )
        await select(browser, "#mode", "hybrid", "hybrid restored")
        report.check(
            "switching back restores the hybrid result",
            await browser.js(SOURCE_TITLES) == hybrid_titles,
        )

        report.section("7. the publication filter restricts the sources")
        await select(browser, "#docs", "02110", "filtered to the Medigap guide")
        titles = await browser.js(SOURCE_TITLES)
        report.check(
            "only the chosen booklet is cited",
            titles and all("Medigap" in t for t in titles),
            str(titles[:3]),
        )
        await select(browser, "#docs", "", "filter cleared")

        report.section("8. the trace shows what actually happened")
        await browser.click("#trace-box summary, #trace-box")
        await asyncio.sleep(0.8)
        steps = await browser.js(TRACE_STEPS)
        report.check("the pipeline steps are listed", len(steps) >= 3, str(steps[:5]))
        report.check(
            "the guard step ran first",
            steps and "guard" in steps[0].lower(),
            str(steps[0]) if steps else "no steps",
        )
        await shot("04_trace.png")

        report.section("9. feedback reaches the API")
        await browser.click("#fb-yes")
        await browser.wait_for(
            "!document.querySelector('#fb-thanks').hidden", 30, "feedback acknowledgement"
        )
        report.check("a thumbs-up is accepted and acknowledged", True)

        report.section("10. the evaluation numbers are served from the live API")
        await browser.js("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.2)
        rows = await browser.js(
            "document.querySelectorAll('#evaluation table.metrics tbody tr').length"
        )
        report.check("the accuracy table is rendered", rows >= 3, f"{rows} rows")
        report.check(
            "it carries real measured numbers",
            "%" in str(await browser.js("document.querySelector('#evaluation').textContent")),
        )
        await shot("05_evaluation.png")

        report.section("11. theme preference survives a reload")
        before = await browser.js("document.documentElement.getAttribute('data-theme')")
        await browser.click("#theme-btn")
        await asyncio.sleep(0.6)
        after = await browser.js("document.documentElement.getAttribute('data-theme')")
        report.check("the theme toggle switches theme", before != after, f"{before} -> {after}")
        await browser.goto(base_url)
        await browser.wait_for("document.querySelectorAll('.chip').length > 0", 120, "reload")
        report.check(
            "the choice survives a reload",
            await browser.js("document.documentElement.getAttribute('data-theme')") == after,
        )

        report.section("12. a phone viewport, a true 390 px")
        await browser.emulate_phone()
        await browser.goto(base_url)
        await browser.wait_for("document.querySelectorAll('.chip').length > 0", 180, "mobile load")
        await ask(browser, "Can I get hospice care at home?")
        await asyncio.sleep(0.8)
        inner = await browser.js("window.innerWidth")
        scroll = await browser.js("document.documentElement.scrollWidth")
        report.check(
            "no horizontal overflow on a phone",
            scroll <= inner + 1,
            f"viewport {inner}px, content {scroll}px",
        )
        report.check("the answer still renders on mobile", len(str(await browser.js(ANSWER))) > 100)
        await shot("06_mobile.png")
        await browser.clear_emulation()

        report.section("13. console and network health")
        report.check("no JavaScript errors", not browser.errors, "; ".join(browser.errors[:2]))
        network_failures = [
            f for f in browser.failed if "favicon" not in f.lower() and "429" not in f
        ]
        report.check("no failed requests", not network_failures, "; ".join(network_failures[:2]))

    return report.summary()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="instance to test")
    parser.add_argument("--shots", type=Path, default=None, help="directory for screenshots")
    parser.add_argument("--port", type=int, default=9222, help="DevTools port")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Testing {base_url}")
    with ChromeProcess(port=args.port):
        return asyncio.run(run(base_url, args.shots))


if __name__ == "__main__":
    raise SystemExit(main())
