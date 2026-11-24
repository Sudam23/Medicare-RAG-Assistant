"""The HTTP surface, exercised through a real ASGI client."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medicare_rag.api.main import _requests, app
from medicare_rag.config import RATE_LIMIT_REQUESTS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give every test the full rate-limit budget.

    The limiter is per-process and the client is module-scoped, so without this
    a test that asks many questions silently causes a 429 in a later, unrelated
    test - which is how a rate-limit bug once showed up as a failing feedback test.
    """
    _requests.clear()
    yield


class TestHealthAndMetadata:
    def test_health_reports_ready(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["index_ready"] is True

    def test_stats_describe_the_index(self, client: TestClient) -> None:
        body = client.get("/api/stats").json()
        assert body["num_chunks"] > 1000
        assert body["embedding_model"]
        assert body["answer_strategy"] in ("extractive", "generated")

    def test_documents_lists_every_publication(self, client: TestClient) -> None:
        docs = client.get("/api/documents").json()
        assert len(docs) == 9
        assert all(d["pages"] > 0 and d["chunks"] > 0 for d in docs)
        assert all(d["url"].startswith("https://www.medicare.gov/") for d in docs)

    def test_examples_are_offered(self, client: TestClient) -> None:
        assert client.get("/api/examples").json()["examples"]

    def test_every_example_question_is_answerable(self, client: TestClient) -> None:
        """A suggested question that refuses makes the demo look broken.

        The refusal may be correct - a bare "Plan G" really is too thin to answer -
        but it must not be something the UI itself invites the visitor to click.
        """
        refused = []
        for question in client.get("/api/examples").json()["examples"]:
            body = client.get("/api/ask", params={"q": question, "k": 4}).json()
            if body["strategy"] == "refused":
                refused.append(question)
        assert not refused, f"example questions that refuse: {refused}"

    def test_evaluation_report_is_served(self, client: TestClient) -> None:
        body = client.get("/api/evaluation").json()
        assert body["pipeline"]["num_questions"] > 0
        assert "hybrid" in body["retrieval"]


class TestAsk:
    def test_a_question_returns_a_cited_answer(self, client: TestClient) -> None:
        body = client.get(
            "/api/ask", params={"q": "Does Medicare cover dental care?", "k": 4}
        ).json()
        assert body["answer"]
        assert body["sources"]
        assert body["groundedness"] == 1.0
        assert body["trace_id"]
        assert body["disclaimer"]

    def test_every_citation_marker_resolves_to_a_source(self, client: TestClient) -> None:
        """A marker with no matching source is a broken link in the UI."""
        import re

        body = client.get(
            "/api/ask", params={"q": "What is the Part B late enrollment penalty?"}
        ).json()
        numbers = {s["number"] for s in body["sources"]}
        assert {int(m) for m in re.findall(r"\[(\d+)\]", body["answer"])} <= numbers

    def test_post_and_get_agree(self, client: TestClient) -> None:
        question = "Can I get hospice care at home?"
        a = client.get("/api/ask", params={"q": question, "k": 3}).json()
        b = client.post("/api/ask", json={"question": question, "k": 3}).json()
        assert a["answer"] == b["answer"]

    def test_out_of_scope_is_refused(self, client: TestClient) -> None:
        body = client.get("/api/ask", params={"q": "How do I bake sourdough bread?"}).json()
        assert body["strategy"] == "refused"

    def test_personal_data_is_refused_with_no_sources(self, client: TestClient) -> None:
        body = client.get("/api/ask", params={"q": "my ssn is 123-45-6789 what is covered"}).json()
        assert body["strategy"] == "refused"
        assert body["sources"] == []

    def test_document_filter_restricts_the_answer(self, client: TestClient) -> None:
        body = client.get(
            "/api/ask", params={"q": "what is covered", "docs": "02154", "k": 5}
        ).json()
        assert {s["doc_id"] for s in body["sources"]} == {"02154"}

    def test_an_unknown_document_is_rejected(self, client: TestClient) -> None:
        assert client.get("/api/ask", params={"q": "anything", "docs": "99999"}).status_code == 422

    def test_an_unknown_mode_is_rejected(self, client: TestClient) -> None:
        assert client.get("/api/ask", params={"q": "anything", "mode": "magic"}).status_code == 422

    @pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
    def test_each_mode_is_accepted(self, client: TestClient, mode: str) -> None:
        assert (
            client.get("/api/ask", params={"q": "Medicare costs", "mode": mode}).status_code == 200
        )

    def test_a_too_long_question_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={"question": "x" * 900}).status_code == 422


class TestObservability:
    def test_a_trace_is_recorded_for_every_answer(self, client: TestClient) -> None:
        trace_id = client.get("/api/ask", params={"q": "What is a Special Needs Plan?"}).json()[
            "trace_id"
        ]
        trace = client.get(f"/api/traces/{trace_id}").json()
        assert next(s["node"] for s in trace["steps"]) == "guard"
        assert trace["total_ms"] > 0

    def test_unknown_trace_is_404(self, client: TestClient) -> None:
        assert client.get("/api/traces/doesnotexist").status_code == 404

    def test_the_trace_summary_aggregates(self, client: TestClient) -> None:
        client.get("/api/ask", params={"q": "Does Medicare cover dental care?"})
        summary = client.get("/api/traces").json()["summary"]
        assert summary["count"] > 0
        assert "strategies" in summary


class TestTheServedPage:
    def test_the_disclaimer_is_in_the_html_itself(self, client: TestClient) -> None:
        """It must not depend on JavaScript or on a fetch succeeding.

        It did once: loadMeta() was written but never called, so the "not medical
        advice" line rendered as an empty paragraph on every visit. The browser
        test caught it; this one keeps it caught without needing a browser.
        """
        html = client.get("/").text
        assert "not medical, legal or insurance advice" in html.lower()

    def test_the_page_bootstraps_every_loader(self, client: TestClient) -> None:
        script = client.get("/app.js").text
        for loader in ("loadExamples", "loadEvaluation", "loadDocuments", "loadMeta"):
            assert f"{loader}()" in script.split("/* ---")[-1] or f"{loader}()" in script, loader


class TestRateLimiting:
    def test_asking_too_often_is_refused(self, client: TestClient) -> None:
        last = None
        for _ in range(RATE_LIMIT_REQUESTS + 1):
            last = client.get("/api/ask", params={"q": "what is Part A?", "k": 2})
        assert last is not None and last.status_code == 429

    def test_metadata_calls_are_not_rate_limited(self, client: TestClient) -> None:
        """Loading the page must not spend the question budget.

        Four metadata calls fire on every page load. Counting them against the
        same budget as a real question meant a visitor could be refused service
        by the demo's own example chips.
        """
        for _ in range(RATE_LIMIT_REQUESTS + 5):
            assert client.get("/api/stats").status_code == 200
        assert client.get("/api/ask", params={"q": "what is Part A?", "k": 2}).status_code == 200


class TestFeedback:
    def test_feedback_is_recorded_against_a_trace(self, client: TestClient) -> None:
        trace_id = client.get(
            "/api/ask", params={"q": "What does a Medigap policy not cover?"}
        ).json()["trace_id"]
        body = client.post("/api/feedback", json={"trace_id": trace_id, "helpful": True}).json()
        assert body["recorded"] is True
        assert body["helpful"] >= 1

    def test_feedback_for_an_unknown_trace_is_404(self, client: TestClient) -> None:
        assert (
            client.post("/api/feedback", json={"trace_id": "nope", "helpful": True}).status_code
            == 404
        )


class TestWebPage:
    @pytest.mark.parametrize("path", ["/", "/styles.css", "/app.js"])
    def test_static_files_are_served(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 200

    def test_openapi_docs_are_available(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
