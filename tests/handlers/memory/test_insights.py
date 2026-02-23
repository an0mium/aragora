"""Tests for InsightsHandler (aragora/server/handlers/memory/insights.py).

Covers all routes and behavior of the InsightsHandler class:
- can_handle() routing for all ROUTES
- GET /api/insights/recent - recent insights retrieval
- GET /api/flips/recent - recent position flips
- GET /api/flips/summary - flip summary statistics
- POST /api/insights/extract-detailed - detailed insight extraction
- Rate limiting (429 on exceeded)
- RBAC auth (401/403 via no_auto_auth)
- Unmatched route returns None
- Edge cases: empty stores, large content, extraction toggles
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class MockHTTPHandler:
    """Mock HTTP request handler for InsightsHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self._request_body = body

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock InsightType enum
# ---------------------------------------------------------------------------


class MockInsightType(Enum):
    POSITION_REVERSAL = "position_reversal"
    CONSENSUS_SHIFT = "consensus_shift"
    EVIDENCE_GAP = "evidence_gap"
    ARGUMENTATION_PATTERN = "argumentation_pattern"


# ---------------------------------------------------------------------------
# Mock Insight object
# ---------------------------------------------------------------------------


@dataclass
class MockInsight:
    id: str = "ins-001"
    type: MockInsightType = MockInsightType.CONSENSUS_SHIFT
    title: str = "Test Insight"
    description: str = "A test insight description"
    confidence: float = 0.85
    agents_involved: list = field(default_factory=lambda: ["claude", "gpt4"])
    evidence: list = field(default_factory=lambda: ["ev1", "ev2", "ev3", "ev4"])
    created_at: str = "2026-01-15T10:00:00"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http():
    """Create a MockHTTPHandler for GET requests."""
    return MockHTTPHandler()


@pytest.fixture
def mock_http_post():
    """Factory for creating MockHTTPHandler for POST requests with body."""
    def _make(body: dict | None = None) -> MockHTTPHandler:
        return MockHTTPHandler(body=body, method="POST")
    return _make


@pytest.fixture
def mock_insight_store():
    """Create a mock insight store."""
    store = MagicMock()

    async def _get_recent(limit=20):
        return [
            MockInsight(
                id="ins-001",
                type=MockInsightType.CONSENSUS_SHIFT,
                title="Consensus Shifted",
                description="Agents converged on option A",
                confidence=0.9,
                agents_involved=["claude", "gpt4"],
                evidence=["ev1", "ev2", "ev3", "ev4"],
            ),
            MockInsight(
                id="ins-002",
                type=MockInsightType.POSITION_REVERSAL,
                title="Agent Reversed Position",
                description="gpt4 changed stance on encryption",
                confidence=0.75,
                agents_involved=["gpt4"],
                evidence=["ev5"],
            ),
            MockInsight(
                id="ins-003",
                type=MockInsightType.EVIDENCE_GAP,
                title="Missing Evidence",
                description="No supporting data for claim X",
                confidence=0.6,
                agents_involved=["claude", "gemini"],
                evidence=[],
            ),
        ]

    store.get_recent_insights = _get_recent
    return store


@pytest.fixture
def handler(mock_insight_store):
    """Create an InsightsHandler with a mocked insight store."""
    from aragora.server.handlers.memory.insights import InsightsHandler

    return InsightsHandler(server_context={"insight_store": mock_insight_store})


@pytest.fixture
def handler_no_store():
    """Create an InsightsHandler with NO insight store (not configured)."""
    from aragora.server.handlers.memory.insights import InsightsHandler

    return InsightsHandler(server_context={})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the insights rate limiter before each test."""
    from aragora.server.handlers.memory.insights import _insights_limiter

    _insights_limiter._buckets = defaultdict(list)
    _insights_limiter._requests = _insights_limiter._buckets
    yield
    _insights_limiter._buckets = defaultdict(list)
    _insights_limiter._requests = _insights_limiter._buckets


# ===========================================================================
# can_handle() Tests
# ===========================================================================


class TestCanHandle:
    """Test can_handle() routing."""

    def test_handles_insights_recent(self, handler):
        assert handler.can_handle("/api/insights/recent") is True

    def test_handles_insights_recent_versioned(self, handler):
        assert handler.can_handle("/api/v1/insights/recent") is True

    def test_handles_insights_extract_detailed(self, handler):
        assert handler.can_handle("/api/insights/extract-detailed") is True

    def test_handles_insights_extract_detailed_versioned(self, handler):
        assert handler.can_handle("/api/v1/insights/extract-detailed") is True

    def test_handles_flips_recent(self, handler):
        assert handler.can_handle("/api/flips/recent") is True

    def test_handles_flips_recent_versioned(self, handler):
        assert handler.can_handle("/api/v1/flips/recent") is True

    def test_handles_flips_summary(self, handler):
        assert handler.can_handle("/api/flips/summary") is True

    def test_handles_flips_summary_versioned(self, handler):
        assert handler.can_handle("/api/v1/flips/summary") is True

    def test_handles_insights_subpath(self, handler):
        """Any path starting with /api/insights/ should be handled."""
        assert handler.can_handle("/api/insights/anything") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates/list") is False

    def test_rejects_memory_path(self, handler):
        assert handler.can_handle("/api/memory/continuum") is False

    def test_rejects_flips_unknown(self, handler):
        """Paths under /api/flips/ that aren't /recent or /summary are rejected."""
        assert handler.can_handle("/api/flips/unknown") is False


# ===========================================================================
# GET /api/insights/recent Tests
# ===========================================================================


class TestRecentInsights:
    """Test GET /api/insights/recent."""

    @pytest.mark.asyncio
    async def test_recent_insights_success(self, handler, mock_http):
        result = await handler.handle("/api/insights/recent", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "insights" in body
        assert "count" in body
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_recent_insights_fields(self, handler, mock_http):
        result = await handler.handle("/api/insights/recent", {}, mock_http)
        body = _body(result)
        first = body["insights"][0]
        assert first["id"] == "ins-001"
        assert first["type"] == "consensus_shift"
        assert first["title"] == "Consensus Shifted"
        assert first["description"] == "Agents converged on option A"
        assert first["confidence"] == 0.9
        assert first["agents_involved"] == ["claude", "gpt4"]

    @pytest.mark.asyncio
    async def test_recent_insights_evidence_truncated(self, handler, mock_http):
        """Evidence should be truncated to first 3 items."""
        result = await handler.handle("/api/insights/recent", {}, mock_http)
        body = _body(result)
        first = body["insights"][0]
        # Original has 4 evidence items, should be truncated to 3
        assert len(first["evidence"]) == 3
        assert first["evidence"] == ["ev1", "ev2", "ev3"]

    @pytest.mark.asyncio
    async def test_recent_insights_empty_evidence(self, handler, mock_http):
        """Insight with empty evidence should return empty list."""
        result = await handler.handle("/api/insights/recent", {}, mock_http)
        body = _body(result)
        third = body["insights"][2]
        assert third["evidence"] == []

    @pytest.mark.asyncio
    async def test_recent_insights_with_limit(self, handler, mock_http):
        result = await handler.handle("/api/insights/recent", {"limit": "5"}, mock_http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_recent_insights_limit_clamped_max(self, handler, mock_http, mock_insight_store):
        """Limit should be clamped to max 100."""
        call_limits = []
        original = mock_insight_store.get_recent_insights

        async def tracking_get_recent(limit=20):
            call_limits.append(limit)
            return await original(limit=limit)

        mock_insight_store.get_recent_insights = tracking_get_recent
        await handler.handle("/api/insights/recent", {"limit": "999"}, mock_http)
        assert call_limits[0] == 100

    @pytest.mark.asyncio
    async def test_recent_insights_limit_clamped_min(self, handler, mock_http, mock_insight_store):
        """Limit should be clamped to min 1."""
        call_limits = []
        original = mock_insight_store.get_recent_insights

        async def tracking_get_recent(limit=20):
            call_limits.append(limit)
            return await original(limit=limit)

        mock_insight_store.get_recent_insights = tracking_get_recent
        await handler.handle("/api/insights/recent", {"limit": "0"}, mock_http)
        assert call_limits[0] == 1

    @pytest.mark.asyncio
    async def test_recent_insights_default_limit(self, handler, mock_http, mock_insight_store):
        """Default limit should be 20."""
        call_limits = []
        original = mock_insight_store.get_recent_insights

        async def tracking_get_recent(limit=20):
            call_limits.append(limit)
            return await original(limit=limit)

        mock_insight_store.get_recent_insights = tracking_get_recent
        await handler.handle("/api/insights/recent", {}, mock_http)
        assert call_limits[0] == 20

    @pytest.mark.asyncio
    async def test_recent_insights_versioned_path(self, handler, mock_http):
        """Versioned path /api/v1/insights/recent should also work."""
        result = await handler.handle("/api/v1/insights/recent", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_recent_insights_no_store(self, handler_no_store, mock_http):
        """When insight_store is not configured, return empty list."""
        result = await handler_no_store.handle("/api/insights/recent", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["insights"] == []
        assert "error" in body


# ===========================================================================
# GET /api/flips/recent Tests
# ===========================================================================


class TestRecentFlips:
    """Test GET /api/flips/recent."""

    @pytest.mark.asyncio
    async def test_recent_flips_success(self, handler, mock_http):
        result = await handler.handle("/api/flips/recent", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "flips" in body
        assert "count" in body

    @pytest.mark.asyncio
    async def test_recent_flips_filters_position_reversals(self, handler, mock_http):
        """Only insights with type=position_reversal should appear as flips."""
        result = await handler.handle("/api/flips/recent", {}, mock_http)
        body = _body(result)
        # Only 1 of the 3 mock insights is a position_reversal
        assert body["count"] == 1
        assert len(body["flips"]) == 1
        flip = body["flips"][0]
        assert flip["id"] == "ins-002"
        assert flip["agent"] == "gpt4"
        assert flip["new_position"] == "Agent Reversed Position"
        assert flip["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_recent_flips_previous_position_truncated(self, handler, mock_http):
        """previous_position should be description truncated to 200 chars."""
        result = await handler.handle("/api/flips/recent", {}, mock_http)
        body = _body(result)
        flip = body["flips"][0]
        assert flip["previous_position"] == "gpt4 changed stance on encryption"
        assert len(flip["previous_position"]) <= 200

    @pytest.mark.asyncio
    async def test_recent_flips_no_store(self, handler_no_store, mock_http):
        """When insight_store is not configured, return empty list with message."""
        result = await handler_no_store.handle("/api/flips/recent", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["flips"] == []
        assert body["count"] == 0
        assert "message" in body

    @pytest.mark.asyncio
    async def test_recent_flips_no_agents(self, handler, mock_http, mock_insight_store):
        """When a position_reversal has empty agents_involved, use 'unknown'."""
        async def _empty_agents(limit=20):
            return [
                MockInsight(
                    id="ins-x",
                    type=MockInsightType.POSITION_REVERSAL,
                    title="Flip",
                    agents_involved=[],
                    confidence=0.5,
                ),
            ]

        mock_insight_store.get_recent_insights = _empty_agents
        result = await handler.handle("/api/flips/recent", {}, mock_http)
        body = _body(result)
        assert body["flips"][0]["agent"] == "unknown"

    @pytest.mark.asyncio
    async def test_recent_flips_store_exception(self, handler, mock_http, mock_insight_store):
        """When store raises an exception, return empty flips gracefully."""
        async def _raise(limit=20):
            raise RuntimeError("store unavailable")

        mock_insight_store.get_recent_insights = _raise
        result = await handler.handle("/api/flips/recent", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["flips"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_recent_flips_versioned_path(self, handler, mock_http):
        result = await handler.handle("/api/v1/flips/recent", {}, mock_http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_recent_flips_limit_respected(self, handler, mock_http, mock_insight_store):
        """With limit=1, only 1 flip should be returned even if more exist."""
        async def _many_flips(limit=20):
            return [
                MockInsight(
                    id=f"ins-{i}",
                    type=MockInsightType.POSITION_REVERSAL,
                    title=f"Flip {i}",
                    agents_involved=[f"agent-{i}"],
                    confidence=0.5 + i * 0.01,
                )
                for i in range(10)
            ]

        mock_insight_store.get_recent_insights = _many_flips
        result = await handler.handle("/api/flips/recent", {"limit": "2"}, mock_http)
        body = _body(result)
        assert body["count"] == 2
        assert len(body["flips"]) == 2

    @pytest.mark.asyncio
    async def test_recent_flips_detected_at_field(self, handler, mock_http):
        """Each flip should include a detected_at field."""
        result = await handler.handle("/api/flips/recent", {}, mock_http)
        body = _body(result)
        if body["flips"]:
            assert "detected_at" in body["flips"][0]


# ===========================================================================
# GET /api/flips/summary Tests
# ===========================================================================


class TestFlipsSummary:
    """Test GET /api/flips/summary."""

    @pytest.mark.asyncio
    async def test_summary_success(self, handler, mock_http):
        result = await handler.handle("/api/flips/summary", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body
        assert "total" in body["summary"]

    @pytest.mark.asyncio
    async def test_summary_counts_reversals(self, handler, mock_http):
        """Summary total should count only position_reversal insights."""
        result = await handler.handle("/api/flips/summary", {}, mock_http)
        body = _body(result)
        # Only 1 of the 3 mock insights is a position_reversal
        assert body["summary"]["total"] == 1

    @pytest.mark.asyncio
    async def test_summary_with_period(self, handler, mock_http):
        """When period param is provided, it should be in response."""
        result = await handler.handle(
            "/api/flips/summary", {"period": "7d"}, mock_http
        )
        body = _body(result)
        assert body["period"] == "7d"

    @pytest.mark.asyncio
    async def test_summary_without_period(self, handler, mock_http):
        """When period param is not provided, it should not be in response."""
        result = await handler.handle("/api/flips/summary", {}, mock_http)
        body = _body(result)
        assert "period" not in body

    @pytest.mark.asyncio
    async def test_summary_no_store(self, handler_no_store, mock_http):
        result = await handler_no_store.handle("/api/flips/summary", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total"] == 0
        assert "message" in body

    @pytest.mark.asyncio
    async def test_summary_store_exception(self, handler, mock_http, mock_insight_store):
        """When store raises, return 0 total gracefully."""
        async def _raise(limit=20):
            raise ValueError("bad query")

        mock_insight_store.get_recent_insights = _raise
        result = await handler.handle("/api/flips/summary", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total"] == 0

    @pytest.mark.asyncio
    async def test_summary_limit_default(self, handler, mock_http, mock_insight_store):
        """Default limit for summary is 200, max 500."""
        call_limits = []
        original = mock_insight_store.get_recent_insights

        async def tracking(limit=20):
            call_limits.append(limit)
            return await original(limit=limit)

        mock_insight_store.get_recent_insights = tracking
        await handler.handle("/api/flips/summary", {}, mock_http)
        assert call_limits[0] == 200

    @pytest.mark.asyncio
    async def test_summary_limit_clamped_max(self, handler, mock_http, mock_insight_store):
        """Limit should be clamped to max 500."""
        call_limits = []
        original = mock_insight_store.get_recent_insights

        async def tracking(limit=20):
            call_limits.append(limit)
            return await original(limit=limit)

        mock_insight_store.get_recent_insights = tracking
        await handler.handle("/api/flips/summary", {"limit": "9999"}, mock_http)
        assert call_limits[0] == 500

    @pytest.mark.asyncio
    async def test_summary_versioned_path(self, handler, mock_http):
        result = await handler.handle("/api/v1/flips/summary", {}, mock_http)
        assert _status(result) == 200


# ===========================================================================
# POST /api/insights/extract-detailed Tests
# ===========================================================================


class TestExtractDetailed:
    """Test POST /api/insights/extract-detailed."""

    @pytest.mark.asyncio
    async def test_extract_success(self, handler, mock_http_post):
        http = mock_http_post({
            "content": "Therefore we should adopt option A. Because option B is worse.",
            "debate_id": "debate-123",
        })
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-123"
        assert body["content_length"] > 0
        assert "claims" in body
        assert "evidence_chains" in body
        assert "patterns" in body

    @pytest.mark.asyncio
    async def test_extract_missing_content(self, handler, mock_http_post):
        http = mock_http_post({"debate_id": "debate-123"})
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "content" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_extract_empty_content(self, handler, mock_http_post):
        http = mock_http_post({"content": "   "})
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_extract_content_too_large(self, handler, mock_http_post):
        from aragora.server.handlers.memory.insights import MAX_CONTENT_SIZE

        big_content = "x" * (MAX_CONTENT_SIZE + 1)
        http = mock_http_post({"content": big_content})
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 413
        body = _body(result)
        assert "large" in body.get("error", "").lower() or "size" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_extract_no_body(self, handler):
        """Missing JSON body should return 400."""
        http = MockHTTPHandler(body=None, method="POST")
        # read_json_body will return {} (empty dict from Content-Length: 2),
        # but "content" key is missing so it should be 400
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_extract_claims_disabled(self, handler, mock_http_post):
        http = mock_http_post({
            "content": "Therefore we should adopt option A.",
            "extract_claims": False,
        })
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "claims" not in body
        assert "evidence_chains" in body
        assert "patterns" in body

    @pytest.mark.asyncio
    async def test_extract_evidence_disabled(self, handler, mock_http_post):
        http = mock_http_post({
            "content": "According to experts, the data indicates improvement.",
            "extract_evidence": False,
        })
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "claims" in body
        assert "evidence_chains" not in body
        assert "patterns" in body

    @pytest.mark.asyncio
    async def test_extract_patterns_disabled(self, handler, mock_http_post):
        http = mock_http_post({
            "content": "On one hand X, on the other hand Y. Because Z.",
            "extract_patterns": False,
        })
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "claims" in body
        assert "evidence_chains" in body
        assert "patterns" not in body

    @pytest.mark.asyncio
    async def test_extract_all_disabled(self, handler, mock_http_post):
        http = mock_http_post({
            "content": "Some content here.",
            "extract_claims": False,
            "extract_evidence": False,
            "extract_patterns": False,
        })
        result = await handler.handle_post("/api/insights/extract-detailed", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "claims" not in body
        assert "evidence_chains" not in body
        assert "patterns" not in body
        assert body["content_length"] == len("Some content here.")

    @pytest.mark.asyncio
    async def test_extract_unmatched_route(self, handler, mock_http_post):
        """POST to unrecognized path should return None."""
        http = mock_http_post({"content": "test"})
        result = await handler.handle_post("/api/insights/unknown", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_versioned_path(self, handler, mock_http_post):
        http = mock_http_post({
            "content": "Therefore we must act now.",
        })
        result = await handler.handle_post(
            "/api/v1/insights/extract-detailed", {}, http
        )
        assert _status(result) == 200


# ===========================================================================
# Claim Extraction Tests
# ===========================================================================


class TestClaimExtraction:
    """Test _extract_claims_from_content logic."""

    def _handler(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        return InsightsHandler(server_context={})

    def test_extracts_therefore_claim(self):
        h = self._handler()
        claims = h._extract_claims_from_content(
            "Therefore we should adopt the new strategy."
        )
        assert len(claims) >= 1
        assert claims[0]["type"] == "argument"  # contains "should"

    def test_extracts_should_claim_as_argument(self):
        h = self._handler()
        claims = h._extract_claims_from_content(
            "We should invest in renewable energy now."
        )
        assert len(claims) >= 1
        assert claims[0]["type"] == "argument"

    def test_extracts_evidence_shows_as_assertion(self):
        h = self._handler()
        claims = h._extract_claims_from_content(
            "Evidence shows that the approach is effective."
        )
        assert len(claims) >= 1
        assert claims[0]["type"] == "assertion"

    def test_extracts_is_better_claim(self):
        h = self._handler()
        claims = h._extract_claims_from_content(
            "Option A is better than option B for our use case."
        )
        assert len(claims) >= 1

    def test_skips_short_sentences(self):
        h = self._handler()
        claims = h._extract_claims_from_content("Thus. Ok.")
        assert len(claims) == 0

    def test_limits_to_20_claims(self):
        h = self._handler()
        content = ". ".join(
            [f"Therefore we should do item number {i} right now" for i in range(30)]
        )
        claims = h._extract_claims_from_content(content)
        assert len(claims) <= 20

    def test_truncates_long_claim_text(self):
        h = self._handler()
        long_sentence = "Therefore " + "x" * 600
        claims = h._extract_claims_from_content(long_sentence)
        if claims:
            assert len(claims[0]["text"]) <= 500

    def test_no_claims_in_plain_text(self):
        h = self._handler()
        claims = h._extract_claims_from_content(
            "The weather is nice today and the sky is blue."
        )
        assert len(claims) == 0


# ===========================================================================
# Evidence Extraction Tests
# ===========================================================================


class TestEvidenceExtraction:
    """Test _extract_evidence_from_content logic."""

    def _handler(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        return InsightsHandler(server_context={})

    def test_extracts_citation(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "According to the research team, the results are conclusive."
        )
        assert len(evidence) >= 1
        assert evidence[0]["type"] == "citation"
        assert "the research team" in evidence[0]["source"]

    def test_extracts_research(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "Research shows the treatment is effective."
        )
        assert len(evidence) >= 1
        assert evidence[0]["type"] == "research"

    def test_extracts_data(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "Data indicates a 50% improvement over baseline."
        )
        assert len(evidence) >= 1
        assert evidence[0]["type"] == "data"

    def test_extracts_example(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "For example, the test suite passed all checks."
        )
        assert len(evidence) >= 1
        assert evidence[0]["type"] == "example"

    def test_extracts_study(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "Studies have shown that this approach scales well."
        )
        assert len(evidence) >= 1
        assert evidence[0]["type"] == "study"

    def test_multiple_evidence_types(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "According to the report, the data indicates improvement. "
            "For example, latency dropped by 30%."
        )
        types = {e["type"] for e in evidence}
        assert len(types) >= 2

    def test_limits_to_15(self):
        h = self._handler()
        content = ". ".join(
            [f"According to source {i}, finding {i}" for i in range(20)]
        )
        evidence = h._extract_evidence_from_content(content)
        assert len(evidence) <= 15

    def test_source_truncated_to_100(self):
        h = self._handler()
        long_source = "x" * 200
        evidence = h._extract_evidence_from_content(
            f"According to {long_source}, the result is clear."
        )
        if evidence and evidence[0]["source"]:
            assert len(evidence[0]["source"]) <= 100

    def test_text_truncated_to_300(self):
        h = self._handler()
        long_finding = "y" * 400
        evidence = h._extract_evidence_from_content(
            f"Research shows {long_finding}."
        )
        if evidence:
            assert len(evidence[0]["text"]) <= 300

    def test_no_evidence_in_plain_text(self):
        h = self._handler()
        evidence = h._extract_evidence_from_content(
            "The project deadline is next Friday."
        )
        assert len(evidence) == 0


# ===========================================================================
# Pattern Extraction Tests
# ===========================================================================


class TestPatternExtraction:
    """Test _extract_patterns_from_content logic."""

    def _handler(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        return InsightsHandler(server_context={})

    def test_balanced_comparison(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "On one hand, it is fast. On the other hand, it is expensive."
        )
        types = [p["type"] for p in patterns]
        assert "balanced_comparison" in types

    def test_concession_rebuttal(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "While the approach is novel, however it lacks evidence."
        )
        types = [p["type"] for p in patterns]
        assert "concession_rebuttal" in types

    def test_enumerated_argument(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "First, we reduce costs. Second, we improve quality."
        )
        types = [p["type"] for p in patterns]
        assert "enumerated_argument" in types

    def test_conditional_reasoning(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "If we invest now, then we will see returns later."
        )
        types = [p["type"] for p in patterns]
        assert "conditional_reasoning" in types

    def test_causal_reasoning_medium(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "This works because the algorithm is efficient."
        )
        causal = [p for p in patterns if p["type"] == "causal_reasoning"]
        assert len(causal) == 1
        assert causal[0]["strength"] == "medium"
        assert causal[0]["instances"] == 1

    def test_causal_reasoning_strong(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "It works because A. Also because B. And because C too."
        )
        causal = [p for p in patterns if p["type"] == "causal_reasoning"]
        assert len(causal) == 1
        assert causal[0]["strength"] == "strong"
        assert causal[0]["instances"] == 3

    def test_no_patterns(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content("Hello world.")
        assert len(patterns) == 0

    def test_multiple_patterns(self):
        h = self._handler()
        patterns = h._extract_patterns_from_content(
            "On one hand X. On the other hand Y. "
            "While A, however B. "
            "If P then Q. "
            "First step. Second step. "
            "Because of Z."
        )
        types = {p["type"] for p in patterns}
        assert len(types) >= 4


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting across endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_get(self, handler, mock_http):
        """GET should return 429 when rate limited."""
        from aragora.server.handlers.memory.insights import _insights_limiter

        with patch.object(_insights_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/insights/recent", {}, mock_http)
            assert _status(result) == 429
            body = _body(result)
            assert "rate limit" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_rate_limit_post(self, handler, mock_http_post):
        """POST should return 429 when rate limited."""
        from aragora.server.handlers.memory.insights import _insights_limiter

        http = mock_http_post({"content": "test content"})
        with patch.object(_insights_limiter, "is_allowed", return_value=False):
            result = await handler.handle_post(
                "/api/insights/extract-detailed", {}, http
            )
            assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_rate_limit_flips(self, handler, mock_http):
        """Flips endpoints should also be rate limited."""
        from aragora.server.handlers.memory.insights import _insights_limiter

        with patch.object(_insights_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/flips/recent", {}, mock_http)
            assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_rate_limit_summary(self, handler, mock_http):
        from aragora.server.handlers.memory.insights import _insights_limiter

        with patch.object(_insights_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/flips/summary", {}, mock_http)
            assert _status(result) == 429


# ===========================================================================
# RBAC / Auth Tests (opt out of auto-auth)
# ===========================================================================


class TestAuth:
    """Test authentication and authorization behavior."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_get_returns_401(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = InsightsHandler(server_context={"insight_store": MagicMock()})
        mock_http = MockHTTPHandler()

        with patch.object(
            SecureHandler, "get_auth_context", side_effect=UnauthorizedError("no token")
        ):
            result = await handler.handle("/api/insights/recent", {}, mock_http)
            assert _status(result) == 401
            body = _body(result)
            assert "authentication" in body.get("error", "").lower()

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_get_returns_403(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = InsightsHandler(server_context={"insight_store": MagicMock()})
        mock_http = MockHTTPHandler()

        mock_ctx = MagicMock()
        with patch.object(
            SecureHandler, "get_auth_context", return_value=mock_ctx
        ), patch.object(
            SecureHandler, "check_permission",
            side_effect=ForbiddenError("no permission"),
        ):
            result = await handler.handle("/api/insights/recent", {}, mock_http)
            assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_post_returns_401(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = InsightsHandler(server_context={})
        http = MockHTTPHandler(body={"content": "test"}, method="POST")

        with patch.object(
            SecureHandler, "get_auth_context", side_effect=UnauthorizedError("no token")
        ):
            result = await handler.handle_post(
                "/api/insights/extract-detailed", {}, http
            )
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_post_returns_403(self):
        from aragora.server.handlers.memory.insights import InsightsHandler
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = InsightsHandler(server_context={})
        http = MockHTTPHandler(body={"content": "test"}, method="POST")

        mock_ctx = MagicMock()
        with patch.object(
            SecureHandler, "get_auth_context", return_value=mock_ctx
        ), patch.object(
            SecureHandler, "check_permission",
            side_effect=ForbiddenError("no permission"),
        ):
            result = await handler.handle_post(
                "/api/insights/extract-detailed", {}, http
            )
            assert _status(result) == 403


# ===========================================================================
# Unmatched Route Tests
# ===========================================================================


class TestUnmatchedRoutes:
    """Test that unmatched routes return None."""

    @pytest.mark.asyncio
    async def test_get_unmatched_returns_none(self, handler, mock_http):
        result = await handler.handle("/api/insights/unknown-route", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_post_unmatched_returns_none(self, handler, mock_http_post):
        http = mock_http_post({"content": "test"})
        result = await handler.handle_post("/api/insights/something", {}, http)
        assert result is None


# ===========================================================================
# Handler Initialization Tests
# ===========================================================================


class TestInit:
    """Test InsightsHandler initialization."""

    def test_init_with_server_context(self):
        from aragora.server.handlers.memory.insights import InsightsHandler

        ctx = {"insight_store": MagicMock()}
        h = InsightsHandler(server_context=ctx)
        assert h.ctx is ctx

    def test_init_with_ctx_kwarg(self):
        from aragora.server.handlers.memory.insights import InsightsHandler

        ctx = {"insight_store": MagicMock()}
        h = InsightsHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_init_prefers_server_context(self):
        from aragora.server.handlers.memory.insights import InsightsHandler

        ctx1 = {"a": 1}
        ctx2 = {"b": 2}
        h = InsightsHandler(ctx=ctx1, server_context=ctx2)
        assert h.ctx is ctx2

    def test_init_no_args(self):
        from aragora.server.handlers.memory.insights import InsightsHandler

        h = InsightsHandler()
        assert h.ctx == {}

    def test_routes_list(self):
        from aragora.server.handlers.memory.insights import InsightsHandler

        assert "/api/insights/recent" in InsightsHandler.ROUTES
        assert "/api/insights/extract-detailed" in InsightsHandler.ROUTES
        assert "/api/flips/recent" in InsightsHandler.ROUTES
        assert "/api/flips/summary" in InsightsHandler.ROUTES


# ===========================================================================
# Invalid JSON Body for POST
# ===========================================================================


class TestInvalidBody:
    """Test POST with invalid/broken JSON body."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, handler):
        """When read_json_body returns None (bad JSON), we get 400."""
        http = MockHTTPHandler(method="POST")
        # Inject bad body bytes that fail JSON parse
        http.rfile.read.return_value = b"not json"
        http.headers["Content-Length"] = "8"

        result = await handler.handle_post(
            "/api/insights/extract-detailed", {}, http
        )
        assert _status(result) == 400
