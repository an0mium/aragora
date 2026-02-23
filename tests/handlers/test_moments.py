"""Tests for moments handler (aragora/server/handlers/moments.py).

Covers all routes and behavior of the MomentsHandler class:
- can_handle() routing for all static and dynamic routes
- GET /api/moments           - Alias for summary
- GET /api/moments/summary   - Global moments overview
- GET /api/moments/timeline  - Chronological moments with pagination
- GET /api/moments/recent    - Recent moments (timeline shortcut)
- GET /api/moments/trending  - Most significant recent moments
- GET /api/moments/by-type/{type} - Filter moments by type
- Rate limiting, error handling, edge cases
- Auth enforcement for non-GET methods
- Detector not available / not configured fallbacks
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.moments import MomentsHandler, VALID_MOMENT_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            return json.loads(raw.decode("utf-8"))
        return json.loads(raw)
    if isinstance(result, dict):
        return result
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    return 200


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to MomentsHandler.handle."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


class MockMoment:
    """Mock SignificantMoment object for testing."""

    def __init__(
        self,
        id: str = "m-001",
        moment_type: str = "upset_victory",
        agent_name: str = "claude",
        description: str = "Claude upset GPT-4 in debate",
        significance_score: float = 0.9,
        debate_id: str = "debate-1",
        other_agents: list[str] | None = None,
        metadata: dict | None = None,
        created_at: str | None = "2026-02-23T10:00:00",
    ):
        self.id = id
        self.moment_type = moment_type
        self.agent_name = agent_name
        self.description = description
        self.significance_score = significance_score
        self.debate_id = debate_id
        self.other_agents = other_agents or []
        self.metadata = metadata or {}
        self.created_at = created_at


def _make_moments(n: int = 5) -> list[MockMoment]:
    """Build a list of diverse mock moments."""
    types = list(VALID_MOMENT_TYPES)
    moments = []
    for i in range(n):
        moments.append(
            MockMoment(
                id=f"m-{i:03d}",
                moment_type=types[i % len(types)],
                agent_name=f"agent-{i % 3}",
                description=f"Moment {i} description",
                significance_score=round(0.5 + (i * 0.1), 2),
                debate_id=f"debate-{i}",
                other_agents=[f"agent-{(i + 1) % 3}"],
                metadata={"round": i},
                created_at=f"2026-02-23T{10 + i:02d}:00:00",
            )
        )
    return moments


def _make_detector(moments: list[MockMoment] | None = None) -> MagicMock:
    """Build a mock moment detector with an internal _moment_cache."""
    detector = MagicMock()
    cache: dict[str, list[MockMoment]] = {}
    if moments:
        for m in moments:
            cache.setdefault(m.agent_name, []).append(m)
    detector._moment_cache = cache
    return detector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter between tests."""
    from aragora.server.handlers.moments import _moments_limiter

    _moments_limiter._buckets.clear()
    yield
    _moments_limiter._buckets.clear()


@pytest.fixture
def handler():
    """Create a MomentsHandler with no context."""
    return MomentsHandler(ctx={})


@pytest.fixture
def handler_with_detector():
    """Create a MomentsHandler with a mock detector and moments."""
    moments = _make_moments(5)
    detector = _make_detector(moments)
    return MomentsHandler(ctx={"moment_detector": detector})


@pytest.fixture
def mock_http():
    """Factory for MockHTTPHandler."""

    def _create(method: str = "GET", body: dict | None = None) -> MockHTTPHandler:
        return MockHTTPHandler(method=method, body=body)

    return _create


# ===========================================================================
# Routing (can_handle)
# ===========================================================================


class TestCanHandle:
    """Test the can_handle routing logic."""

    def test_can_handle_moments_root(self, handler):
        assert handler.can_handle("/api/moments") is True

    def test_can_handle_v1_moments_root(self, handler):
        assert handler.can_handle("/api/v1/moments") is True

    def test_can_handle_summary(self, handler):
        assert handler.can_handle("/api/moments/summary") is True

    def test_can_handle_v1_summary(self, handler):
        assert handler.can_handle("/api/v1/moments/summary") is True

    def test_can_handle_timeline(self, handler):
        assert handler.can_handle("/api/moments/timeline") is True

    def test_can_handle_trending(self, handler):
        assert handler.can_handle("/api/moments/trending") is True

    def test_can_handle_recent(self, handler):
        assert handler.can_handle("/api/moments/recent") is True

    def test_can_handle_v1_recent(self, handler):
        assert handler.can_handle("/api/v1/moments/recent") is True

    def test_can_handle_by_type_with_param(self, handler):
        assert handler.can_handle("/api/moments/by-type/upset_victory") is True

    def test_can_handle_v1_by_type(self, handler):
        assert handler.can_handle("/api/v1/moments/by-type/consensus_breakthrough") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/other") is False

    def test_cannot_handle_moments_subpath_unknown(self, handler):
        assert handler.can_handle("/api/moments/unknown_sub") is False


# ===========================================================================
# Summary endpoint
# ===========================================================================


class TestSummary:
    """Test GET /api/moments/summary and /api/moments."""

    @pytest.mark.asyncio
    async def test_summary_no_detector_available(self, handler, mock_http):
        """When MOMENT_DETECTOR_AVAILABLE is False, returns empty summary."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False):
            result = await handler.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_moments"] == 0
        assert body["message"] == "Moment detection not available"

    @pytest.mark.asyncio
    async def test_summary_no_detector_configured(self, handler, mock_http):
        """When detector is None in context, returns not-configured response."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_moments"] == 0
        assert body["message"] == "Moment detector not configured"

    @pytest.mark.asyncio
    async def test_summary_with_moments(self, handler_with_detector, mock_http):
        """Summary returns aggregated data from all moments."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_moments"] == 5
        assert isinstance(body["by_type"], dict)
        assert isinstance(body["by_agent"], dict)
        assert body["most_significant"] is not None
        # Most significant should be the one with highest score
        assert isinstance(body["recent"], list)
        assert len(body["recent"]) == 5  # all 5 moments are "recent"

    @pytest.mark.asyncio
    async def test_summary_via_moments_root(self, handler_with_detector, mock_http):
        """GET /api/moments is aliased to summary."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_moments"] == 5

    @pytest.mark.asyncio
    async def test_summary_most_significant(self, mock_http):
        """Verify that most_significant is the highest significance_score moment."""
        m1 = MockMoment(id="m-lo", significance_score=0.1)
        m2 = MockMoment(id="m-hi", significance_score=0.99)
        det = _make_detector([m1, m2])
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert body["most_significant"]["id"] == "m-hi"
        assert body["most_significant"]["significance"] == 0.99

    @pytest.mark.asyncio
    async def test_summary_by_type_counts(self, mock_http):
        """Verify by_type counts are correct."""
        moments = [
            MockMoment(id="a", moment_type="upset_victory"),
            MockMoment(id="b", moment_type="upset_victory"),
            MockMoment(id="c", moment_type="alliance_shift"),
        ]
        det = _make_detector(moments)
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert body["by_type"]["upset_victory"] == 2
        assert body["by_type"]["alliance_shift"] == 1

    @pytest.mark.asyncio
    async def test_summary_by_agent_counts(self, mock_http):
        """Verify by_agent counts are correct."""
        moments = [
            MockMoment(id="a", agent_name="claude"),
            MockMoment(id="b", agent_name="claude"),
            MockMoment(id="c", agent_name="gpt4"),
        ]
        det = _make_detector(moments)
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert body["by_agent"]["claude"] == 2
        assert body["by_agent"]["gpt4"] == 1

    @pytest.mark.asyncio
    async def test_summary_empty_cache(self, mock_http):
        """Detector present but empty cache returns empty summary."""
        det = _make_detector([])
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        body = _body(result)
        assert body["total_moments"] == 0
        assert body["most_significant"] is None
        assert body["recent"] == []

    @pytest.mark.asyncio
    async def test_summary_data_error(self, mock_http):
        """AttributeError in moments iteration returns 400."""
        det = MagicMock()
        # _moment_cache contains a broken moment that raises AttributeError
        bad_moment = MagicMock()
        bad_moment.moment_type = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no type"))
        )
        del bad_moment.moment_type
        det._moment_cache = {"agent": [bad_moment]}
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        assert _status(result) in (400, 500)

    @pytest.mark.asyncio
    async def test_summary_runtime_error(self, mock_http):
        """RuntimeError in get_all_moments returns 500."""
        h_obj = MomentsHandler(ctx={"moment_detector": MagicMock()})
        h = mock_http()
        with (
            patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True),
            patch.object(h_obj, "_get_all_moments", side_effect=RuntimeError("boom")),
        ):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        assert _status(result) == 500


# ===========================================================================
# Timeline endpoint
# ===========================================================================


class TestTimeline:
    """Test GET /api/moments/timeline."""

    @pytest.mark.asyncio
    async def test_timeline_not_available(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False):
            result = await handler.handle("/api/moments/timeline", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["moments"] == []
        assert body["message"] == "Moment detection not available"

    @pytest.mark.asyncio
    async def test_timeline_no_detector(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler.handle("/api/moments/timeline", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["moments"] == []
        assert body["message"] == "Moment detector not configured"

    @pytest.mark.asyncio
    async def test_timeline_default_params(self, handler_with_detector, mock_http):
        """Default limit=50, offset=0."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/timeline", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 5
        assert body["limit"] == 50
        assert body["offset"] == 0
        assert body["has_more"] is False
        assert len(body["moments"]) == 5

    @pytest.mark.asyncio
    async def test_timeline_pagination(self, handler_with_detector, mock_http):
        """Limit and offset are respected."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle(
                "/api/moments/timeline", {"limit": "2", "offset": "0"}, h
            )
        body = _body(result)
        assert len(body["moments"]) == 2
        assert body["has_more"] is True

    @pytest.mark.asyncio
    async def test_timeline_offset_past_end(self, handler_with_detector, mock_http):
        """Offset beyond total returns empty page."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle(
                "/api/moments/timeline", {"limit": "10", "offset": "100"}, h
            )
        body = _body(result)
        assert len(body["moments"]) == 0
        assert body["has_more"] is False

    @pytest.mark.asyncio
    async def test_timeline_limit_clamped(self, handler_with_detector, mock_http):
        """Limit above 200 is clamped to 200; below 1 is clamped to 1."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle(
                "/api/moments/timeline", {"limit": "999"}, h
            )
        body = _body(result)
        assert body["limit"] == 200

    @pytest.mark.asyncio
    async def test_timeline_limit_min_clamped(self, handler_with_detector, mock_http):
        """Limit of 0 or negative is clamped to 1."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/timeline", {"limit": "0"}, h)
        body = _body(result)
        assert body["limit"] == 1

    @pytest.mark.asyncio
    async def test_timeline_sorted_by_created_at(self, mock_http):
        """Moments are returned in reverse chronological order."""
        m1 = MockMoment(id="old", created_at="2026-01-01T00:00:00")
        m2 = MockMoment(id="new", created_at="2026-02-01T00:00:00")
        det = _make_detector([m1, m2])
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/timeline", {}, h)
        body = _body(result)
        assert body["moments"][0]["id"] == "new"
        assert body["moments"][1]["id"] == "old"

    @pytest.mark.asyncio
    async def test_timeline_runtime_error(self, mock_http):
        """RuntimeError returns 500."""
        h_obj = MomentsHandler(ctx={"moment_detector": MagicMock()})
        h = mock_http()
        with (
            patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True),
            patch.object(h_obj, "_get_all_moments", side_effect=ValueError("bad")),
        ):
            result = await h_obj.handle("/api/moments/timeline", {}, h)
        assert _status(result) == 500


# ===========================================================================
# Recent endpoint
# ===========================================================================


class TestRecent:
    """Test GET /api/moments/recent."""

    @pytest.mark.asyncio
    async def test_recent_default_limit(self, handler_with_detector, mock_http):
        """Recent uses default limit=20, offset=0."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/recent", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["limit"] == 20
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_recent_custom_limit(self, handler_with_detector, mock_http):
        """Custom limit is respected."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/recent", {"limit": "3"}, h)
        body = _body(result)
        assert body["limit"] == 3
        assert len(body["moments"]) == 3

    @pytest.mark.asyncio
    async def test_recent_v1_path(self, handler_with_detector, mock_http):
        """GET /api/v1/moments/recent works."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/v1/moments/recent", {}, h)
        body = _body(result)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_recent_not_available(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False):
            result = await handler.handle("/api/moments/recent", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["message"] == "Moment detection not available"


# ===========================================================================
# Trending endpoint
# ===========================================================================


class TestTrending:
    """Test GET /api/moments/trending."""

    @pytest.mark.asyncio
    async def test_trending_not_available(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False):
            result = await handler.handle("/api/moments/trending", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["trending"] == []
        assert body["message"] == "Moment detection not available"

    @pytest.mark.asyncio
    async def test_trending_no_detector(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler.handle("/api/moments/trending", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["trending"] == []
        assert body["message"] == "Moment detector not configured"

    @pytest.mark.asyncio
    async def test_trending_default_limit(self, handler_with_detector, mock_http):
        """Trending returns top 10 by default."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/trending", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert len(body["trending"]) == 5  # only 5 moments total

    @pytest.mark.asyncio
    async def test_trending_custom_limit(self, handler_with_detector, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/trending", {"limit": "2"}, h)
        body = _body(result)
        assert len(body["trending"]) == 2
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_trending_sorted_by_significance(self, mock_http):
        """Trending is sorted by significance_score descending."""
        m1 = MockMoment(id="lo", significance_score=0.1)
        m2 = MockMoment(id="hi", significance_score=0.95)
        m3 = MockMoment(id="mid", significance_score=0.5)
        det = _make_detector([m1, m2, m3])
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/trending", {}, h)
        body = _body(result)
        scores = [t["significance"] for t in body["trending"]]
        assert scores == sorted(scores, reverse=True)
        assert body["trending"][0]["id"] == "hi"

    @pytest.mark.asyncio
    async def test_trending_limit_clamped_to_50(self, handler_with_detector, mock_http):
        """Limit above 50 is clamped."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle(
                "/api/moments/trending", {"limit": "999"}, h
            )
        body = _body(result)
        # max(1, min(999, 50)) = 50, but only 5 moments exist
        assert len(body["trending"]) == 5

    @pytest.mark.asyncio
    async def test_trending_runtime_error(self, mock_http):
        h_obj = MomentsHandler(ctx={"moment_detector": MagicMock()})
        h = mock_http()
        with (
            patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True),
            patch.object(h_obj, "_get_all_moments", side_effect=RuntimeError("fail")),
        ):
            result = await h_obj.handle("/api/moments/trending", {}, h)
        assert _status(result) == 500


# ===========================================================================
# By-type endpoint
# ===========================================================================


class TestByType:
    """Test GET /api/moments/by-type/{type}."""

    @pytest.mark.asyncio
    async def test_by_type_not_available(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False):
            result = await handler.handle("/api/moments/by-type/upset_victory", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["moments"] == []
        assert body["type"] == "upset_victory"
        assert body["message"] == "Moment detection not available"

    @pytest.mark.asyncio
    async def test_by_type_no_detector(self, handler, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler.handle("/api/moments/by-type/upset_victory", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["moments"] == []
        assert body["message"] == "Moment detector not configured"

    @pytest.mark.asyncio
    async def test_by_type_filters_correctly(self, mock_http):
        """Only moments of the requested type are returned."""
        moments = [
            MockMoment(id="a", moment_type="upset_victory", significance_score=0.8),
            MockMoment(id="b", moment_type="alliance_shift", significance_score=0.9),
            MockMoment(id="c", moment_type="upset_victory", significance_score=0.5),
        ]
        det = _make_detector(moments)
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/by-type/upset_victory", {}, h)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 2
        assert all(m["type"] == "upset_victory" for m in body["moments"])

    @pytest.mark.asyncio
    async def test_by_type_sorted_by_significance(self, mock_http):
        """By-type results are sorted by significance descending."""
        moments = [
            MockMoment(id="lo", moment_type="upset_victory", significance_score=0.2),
            MockMoment(id="hi", moment_type="upset_victory", significance_score=0.9),
        ]
        det = _make_detector(moments)
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/by-type/upset_victory", {}, h)
        body = _body(result)
        assert body["moments"][0]["id"] == "hi"
        assert body["moments"][1]["id"] == "lo"

    @pytest.mark.asyncio
    async def test_by_type_invalid_type(self, handler, mock_http):
        """Invalid moment type returns 400 with valid types listed."""
        h = mock_http()
        result = await handler.handle("/api/moments/by-type/not_a_type", {}, h)
        body = _body(result)
        assert _status(result) == 400
        assert "Invalid moment type" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_by_type_all_valid_types_accepted(self, handler_with_detector, mock_http):
        """All entries in VALID_MOMENT_TYPES can be queried without error."""
        for mtype in VALID_MOMENT_TYPES:
            h = mock_http()
            with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
                result = await handler_with_detector.handle(f"/api/moments/by-type/{mtype}", {}, h)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_by_type_custom_limit(self, mock_http):
        """Limit query param is respected."""
        moments = [
            MockMoment(id=f"m-{i}", moment_type="upset_victory", significance_score=0.5 + i * 0.01)
            for i in range(10)
        ]
        det = _make_detector(moments)
        h_obj = MomentsHandler(ctx={"moment_detector": det})
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await h_obj.handle("/api/moments/by-type/upset_victory", {"limit": "3"}, h)
        body = _body(result)
        assert len(body["moments"]) == 3
        assert body["total"] == 10
        assert body["limit"] == 3

    @pytest.mark.asyncio
    async def test_by_type_v1_path(self, handler_with_detector, mock_http):
        """v1 path works for by-type."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle(
                "/api/v1/moments/by-type/upset_victory", {}, h
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_by_type_runtime_error(self, mock_http):
        h_obj = MomentsHandler(ctx={"moment_detector": MagicMock()})
        h = mock_http()
        with (
            patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True),
            patch.object(h_obj, "_get_all_moments", side_effect=RuntimeError("fail")),
        ):
            result = await h_obj.handle("/api/moments/by-type/upset_victory", {}, h)
        assert _status(result) == 500


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on moments endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler, mock_http):
        """When rate limiter denies, return 429."""
        h = mock_http()
        with patch("aragora.server.handlers.moments._moments_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle("/api/moments/summary", {}, h)
        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, handler_with_detector, mock_http):
        """Normal requests pass through rate limiter."""
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/summary", {}, h)
        assert _status(result) == 200


# ===========================================================================
# Auth enforcement for non-GET methods
# ===========================================================================


class TestAuthEnforcement:
    """Test that non-GET methods require authentication."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_non_get_requires_auth(self, handler, mock_http):
        """POST requests require authentication."""
        h = mock_http(method="POST")
        with patch("aragora.server.handlers.moments._moments_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = await handler.handle("/api/moments/summary", {}, h)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_get_does_not_require_auth(self, handler_with_detector, mock_http):
        """GET requests skip auth (public dashboard data)."""
        h = mock_http(method="GET")
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/moments/summary", {}, h)
        assert _status(result) == 200


# ===========================================================================
# Moment serialization
# ===========================================================================


class TestMomentToDict:
    """Test the _moment_to_dict helper."""

    def test_moment_to_dict_all_fields(self, handler):
        """All fields are serialized correctly."""
        m = MockMoment(
            id="m-x",
            moment_type="streak_achievement",
            agent_name="gemini",
            description="5-debate streak",
            significance_score=0.88,
            debate_id="d-42",
            other_agents=["claude", "gpt4"],
            metadata={"streak": 5},
            created_at="2026-02-20T12:00:00",
        )
        d = handler._moment_to_dict(m)
        assert d["id"] == "m-x"
        assert d["type"] == "streak_achievement"
        assert d["agent"] == "gemini"
        assert d["description"] == "5-debate streak"
        assert d["significance"] == 0.88
        assert d["debate_id"] == "d-42"
        assert d["other_agents"] == ["claude", "gpt4"]
        assert d["metadata"] == {"streak": 5}
        assert d["created_at"] == "2026-02-20T12:00:00"

    def test_moment_to_dict_no_created_at(self, handler):
        """Moments without created_at return None for that field."""
        m = MockMoment()
        delattr(m, "created_at")
        d = handler._moment_to_dict(m)
        assert d["created_at"] is None

    def test_moment_to_dict_empty_lists(self, handler):
        """Null other_agents/metadata default to []/{} respectively."""
        m = MockMoment(other_agents=None, metadata=None)
        d = handler._moment_to_dict(m)
        assert d["other_agents"] == []
        assert d["metadata"] == {}


# ===========================================================================
# Internal helpers
# ===========================================================================


class TestInternalHelpers:
    """Test _get_moment_detector and _get_all_moments."""

    def test_get_moment_detector_missing(self, handler):
        assert handler._get_moment_detector() is None

    def test_get_moment_detector_present(self, handler_with_detector):
        assert handler_with_detector._get_moment_detector() is not None

    def test_get_all_moments_no_detector(self, handler):
        assert handler._get_all_moments() == []

    def test_get_all_moments_no_cache(self):
        """Detector without _moment_cache returns empty."""
        det = MagicMock(spec=[])  # no _moment_cache attribute
        h = MomentsHandler(ctx={"moment_detector": det})
        assert h._get_all_moments() == []

    def test_get_all_moments_with_data(self, handler_with_detector):
        moments = handler_with_detector._get_all_moments()
        assert len(moments) == 5


# ===========================================================================
# Unhandled path returns None
# ===========================================================================


class TestUnhandledPath:
    """Test that paths not matching any route return None."""

    @pytest.mark.asyncio
    async def test_unmatched_path_returns_none(self, handler, mock_http):
        h = mock_http()
        result = await handler.handle("/api/moments/unknown", {}, h)
        assert result is None


# ===========================================================================
# VALID_MOMENT_TYPES
# ===========================================================================


class TestValidMomentTypes:
    """Verify the VALID_MOMENT_TYPES constant."""

    def test_expected_types_present(self):
        expected = {
            "upset_victory",
            "position_reversal",
            "calibration_vindication",
            "alliance_shift",
            "consensus_breakthrough",
            "streak_achievement",
            "domain_mastery",
        }
        assert VALID_MOMENT_TYPES == expected

    def test_types_are_set(self):
        assert isinstance(VALID_MOMENT_TYPES, set)


# ===========================================================================
# Permission denied for non-GET
# ===========================================================================


class TestPermissionDenied:
    """Test that permission denied returns 403 for non-GET."""

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, mock_http):
        """When check_permission raises ForbiddenError, return 403."""
        from aragora.server.handlers.utils.auth import ForbiddenError

        h_obj = MomentsHandler(ctx={})
        h = mock_http(method="PUT")

        async def mock_get_auth(*args, **kwargs):
            return MagicMock()

        with (
            patch.object(h_obj, "get_auth_context", side_effect=mock_get_auth),
            patch.object(h_obj, "check_permission", side_effect=ForbiddenError("denied")),
        ):
            result = await h_obj.handle("/api/moments/summary", {}, h)
        assert _status(result) == 403


# ===========================================================================
# Version prefix handling
# ===========================================================================


class TestVersionPrefixHandling:
    """Test that /api/v1/... paths are normalized correctly."""

    @pytest.mark.asyncio
    async def test_v1_summary(self, handler_with_detector, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/v1/moments/summary", {}, h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_timeline(self, handler_with_detector, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/v1/moments/timeline", {}, h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_trending(self, handler_with_detector, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle("/api/v1/moments/trending", {}, h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_by_type(self, handler_with_detector, mock_http):
        h = mock_http()
        with patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True):
            result = await handler_with_detector.handle(
                "/api/v1/moments/by-type/upset_victory", {}, h
            )
        assert _status(result) == 200
