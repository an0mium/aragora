"""
Tests for aragora.server.handlers.moments - Moments HTTP Handlers.

Tests cover:
- MomentsHandler: instantiation, RESOURCE_TYPE, ROUTES, can_handle
- GET /api/moments/summary: no detector, with moments, empty moments
- GET /api/moments/timeline: no detector, pagination, empty
- GET /api/moments/trending: no detector, with moments, limit
- GET /api/moments/by-type/{type}: valid type, invalid type, no detector
- RBAC: unauthorized, forbidden
- handle() routing: returns None for unmatched paths
- VALID_MOMENT_TYPES: contains expected types
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.moments import (
    MomentsHandler,
    VALID_MOMENT_TYPES,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
        "Authorization": "Bearer test-token",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Objects
# ===========================================================================


class MockMoment:
    """Mock SignificantMoment object."""

    def __init__(
        self,
        moment_id: str = "mom-001",
        moment_type: str = "upset_victory",
        agent_name: str = "claude",
        significance: float = 0.85,
        created_at: str = "2026-02-14T10:00:00Z",
    ):
        self.id = moment_id
        self.moment_type = moment_type
        self.agent_name = agent_name
        self.description = f"Agent {agent_name} had a {moment_type}"
        self.significance_score = significance
        self.debate_id = "debate-001"
        self.other_agents = ["gpt4", "gemini"]
        self.metadata = {"round": 3}
        self.created_at = created_at


class MockMomentDetector:
    """Mock MomentDetector with cache."""

    def __init__(self, moments: dict[str, list] | None = None):
        self._moment_cache = moments or {}


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a MomentsHandler with mocked dependencies."""
    h = MomentsHandler(server_context={})
    return h


@pytest.fixture
def handler_with_detector():
    """Create a MomentsHandler with a populated detector."""
    moments = {
        "claude": [
            MockMoment("mom-001", "upset_victory", "claude", 0.85),
            MockMoment("mom-002", "consensus_breakthrough", "claude", 0.92),
        ],
        "gpt4": [
            MockMoment("mom-003", "upset_victory", "gpt4", 0.70),
            MockMoment("mom-004", "alliance_shift", "gpt4", 0.60),
        ],
    }
    detector = MockMomentDetector(moments)
    h = MomentsHandler(server_context={"moment_detector": detector})
    return h


@pytest.fixture
def handler_empty_detector():
    """Create a MomentsHandler with an empty detector."""
    detector = MockMomentDetector({})
    h = MomentsHandler(server_context={"moment_detector": detector})
    return h


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestMomentsHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, MomentsHandler)

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "moments"

    def test_has_routes(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0

    def test_routes_contain_summary(self, handler):
        assert "/api/moments/summary" in handler.ROUTES

    def test_routes_contain_timeline(self, handler):
        assert "/api/moments/timeline" in handler.ROUTES

    def test_routes_contain_trending(self, handler):
        assert "/api/moments/trending" in handler.ROUTES

    def test_routes_contain_by_type(self, handler):
        assert "/api/moments/by-type/*" in handler.ROUTES


# ===========================================================================
# Test can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing logic."""

    def test_can_handle_summary(self, handler):
        assert handler.can_handle("/api/moments/summary") is True

    def test_can_handle_timeline(self, handler):
        assert handler.can_handle("/api/moments/timeline") is True

    def test_can_handle_trending(self, handler):
        assert handler.can_handle("/api/moments/trending") is True

    def test_can_handle_by_type(self, handler):
        assert handler.can_handle("/api/moments/by-type/upset_victory") is True

    def test_can_handle_versioned_summary(self, handler):
        assert handler.can_handle("/api/v1/moments/summary") is True

    def test_can_handle_versioned_by_type(self, handler):
        assert handler.can_handle("/api/v1/moments/by-type/alliance_shift") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_moments_root(self, handler):
        assert handler.can_handle("/api/moments") is False


# ===========================================================================
# Test VALID_MOMENT_TYPES
# ===========================================================================


class TestValidMomentTypes:
    """Tests for moment type constants."""

    def test_contains_upset_victory(self):
        assert "upset_victory" in VALID_MOMENT_TYPES

    def test_contains_position_reversal(self):
        assert "position_reversal" in VALID_MOMENT_TYPES

    def test_contains_consensus_breakthrough(self):
        assert "consensus_breakthrough" in VALID_MOMENT_TYPES

    def test_contains_alliance_shift(self):
        assert "alliance_shift" in VALID_MOMENT_TYPES

    def test_contains_streak_achievement(self):
        assert "streak_achievement" in VALID_MOMENT_TYPES

    def test_contains_domain_mastery(self):
        assert "domain_mastery" in VALID_MOMENT_TYPES

    def test_contains_calibration_vindication(self):
        assert "calibration_vindication" in VALID_MOMENT_TYPES


# ===========================================================================
# Test _get_summary
# ===========================================================================


class TestGetSummary:
    """Tests for the summary endpoint."""

    def test_summary_no_detector(self, handler):
        result = handler._get_summary()
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["total_moments"] == 0

    def test_summary_with_moments(self, handler_with_detector):
        result = handler_with_detector._get_summary()
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["total_moments"] == 4
        assert "by_type" in data
        assert "by_agent" in data
        assert data["most_significant"] is not None
        # Most significant should be mom-002 with score 0.92
        assert data["most_significant"]["significance"] == 0.92

    def test_summary_by_type_counts(self, handler_with_detector):
        result = handler_with_detector._get_summary()
        data = _parse_body(result)
        assert data["by_type"]["upset_victory"] == 2
        assert data["by_type"]["consensus_breakthrough"] == 1
        assert data["by_type"]["alliance_shift"] == 1

    def test_summary_by_agent_counts(self, handler_with_detector):
        result = handler_with_detector._get_summary()
        data = _parse_body(result)
        assert data["by_agent"]["claude"] == 2
        assert data["by_agent"]["gpt4"] == 2

    def test_summary_recent_limited(self, handler_with_detector):
        result = handler_with_detector._get_summary()
        data = _parse_body(result)
        assert len(data["recent"]) <= 5

    def test_summary_empty_detector(self, handler_empty_detector):
        result = handler_empty_detector._get_summary()
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["total_moments"] == 0
        assert data["most_significant"] is None


# ===========================================================================
# Test _get_timeline
# ===========================================================================


class TestGetTimeline:
    """Tests for the timeline endpoint."""

    def test_timeline_no_detector(self, handler):
        result = handler._get_timeline(50, 0)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["moments"] == []
        assert data["total"] == 0

    def test_timeline_with_moments(self, handler_with_detector):
        result = handler_with_detector._get_timeline(50, 0)
        assert result.status_code == 200
        data = _parse_body(result)
        assert len(data["moments"]) == 4
        assert data["total"] == 4
        assert data["has_more"] is False

    def test_timeline_pagination(self, handler_with_detector):
        result = handler_with_detector._get_timeline(2, 0)
        data = _parse_body(result)
        assert len(data["moments"]) == 2
        assert data["has_more"] is True
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_timeline_offset(self, handler_with_detector):
        result = handler_with_detector._get_timeline(2, 2)
        data = _parse_body(result)
        assert len(data["moments"]) == 2
        assert data["offset"] == 2

    def test_timeline_empty_detector(self, handler_empty_detector):
        result = handler_empty_detector._get_timeline(50, 0)
        data = _parse_body(result)
        assert data["moments"] == []
        assert data["has_more"] is False


# ===========================================================================
# Test _get_trending
# ===========================================================================


class TestGetTrending:
    """Tests for the trending endpoint."""

    def test_trending_no_detector(self, handler):
        result = handler._get_trending(10)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["trending"] == []
        assert data["count"] == 0

    def test_trending_with_moments(self, handler_with_detector):
        result = handler_with_detector._get_trending(10)
        assert result.status_code == 200
        data = _parse_body(result)
        assert len(data["trending"]) == 4
        # First should be the most significant
        assert data["trending"][0]["significance"] == 0.92

    def test_trending_limited(self, handler_with_detector):
        result = handler_with_detector._get_trending(2)
        data = _parse_body(result)
        assert len(data["trending"]) == 2

    def test_trending_empty_detector(self, handler_empty_detector):
        result = handler_empty_detector._get_trending(10)
        data = _parse_body(result)
        assert data["trending"] == []


# ===========================================================================
# Test _get_by_type
# ===========================================================================


class TestGetByType:
    """Tests for the by-type endpoint."""

    def test_by_type_no_detector(self, handler):
        result = handler._get_by_type("upset_victory", 50)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["type"] == "upset_victory"
        assert data["moments"] == []

    def test_by_type_with_moments(self, handler_with_detector):
        result = handler_with_detector._get_by_type("upset_victory", 50)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["type"] == "upset_victory"
        assert len(data["moments"]) == 2
        assert data["total"] == 2

    def test_by_type_limited(self, handler_with_detector):
        result = handler_with_detector._get_by_type("upset_victory", 1)
        data = _parse_body(result)
        assert len(data["moments"]) == 1

    def test_by_type_no_matches(self, handler_with_detector):
        result = handler_with_detector._get_by_type("streak_achievement", 50)
        data = _parse_body(result)
        assert data["moments"] == []
        assert data["total"] == 0

    def test_by_type_empty_detector(self, handler_empty_detector):
        result = handler_empty_detector._get_by_type("upset_victory", 50)
        data = _parse_body(result)
        assert data["moments"] == []


# ===========================================================================
# Test handle() Routing with RBAC
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    @pytest.mark.asyncio
    async def test_handle_unauthorized(self, handler):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        mock_handler = _make_mock_handler()
        with patch.object(
            handler, "get_auth_context", side_effect=UnauthorizedError()
        ):
            result = await handler.handle("/api/moments/summary", {}, mock_handler)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_handle_forbidden(self, handler):
        from aragora.server.handlers.utils.auth import ForbiddenError

        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("Denied"),
            ):
                result = await handler.handle("/api/moments/summary", {}, mock_handler)
                assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_handle_summary_authorized(self, handler):
        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/moments/summary", {}, mock_handler
                    )
                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_timeline_authorized(self, handler):
        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/moments/timeline", {"limit": "10", "offset": "0"}, mock_handler
                    )
                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_trending_authorized(self, handler):
        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/moments/trending", {}, mock_handler
                    )
                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_by_type_invalid(self, handler):
        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/moments/by-type/invalid_type", {}, mock_handler
                    )
                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_rate_limited(self, handler):
        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = False
                    result = await handler.handle(
                        "/api/moments/summary", {}, mock_handler
                    )
                    assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        auth_ctx = MagicMock()
        with patch.object(handler, "get_auth_context", return_value=auth_ctx):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True
                    result = await handler.handle(
                        "/api/moments/unknown", {}, mock_handler
                    )
                    assert result is None


# ===========================================================================
# Test _moment_to_dict
# ===========================================================================


class TestMomentToDict:
    """Tests for moment serialization."""

    def test_moment_to_dict(self, handler):
        moment = MockMoment()
        d = handler._moment_to_dict(moment)
        assert d["id"] == "mom-001"
        assert d["type"] == "upset_victory"
        assert d["agent"] == "claude"
        assert d["significance"] == 0.85
        assert d["debate_id"] == "debate-001"
        assert d["other_agents"] == ["gpt4", "gemini"]
        assert d["metadata"] == {"round": 3}

    def test_moment_to_dict_fields(self, handler):
        moment = MockMoment()
        d = handler._moment_to_dict(moment)
        expected_keys = {
            "id", "type", "agent", "description", "significance",
            "debate_id", "other_agents", "metadata", "created_at",
        }
        assert set(d.keys()) == expected_keys
