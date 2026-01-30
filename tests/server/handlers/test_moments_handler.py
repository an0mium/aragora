"""
Tests for aragora.server.handlers.moments - Moments endpoint handler.

Tests cover:
- MomentsHandler initialization
- can_handle() route matching for all endpoints
- handle() with RBAC verification
- _get_summary() - Global moments overview
- _get_timeline() - Chronological moments with pagination
- _get_trending() - Most significant recent moments
- _get_by_type() - Filter moments by type
- Rate limiting
- Error handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.moments import (
    MomentsHandler,
    VALID_MOMENT_TYPES,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authentication context for testing."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    org_id: str = "org-123"
    role: str = "admin"
    permissions: set = field(default_factory=lambda: {"moments:read", "*"})

    @property
    def authenticated(self) -> bool:
        return self.is_authenticated


@dataclass
class MockHandler:
    """Mock HTTP handler for testing."""

    headers: dict = field(default_factory=dict)
    client_address: tuple = ("127.0.0.1", 12345)
    command: str = "GET"


@dataclass
class MockSignificantMoment:
    """Mock significant moment for testing."""

    id: str = "moment-1"
    moment_type: str = "consensus_breakthrough"
    agent_name: str = "claude"
    description: str = "Achieved strong consensus on policy decision."
    significance_score: float = 0.85
    debate_id: str = "debate-123"
    other_agents: list = field(default_factory=lambda: ["gpt4", "gemini"])
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MockMomentDetector:
    """Mock moment detector for testing."""

    def __init__(self, moments: list = None):
        self._moment_cache = {
            "claude": moments or [],
        }


def create_handler(ctx: dict = None) -> MomentsHandler:
    """Create a MomentsHandler with context."""
    return MomentsHandler(ctx or {})


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body as dict from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    return result[0] if isinstance(result[0], dict) else json.loads(result[0])


# ===========================================================================
# Tests for VALID_MOMENT_TYPES
# ===========================================================================


class TestValidMomentTypes:
    """Tests for valid moment type definitions."""

    def test_valid_moment_types_defined(self):
        """Should have all expected moment types defined."""
        assert "upset_victory" in VALID_MOMENT_TYPES
        assert "position_reversal" in VALID_MOMENT_TYPES
        assert "calibration_vindication" in VALID_MOMENT_TYPES
        assert "alliance_shift" in VALID_MOMENT_TYPES
        assert "consensus_breakthrough" in VALID_MOMENT_TYPES
        assert "streak_achievement" in VALID_MOMENT_TYPES
        assert "domain_mastery" in VALID_MOMENT_TYPES

    def test_valid_moment_types_is_frozenset(self):
        """Valid moment types should be immutable."""
        # Should be a set (actually frozenset for immutability)
        assert isinstance(VALID_MOMENT_TYPES, (set, frozenset))


# ===========================================================================
# Tests for can_handle() Route Matching
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_summary_endpoint(self):
        """Should handle /api/moments/summary."""
        handler = create_handler()
        assert handler.can_handle("/api/moments/summary") is True

    def test_handles_timeline_endpoint(self):
        """Should handle /api/moments/timeline."""
        handler = create_handler()
        assert handler.can_handle("/api/moments/timeline") is True

    def test_handles_trending_endpoint(self):
        """Should handle /api/moments/trending."""
        handler = create_handler()
        assert handler.can_handle("/api/moments/trending") is True

    def test_handles_by_type_endpoint(self):
        """Should handle /api/moments/by-type/{type}."""
        handler = create_handler()
        assert handler.can_handle("/api/moments/by-type/upset_victory") is True
        assert handler.can_handle("/api/moments/by-type/consensus_breakthrough") is True

    def test_handles_v1_routes(self):
        """Should handle v1 prefixed routes."""
        handler = create_handler()
        # The handler uses strip_version_prefix, so v1 paths should work
        assert handler.can_handle("/api/v1/moments/summary") is True
        assert handler.can_handle("/api/v1/moments/timeline") is True

    def test_rejects_unrelated_paths(self):
        """Should reject unrelated API paths."""
        handler = create_handler()
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/users") is False
        assert handler.can_handle("/api/health") is False


# ===========================================================================
# Tests for handle() with RBAC
# ===========================================================================


class TestHandle:
    """Tests for handle() method with RBAC verification."""

    @pytest.mark.asyncio
    async def test_handle_requires_authentication(self):
        """Should return 401 when not authenticated."""
        handler = create_handler()
        mock_http = MockHandler()

        from aragora.server.handlers.secure import UnauthorizedError

        with patch.object(
            handler,
            "get_auth_context",
            side_effect=UnauthorizedError("Not authenticated"),
        ):
            result = await handler.handle("/api/moments/summary", {}, mock_http)

        assert get_status(result) == 401
        body = get_body(result)
        assert "Authentication required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_requires_moments_read_permission(self):
        """Should return 403 when user lacks moments:read permission."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext(permissions=set())

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            from aragora.server.handlers.utils.auth import ForbiddenError

            with patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ):
                result = await handler.handle("/api/moments/summary", {}, mock_http)

        assert get_status(result) == 403

    @pytest.mark.asyncio
    async def test_handle_rate_limit_exceeded(self):
        """Should return 429 when rate limit is exceeded."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter.is_allowed",
                    return_value=False,
                ):
                    result = await handler.handle("/api/moments/summary", {}, mock_http)

        assert get_status(result) == 429
        body = get_body(result)
        assert "Rate limit" in body.get("error", "")


# ===========================================================================
# Tests for _get_summary()
# ===========================================================================


class TestGetSummary:
    """Tests for _get_summary() method."""

    def test_summary_when_moment_detector_not_available(self):
        """Should return empty summary when moment detection is not available."""
        handler = create_handler()

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            False,
        ):
            result = handler._get_summary()

        assert get_status(result) == 200
        body = get_body(result)
        assert body["total_moments"] == 0
        assert body["message"] == "Moment detection not available"

    def test_summary_when_detector_not_configured(self):
        """Should return empty summary when detector is not in context."""
        handler = create_handler({})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_summary()

        body = get_body(result)
        assert body["total_moments"] == 0
        assert body["message"] == "Moment detector not configured"

    def test_summary_with_moments(self):
        """Should return summary with moment statistics."""
        moments = [
            MockSignificantMoment(
                id="m1",
                moment_type="upset_victory",
                agent_name="claude",
                significance_score=0.9,
            ),
            MockSignificantMoment(
                id="m2",
                moment_type="consensus_breakthrough",
                agent_name="claude",
                significance_score=0.85,
            ),
            MockSignificantMoment(
                id="m3",
                moment_type="upset_victory",
                agent_name="gpt4",
                significance_score=0.75,
            ),
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {
            "claude": moments[:2],
            "gpt4": [moments[2]],
        }
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_summary()

        body = get_body(result)
        assert body["total_moments"] == 3
        assert body["by_type"]["upset_victory"] == 2
        assert body["by_type"]["consensus_breakthrough"] == 1
        assert body["by_agent"]["claude"] == 2
        assert body["by_agent"]["gpt4"] == 1
        assert body["most_significant"]["significance"] == 0.9


# ===========================================================================
# Tests for _get_timeline()
# ===========================================================================


class TestGetTimeline:
    """Tests for _get_timeline() method."""

    def test_timeline_when_detector_not_available(self):
        """Should return empty timeline when moment detection is not available."""
        handler = create_handler()

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            False,
        ):
            result = handler._get_timeline(50, 0)

        body = get_body(result)
        assert body["moments"] == []
        assert body["total"] == 0
        assert body["has_more"] is False

    def test_timeline_with_pagination(self):
        """Should return paginated timeline."""
        moments = [
            MockSignificantMoment(
                id=f"m{i}",
                significance_score=0.9 - i * 0.1,
                created_at=f"2025-01-{15 - i:02d}T10:00:00Z",
            )
            for i in range(10)
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {"claude": moments}
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_timeline(5, 0)

        body = get_body(result)
        assert len(body["moments"]) == 5
        assert body["total"] == 10
        assert body["has_more"] is True

    def test_timeline_with_offset(self):
        """Should return timeline with offset."""
        moments = [
            MockSignificantMoment(
                id=f"m{i}",
                created_at=f"2025-01-{15 - i:02d}T10:00:00Z",
            )
            for i in range(10)
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {"claude": moments}
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_timeline(5, 5)

        body = get_body(result)
        assert len(body["moments"]) == 5
        assert body["offset"] == 5
        assert body["has_more"] is False


# ===========================================================================
# Tests for _get_trending()
# ===========================================================================


class TestGetTrending:
    """Tests for _get_trending() method."""

    def test_trending_when_detector_not_available(self):
        """Should return empty trending when moment detection is not available."""
        handler = create_handler()

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            False,
        ):
            result = handler._get_trending(10)

        body = get_body(result)
        assert body["trending"] == []
        assert body["count"] == 0

    def test_trending_sorted_by_significance(self):
        """Should return moments sorted by significance score."""
        moments = [
            MockSignificantMoment(id="m1", significance_score=0.5),
            MockSignificantMoment(id="m2", significance_score=0.9),
            MockSignificantMoment(id="m3", significance_score=0.7),
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {"claude": moments}
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_trending(10)

        body = get_body(result)
        assert body["count"] == 3
        # Should be sorted by significance descending
        assert body["trending"][0]["significance"] == 0.9
        assert body["trending"][1]["significance"] == 0.7
        assert body["trending"][2]["significance"] == 0.5

    def test_trending_respects_limit(self):
        """Should respect the limit parameter."""
        moments = [
            MockSignificantMoment(id=f"m{i}", significance_score=0.9 - i * 0.1) for i in range(10)
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {"claude": moments}
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_trending(3)

        body = get_body(result)
        assert body["count"] == 3


# ===========================================================================
# Tests for _get_by_type()
# ===========================================================================


class TestGetByType:
    """Tests for _get_by_type() method."""

    def test_by_type_when_detector_not_available(self):
        """Should return empty result when moment detection is not available."""
        handler = create_handler()

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            False,
        ):
            result = handler._get_by_type("upset_victory", 50)

        body = get_body(result)
        assert body["moments"] == []
        assert body["total"] == 0

    def test_by_type_filters_correctly(self):
        """Should return only moments of the specified type."""
        moments = [
            MockSignificantMoment(id="m1", moment_type="upset_victory"),
            MockSignificantMoment(id="m2", moment_type="consensus_breakthrough"),
            MockSignificantMoment(id="m3", moment_type="upset_victory"),
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {"claude": moments}
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_by_type("upset_victory", 50)

        body = get_body(result)
        assert body["type"] == "upset_victory"
        assert body["total"] == 2
        assert len(body["moments"]) == 2
        assert all(m["type"] == "upset_victory" for m in body["moments"])

    def test_by_type_sorted_by_significance(self):
        """Should return moments sorted by significance."""
        moments = [
            MockSignificantMoment(id="m1", moment_type="upset_victory", significance_score=0.5),
            MockSignificantMoment(id="m2", moment_type="upset_victory", significance_score=0.9),
            MockSignificantMoment(id="m3", moment_type="upset_victory", significance_score=0.7),
        ]
        detector = MockMomentDetector()
        detector._moment_cache = {"claude": moments}
        handler = create_handler({"moment_detector": detector})

        with patch(
            "aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE",
            True,
        ):
            result = handler._get_by_type("upset_victory", 50)

        body = get_body(result)
        assert body["moments"][0]["significance"] == 0.9
        assert body["moments"][1]["significance"] == 0.7
        assert body["moments"][2]["significance"] == 0.5


# ===========================================================================
# Tests for _moment_to_dict()
# ===========================================================================


class TestMomentToDict:
    """Tests for _moment_to_dict() conversion."""

    def test_converts_moment_to_dict(self):
        """Should convert a SignificantMoment to dict properly."""
        moment = MockSignificantMoment(
            id="moment-123",
            moment_type="consensus_breakthrough",
            agent_name="claude",
            description="Achieved consensus",
            significance_score=0.85,
            debate_id="debate-456",
            other_agents=["gpt4", "gemini"],
            metadata={"round": 3},
            created_at="2025-01-15T10:00:00Z",
        )
        handler = create_handler({})

        result = handler._moment_to_dict(moment)

        assert result["id"] == "moment-123"
        assert result["type"] == "consensus_breakthrough"
        assert result["agent"] == "claude"
        assert result["description"] == "Achieved consensus"
        assert result["significance"] == 0.85
        assert result["debate_id"] == "debate-456"
        assert result["other_agents"] == ["gpt4", "gemini"]
        assert result["metadata"] == {"round": 3}
        assert result["created_at"] == "2025-01-15T10:00:00Z"


# ===========================================================================
# Tests for handle() Routing
# ===========================================================================


class TestHandleRouting:
    """Tests for handle() method routing to correct endpoints."""

    @pytest.mark.asyncio
    async def test_routes_to_summary(self):
        """Should route /api/moments/summary to _get_summary."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter.is_allowed",
                    return_value=True,
                ):
                    with patch.object(handler, "_get_summary") as mock_summary:
                        mock_summary.return_value = MagicMock(status_code=200)
                        await handler.handle("/api/moments/summary", {}, mock_http)
                        mock_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_timeline(self):
        """Should route /api/moments/timeline to _get_timeline."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter.is_allowed",
                    return_value=True,
                ):
                    with patch.object(handler, "_get_timeline") as mock_timeline:
                        mock_timeline.return_value = MagicMock(status_code=200)
                        await handler.handle(
                            "/api/moments/timeline",
                            {"limit": "20", "offset": "0"},
                            mock_http,
                        )
                        mock_timeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_trending(self):
        """Should route /api/moments/trending to _get_trending."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter.is_allowed",
                    return_value=True,
                ):
                    with patch.object(handler, "_get_trending") as mock_trending:
                        mock_trending.return_value = MagicMock(status_code=200)
                        await handler.handle("/api/moments/trending", {}, mock_http)
                        mock_trending.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_by_type(self):
        """Should route /api/moments/by-type/{type} to _get_by_type."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter.is_allowed",
                    return_value=True,
                ):
                    with patch.object(handler, "_get_by_type") as mock_by_type:
                        mock_by_type.return_value = MagicMock(status_code=200)
                        await handler.handle("/api/moments/by-type/upset_victory", {}, mock_http)
                        mock_by_type.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_moment_type_returns_400(self):
        """Should return 400 for invalid moment type."""
        handler = create_handler()
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.moments._moments_limiter.is_allowed",
                    return_value=True,
                ):
                    result = await handler.handle(
                        "/api/moments/by-type/invalid_type", {}, mock_http
                    )

        assert get_status(result) == 400
        body = get_body(result)
        assert "Invalid moment type" in body.get("error", "")


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestMomentsHandlerIntegration:
    """Integration tests for moments handler."""

    def test_handler_has_resource_type(self):
        """Handler should have RESOURCE_TYPE defined for RBAC."""
        handler = create_handler({})
        assert handler.RESOURCE_TYPE == "moments"

    def test_handler_has_routes(self):
        """Handler should have ROUTES defined."""
        handler = create_handler({})
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0
        assert "/api/moments/summary" in handler.ROUTES

    def test_rate_limiter_configured(self):
        """Rate limiter should be configured for moments endpoint."""
        from aragora.server.handlers.moments import _moments_limiter

        assert _moments_limiter is not None
        assert hasattr(_moments_limiter, "is_allowed")
