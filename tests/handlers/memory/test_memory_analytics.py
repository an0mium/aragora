"""Tests for MemoryAnalyticsHandler (aragora/server/handlers/memory/memory_analytics.py).

Covers all routes and behavior:
- MemoryAnalyticsHandler.__init__ and context handling
- can_handle() for all ROUTES and tier-specific paths
- handle() routing: GET /api/v1/memory/analytics
- handle() routing: GET /api/v1/memory/analytics/tier/{tier}
- handle_post() routing: POST /api/v1/memory/analytics/snapshot
- _get_analytics() with tracker available/unavailable
- _get_tier_stats() with valid/invalid tiers and import errors
- _take_snapshot() with tracker available/unavailable
- Rate limiting (module-level _memory_analytics_limiter)
- RBAC permission checks (authentication and authorization)
- Lazy tracker initialization and ImportError fallback
- Edge cases (unknown paths, missing params, clamped days)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.memory.memory_analytics import (
    MemoryAnalyticsHandler,
    MEMORY_ANALYTICS_PERMISSION,
    MEMORY_READ_PERMISSION,
    _memory_analytics_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for analytics tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock MemoryTier enum
# ---------------------------------------------------------------------------


class _MockMemoryTier(Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


# ---------------------------------------------------------------------------
# Mock analytics data
# ---------------------------------------------------------------------------


def _mock_tier_stats() -> MagicMock:
    """Create a mock TierStats with to_dict."""
    stats = MagicMock()
    stats.to_dict.return_value = {
        "tier": "fast",
        "count": 42,
        "avg_importance": 0.75,
        "avg_surprise": 0.3,
    }
    return stats


def _mock_analytics() -> MagicMock:
    """Create a mock analytics result with to_dict."""
    analytics = MagicMock()
    analytics.to_dict.return_value = {
        "total_memories": 200,
        "tier_breakdown": {
            "fast": 50,
            "medium": 80,
            "slow": 50,
            "glacial": 20,
        },
        "promotion_effectiveness": 0.85,
        "learning_velocity": 0.72,
        "recommendations": ["Increase consolidation frequency"],
    }
    return analytics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> MemoryAnalyticsHandler:
    """Create a MemoryAnalyticsHandler with default context."""
    return MemoryAnalyticsHandler(ctx={"analytics_db": ":memory:"})


@pytest.fixture
def handler_no_ctx() -> MemoryAnalyticsHandler:
    """Create a handler with None context (defaults to empty dict)."""
    return MemoryAnalyticsHandler(ctx=None)


@pytest.fixture
def mock_http() -> MockHTTPHandler:
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset rate limiter state between tests."""
    _memory_analytics_limiter._buckets.clear()
    yield
    _memory_analytics_limiter._buckets.clear()


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestInit:
    """Test handler initialization."""

    def test_init_with_context(self, handler: MemoryAnalyticsHandler):
        assert handler.ctx == {"analytics_db": ":memory:"}
        assert handler._tracker is None

    def test_init_with_none_context(self, handler_no_ctx: MemoryAnalyticsHandler):
        assert handler_no_ctx.ctx == {}
        assert handler_no_ctx._tracker is None

    def test_init_with_empty_context(self):
        h = MemoryAnalyticsHandler(ctx={})
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# can_handle Tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test route matching in can_handle()."""

    def test_can_handle_analytics_root(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/analytics") is True

    def test_can_handle_analytics_snapshot(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/analytics/snapshot") is True

    def test_can_handle_tier_fast(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/analytics/tier/fast") is True

    def test_can_handle_tier_medium(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/analytics/tier/medium") is True

    def test_can_handle_tier_slow(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/analytics/tier/slow") is True

    def test_can_handle_tier_glacial(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/analytics/tier/glacial") is True

    def test_cannot_handle_unknown_path(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory/something-else") is False

    def test_cannot_handle_empty_path(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("") is False

    def test_cannot_handle_partial_match(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/memory") is False

    def test_cannot_handle_different_api(self, handler: MemoryAnalyticsHandler):
        assert handler.can_handle("/api/v1/debates") is False


# ---------------------------------------------------------------------------
# Tracker Lazy Loading Tests
# ---------------------------------------------------------------------------


class TestTrackerProperty:
    """Test lazy-loading of TierAnalyticsTracker."""

    def test_tracker_lazy_load_success(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        with patch(
            "aragora.server.handlers.memory.memory_analytics.TierAnalyticsTracker",
            create=True,
        ) as mock_cls:
            # We need to patch the import inside the property
            import importlib

            with patch.dict(
                "sys.modules",
                {
                    "aragora.memory.tier_analytics": MagicMock(
                        TierAnalyticsTracker=lambda db_path: mock_tracker
                    )
                },
            ):
                handler._tracker = None
                result = handler.tracker
                assert result is mock_tracker

    def test_tracker_import_error_returns_none(self, handler: MemoryAnalyticsHandler):
        """When TierAnalyticsTracker cannot be imported, tracker returns None."""
        handler._tracker = None
        with patch.dict("sys.modules", {"aragora.memory.tier_analytics": None}):
            result = handler.tracker
            assert result is None

    def test_tracker_cached_after_first_load(self, handler: MemoryAnalyticsHandler):
        """Once loaded, tracker is cached and reused."""
        mock_tracker = MagicMock()
        handler._tracker = mock_tracker
        assert handler.tracker is mock_tracker

    def test_tracker_uses_analytics_db_from_ctx(self):
        """Tracker uses analytics_db path from context."""
        mock_tracker = MagicMock()
        h = MemoryAnalyticsHandler(ctx={"analytics_db": "/custom/path.db"})
        with patch.dict(
            "sys.modules",
            {
                "aragora.memory.tier_analytics": MagicMock(
                    TierAnalyticsTracker=MagicMock(return_value=mock_tracker)
                )
            },
        ):
            result = h.tracker
            assert result is mock_tracker

    def test_tracker_defaults_db_path_when_missing(self):
        """When analytics_db not in ctx, defaults to memory_analytics.db."""
        mock_cls = MagicMock()
        h = MemoryAnalyticsHandler(ctx={})
        with patch.dict(
            "sys.modules",
            {"aragora.memory.tier_analytics": MagicMock(TierAnalyticsTracker=mock_cls)},
        ):
            h.tracker
            mock_cls.assert_called_once_with(db_path="memory_analytics.db")


# ---------------------------------------------------------------------------
# GET /api/v1/memory/analytics Tests
# ---------------------------------------------------------------------------


class TestGetAnalytics:
    """Test the _get_analytics method and GET analytics route."""

    def test_get_analytics_success(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        result = handler._get_analytics(30)

        assert _status(result) == 200
        body = _body(result)
        assert body["total_memories"] == 200
        assert "tier_breakdown" in body
        mock_tracker.get_analytics.assert_called_once_with(days=30)

    def test_get_analytics_custom_days(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        handler._get_analytics(7)

        mock_tracker.get_analytics.assert_called_once_with(days=7)

    def test_get_analytics_no_tracker_returns_503(self, handler: MemoryAnalyticsHandler):
        handler._tracker = None
        # Ensure tracker property also returns None
        with patch.dict("sys.modules", {"aragora.memory.tier_analytics": None}):
            result = handler._get_analytics(30)
            assert _status(result) == 503
            assert "not available" in _body(result).get("error", "")

    def test_get_analytics_via_handle(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        result = asyncio.run(handler.handle("/api/v1/memory/analytics", {}, mock_http))

        assert _status(result) == 200
        body = _body(result)
        assert body["total_memories"] == 200

    def test_get_analytics_via_handle_with_days_param(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        result = asyncio.run(handler.handle("/api/v1/memory/analytics", {"days": "7"}, mock_http))

        assert _status(result) == 200
        mock_tracker.get_analytics.assert_called_once_with(days=7)


# ---------------------------------------------------------------------------
# GET /api/v1/memory/analytics/tier/{tier} Tests
# ---------------------------------------------------------------------------


class TestGetTierStats:
    """Test the _get_tier_stats method and tier-specific route."""

    def test_get_tier_stats_valid_tier(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            result = handler._get_tier_stats("fast", 30)

        assert _status(result) == 200
        body = _body(result)
        assert body["tier"] == "fast"
        assert body["count"] == 42

    def test_get_tier_stats_case_insensitive(self, handler: MemoryAnalyticsHandler):
        """Tier name should be case-insensitive (FAST, Fast, fast all work)."""
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            result = handler._get_tier_stats("FAST", 30)

        assert _status(result) == 200

    def test_get_tier_stats_medium(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            result = handler._get_tier_stats("medium", 30)

        assert _status(result) == 200

    def test_get_tier_stats_slow(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            result = handler._get_tier_stats("slow", 30)

        assert _status(result) == 200

    def test_get_tier_stats_glacial(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            result = handler._get_tier_stats("glacial", 30)

        assert _status(result) == 200

    def test_get_tier_stats_invalid_tier_returns_400(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            result = handler._get_tier_stats("nonexistent", 30)

        assert _status(result) == 400
        body = _body(result)
        assert "Invalid tier" in body.get("error", "")

    def test_get_tier_stats_no_tracker_returns_503(self, handler: MemoryAnalyticsHandler):
        handler._tracker = None
        with patch.dict("sys.modules", {"aragora.memory.tier_analytics": None}):
            result = handler._get_tier_stats("fast", 30)
            assert _status(result) == 503

    def test_get_tier_stats_import_error_returns_503(self, handler: MemoryAnalyticsHandler):
        """When tier_manager cannot be imported, returns 503."""
        mock_tracker = MagicMock()
        handler._tracker = mock_tracker

        with patch.dict("sys.modules", {"aragora.memory.tier_manager": None}):
            result = handler._get_tier_stats("fast", 30)

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    def test_get_tier_stats_custom_days(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            handler._get_tier_stats("fast", 90)

        mock_tracker.get_tier_stats.assert_called_once_with(_MockMemoryTier.FAST, days=90)

    def test_get_tier_stats_via_handle(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            import asyncio

            result = asyncio.run(
                handler.handle("/api/v1/memory/analytics/tier/fast", {}, mock_http)
            )

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# POST /api/v1/memory/analytics/snapshot Tests
# ---------------------------------------------------------------------------


class TestTakeSnapshot:
    """Test the _take_snapshot method and POST snapshot route."""

    def test_take_snapshot_success(self, handler: MemoryAnalyticsHandler):
        mock_tracker = MagicMock()
        handler._tracker = mock_tracker

        result = handler._take_snapshot()

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "success"
        assert "Snapshot recorded" in body["message"]
        mock_tracker.take_snapshot.assert_called_once()

    def test_take_snapshot_no_tracker_returns_503(self, handler: MemoryAnalyticsHandler):
        handler._tracker = None
        with patch.dict("sys.modules", {"aragora.memory.tier_analytics": None}):
            result = handler._take_snapshot()
            assert _status(result) == 503
            assert "not available" in _body(result).get("error", "")

    def test_take_snapshot_via_handle_post(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        handler._tracker = mock_tracker

        import asyncio

        result = asyncio.run(
            handler.handle_post("/api/v1/memory/analytics/snapshot", {}, mock_http)
        )

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "success"

    def test_take_snapshot_unknown_post_path_returns_none(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        import asyncio

        result = asyncio.run(handler.handle_post("/api/v1/memory/analytics/unknown", {}, mock_http))
        assert result is None


# ---------------------------------------------------------------------------
# Rate Limiting Tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Test rate limiting on the handle() method."""

    @pytest.mark.no_auto_auth
    def test_rate_limit_exceeded_returns_429(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        """When rate limiter denies request, returns 429."""
        with patch.object(_memory_analytics_limiter, "is_allowed", return_value=False):
            import asyncio

            result = asyncio.run(handler.handle("/api/v1/memory/analytics", {}, mock_http))
            assert _status(result) == 429
            assert "Rate limit" in _body(result).get("error", "")

    def test_rate_limit_allowed_proceeds(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        """When rate limiter allows request, proceeds normally."""
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        with patch.object(_memory_analytics_limiter, "is_allowed", return_value=True):
            import asyncio

            result = asyncio.run(handler.handle("/api/v1/memory/analytics", {}, mock_http))
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# RBAC / Auth Tests
# ---------------------------------------------------------------------------


class TestRBAC:
    """Test authentication and authorization checks."""

    @pytest.mark.no_auto_auth
    def test_handle_unauthenticated_returns_401(self, mock_http: MockHTTPHandler):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        h = MemoryAnalyticsHandler(ctx={})

        async def raise_unauth(self, handler, require_auth=False):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            import asyncio

            result = asyncio.run(h.handle("/api/v1/memory/analytics", {}, mock_http))
            assert _status(result) == 401
            assert "Authentication required" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    def test_handle_forbidden_returns_403(self, mock_http: MockHTTPHandler):
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        h = MemoryAnalyticsHandler(ctx={})

        async def mock_auth(self, handler, require_auth=False):
            return MagicMock()

        def raise_forbidden(self, auth_ctx, perm, resource_id=None):
            raise ForbiddenError("No permission")

        with (
            patch.object(SecureHandler, "get_auth_context", mock_auth),
            patch.object(SecureHandler, "check_permission", raise_forbidden),
        ):
            import asyncio

            result = asyncio.run(h.handle("/api/v1/memory/analytics", {}, mock_http))
            assert _status(result) == 403
            assert "Permission denied" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    def test_handle_post_unauthenticated_returns_401(self, mock_http: MockHTTPHandler):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        h = MemoryAnalyticsHandler(ctx={})

        async def raise_unauth(self, handler, require_auth=False):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            import asyncio

            result = asyncio.run(h.handle_post("/api/v1/memory/analytics/snapshot", {}, mock_http))
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_handle_post_forbidden_returns_403(self, mock_http: MockHTTPHandler):
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        h = MemoryAnalyticsHandler(ctx={})

        async def mock_auth(self, handler, require_auth=False):
            return MagicMock()

        def raise_forbidden(self, auth_ctx, perm, resource_id=None):
            raise ForbiddenError("No permission")

        with (
            patch.object(SecureHandler, "get_auth_context", mock_auth),
            patch.object(SecureHandler, "check_permission", raise_forbidden),
        ):
            import asyncio

            result = asyncio.run(h.handle_post("/api/v1/memory/analytics/snapshot", {}, mock_http))
            assert _status(result) == 403


# ---------------------------------------------------------------------------
# Handle() Routing Edge Cases
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Test handle() routing to correct internal methods."""

    def test_handle_unknown_path_returns_none(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        import asyncio

        result = asyncio.run(handler.handle("/api/v1/memory/analytics/unknown", {}, mock_http))
        assert result is None

    def test_handle_tier_path_extracts_tier_name(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        """Tier name is extracted from last segment of the path."""
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            import asyncio

            result = asyncio.run(
                handler.handle("/api/v1/memory/analytics/tier/glacial", {"days": "14"}, mock_http)
            )

        assert _status(result) == 200
        mock_tracker.get_tier_stats.assert_called_once_with(_MockMemoryTier.GLACIAL, days=14)

    def test_handle_null_handler_rate_limit(self, handler: MemoryAnalyticsHandler):
        """When handler is None, get_client_ip returns 'unknown'."""
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        result = asyncio.run(handler.handle("/api/v1/memory/analytics", {}, None))
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Days Parameter Clamping Tests
# ---------------------------------------------------------------------------


class TestDaysClamping:
    """Test days parameter is clamped to [1, 365]."""

    def test_days_defaults_to_30(self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        asyncio.run(handler.handle("/api/v1/memory/analytics", {}, mock_http))
        mock_tracker.get_analytics.assert_called_once_with(days=30)

    def test_days_clamped_to_min_1(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        asyncio.run(handler.handle("/api/v1/memory/analytics", {"days": "0"}, mock_http))
        mock_tracker.get_analytics.assert_called_once_with(days=1)

    def test_days_clamped_to_max_365(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        asyncio.run(handler.handle("/api/v1/memory/analytics", {"days": "999"}, mock_http))
        mock_tracker.get_analytics.assert_called_once_with(days=365)

    def test_days_negative_clamped_to_1(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_analytics.return_value = _mock_analytics()
        handler._tracker = mock_tracker

        import asyncio

        asyncio.run(handler.handle("/api/v1/memory/analytics", {"days": "-5"}, mock_http))
        mock_tracker.get_analytics.assert_called_once_with(days=1)

    def test_days_clamped_for_tier_route(
        self, handler: MemoryAnalyticsHandler, mock_http: MockHTTPHandler
    ):
        mock_tracker = MagicMock()
        mock_tracker.get_tier_stats.return_value = _mock_tier_stats()
        handler._tracker = mock_tracker

        with patch.dict(
            "sys.modules", {"aragora.memory.tier_manager": MagicMock(MemoryTier=_MockMemoryTier)}
        ):
            import asyncio

            asyncio.run(
                handler.handle("/api/v1/memory/analytics/tier/fast", {"days": "500"}, mock_http)
            )

        mock_tracker.get_tier_stats.assert_called_once_with(_MockMemoryTier.FAST, days=365)


# ---------------------------------------------------------------------------
# Constants and Module-Level Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_memory_analytics_permission_value(self):
        assert MEMORY_ANALYTICS_PERMISSION == "memory:analytics"

    def test_memory_read_permission_value(self):
        assert MEMORY_READ_PERMISSION == "memory:read"

    def test_rate_limiter_rpm(self):
        assert _memory_analytics_limiter.rpm == 30

    def test_routes_list(self, handler: MemoryAnalyticsHandler):
        assert "/api/v1/memory/analytics" in handler.ROUTES
        assert "/api/v1/memory/analytics/snapshot" in handler.ROUTES
        assert len(handler.ROUTES) == 2
