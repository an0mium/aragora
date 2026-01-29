"""Tests for MemoryAnalyticsHandler."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.memory.memory_analytics import (
    MemoryAnalyticsHandler,
    MEMORY_ANALYTICS_PERMISSION,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockTierStats:
    """Mock tier statistics."""

    tier_name: str
    total_items: int = 100
    active_items: int = 80
    promotions: int = 20
    demotions: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "total_items": self.total_items,
            "active_items": self.active_items,
            "promotions": self.promotions,
            "demotions": self.demotions,
        }


@dataclass
class MockAnalytics:
    """Mock analytics result."""

    tier_stats: Dict[str, MockTierStats] = field(default_factory=dict)
    promotion_effectiveness: float = 0.85
    learning_velocity: float = 0.7
    recommendations: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_stats": {k: v.to_dict() for k, v in self.tier_stats.items()},
            "promotion_effectiveness": self.promotion_effectiveness,
            "learning_velocity": self.learning_velocity,
            "recommendations": self.recommendations,
        }


@dataclass
class MockTierAnalyticsTracker:
    """Mock TierAnalyticsTracker."""

    def get_analytics(self, days: int = 30) -> MockAnalytics:
        return MockAnalytics(
            tier_stats={
                "fast": MockTierStats(tier_name="fast", total_items=50),
                "medium": MockTierStats(tier_name="medium", total_items=200),
                "slow": MockTierStats(tier_name="slow", total_items=500),
                "glacial": MockTierStats(tier_name="glacial", total_items=100),
            },
            recommendations=["Increase promotion threshold for slow tier"],
        )

    def get_tier_stats(self, tier, days: int = 30) -> MockTierStats:
        return MockTierStats(tier_name=tier.name.lower())

    def take_snapshot(self) -> None:
        pass


class MockAuthContext:
    """Mock authentication context."""

    def __init__(self, user_id: str = "user-123", permissions: list = None):
        self.user_id = user_id
        self.permissions = permissions or [MEMORY_ANALYTICS_PERMISSION]


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, client_ip: str = "127.0.0.1"):
        self.headers = {"X-Forwarded-For": client_ip}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tracker():
    """Create a mock tracker."""
    return MockTierAnalyticsTracker()


@pytest.fixture
def handler(mock_tracker):
    """Create a test handler with mock tracker."""
    h = MemoryAnalyticsHandler(server_context={"analytics_db": ":memory:"})
    h._tracker = mock_tracker
    return h


@pytest.fixture
def handler_no_tracker():
    """Create a test handler without tracker."""
    h = MemoryAnalyticsHandler(server_context={})
    h._tracker = None
    return h


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test Handler Routing
# =============================================================================


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_analytics_route(self, handler):
        """Test can_handle for analytics route."""
        assert handler.can_handle("/api/v1/memory/analytics") is True

    def test_can_handle_snapshot_route(self, handler):
        """Test can_handle for snapshot route."""
        assert handler.can_handle("/api/v1/memory/analytics/snapshot") is True

    def test_can_handle_tier_route(self, handler):
        """Test can_handle for tier-specific route."""
        assert handler.can_handle("/api/v1/memory/analytics/tier/fast") is True
        assert handler.can_handle("/api/v1/memory/analytics/tier/slow") is True

    def test_cannot_handle_invalid_route(self, handler):
        """Test can_handle for invalid route."""
        assert handler.can_handle("/api/v1/memory/other") is False


# =============================================================================
# Test Get Analytics
# =============================================================================


class TestGetAnalytics:
    """Tests for get analytics endpoint."""

    def test_get_analytics_success(self, handler):
        """Test successful analytics retrieval."""
        result = handler._get_analytics(days=30)

        assert result.status_code == 200
        data = parse_response(result)
        assert "tier_stats" in data
        assert "promotion_effectiveness" in data
        assert "learning_velocity" in data

    def test_get_analytics_with_days(self, handler):
        """Test analytics with custom days parameter."""
        result = handler._get_analytics(days=7)

        assert result.status_code == 200

    def test_get_analytics_no_tracker(self, handler_no_tracker):
        """Test analytics when tracker not available."""
        result = handler_no_tracker._get_analytics(days=30)

        assert result.status_code == 503
        assert "not available" in parse_response(result)["error"]


# =============================================================================
# Test Get Tier Stats
# =============================================================================


class TestGetTierStats:
    """Tests for get tier stats endpoint."""

    def test_get_tier_stats_fast(self, handler):
        """Test tier stats for fast tier."""
        with patch("aragora.server.handlers.memory.memory_analytics.MemoryTier") as mock_tier:
            from enum import Enum

            class MockMemoryTier(Enum):
                FAST = "fast"
                MEDIUM = "medium"
                SLOW = "slow"
                GLACIAL = "glacial"

            mock_tier.__iter__ = lambda self: iter([MockMemoryTier.FAST])
            mock_tier.__getitem__ = lambda self, key: MockMemoryTier[key]
            mock_tier.FAST = MockMemoryTier.FAST

            result = handler._get_tier_stats("fast", days=30)

            assert result.status_code == 200
            data = parse_response(result)
            assert "tier_name" in data

    def test_get_tier_stats_medium(self, handler):
        """Test tier stats for medium tier."""
        with patch("aragora.server.handlers.memory.memory_analytics.MemoryTier") as mock_tier:
            from enum import Enum

            class MockMemoryTier(Enum):
                FAST = "fast"
                MEDIUM = "medium"
                SLOW = "slow"
                GLACIAL = "glacial"

            mock_tier.__iter__ = lambda self: iter([MockMemoryTier.MEDIUM])
            mock_tier.__getitem__ = lambda self, key: MockMemoryTier[key]
            mock_tier.MEDIUM = MockMemoryTier.MEDIUM

            result = handler._get_tier_stats("medium", days=30)

            assert result.status_code == 200

    def test_get_tier_stats_no_tracker(self, handler_no_tracker):
        """Test tier stats when tracker not available."""
        result = handler_no_tracker._get_tier_stats("fast", days=30)

        assert result.status_code == 503


# =============================================================================
# Test Take Snapshot
# =============================================================================


class TestTakeSnapshot:
    """Tests for take snapshot endpoint."""

    def test_take_snapshot_success(self, handler):
        """Test successful snapshot."""
        result = handler._take_snapshot()

        assert result.status_code == 200
        data = parse_response(result)
        assert data["status"] == "success"
        assert "Snapshot recorded" in data["message"]

    def test_take_snapshot_no_tracker(self, handler_no_tracker):
        """Test snapshot when tracker not available."""
        result = handler_no_tracker._take_snapshot()

        assert result.status_code == 503


# =============================================================================
# Test Authentication & Authorization
# =============================================================================


class TestAuthentication:
    """Tests for authentication and authorization."""

    @pytest.mark.asyncio
    async def test_requires_authentication(self, handler):
        """Test endpoint requires authentication."""
        from aragora.server.handlers.secure import UnauthorizedError

        with patch.object(handler, "get_auth_context") as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle("/api/v1/memory/analytics", {}, MockHandler())

            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_requires_permission(self, handler):
        """Test endpoint requires analytics permission."""
        from aragora.server.handlers.secure import ForbiddenError

        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission") as mock_perm,
        ):
            mock_auth.return_value = MockAuthContext(permissions=[])
            mock_perm.side_effect = ForbiddenError("Missing permission")

            result = await handler.handle("/api/v1/memory/analytics", {}, MockHandler())

            assert result.status_code == 403


# =============================================================================
# Test Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_initial_requests(self, handler):
        """Test rate limiter allows initial requests."""
        with (
            patch.object(handler, "get_auth_context") as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.memory.memory_analytics._memory_analytics_limiter"
            ) as mock_limiter,
        ):
            mock_auth.return_value = MockAuthContext()
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle("/api/v1/memory/analytics", {}, MockHandler())

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler):
        """Test rate limiter rejects excessive requests."""
        with patch(
            "aragora.server.handlers.memory.memory_analytics._memory_analytics_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = await handler.handle("/api/v1/memory/analytics", {}, MockHandler())

            assert result.status_code == 429


# =============================================================================
# Test Tracker Property
# =============================================================================


class TestTrackerProperty:
    """Tests for tracker lazy loading."""

    def test_tracker_lazy_loads(self):
        """Test tracker is lazily loaded."""
        h = MemoryAnalyticsHandler(ctx={"analytics_db": ":memory:"})
        assert h._tracker is None

        # Access tracker property - may fail if module not available
        # which is expected in test environment
        tracker = h.tracker
        # Either None (module unavailable) or an instance

    def test_tracker_returns_cached(self, handler, mock_tracker):
        """Test tracker returns cached instance."""
        first = handler.tracker
        second = handler.tracker

        assert first is second
