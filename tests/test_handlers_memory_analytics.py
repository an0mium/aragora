"""
Tests for Memory Analytics Handler.

Tests cover:
- GET /api/memory/analytics - Get comprehensive memory tier analytics
- GET /api/memory/analytics/tier/{tier} - Get stats for specific tier
- POST /api/memory/analytics/snapshot - Take a manual snapshot
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from tests.conftest import requires_memory_analytics, REQUIRES_MEMORY_ANALYTICS
from tests.fixtures.shared.auth import setup_full_auth_bypass

# Skip entire module if MemoryAnalyticsHandler is not available
pytestmark = pytest.mark.skipif(requires_memory_analytics, reason=REQUIRES_MEMORY_ANALYTICS)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def bypass_rbac(monkeypatch):
    """Bypass RBAC checks for all tests in this module."""
    setup_full_auth_bypass(monkeypatch)


@pytest.fixture
def mock_tracker():
    """Create mock analytics tracker."""
    tracker = Mock()

    # Mock analytics response
    mock_analytics = Mock()
    mock_analytics.to_dict.return_value = {
        "total_memories": 1000,
        "tier_stats": {
            "fast": {"count": 100, "avg_importance": 0.8},
            "medium": {"count": 300, "avg_importance": 0.6},
            "slow": {"count": 400, "avg_importance": 0.4},
            "glacial": {"count": 200, "avg_importance": 0.2},
        },
        "learning_velocity": 0.75,
        "promotion_effectiveness": 0.82,
    }
    tracker.get_analytics.return_value = mock_analytics

    # Mock tier stats response
    mock_tier_stats = Mock()
    mock_tier_stats.to_dict.return_value = {
        "tier": "fast",
        "count": 100,
        "avg_importance": 0.8,
        "hit_rate": 0.95,
        "promotion_rate": 0.3,
    }
    tracker.get_tier_stats.return_value = mock_tier_stats

    return tracker


@pytest.fixture
def handler_with_tracker(mock_tracker):
    """Create handler with mocked tracker."""
    from aragora.server.handlers.memory import MemoryAnalyticsHandler

    handler = MemoryAnalyticsHandler({})
    handler._tracker = mock_tracker
    return handler


@pytest.fixture
def handler_without_tracker():
    """Create handler without tracker (simulates module not available)."""
    from aragora.server.handlers.memory import MemoryAnalyticsHandler

    handler = MemoryAnalyticsHandler({})
    # Use a sentinel to indicate "no tracker" - set to False to bypass lazy loading
    # The tracker property checks `if self._tracker is None` to lazy load,
    # so we need to make it return falsy without triggering lazy load
    handler._tracker = False  # Falsy but not None
    return handler


# ============================================================================
# Route Recognition Tests
# ============================================================================


class TestMemoryAnalyticsRoutes:
    """Tests for route recognition."""

    def test_handler_routes(self):
        """Test handler recognizes all memory analytics routes."""
        from aragora.server.handlers.memory import MemoryAnalyticsHandler

        handler = MemoryAnalyticsHandler({})

        assert "/api/v1/memory/analytics" in handler.ROUTES
        assert "/api/v1/memory/analytics/snapshot" in handler.ROUTES

    def test_can_handle_analytics_route(self):
        """Test can_handle returns True for analytics route."""
        from aragora.server.handlers.memory import MemoryAnalyticsHandler

        handler = MemoryAnalyticsHandler({})

        assert handler.can_handle("/api/v1/memory/analytics") is True

    def test_can_handle_snapshot_route(self):
        """Test can_handle returns True for snapshot route."""
        from aragora.server.handlers.memory import MemoryAnalyticsHandler

        handler = MemoryAnalyticsHandler({})

        assert handler.can_handle("/api/v1/memory/analytics/snapshot") is True

    def test_can_handle_tier_route(self):
        """Test can_handle returns True for tier-specific routes."""
        from aragora.server.handlers.memory import MemoryAnalyticsHandler

        handler = MemoryAnalyticsHandler({})

        assert handler.can_handle("/api/v1/memory/analytics/tier/fast") is True
        assert handler.can_handle("/api/v1/memory/analytics/tier/medium") is True
        assert handler.can_handle("/api/v1/memory/analytics/tier/slow") is True
        assert handler.can_handle("/api/v1/memory/analytics/tier/glacial") is True

    def test_cannot_handle_other_routes(self):
        """Test can_handle returns False for unrelated routes."""
        from aragora.server.handlers.memory import MemoryAnalyticsHandler

        handler = MemoryAnalyticsHandler({})

        assert handler.can_handle("/api/v1/memory/other") is False
        assert handler.can_handle("/api/v1/debates") is False


# ============================================================================
# GET Analytics Tests
# ============================================================================


class TestGetAnalytics:
    """Tests for GET /api/memory/analytics endpoint."""

    def test_get_analytics_success(self, handler_with_tracker, mock_tracker):
        """Test successful analytics retrieval."""
        result = handler_with_tracker._get_analytics(days=30)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total_memories" in data
        assert "tier_stats" in data
        mock_tracker.get_analytics.assert_called_once_with(days=30)

    def test_get_analytics_custom_days(self, handler_with_tracker, mock_tracker):
        """Test analytics with custom day range."""
        result = handler_with_tracker._get_analytics(days=7)

        mock_tracker.get_analytics.assert_called_once_with(days=7)

    def test_get_analytics_no_tracker(self, handler_without_tracker):
        """Test 503 response when tracker not available."""
        result = handler_without_tracker._get_analytics(days=30)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_routes_to_get_analytics(self, handler_with_tracker):
        """Test handle method routes to _get_analytics."""
        result = await handler_with_tracker.handle(
            "/api/v1/memory/analytics",
            {"days": "30"},
        )

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_with_days_param(self, handler_with_tracker, mock_tracker):
        """Test handle extracts days parameter."""
        await handler_with_tracker.handle(
            "/api/v1/memory/analytics",
            {"days": "7"},
        )

        mock_tracker.get_analytics.assert_called_once_with(days=7)


# ============================================================================
# GET Tier Stats Tests
# ============================================================================


class TestGetTierStats:
    """Tests for GET /api/memory/analytics/tier/{tier} endpoint."""

    def test_get_tier_stats_no_tracker(self, handler_without_tracker):
        """Test 503 response when tracker not available."""
        result = handler_without_tracker._get_tier_stats("fast", days=30)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_routes_to_tier_stats(self, handler_with_tracker):
        """Test handle method routes tier paths correctly."""
        result = await handler_with_tracker.handle(
            "/api/v1/memory/analytics/tier/fast",
            {},
        )

        # Should not return None (route is recognized)
        assert result is not None


# ============================================================================
# POST Snapshot Tests
# ============================================================================


class TestTakeSnapshot:
    """Tests for POST /api/memory/analytics/snapshot endpoint."""

    def test_take_snapshot_success(self, handler_with_tracker, mock_tracker):
        """Test successful snapshot."""
        result = handler_with_tracker._take_snapshot()

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "success"
        mock_tracker.take_snapshot.assert_called_once()

    def test_take_snapshot_no_tracker(self, handler_without_tracker):
        """Test 503 response when tracker not available."""
        result = handler_without_tracker._take_snapshot()

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_snapshot(self, handler_with_tracker, mock_tracker):
        """Test handle_post routes to _take_snapshot."""
        result = await handler_with_tracker.handle_post(
            "/api/v1/memory/analytics/snapshot",
            {},
        )

        assert result.status_code == 200
        mock_tracker.take_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_post_wrong_path_returns_none(self, handler_with_tracker):
        """Test handle_post returns None for unrecognized paths."""
        result = await handler_with_tracker.handle_post(
            "/api/v1/memory/other",
            {},
        )

        assert result is None


# ============================================================================
# Tracker Lazy Loading Tests
# ============================================================================


class TestTrackerLazyLoading:
    """Tests for lazy loading of analytics tracker."""

    def test_tracker_caches_instance(self):
        """Test tracker property caches the tracker instance."""
        from aragora.server.handlers.memory import MemoryAnalyticsHandler

        handler = MemoryAnalyticsHandler({})

        mock_tracker = Mock()
        handler._tracker = mock_tracker

        # Should return cached instance
        assert handler.tracker is mock_tracker
        assert handler.tracker is mock_tracker  # Same instance


# ============================================================================
# Parameter Handling Tests
# ============================================================================


class TestParameterHandling:
    """Tests for query parameter handling."""

    @pytest.mark.asyncio
    async def test_days_parameter_clamped_to_max(self, handler_with_tracker, mock_tracker):
        """Test days parameter is clamped to maximum value."""
        await handler_with_tracker.handle(
            "/api/v1/memory/analytics",
            {"days": "1000"},  # Exceeds max of 365
        )

        # Should be clamped to 365
        mock_tracker.get_analytics.assert_called_once_with(days=365)

    @pytest.mark.asyncio
    async def test_days_parameter_clamped_to_min(self, handler_with_tracker, mock_tracker):
        """Test days parameter is clamped to minimum value."""
        await handler_with_tracker.handle(
            "/api/v1/memory/analytics",
            {"days": "0"},  # Below min of 1
        )

        # Should be clamped to 1
        mock_tracker.get_analytics.assert_called_once_with(days=1)

    @pytest.mark.asyncio
    async def test_days_parameter_default_value(self, handler_with_tracker, mock_tracker):
        """Test days parameter uses default when not provided."""
        await handler_with_tracker.handle(
            "/api/v1/memory/analytics",
            {},  # No days param
        )

        # Should use default of 30
        mock_tracker.get_analytics.assert_called_once_with(days=30)

    @pytest.mark.asyncio
    async def test_invalid_days_parameter_uses_default(self, handler_with_tracker, mock_tracker):
        """Test invalid days parameter uses default value."""
        await handler_with_tracker.handle(
            "/api/v1/memory/analytics",
            {"days": "invalid"},
        )

        # Should use default of 30
        mock_tracker.get_analytics.assert_called_once_with(days=30)


# ============================================================================
# Integration Tests
# ============================================================================


class TestMemoryAnalyticsIntegration:
    """Integration tests for memory analytics handler."""

    @pytest.mark.asyncio
    async def test_full_analytics_flow(self, handler_with_tracker, mock_tracker):
        """Test complete analytics retrieval flow."""
        # Get overall analytics
        result1 = await handler_with_tracker.handle("/api/v1/memory/analytics", {})
        assert result1.status_code == 200

        # Take a snapshot
        result2 = await handler_with_tracker.handle_post("/api/v1/memory/analytics/snapshot", {})
        assert result2.status_code == 200

        # Verify calls
        mock_tracker.get_analytics.assert_called()
        mock_tracker.take_snapshot.assert_called()

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self, handler_with_tracker):
        """Test handle returns None for paths it doesn't handle."""
        result = await handler_with_tracker.handle("/api/v1/other/route", {})
        assert result is None
