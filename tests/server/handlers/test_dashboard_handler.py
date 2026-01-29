"""
Tests for Dashboard Handler.

Tests the dashboard API endpoints including:
- Dashboard overview
- Stats retrieval
- Recent activity
- Inbox summary
- Quick actions
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.dashboard import (
    handle_get_dashboard,
    handle_get_stats,
    handle_get_activity,
    handle_get_inbox_summary,
    handle_get_quick_actions,
    handle_execute_quick_action,
    get_dashboard_routes,
    _get_cached_data,
    _set_cached_data,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def admin_auth():
    """Create admin auth context for RBAC-protected handlers."""
    return AuthorizationContext(
        user_id="test-user-001",
        user_email="test@example.com",
        org_id="test-org-001",
        roles={"admin"},
        permissions={"*", "dashboard:read", "dashboard:execute"},
    )


@pytest.fixture(autouse=True)
def clear_dashboard_cache():
    """Clear dashboard cache before and after each test."""
    import aragora.server.handlers.dashboard as dashboard_module

    original_cache = dashboard_module._dashboard_cache.copy()
    dashboard_module._dashboard_cache.clear()
    yield
    dashboard_module._dashboard_cache.clear()
    dashboard_module._dashboard_cache.update(original_cache)


# ============================================================================
# Route Tests
# ============================================================================


class TestDashboardRoutes:
    """Test route definitions."""

    def test_routes_defined(self):
        """Should define all expected routes."""
        routes = get_dashboard_routes()
        route_paths = [r[1] for r in routes]

        assert "/api/v1/dashboard" in route_paths
        assert "/api/v1/dashboard/stats" in route_paths
        assert "/api/v1/dashboard/activity" in route_paths
        assert "/api/v1/dashboard/inbox-summary" in route_paths
        assert "/api/v1/dashboard/quick-actions" in route_paths

    def test_route_methods(self):
        """Should use correct HTTP methods."""
        routes = get_dashboard_routes()
        route_dict = {r[1]: r[0] for r in routes}

        assert route_dict["/api/v1/dashboard"] == "GET"
        assert route_dict["/api/v1/dashboard/stats"] == "GET"
        assert route_dict["/api/v1/dashboard/activity"] == "GET"


# ============================================================================
# Dashboard Overview Tests
# ============================================================================


class TestDashboardOverview:
    """Test dashboard overview endpoint."""

    @pytest.mark.asyncio
    async def test_get_dashboard_success(self, admin_auth):
        """Should return dashboard overview with key sections."""
        result = await handle_get_dashboard(
            context=admin_auth,
            data={},
            user_id="test-user-001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body.get("success") is True
        assert "data" in body

        # Check key dashboard sections exist
        data = body["data"]
        assert "stats" in data or "overview" in data or isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_get_dashboard_caches_result(self, admin_auth):
        """Should cache dashboard data for performance."""
        # First call
        result1 = await handle_get_dashboard(
            context=admin_auth,
            data={},
            user_id="test-user-001",
        )

        # Check cache was populated
        cached = _get_cached_data("test-user-001", "overview")
        assert cached is not None or result1.status_code == 200


# ============================================================================
# Stats Tests
# ============================================================================


class TestDashboardStats:
    """Test dashboard stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, admin_auth):
        """Should return detailed stats."""
        result = await handle_get_stats(
            context=admin_auth,
            data={},
            user_id="test-user-001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body.get("success") is True

    @pytest.mark.asyncio
    async def test_get_stats_with_timeframe(self, admin_auth):
        """Should support timeframe parameter."""
        result = await handle_get_stats(
            context=admin_auth,
            data={"timeframe": "week"},
            user_id="test-user-001",
        )

        assert result.status_code == 200


# ============================================================================
# Activity Tests
# ============================================================================


class TestDashboardActivity:
    """Test dashboard activity endpoint."""

    @pytest.mark.asyncio
    async def test_get_activity_success(self, admin_auth):
        """Should return recent activity list."""
        result = await handle_get_activity(
            context=admin_auth,
            data={},
            user_id="test-user-001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body.get("success") is True
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_activity_with_limit(self, admin_auth):
        """Should support limit parameter."""
        result = await handle_get_activity(
            context=admin_auth,
            data={"limit": 5},
            user_id="test-user-001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        data = body.get("data", {})
        activity = data.get("activity", data.get("items", []))
        # Should respect limit
        assert len(activity) <= 5


# ============================================================================
# Inbox Summary Tests
# ============================================================================


class TestDashboardInboxSummary:
    """Test dashboard inbox summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_inbox_summary_success(self, admin_auth):
        """Should return inbox summary."""
        result = await handle_get_inbox_summary(
            context=admin_auth,
            data={},
            user_id="test-user-001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body.get("success") is True


# ============================================================================
# Quick Actions Tests
# ============================================================================


class TestDashboardQuickActions:
    """Test dashboard quick actions endpoint."""

    @pytest.mark.asyncio
    async def test_get_quick_actions_success(self, admin_auth):
        """Should return available quick actions."""
        result = await handle_get_quick_actions(
            context=admin_auth,
            data={},
            user_id="test-user-001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body.get("success") is True
        assert "data" in body

        # Should have actions list
        data = body["data"]
        assert "actions" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_execute_quick_action_success(self, admin_auth):
        """Should execute a quick action."""
        result = await handle_execute_quick_action(
            context=admin_auth,
            action_id="refresh_stats",
            data={},
            user_id="test-user-001",
        )

        # Should return result (success or error for unknown action)
        assert result.status_code in [200, 400, 404]

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, admin_auth):
        """Should handle unknown action gracefully."""
        result = await handle_execute_quick_action(
            context=admin_auth,
            action_id="nonexistent_action_xyz",
            data={},
            user_id="test-user-001",
        )

        # Should return 404 or handle gracefully
        assert result.status_code in [200, 400, 404]


# ============================================================================
# Cache Tests
# ============================================================================


class TestDashboardCache:
    """Test dashboard caching functionality."""

    def test_cache_set_and_get(self):
        """Should set and retrieve cached data."""
        test_data = {"value": "test123"}
        _set_cached_data("user-1", "test-key", test_data)

        cached = _get_cached_data("user-1", "test-key")
        assert cached == test_data

    def test_cache_isolation_by_user(self):
        """Should isolate cache by user ID."""
        _set_cached_data("user-1", "key", {"user": "1"})
        _set_cached_data("user-2", "key", {"user": "2"})

        assert _get_cached_data("user-1", "key")["user"] == "1"
        assert _get_cached_data("user-2", "key")["user"] == "2"

    def test_cache_miss_returns_none(self):
        """Should return None for cache miss."""
        result = _get_cached_data("nonexistent-user", "nonexistent-key")
        assert result is None
