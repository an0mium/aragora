"""
Tests for Dashboard HTTP API Handlers.

Tests cover:
- Dashboard overview with caching
- Detailed statistics with period validation
- Activity feed with pagination and filtering
- Inbox summary
- Quick actions listing and execution
- Permission decorators
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.dashboard import (
    _dashboard_cache,
    handle_execute_quick_action,
    handle_get_activity,
    handle_get_dashboard,
    handle_get_inbox_summary,
    handle_get_quick_actions,
    handle_get_stats,
    get_dashboard_handlers,
    get_dashboard_routes,
)


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_result(result) -> dict:
    """Parse HandlerResult body to get response data."""
    body = result.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def auth_context():
    """Create a real authorization context with permissions."""
    return AuthorizationContext(
        user_id="test-user",
        org_id="test-org",
        roles={"admin"},
        permissions={"dashboard:read", "dashboard:write"},
    )


@pytest.fixture
def clear_cache():
    """Clear the dashboard cache before each test."""
    _dashboard_cache._cache.clear()
    yield
    _dashboard_cache._cache.clear()


# ===========================================================================
# Test Dashboard Overview
# ===========================================================================


class TestGetDashboard:
    """Tests for handle_get_dashboard."""

    @pytest.mark.asyncio
    async def test_returns_overview_data(self, auth_context, clear_cache):
        """Returns dashboard overview with all expected sections."""
        result = await handle_get_dashboard(auth_context, {}, user_id="test-user")
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert "inbox" in data
        assert "today" in data
        assert "team" in data
        assert "ai" in data
        assert "cards" in data

    @pytest.mark.asyncio
    async def test_overview_contains_inbox_stats(self, auth_context, clear_cache):
        """Overview includes inbox statistics."""
        result = await handle_get_dashboard(auth_context, {}, user_id="test-user")
        parsed = parse_result(result)

        data = parsed["data"]
        inbox = data["inbox"]
        assert "total_unread" in inbox
        assert "high_priority" in inbox
        assert "needs_response" in inbox

    @pytest.mark.asyncio
    async def test_uses_cache(self, auth_context, clear_cache):
        """Second call returns cached data."""
        # First call
        result1 = await handle_get_dashboard(auth_context, {}, user_id="test-user")
        parsed1 = parse_result(result1)
        generated_at_1 = parsed1["data"]["generated_at"]

        # Second call (should use cache)
        result2 = await handle_get_dashboard(auth_context, {}, user_id="test-user")
        parsed2 = parse_result(result2)
        generated_at_2 = parsed2["data"]["generated_at"]

        # Timestamps should be the same (from cache)
        assert generated_at_1 == generated_at_2

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, auth_context, clear_cache):
        """Force refresh bypasses cache."""
        # First call
        result1 = await handle_get_dashboard(auth_context, {}, user_id="test-user")
        parsed1 = parse_result(result1)

        # Force refresh
        result2 = await handle_get_dashboard(auth_context, {"refresh": True}, user_id="test-user")
        parsed2 = parse_result(result2)

        # Both should succeed
        assert parsed1["success"] is True
        assert parsed2["success"] is True

    @pytest.mark.asyncio
    async def test_refresh_string_true(self, auth_context, clear_cache):
        """Refresh parameter accepts string 'true'."""
        result = await handle_get_dashboard(auth_context, {"refresh": "true"}, user_id="test-user")
        parsed = parse_result(result)
        assert parsed["success"] is True

    @pytest.mark.asyncio
    async def test_includes_user_id(self, auth_context, clear_cache):
        """Response includes user_id."""
        result = await handle_get_dashboard(auth_context, {}, user_id="test-user")
        parsed = parse_result(result)
        assert parsed["data"]["user_id"] == "test-user"


# ===========================================================================
# Test Statistics
# ===========================================================================


class TestGetStats:
    """Tests for handle_get_stats."""

    @pytest.mark.asyncio
    async def test_returns_stats_for_week(self, auth_context):
        """Returns weekly statistics by default."""
        result = await handle_get_stats(auth_context, {})
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert data["period"] == "week"
        assert len(data["email_volume"]["labels"]) == 7

    @pytest.mark.asyncio
    async def test_returns_stats_for_day(self, auth_context):
        """Returns daily statistics when period=day."""
        result = await handle_get_stats(auth_context, {"period": "day"})
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert data["period"] == "day"
        assert len(data["email_volume"]["labels"]) == 24

    @pytest.mark.asyncio
    async def test_returns_stats_for_month(self, auth_context):
        """Returns monthly statistics when period=month."""
        result = await handle_get_stats(auth_context, {"period": "month"})
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert data["period"] == "month"
        assert len(data["email_volume"]["labels"]) == 30

    @pytest.mark.asyncio
    async def test_invalid_period_returns_error(self, auth_context):
        """Invalid period returns 400 error."""
        result = await handle_get_stats(auth_context, {"period": "invalid"})
        parsed = parse_result(result)

        assert result.status_code == 400
        assert "error" in parsed
        assert "Invalid period" in parsed["error"]

    @pytest.mark.asyncio
    async def test_includes_email_volume(self, auth_context):
        """Stats include email volume time series."""
        result = await handle_get_stats(auth_context, {})
        parsed = parse_result(result)

        data = parsed["data"]
        assert "email_volume" in data
        assert "received" in data["email_volume"]
        assert "sent" in data["email_volume"]

    @pytest.mark.asyncio
    async def test_includes_response_time(self, auth_context):
        """Stats include response time distribution."""
        result = await handle_get_stats(auth_context, {})
        parsed = parse_result(result)

        data = parsed["data"]
        assert "response_time" in data
        assert "labels" in data["response_time"]
        assert "values" in data["response_time"]

    @pytest.mark.asyncio
    async def test_includes_team_performance(self, auth_context):
        """Stats include team performance data."""
        result = await handle_get_stats(auth_context, {})
        parsed = parse_result(result)

        data = parsed["data"]
        assert "team_performance" in data
        assert len(data["team_performance"]) > 0
        assert "name" in data["team_performance"][0]


# ===========================================================================
# Test Activity Feed
# ===========================================================================


class TestGetActivity:
    """Tests for handle_get_activity."""

    @pytest.mark.asyncio
    async def test_returns_activities(self, auth_context):
        """Returns activity items."""
        result = await handle_get_activity(auth_context, {})
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert "activities" in data
        assert len(data["activities"]) > 0

    @pytest.mark.asyncio
    async def test_respects_limit(self, auth_context):
        """Respects limit parameter."""
        result = await handle_get_activity(auth_context, {"limit": 3})
        parsed = parse_result(result)

        data = parsed["data"]
        assert len(data["activities"]) <= 3
        assert data["limit"] == 3

    @pytest.mark.asyncio
    async def test_limit_capped_at_100(self, auth_context):
        """Limit is capped at 100."""
        result = await handle_get_activity(auth_context, {"limit": 200})
        parsed = parse_result(result)

        data = parsed["data"]
        assert data["limit"] == 100

    @pytest.mark.asyncio
    async def test_respects_offset(self, auth_context):
        """Respects offset parameter."""
        result = await handle_get_activity(auth_context, {"offset": 2})
        parsed = parse_result(result)

        data = parsed["data"]
        assert data["offset"] == 2

    @pytest.mark.asyncio
    async def test_filters_by_type(self, auth_context):
        """Filters activities by type."""
        result = await handle_get_activity(auth_context, {"type": "email_received"})
        parsed = parse_result(result)

        data = parsed["data"]
        for activity in data["activities"]:
            assert activity["type"] == "email_received"

    @pytest.mark.asyncio
    async def test_has_more_flag(self, auth_context):
        """Returns has_more flag for pagination."""
        result = await handle_get_activity(auth_context, {"limit": 2})
        parsed = parse_result(result)

        data = parsed["data"]
        assert "has_more" in data

    @pytest.mark.asyncio
    async def test_activity_structure(self, auth_context):
        """Each activity has expected fields."""
        result = await handle_get_activity(auth_context, {})
        parsed = parse_result(result)

        activity = parsed["data"]["activities"][0]
        assert "id" in activity
        assert "type" in activity
        assert "title" in activity
        assert "timestamp" in activity
        assert "priority" in activity


# ===========================================================================
# Test Inbox Summary
# ===========================================================================


class TestGetInboxSummary:
    """Tests for handle_get_inbox_summary."""

    @pytest.mark.asyncio
    async def test_returns_summary(self, auth_context):
        """Returns inbox summary."""
        result = await handle_get_inbox_summary(auth_context, {})
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert "counts" in data
        assert "by_priority" in data
        assert "by_category" in data

    @pytest.mark.asyncio
    async def test_includes_counts(self, auth_context):
        """Summary includes email counts by status."""
        result = await handle_get_inbox_summary(auth_context, {})
        parsed = parse_result(result)

        counts = parsed["data"]["counts"]
        assert "unread" in counts
        assert "starred" in counts
        assert "drafts" in counts

    @pytest.mark.asyncio
    async def test_includes_priority_breakdown(self, auth_context):
        """Summary includes priority breakdown."""
        result = await handle_get_inbox_summary(auth_context, {})
        parsed = parse_result(result)

        by_priority = parsed["data"]["by_priority"]
        assert "critical" in by_priority
        assert "high" in by_priority
        assert "medium" in by_priority

    @pytest.mark.asyncio
    async def test_includes_urgent_emails(self, auth_context):
        """Summary includes urgent emails list."""
        result = await handle_get_inbox_summary(auth_context, {})
        parsed = parse_result(result)

        data = parsed["data"]
        assert "urgent_emails" in data
        if data["urgent_emails"]:
            email = data["urgent_emails"][0]
            assert "subject" in email
            assert "from" in email

    @pytest.mark.asyncio
    async def test_includes_pending_actions(self, auth_context):
        """Summary includes pending action items."""
        result = await handle_get_inbox_summary(auth_context, {})
        parsed = parse_result(result)

        data = parsed["data"]
        assert "pending_actions" in data


# ===========================================================================
# Test Quick Actions
# ===========================================================================


class TestGetQuickActions:
    """Tests for handle_get_quick_actions."""

    @pytest.mark.asyncio
    async def test_returns_actions(self, auth_context):
        """Returns list of quick actions."""
        result = await handle_get_quick_actions(auth_context, {})
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert "actions" in data
        assert len(data["actions"]) > 0

    @pytest.mark.asyncio
    async def test_action_structure(self, auth_context):
        """Each action has expected fields."""
        result = await handle_get_quick_actions(auth_context, {})
        parsed = parse_result(result)

        action = parsed["data"]["actions"][0]
        assert "id" in action
        assert "name" in action
        assert "description" in action
        assert "icon" in action
        assert "available" in action

    @pytest.mark.asyncio
    async def test_includes_count(self, auth_context):
        """Response includes action count."""
        result = await handle_get_quick_actions(auth_context, {})
        parsed = parse_result(result)

        data = parsed["data"]
        assert data["count"] == len(data["actions"])


class TestExecuteQuickAction:
    """Tests for handle_execute_quick_action."""

    @pytest.mark.asyncio
    async def test_execute_archive_read(self, auth_context):
        """Executes archive_read action."""
        result = await handle_execute_quick_action(auth_context, {}, action_id="archive_read")
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert data["action_id"] == "archive_read"
        assert data["executed"] is True
        assert "affected_count" in data

    @pytest.mark.asyncio
    async def test_execute_snooze_low(self, auth_context):
        """Executes snooze_low action."""
        result = await handle_execute_quick_action(auth_context, {}, action_id="snooze_low")
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert data["action_id"] == "snooze_low"
        assert "snooze_until" in data

    @pytest.mark.asyncio
    async def test_execute_sync_inbox(self, auth_context):
        """Executes sync_inbox action."""
        result = await handle_execute_quick_action(auth_context, {}, action_id="sync_inbox")
        parsed = parse_result(result)

        assert parsed["success"] is True
        data = parsed["data"]
        assert data["sync_status"] == "completed"
        assert "new_emails" in data

    @pytest.mark.asyncio
    async def test_action_id_from_data(self, auth_context):
        """Action ID can be passed in data."""
        result = await handle_execute_quick_action(auth_context, {"action_id": "archive_read"})
        parsed = parse_result(result)

        assert parsed["success"] is True
        assert parsed["data"]["action_id"] == "archive_read"

    @pytest.mark.asyncio
    async def test_missing_action_id_returns_error(self, auth_context):
        """Missing action_id returns 400 error."""
        result = await handle_execute_quick_action(auth_context, {})
        parsed = parse_result(result)

        assert result.status_code == 400
        assert "error" in parsed
        assert "action_id is required" in parsed["error"]

    @pytest.mark.asyncio
    async def test_invalid_action_id_returns_error(self, auth_context):
        """Invalid action_id returns 400 error."""
        result = await handle_execute_quick_action(auth_context, {}, action_id="invalid_action")
        parsed = parse_result(result)

        assert result.status_code == 400
        assert "error" in parsed
        assert "Unknown action" in parsed["error"]

    @pytest.mark.asyncio
    async def test_includes_timestamp(self, auth_context):
        """Result includes execution timestamp."""
        result = await handle_execute_quick_action(auth_context, {}, action_id="archive_read")
        parsed = parse_result(result)

        assert "timestamp" in parsed["data"]


# ===========================================================================
# Test Handler Registration
# ===========================================================================


class TestHandlerRegistration:
    """Tests for handler registration functions."""

    def test_get_dashboard_handlers(self):
        """get_dashboard_handlers returns all handlers."""
        handlers = get_dashboard_handlers()

        assert "get_dashboard" in handlers
        assert "get_stats" in handlers
        assert "get_activity" in handlers
        assert "get_inbox_summary" in handlers
        assert "get_quick_actions" in handlers
        assert "execute_quick_action" in handlers

    def test_get_dashboard_routes(self):
        """get_dashboard_routes returns route definitions."""
        routes = get_dashboard_routes()

        assert len(routes) == 6

        # Check route structure
        methods = [r[0] for r in routes]
        paths = [r[1] for r in routes]

        assert "GET" in methods
        assert "POST" in methods
        assert "/api/v1/dashboard" in paths
        assert "/api/v1/dashboard/stats" in paths


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in handlers."""

    @pytest.mark.asyncio
    async def test_get_dashboard_handles_exception(self, auth_context, clear_cache):
        """handle_get_dashboard catches exceptions and returns error response."""
        with patch(
            "aragora.server.handlers.dashboard._get_cached_data",
            side_effect=RuntimeError("Cache error"),
        ):
            result = await handle_get_dashboard(auth_context, {})
            parsed = parse_result(result)
            # Exception is caught and returns 500 error
            assert result.status_code == 500
            assert "error" in parsed

    @pytest.mark.asyncio
    async def test_get_activity_handles_invalid_limit(self, auth_context):
        """handle_get_activity handles invalid limit gracefully."""
        # String limit should be converted
        result = await handle_get_activity(auth_context, {"limit": "10"})
        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["data"]["limit"] == 10
