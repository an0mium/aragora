"""Tests for dashboard handler endpoints.

Covers all routes and behavior of the dashboard handler functions:
- GET    /api/v1/dashboard                          - Dashboard overview
- GET    /api/v1/dashboard/stats                    - Detailed stats
- GET    /api/v1/dashboard/activity                 - Recent activity feed
- GET    /api/v1/dashboard/inbox-summary            - Inbox summary
- GET    /api/v1/dashboard/quick-actions            - Available quick actions
- POST   /api/v1/dashboard/quick-actions/{action}   - Execute quick action

Also covers:
- TTL cache behaviour (cache hit, miss, forced refresh, per-user isolation)
- Period validation for stats endpoint
- Activity type filtering, pagination, offset, limit clamping
- Quick-action validation (missing/unknown action_id)
- Each individual quick-action branch
- Error paths (exception handling)
- get_dashboard_handlers() and get_dashboard_routes() registration helpers
- RBAC enforcement via @require_permission
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import HandlerResult

# Import all handler functions under test
from aragora.server.handlers.dashboard import (
    CACHE_TTL,
    _dashboard_cache,
    _get_cached_data,
    _set_cached_data,
    get_dashboard_handlers,
    get_dashboard_routes,
    handle_execute_quick_action,
    handle_get_activity,
    handle_get_dashboard,
    handle_get_inbox_summary,
    handle_get_quick_actions,
    handle_get_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


def _data(result: HandlerResult) -> dict:
    """Extract the nested 'data' payload from a success_response envelope."""
    body = _body(result)
    return body.get("data", body)


def _make_context(
    user_id: str = "test-user-001",
    permissions: set[str] | None = None,
    roles: set[str] | None = None,
) -> AuthorizationContext:
    """Build an AuthorizationContext for testing."""
    return AuthorizationContext(
        user_id=user_id,
        user_email="test@example.com",
        org_id="test-org-001",
        roles=roles if roles is not None else {"admin", "owner"},
        permissions=permissions if permissions is not None else {"*"},
    )


# Default admin context used by most tests
CTX = _make_context()


# ---------------------------------------------------------------------------
# Autouse fixture -- reset module-level cache between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_dashboard_cache():
    """Clear the module-level TTLCache before and after each test."""
    _dashboard_cache.clear()
    yield
    _dashboard_cache.clear()


# ===========================================================================
# GET /api/v1/dashboard  (handle_get_dashboard)
# ===========================================================================


class TestGetDashboard:
    """Tests for handle_get_dashboard."""

    @pytest.mark.asyncio
    async def test_returns_success(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_overview_structure(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        data = _data(result)
        for section in ("inbox", "today", "team", "ai", "cards", "user_id", "generated_at"):
            assert section in data, f"Missing section: {section}"

    @pytest.mark.asyncio
    async def test_user_id_in_response(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="alice")
        data = _data(result)
        assert data["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_default_user_id(self):
        result = await handle_get_dashboard(context=CTX, data={})
        data = _data(result)
        assert data["user_id"] == "default"

    @pytest.mark.asyncio
    async def test_inbox_stats(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        inbox = _data(result)["inbox"]
        assert "total_unread" in inbox
        assert "high_priority" in inbox
        assert "needs_response" in inbox
        assert "snoozed" in inbox
        assert "assigned_to_me" in inbox

    @pytest.mark.asyncio
    async def test_today_stats(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        today = _data(result)["today"]
        for key in ("emails_received", "emails_sent", "emails_archived",
                     "meetings_scheduled", "action_items_completed", "action_items_created"):
            assert key in today

    @pytest.mark.asyncio
    async def test_team_stats(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        team = _data(result)["team"]
        assert "active_members" in team
        assert "open_tickets" in team
        assert "avg_response_time_mins" in team
        assert "resolved_today" in team

    @pytest.mark.asyncio
    async def test_ai_stats(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        ai = _data(result)["ai"]
        for key in ("emails_categorized", "auto_responses_suggested",
                     "priority_predictions", "debates_run"):
            assert key in ai

    @pytest.mark.asyncio
    async def test_cards_list(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        cards = _data(result)["cards"]
        assert isinstance(cards, list)
        assert len(cards) == 4
        card_ids = {c["id"] for c in cards}
        assert card_ids == {"unread", "high_priority", "response_time", "resolved"}

    @pytest.mark.asyncio
    async def test_card_structure(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        card = _data(result)["cards"][0]
        for key in ("id", "title", "value", "change", "change_type", "icon"):
            assert key in card

    @pytest.mark.asyncio
    async def test_generated_at_is_iso(self):
        result = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        generated = _data(result)["generated_at"]
        # Should parse without error
        datetime.fromisoformat(generated)

    # ---- Caching ----

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Second call without refresh should return cached data."""
        r1 = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        r2 = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        assert _data(r1)["generated_at"] == _data(r2)["generated_at"]

    @pytest.mark.asyncio
    async def test_cache_per_user(self):
        """Different users get separate caches."""
        r1 = await handle_get_dashboard(context=CTX, data={}, user_id="alice")
        r2 = await handle_get_dashboard(context=CTX, data={}, user_id="bob")
        assert _data(r1)["user_id"] == "alice"
        assert _data(r2)["user_id"] == "bob"

    @pytest.mark.asyncio
    async def test_force_refresh_bool(self):
        """refresh=True bypasses cache."""
        r1 = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        # Force refresh
        r2 = await handle_get_dashboard(context=CTX, data={"refresh": True}, user_id="u1")
        # The generated_at may differ (new datetime.now call)
        assert _status(r2) == 200
        assert _data(r2)["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_force_refresh_string_true(self):
        """refresh='true' (string) bypasses cache."""
        await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        r = await handle_get_dashboard(context=CTX, data={"refresh": "true"}, user_id="u1")
        assert _status(r) == 200

    @pytest.mark.asyncio
    async def test_force_refresh_string_false(self):
        """refresh='false' uses cache."""
        r1 = await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        r2 = await handle_get_dashboard(context=CTX, data={"refresh": "false"}, user_id="u1")
        assert _data(r1)["generated_at"] == _data(r2)["generated_at"]

    @pytest.mark.asyncio
    async def test_force_refresh_string_TRUE_upper(self):
        """refresh='TRUE' (uppercase) bypasses cache."""
        await handle_get_dashboard(context=CTX, data={}, user_id="u1")
        r = await handle_get_dashboard(context=CTX, data={"refresh": "TRUE"}, user_id="u1")
        assert _status(r) == 200


# ===========================================================================
# GET /api/v1/dashboard/stats  (handle_get_stats)
# ===========================================================================


class TestGetStats:
    """Tests for handle_get_stats."""

    @pytest.mark.asyncio
    async def test_returns_success(self):
        result = await handle_get_stats(context=CTX, data={})
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_default_period_is_week(self):
        result = await handle_get_stats(context=CTX, data={})
        data = _data(result)
        assert data["period"] == "week"

    @pytest.mark.asyncio
    async def test_period_day(self):
        result = await handle_get_stats(context=CTX, data={"period": "day"})
        data = _data(result)
        assert data["period"] == "day"
        assert len(data["email_volume"]["labels"]) == 24

    @pytest.mark.asyncio
    async def test_period_week(self):
        result = await handle_get_stats(context=CTX, data={"period": "week"})
        data = _data(result)
        assert data["period"] == "week"
        assert len(data["email_volume"]["labels"]) == 7

    @pytest.mark.asyncio
    async def test_period_month(self):
        result = await handle_get_stats(context=CTX, data={"period": "month"})
        data = _data(result)
        assert data["period"] == "month"
        assert len(data["email_volume"]["labels"]) == 30

    @pytest.mark.asyncio
    async def test_invalid_period(self):
        result = await handle_get_stats(context=CTX, data={"period": "year"})
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid period" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_invalid_period_empty(self):
        result = await handle_get_stats(context=CTX, data={"period": ""})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_email_volume_structure(self):
        result = await handle_get_stats(context=CTX, data={"period": "week"})
        ev = _data(result)["email_volume"]
        assert "labels" in ev
        assert "received" in ev
        assert "sent" in ev
        assert "archived" in ev
        assert len(ev["received"]) == 7

    @pytest.mark.asyncio
    async def test_response_time_distribution(self):
        result = await handle_get_stats(context=CTX, data={})
        rt = _data(result)["response_time"]
        assert "labels" in rt
        assert "values" in rt
        assert len(rt["labels"]) == len(rt["values"])

    @pytest.mark.asyncio
    async def test_priority_distribution(self):
        result = await handle_get_stats(context=CTX, data={})
        pd = _data(result)["priority_distribution"]
        assert "labels" in pd
        assert "values" in pd
        assert len(pd["labels"]) == 4

    @pytest.mark.asyncio
    async def test_categories(self):
        result = await handle_get_stats(context=CTX, data={})
        cats = _data(result)["categories"]
        assert len(cats["labels"]) == 5

    @pytest.mark.asyncio
    async def test_team_performance(self):
        result = await handle_get_stats(context=CTX, data={})
        tp = _data(result)["team_performance"]
        assert isinstance(tp, list)
        assert len(tp) == 4
        for member in tp:
            assert "name" in member
            assert "resolved" in member
            assert "avg_response" in member

    @pytest.mark.asyncio
    async def test_top_senders(self):
        result = await handle_get_stats(context=CTX, data={})
        ts = _data(result)["top_senders"]
        assert isinstance(ts, list)
        assert len(ts) == 4
        for sender in ts:
            assert "email" in sender
            assert "count" in sender
            assert "priority" in sender

    @pytest.mark.asyncio
    async def test_summary_metrics(self):
        result = await handle_get_stats(context=CTX, data={})
        s = _data(result)["summary"]
        for key in ("total_emails", "avg_daily_emails", "response_rate",
                     "avg_response_time_mins", "ai_accuracy"):
            assert key in s

    @pytest.mark.asyncio
    async def test_generated_at_present(self):
        result = await handle_get_stats(context=CTX, data={})
        assert "generated_at" in _data(result)

    @pytest.mark.asyncio
    async def test_day_labels_format(self):
        """Day period labels should be like '00:00' .. '23:00'."""
        result = await handle_get_stats(context=CTX, data={"period": "day"})
        labels = _data(result)["email_volume"]["labels"]
        assert labels[0] == "00:00"
        assert labels[-1] == "23:00"

    @pytest.mark.asyncio
    async def test_week_labels_format(self):
        result = await handle_get_stats(context=CTX, data={"period": "week"})
        labels = _data(result)["email_volume"]["labels"]
        assert labels == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    @pytest.mark.asyncio
    async def test_month_labels_format(self):
        result = await handle_get_stats(context=CTX, data={"period": "month"})
        labels = _data(result)["email_volume"]["labels"]
        assert labels[0] == "1"
        assert labels[-1] == "30"


# ===========================================================================
# GET /api/v1/dashboard/activity  (handle_get_activity)
# ===========================================================================


class TestGetActivity:
    """Tests for handle_get_activity."""

    @pytest.mark.asyncio
    async def test_returns_success(self):
        result = await handle_get_activity(context=CTX, data={})
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_default_pagination(self):
        result = await handle_get_activity(context=CTX, data={})
        data = _data(result)
        assert data["limit"] == 20
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_returns_activities(self):
        result = await handle_get_activity(context=CTX, data={})
        data = _data(result)
        assert "activities" in data
        assert isinstance(data["activities"], list)
        assert len(data["activities"]) == 8

    @pytest.mark.asyncio
    async def test_activity_structure(self):
        result = await handle_get_activity(context=CTX, data={})
        act = _data(result)["activities"][0]
        for key in ("id", "type", "title", "description", "timestamp", "priority", "icon"):
            assert key in act

    @pytest.mark.asyncio
    async def test_total_count(self):
        result = await handle_get_activity(context=CTX, data={})
        data = _data(result)
        assert data["total"] == 8

    @pytest.mark.asyncio
    async def test_has_more_false(self):
        result = await handle_get_activity(context=CTX, data={})
        assert _data(result)["has_more"] is False

    @pytest.mark.asyncio
    async def test_limit_param(self):
        result = await handle_get_activity(context=CTX, data={"limit": 3})
        data = _data(result)
        assert len(data["activities"]) == 3
        assert data["limit"] == 3
        assert data["has_more"] is True

    @pytest.mark.asyncio
    async def test_offset_param(self):
        result = await handle_get_activity(context=CTX, data={"offset": 5})
        data = _data(result)
        assert data["offset"] == 5
        assert len(data["activities"]) == 3  # 8 - 5

    @pytest.mark.asyncio
    async def test_limit_and_offset(self):
        result = await handle_get_activity(context=CTX, data={"limit": 2, "offset": 3})
        data = _data(result)
        assert data["limit"] == 2
        assert data["offset"] == 3
        assert len(data["activities"]) == 2

    @pytest.mark.asyncio
    async def test_limit_clamped_to_100(self):
        result = await handle_get_activity(context=CTX, data={"limit": 200})
        data = _data(result)
        assert data["limit"] == 100

    @pytest.mark.asyncio
    async def test_limit_clamped_to_1(self):
        result = await handle_get_activity(context=CTX, data={"limit": 0})
        data = _data(result)
        assert data["limit"] == 1

    @pytest.mark.asyncio
    async def test_negative_limit_clamped(self):
        result = await handle_get_activity(context=CTX, data={"limit": -5})
        data = _data(result)
        assert data["limit"] == 1

    @pytest.mark.asyncio
    async def test_negative_offset_clamped(self):
        result = await handle_get_activity(context=CTX, data={"offset": -10})
        data = _data(result)
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_filter_by_type_email_received(self):
        result = await handle_get_activity(context=CTX, data={"type": "email_received"})
        data = _data(result)
        assert all(a["type"] == "email_received" for a in data["activities"])
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_filter_by_type_mention(self):
        result = await handle_get_activity(context=CTX, data={"type": "mention"})
        data = _data(result)
        assert data["total"] == 1
        assert data["activities"][0]["id"] == "act_004"

    @pytest.mark.asyncio
    async def test_filter_by_type_nonexistent(self):
        result = await handle_get_activity(context=CTX, data={"type": "nonexistent"})
        data = _data(result)
        assert data["total"] == 0
        assert data["activities"] == []

    @pytest.mark.asyncio
    async def test_filter_combined_with_pagination(self):
        result = await handle_get_activity(
            context=CTX, data={"type": "email_received", "limit": 1, "offset": 0}
        )
        data = _data(result)
        assert data["total"] == 1
        assert len(data["activities"]) == 1

    @pytest.mark.asyncio
    async def test_offset_beyond_total(self):
        result = await handle_get_activity(context=CTX, data={"offset": 100})
        data = _data(result)
        assert data["activities"] == []
        assert data["has_more"] is False

    @pytest.mark.asyncio
    async def test_activity_types_present(self):
        """All 8 activity types should be present."""
        result = await handle_get_activity(context=CTX, data={})
        types = {a["type"] for a in _data(result)["activities"]}
        expected = {
            "email_received", "email_sent", "action_completed",
            "mention", "assignment", "email_archived",
            "meeting_scheduled", "ai_suggestion",
        }
        assert types == expected


# ===========================================================================
# GET /api/v1/dashboard/inbox-summary  (handle_get_inbox_summary)
# ===========================================================================


class TestGetInboxSummary:
    """Tests for handle_get_inbox_summary."""

    @pytest.mark.asyncio
    async def test_returns_success(self):
        result = await handle_get_inbox_summary(context=CTX, data={})
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_generated_at_present(self):
        data = _data(await handle_get_inbox_summary(context=CTX, data={}))
        assert "generated_at" in data
        datetime.fromisoformat(data["generated_at"])

    @pytest.mark.asyncio
    async def test_counts_structure(self):
        counts = _data(await handle_get_inbox_summary(context=CTX, data={}))["counts"]
        for key in ("unread", "starred", "snoozed", "drafts", "trash"):
            assert key in counts
            assert isinstance(counts[key], int)

    @pytest.mark.asyncio
    async def test_by_priority(self):
        bp = _data(await handle_get_inbox_summary(context=CTX, data={}))["by_priority"]
        for key in ("critical", "high", "medium", "low"):
            assert key in bp

    @pytest.mark.asyncio
    async def test_by_category(self):
        bc = _data(await handle_get_inbox_summary(context=CTX, data={}))["by_category"]
        for key in ("inbox", "updates", "promotions", "social", "forums"):
            assert key in bc

    @pytest.mark.asyncio
    async def test_top_labels(self):
        labels = _data(await handle_get_inbox_summary(context=CTX, data={}))["top_labels"]
        assert isinstance(labels, list)
        assert len(labels) == 4
        for label in labels:
            assert "name" in label
            assert "count" in label
            assert "color" in label

    @pytest.mark.asyncio
    async def test_urgent_emails(self):
        urgent = _data(await handle_get_inbox_summary(context=CTX, data={}))["urgent_emails"]
        assert isinstance(urgent, list)
        assert len(urgent) == 2
        for email in urgent:
            assert "id" in email
            assert "subject" in email
            assert "from" in email
            assert "received_at" in email
            assert "snippet" in email

    @pytest.mark.asyncio
    async def test_pending_actions(self):
        actions = _data(await handle_get_inbox_summary(context=CTX, data={}))["pending_actions"]
        assert isinstance(actions, list)
        assert len(actions) == 2
        for action in actions:
            assert "id" in action
            assert "title" in action
            assert "deadline" in action
            assert "from_email" in action

    @pytest.mark.asyncio
    async def test_urgent_email_ids(self):
        urgent = _data(await handle_get_inbox_summary(context=CTX, data={}))["urgent_emails"]
        ids = {e["id"] for e in urgent}
        assert ids == {"msg_001", "msg_002"}

    @pytest.mark.asyncio
    async def test_pending_action_ids(self):
        actions = _data(await handle_get_inbox_summary(context=CTX, data={}))["pending_actions"]
        ids = {a["id"] for a in actions}
        assert ids == {"action_001", "action_002"}


# ===========================================================================
# GET /api/v1/dashboard/quick-actions  (handle_get_quick_actions)
# ===========================================================================


class TestGetQuickActions:
    """Tests for handle_get_quick_actions."""

    @pytest.mark.asyncio
    async def test_returns_success(self):
        result = await handle_get_quick_actions(context=CTX, data={})
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_actions_list(self):
        data = _data(await handle_get_quick_actions(context=CTX, data={}))
        assert "actions" in data
        assert isinstance(data["actions"], list)
        assert len(data["actions"]) == 6

    @pytest.mark.asyncio
    async def test_count_field(self):
        data = _data(await handle_get_quick_actions(context=CTX, data={}))
        assert data["count"] == 6

    @pytest.mark.asyncio
    async def test_action_ids(self):
        actions = _data(await handle_get_quick_actions(context=CTX, data={}))["actions"]
        ids = {a["id"] for a in actions}
        expected = {
            "archive_read", "snooze_low", "mark_spam",
            "complete_actions", "ai_respond", "sync_inbox",
        }
        assert ids == expected

    @pytest.mark.asyncio
    async def test_action_structure(self):
        actions = _data(await handle_get_quick_actions(context=CTX, data={}))["actions"]
        for action in actions:
            for key in ("id", "name", "description", "icon", "available"):
                assert key in action, f"Action {action.get('id')} missing key '{key}'"
            assert "estimated_count" in action

    @pytest.mark.asyncio
    async def test_all_available(self):
        actions = _data(await handle_get_quick_actions(context=CTX, data={}))["actions"]
        assert all(a["available"] is True for a in actions)

    @pytest.mark.asyncio
    async def test_sync_inbox_has_null_count(self):
        actions = _data(await handle_get_quick_actions(context=CTX, data={}))["actions"]
        sync = next(a for a in actions if a["id"] == "sync_inbox")
        assert sync["estimated_count"] is None


# ===========================================================================
# POST /api/v1/dashboard/quick-actions/{action}  (handle_execute_quick_action)
# ===========================================================================


class TestExecuteQuickAction:
    """Tests for handle_execute_quick_action."""

    # ---- Missing / invalid action_id ----

    @pytest.mark.asyncio
    async def test_missing_action_id(self):
        result = await handle_execute_quick_action(context=CTX, data={})
        assert _status(result) == 400
        assert "action_id is required" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_empty_action_id(self):
        result = await handle_execute_quick_action(context=CTX, data={"action_id": ""})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unknown_action_id(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="nope")
        assert _status(result) == 400
        assert "Unknown action" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_action_id_from_param(self):
        """action_id passed as keyword takes precedence."""
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="sync_inbox")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_action_id_from_body(self):
        """action_id can come from data dict when param is empty."""
        result = await handle_execute_quick_action(context=CTX, data={"action_id": "sync_inbox"})
        assert _status(result) == 200

    # ---- Individual action branches ----

    @pytest.mark.asyncio
    async def test_archive_read(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="archive_read")
        data = _data(result)
        assert data["action_id"] == "archive_read"
        assert data["executed"] is True
        assert data["affected_count"] == 45
        assert "Archived" in data["message"]

    @pytest.mark.asyncio
    async def test_snooze_low(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="snooze_low")
        data = _data(result)
        assert data["action_id"] == "snooze_low"
        assert data["affected_count"] == 12
        assert "snooze_until" in data
        assert "Snoozed" in data["message"]

    @pytest.mark.asyncio
    async def test_mark_spam(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="mark_spam")
        data = _data(result)
        assert data["action_id"] == "mark_spam"
        assert data["affected_count"] == 8
        assert "spam" in data["message"]

    @pytest.mark.asyncio
    async def test_complete_actions(self):
        result = await handle_execute_quick_action(
            context=CTX, data={}, action_id="complete_actions"
        )
        data = _data(result)
        assert data["action_id"] == "complete_actions"
        assert data["affected_count"] == 3
        assert "Completed" in data["message"]

    @pytest.mark.asyncio
    async def test_ai_respond(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="ai_respond")
        data = _data(result)
        assert data["action_id"] == "ai_respond"
        assert data["affected_count"] == 5
        assert data["drafts_created"] == 5
        assert "draft" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_sync_inbox(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="sync_inbox")
        data = _data(result)
        assert data["action_id"] == "sync_inbox"
        assert data["affected_count"] == 0
        assert data["sync_status"] == "completed"
        assert data["new_emails"] == 3

    # ---- Common fields ----

    @pytest.mark.asyncio
    async def test_executed_flag(self):
        for action in ("archive_read", "snooze_low", "mark_spam",
                        "complete_actions", "ai_respond", "sync_inbox"):
            result = await handle_execute_quick_action(context=CTX, data={}, action_id=action)
            assert _data(result)["executed"] is True

    @pytest.mark.asyncio
    async def test_timestamp_in_result(self):
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="sync_inbox")
        data = _data(result)
        assert "timestamp" in data
        datetime.fromisoformat(data["timestamp"])

    @pytest.mark.asyncio
    async def test_user_id_default(self):
        """user_id defaults to 'default' -- action still succeeds."""
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="archive_read")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_with_options(self):
        """Options dict is accepted (but not validated currently)."""
        result = await handle_execute_quick_action(
            context=CTX,
            data={"options": {"dry_run": True}},
            action_id="archive_read",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_with_confirm(self):
        result = await handle_execute_quick_action(
            context=CTX,
            data={"confirm": True},
            action_id="mark_spam",
        )
        assert _status(result) == 200


# ===========================================================================
# Cache helper functions
# ===========================================================================


class TestCacheHelpers:
    """Tests for _get_cached_data and _set_cached_data."""

    def test_cache_miss(self):
        assert _get_cached_data("u1", "overview") is None

    def test_cache_set_and_get(self):
        _set_cached_data("u1", "overview", {"hello": "world"})
        result = _get_cached_data("u1", "overview")
        assert result == {"hello": "world"}

    def test_cache_different_users(self):
        _set_cached_data("alice", "overview", {"user": "alice"})
        _set_cached_data("bob", "overview", {"user": "bob"})
        assert _get_cached_data("alice", "overview")["user"] == "alice"
        assert _get_cached_data("bob", "overview")["user"] == "bob"

    def test_cache_different_keys(self):
        _set_cached_data("u1", "key_a", {"key": "a"})
        _set_cached_data("u1", "key_b", {"key": "b"})
        assert _get_cached_data("u1", "key_a")["key"] == "a"
        assert _get_cached_data("u1", "key_b")["key"] == "b"

    def test_cache_ttl_constant(self):
        assert CACHE_TTL == 30


# ===========================================================================
# Handler / Route Registration
# ===========================================================================


class TestRegistration:
    """Tests for get_dashboard_handlers and get_dashboard_routes."""

    def test_get_dashboard_handlers_returns_dict(self):
        handlers = get_dashboard_handlers()
        assert isinstance(handlers, dict)

    def test_get_dashboard_handlers_keys(self):
        handlers = get_dashboard_handlers()
        expected_keys = {
            "get_dashboard", "get_stats", "get_activity",
            "get_inbox_summary", "get_quick_actions", "execute_quick_action",
        }
        assert set(handlers.keys()) == expected_keys

    def test_get_dashboard_handlers_values_callable(self):
        handlers = get_dashboard_handlers()
        for name, handler in handlers.items():
            assert callable(handler), f"Handler '{name}' is not callable"

    def test_get_dashboard_routes_returns_list(self):
        routes = get_dashboard_routes()
        assert isinstance(routes, list)

    def test_get_dashboard_routes_count(self):
        routes = get_dashboard_routes()
        assert len(routes) == 6

    def test_get_dashboard_routes_structure(self):
        routes = get_dashboard_routes()
        for method, path, handler in routes:
            assert method in ("GET", "POST")
            assert path.startswith("/api/v1/dashboard")
            assert callable(handler)

    def test_get_routes_methods(self):
        routes = get_dashboard_routes()
        methods = [m for m, _, _ in routes]
        assert methods.count("GET") == 5
        assert methods.count("POST") == 1

    def test_get_routes_paths(self):
        routes = get_dashboard_routes()
        paths = [p for _, p, _ in routes]
        expected = [
            "/api/v1/dashboard",
            "/api/v1/dashboard/stats",
            "/api/v1/dashboard/activity",
            "/api/v1/dashboard/inbox-summary",
            "/api/v1/dashboard/quick-actions",
            "/api/v1/dashboard/quick-actions/{action}",
        ]
        assert paths == expected

    def test_get_routes_handler_mapping(self):
        routes = get_dashboard_routes()
        handler_map = {p: h for _, p, h in routes}
        assert handler_map["/api/v1/dashboard"] is handle_get_dashboard
        assert handler_map["/api/v1/dashboard/stats"] is handle_get_stats
        assert handler_map["/api/v1/dashboard/activity"] is handle_get_activity
        assert handler_map["/api/v1/dashboard/inbox-summary"] is handle_get_inbox_summary
        assert handler_map["/api/v1/dashboard/quick-actions"] is handle_get_quick_actions
        assert handler_map["/api/v1/dashboard/quick-actions/{action}"] is handle_execute_quick_action


# ===========================================================================
# Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error paths in dashboard handlers."""

    @pytest.mark.asyncio
    async def test_get_dashboard_handles_type_error(self):
        """Passing non-dict data that triggers TypeError in .get()."""
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=TypeError("bad"))
        result = await handle_get_dashboard(context=CTX, data=bad_data, user_id="u1")
        assert _status(result) == 500
        assert "Dashboard loading failed" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_get_stats_handles_type_error(self):
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=TypeError("bad"))
        result = await handle_get_stats(context=CTX, data=bad_data)
        assert _status(result) == 500
        assert "Statistics retrieval failed" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_get_activity_handles_type_error(self):
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=TypeError("bad"))
        result = await handle_get_activity(context=CTX, data=bad_data)
        assert _status(result) == 500
        assert "Activity retrieval failed" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_get_inbox_summary_handles_type_error(self):
        """TypeError in inbox summary handler triggers 500."""
        # inbox_summary doesn't call data.get, so we need to trigger
        # the exception differently -- patch datetime.now to raise TypeError
        from unittest.mock import patch as _patch

        with _patch(
            "aragora.server.handlers.dashboard.datetime"
        ) as mock_dt:
            mock_dt.now.side_effect = TypeError("bad")
            mock_dt.side_effect = TypeError("bad")
            result = await handle_get_inbox_summary(context=CTX, data={})
        assert _status(result) == 500
        assert "Inbox summary failed" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_get_quick_actions_handles_type_error(self):
        """TypeError from quick-actions handler triggers 500 -- patch list creation."""
        from unittest.mock import patch as _patch

        with _patch(
            "aragora.server.handlers.dashboard.success_response",
            side_effect=TypeError("bad"),
        ):
            result = await handle_get_quick_actions(context=CTX, data={})
        assert _status(result) == 500
        assert "Failed to retrieve actions" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_execute_quick_action_handles_type_error(self):
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=TypeError("bad"))
        result = await handle_execute_quick_action(context=CTX, data=bad_data)
        assert _status(result) == 500
        assert "Action execution failed" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_get_stats_value_error(self):
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=ValueError("bad"))
        result = await handle_get_stats(context=CTX, data=bad_data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_activity_value_error(self):
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=ValueError("bad"))
        result = await handle_get_activity(context=CTX, data=bad_data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_dashboard_key_error(self):
        """KeyError in dashboard handler triggers 500."""
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=KeyError("bad"))
        result = await handle_get_dashboard(context=CTX, data=bad_data, user_id="u1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_execute_quick_action_value_error(self):
        bad_data = MagicMock()
        bad_data.get = MagicMock(side_effect=ValueError("bad"))
        result = await handle_execute_quick_action(context=CTX, data=bad_data)
        assert _status(result) == 500


# ===========================================================================
# RBAC Enforcement
# ===========================================================================


class TestRBACEnforcement:
    """Tests to verify RBAC permissions are correctly declared."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_dashboard_requires_permission(self):
        """handle_get_dashboard requires 'dashboard:read'."""
        from aragora.rbac.decorators import PermissionDeniedError

        ctx = _make_context(permissions=set())  # No permissions
        with pytest.raises(PermissionDeniedError):
            await handle_get_dashboard(context=ctx, data={})

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_stats_requires_permission(self):
        from aragora.rbac.decorators import PermissionDeniedError

        ctx = _make_context(permissions=set())
        with pytest.raises(PermissionDeniedError):
            await handle_get_stats(context=ctx, data={})

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_activity_requires_permission(self):
        from aragora.rbac.decorators import PermissionDeniedError

        ctx = _make_context(permissions=set())
        with pytest.raises(PermissionDeniedError):
            await handle_get_activity(context=ctx, data={})

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_inbox_summary_requires_permission(self):
        from aragora.rbac.decorators import PermissionDeniedError

        ctx = _make_context(permissions=set())
        with pytest.raises(PermissionDeniedError):
            await handle_get_inbox_summary(context=ctx, data={})

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_quick_actions_requires_permission(self):
        from aragora.rbac.decorators import PermissionDeniedError

        ctx = _make_context(permissions=set())
        with pytest.raises(PermissionDeniedError):
            await handle_get_quick_actions(context=ctx, data={})

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_execute_quick_action_requires_write_permission(self):
        """handle_execute_quick_action requires 'dashboard:write'."""
        from aragora.rbac.decorators import PermissionDeniedError

        ctx = _make_context(permissions={"dashboard:read"})  # read but not write
        with pytest.raises(PermissionDeniedError):
            await handle_execute_quick_action(context=ctx, data={}, action_id="sync_inbox")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_read_permission_grants_dashboard(self):
        """dashboard:read is sufficient for read endpoints."""
        ctx = _make_context(permissions={"dashboard:read"})
        result = await handle_get_dashboard(context=ctx, data={})
        assert _status(result) == 200

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_write_permission_grants_execute(self):
        """dashboard:write is sufficient for execute endpoint."""
        ctx = _make_context(permissions={"dashboard:write"})
        result = await handle_execute_quick_action(
            context=ctx, data={}, action_id="sync_inbox"
        )
        assert _status(result) == 200

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_wildcard_permission_grants_all(self):
        """Wildcard '*' permission grants access to everything."""
        ctx = _make_context(permissions={"*"})
        result = await handle_get_dashboard(context=ctx, data={})
        assert _status(result) == 200
        result2 = await handle_execute_quick_action(
            context=ctx, data={}, action_id="sync_inbox"
        )
        assert _status(result2) == 200


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Additional edge-case tests."""

    @pytest.mark.asyncio
    async def test_activity_string_limit(self):
        """Limit as string should work (int conversion)."""
        result = await handle_get_activity(context=CTX, data={"limit": "5"})
        data = _data(result)
        assert data["limit"] == 5

    @pytest.mark.asyncio
    async def test_activity_string_offset(self):
        """Offset as string should work (int conversion)."""
        result = await handle_get_activity(context=CTX, data={"offset": "2"})
        data = _data(result)
        assert data["offset"] == 2

    @pytest.mark.asyncio
    async def test_execute_action_all_valid_actions(self):
        """All valid action IDs should return 200."""
        valid = ["archive_read", "snooze_low", "mark_spam",
                 "complete_actions", "ai_respond", "sync_inbox"]
        for action in valid:
            result = await handle_execute_quick_action(context=CTX, data={}, action_id=action)
            assert _status(result) == 200, f"Action '{action}' failed"

    @pytest.mark.asyncio
    async def test_execute_action_case_sensitive(self):
        """Action IDs are case-sensitive."""
        result = await handle_execute_quick_action(context=CTX, data={}, action_id="SYNC_INBOX")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_stats_period_case_sensitive(self):
        """Period values are case-sensitive."""
        result = await handle_get_stats(context=CTX, data={"period": "Day"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_snooze_until_is_future(self):
        """snooze_until should be approximately 1 day in the future."""
        result = await handle_execute_quick_action(
            context=CTX, data={}, action_id="snooze_low"
        )
        snooze_until = datetime.fromisoformat(_data(result)["snooze_until"])
        now = datetime.now(timezone.utc)
        diff = snooze_until - now
        # Should be roughly 1 day (between 23h and 25h to allow timing variance)
        assert 23 * 3600 < diff.total_seconds() < 25 * 3600

    @pytest.mark.asyncio
    async def test_activity_limit_exactly_1(self):
        result = await handle_get_activity(context=CTX, data={"limit": 1})
        data = _data(result)
        assert data["limit"] == 1
        assert len(data["activities"]) == 1

    @pytest.mark.asyncio
    async def test_activity_limit_exactly_100(self):
        result = await handle_get_activity(context=CTX, data={"limit": 100})
        data = _data(result)
        assert data["limit"] == 100

    @pytest.mark.asyncio
    async def test_multiple_quick_actions_in_sequence(self):
        """Execute multiple actions in sequence -- all should succeed independently."""
        for action in ("archive_read", "ai_respond", "sync_inbox"):
            result = await handle_execute_quick_action(context=CTX, data={}, action_id=action)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_execute_with_explicit_user_id(self):
        result = await handle_execute_quick_action(
            context=CTX, data={}, action_id="sync_inbox", user_id="alice"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_activity_filter_email_sent(self):
        result = await handle_get_activity(context=CTX, data={"type": "email_sent"})
        data = _data(result)
        assert data["total"] == 1
        assert data["activities"][0]["type"] == "email_sent"

    @pytest.mark.asyncio
    async def test_activity_filter_ai_suggestion(self):
        result = await handle_get_activity(context=CTX, data={"type": "ai_suggestion"})
        data = _data(result)
        assert data["total"] == 1
        assert data["activities"][0]["id"] == "act_008"

    @pytest.mark.asyncio
    async def test_dashboard_overview_populates_cache(self):
        """After calling get_dashboard, cache should contain data."""
        await handle_get_dashboard(context=CTX, data={}, user_id="cachetest")
        cached = _get_cached_data("cachetest", "overview")
        assert cached is not None
        assert cached["user_id"] == "cachetest"

    @pytest.mark.asyncio
    async def test_stats_email_volume_data_lengths_match(self):
        """received/sent/archived arrays match label count."""
        result = await handle_get_stats(context=CTX, data={"period": "day"})
        ev = _data(result)["email_volume"]
        n = len(ev["labels"])
        assert len(ev["received"]) == n
        assert len(ev["sent"]) == n
        assert len(ev["archived"]) == n

    @pytest.mark.asyncio
    async def test_inbox_summary_label_colors_are_hex(self):
        labels = _data(await handle_get_inbox_summary(context=CTX, data={}))["top_labels"]
        for label in labels:
            assert label["color"].startswith("#")
            assert len(label["color"]) == 7  # #RRGGBB

    @pytest.mark.asyncio
    async def test_inbox_counts_values(self):
        counts = _data(await handle_get_inbox_summary(context=CTX, data={}))["counts"]
        assert counts["unread"] == 42
        assert counts["starred"] == 15
        assert counts["snoozed"] == 5
        assert counts["drafts"] == 3
        assert counts["trash"] == 28

    @pytest.mark.asyncio
    async def test_priority_values(self):
        bp = _data(await handle_get_inbox_summary(context=CTX, data={}))["by_priority"]
        assert bp["critical"] == 2
        assert bp["high"] == 7
        assert bp["medium"] == 25
        assert bp["low"] == 8

    @pytest.mark.asyncio
    async def test_dashboard_inbox_values(self):
        inbox = _data(await handle_get_dashboard(context=CTX, data={}, user_id="u1"))["inbox"]
        assert inbox["total_unread"] == 42
        assert inbox["high_priority"] == 7
        assert inbox["needs_response"] == 12
        assert inbox["snoozed"] == 5
        assert inbox["assigned_to_me"] == 8

    @pytest.mark.asyncio
    async def test_dashboard_today_values(self):
        today = _data(await handle_get_dashboard(context=CTX, data={}, user_id="u1"))["today"]
        assert today["emails_received"] == 28
        assert today["emails_sent"] == 15
        assert today["emails_archived"] == 23
        assert today["meetings_scheduled"] == 3
        assert today["action_items_completed"] == 5
        assert today["action_items_created"] == 8

    @pytest.mark.asyncio
    async def test_dashboard_team_values(self):
        team = _data(await handle_get_dashboard(context=CTX, data={}, user_id="u1"))["team"]
        assert team["active_members"] == 5
        assert team["open_tickets"] == 34
        assert team["avg_response_time_mins"] == 45
        assert team["resolved_today"] == 12

    @pytest.mark.asyncio
    async def test_dashboard_ai_values(self):
        ai = _data(await handle_get_dashboard(context=CTX, data={}, user_id="u1"))["ai"]
        assert ai["emails_categorized"] == 156
        assert ai["auto_responses_suggested"] == 23
        assert ai["priority_predictions"] == 89
        assert ai["debates_run"] == 3

    @pytest.mark.asyncio
    async def test_stats_summary_values(self):
        s = _data(await handle_get_stats(context=CTX, data={}))["summary"]
        assert s["total_emails"] == 342
        assert s["avg_daily_emails"] == 49
        assert s["response_rate"] == 0.87
        assert s["avg_response_time_mins"] == 45
        assert s["ai_accuracy"] == 0.94

    @pytest.mark.asyncio
    async def test_activity_assignment_item(self):
        result = await handle_get_activity(context=CTX, data={"type": "assignment"})
        data = _data(result)
        assert data["total"] == 1
        act = data["activities"][0]
        assert act["id"] == "act_005"
        assert act["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_activity_meeting_scheduled(self):
        result = await handle_get_activity(context=CTX, data={"type": "meeting_scheduled"})
        data = _data(result)
        assert data["total"] == 1
        assert data["activities"][0]["id"] == "act_007"

    @pytest.mark.asyncio
    async def test_activity_action_completed(self):
        result = await handle_get_activity(context=CTX, data={"type": "action_completed"})
        data = _data(result)
        assert data["total"] == 1
        assert data["activities"][0]["id"] == "act_003"

    @pytest.mark.asyncio
    async def test_activity_email_archived(self):
        result = await handle_get_activity(context=CTX, data={"type": "email_archived"})
        data = _data(result)
        assert data["total"] == 1
        assert data["activities"][0]["id"] == "act_006"
