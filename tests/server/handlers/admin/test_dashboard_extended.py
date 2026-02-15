"""
Extended tests for the admin dashboard handler.

Tests cover methods and edge cases NOT covered in test_dashboard.py:
- can_handle edge cases (gastown exclusion, path segment counts, detail routes)
- handle routing for labels, inbox-summary, quick-actions, urgent, pending, search, export
- handle_post routing for quick-action execution, urgent dismiss, pending complete, export
- Auth error paths (401 Unauthorized, 403 Forbidden) for both GET and POST
- _get_dashboard_debate with empty debate_id
- _execute_quick_action with empty and valid action IDs
- _dismiss_urgent_item with storage not found, storage error, no storage
- _complete_pending_action with not found, storage error, no storage
- _get_labels with and without storage
- _get_inbox_summary with and without storage
- _get_top_senders with and without storage
- _call_bypassing_decorators utility
- _get_connector_type with various connector classes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.admin.dashboard import (
    DashboardHandler,
    _call_bypassing_decorators,
)


# =============================================================================
# Helpers
# =============================================================================


def _parse_body(result) -> dict:
    """Parse HandlerResult body bytes into a dict."""
    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a DashboardHandler with mocked context."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "debate_embeddings": None,
        "critique_store": None,
        "calibration_tracker": None,
        "performance_monitor": None,
        "prompt_evolver": None,
    }
    return DashboardHandler(ctx)


@pytest.fixture
def mock_storage():
    """Create mock storage with connection context manager."""
    storage = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    storage.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    storage.connection.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor

    return storage, mock_cursor


def _make_auth_handler(handler):
    """Patch handler to bypass auth and rate limiting, returning mock_handler."""
    mock_http = MagicMock()
    mock_http.client_address = ("127.0.0.1", 12345)
    mock_auth_context = MagicMock()
    return mock_http, mock_auth_context


# =============================================================================
# can_handle Edge Cases
# =============================================================================


class TestCanHandleEdgeCases:
    """Tests for can_handle with various path patterns."""

    def test_excludes_gastown_prefix(self, handler):
        """Paths under /api/v1/dashboard/gastown/ are excluded."""
        assert handler.can_handle("/api/v1/dashboard/gastown/something") is False

    def test_debate_detail_valid_path(self, handler):
        """Debate detail path with exactly 6 segments is handled."""
        assert handler.can_handle("/api/v1/dashboard/debates/abc123") is True

    def test_debate_detail_too_many_segments(self, handler):
        """Debate detail path with more than 6 segments is not handled."""
        assert handler.can_handle("/api/v1/dashboard/debates/abc/extra") is False

    def test_team_performance_detail_valid(self, handler):
        """Team performance detail with 6 segments is handled."""
        assert handler.can_handle("/api/v1/dashboard/team-performance/claude") is True

    def test_quick_action_detail_valid(self, handler):
        """Quick action detail path with 6 segments is handled."""
        assert handler.can_handle("/api/v1/dashboard/quick-actions/archive_read") is True

    def test_urgent_dismiss_valid(self, handler):
        """Urgent dismiss path with correct format is handled."""
        assert handler.can_handle("/api/v1/dashboard/urgent/item1/dismiss") is True

    def test_urgent_dismiss_wrong_segments(self, handler):
        """Urgent dismiss with wrong segment count is not handled."""
        assert handler.can_handle("/api/v1/dashboard/urgent/dismiss") is False

    def test_pending_complete_valid(self, handler):
        """Pending action complete path with correct format is handled."""
        assert handler.can_handle("/api/v1/dashboard/pending-actions/a1/complete") is True

    def test_pending_complete_wrong_segments(self, handler):
        """Pending complete with wrong segment count is not handled."""
        assert handler.can_handle("/api/v1/dashboard/pending-actions/complete") is False

    def test_unrecognized_dashboard_path(self, handler):
        """Unrecognized sub-path under /api/v1/dashboard/ is not handled."""
        assert handler.can_handle("/api/v1/dashboard/unknown-path") is False


# =============================================================================
# handle Routing Tests (GET paths not covered in test_dashboard.py)
# =============================================================================


class TestHandleRoutingExtended:
    """Tests for handle() routing to various GET endpoints."""

    @pytest.mark.asyncio
    async def test_handle_routes_to_stats(self, handler):
        """Handle routes /api/v1/dashboard/stats correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_get_dashboard_stats") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle("/api/v1/dashboard/stats", {}, mock_http)
                        mock_m.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_routes_to_labels(self, handler):
        """Handle routes /api/v1/dashboard/labels correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_get_labels") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle("/api/v1/dashboard/labels", {}, mock_http)
                        mock_m.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_routes_to_inbox_summary(self, handler):
        """Handle routes /api/v1/dashboard/inbox-summary correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_get_inbox_summary") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle("/api/v1/dashboard/inbox-summary", {}, mock_http)
                        mock_m.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_routes_to_search(self, handler):
        """Handle routes /api/v1/dashboard/search with query param."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_search_dashboard") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle("/api/v1/dashboard/search", {"q": "test"}, mock_http)
                        mock_m.assert_called_once_with("test")


# =============================================================================
# handle Auth Error Paths
# =============================================================================


class TestHandleAuthErrors:
    """Tests for authentication and authorization error handling."""

    @pytest.mark.asyncio
    async def test_handle_returns_401_on_unauthorized(self, handler):
        """Handle returns 401 when authentication fails."""
        from aragora.server.handlers.secure import UnauthorizedError

        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(
                handler,
                "get_auth_context",
                new_callable=AsyncMock,
                side_effect=UnauthorizedError("No token"),
            ):
                result = await handler.handle("/api/v1/dashboard/debates", {}, mock_http)
                data = _parse_body(result)
                assert data.get("error") == "Authentication required"

    @pytest.mark.asyncio
    async def test_handle_returns_403_on_forbidden(self, handler):
        """Handle returns 403 when permission check fails."""
        from aragora.server.handlers.secure import ForbiddenError

        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(
                    handler,
                    "check_permission",
                    side_effect=ForbiddenError("Denied"),
                ):
                    result = await handler.handle("/api/v1/dashboard/debates", {}, mock_http)
                    data = _parse_body(result)
                    assert "denied" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_post_returns_401_on_unauthorized(self, handler):
        """handle_post returns 401 when authentication fails."""
        from aragora.server.handlers.secure import UnauthorizedError

        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(
                handler,
                "get_auth_context",
                new_callable=AsyncMock,
                side_effect=UnauthorizedError("No token"),
            ):
                result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)
                data = _parse_body(result)
                assert data.get("error") == "Authentication required"

    @pytest.mark.asyncio
    async def test_handle_post_returns_403_on_forbidden(self, handler):
        """handle_post returns 403 when permission check fails."""
        from aragora.server.handlers.secure import ForbiddenError

        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(
                    handler,
                    "check_permission",
                    side_effect=ForbiddenError("Write denied"),
                ):
                    result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)
                    data = _parse_body(result)
                    assert "denied" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_post_rate_limited(self, handler):
        """handle_post returns 429 when rate limited."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = False
            result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)
            assert result is not None
            data = _parse_body(result)
            assert "Rate limit" in data.get("error", "")


# =============================================================================
# handle_post Routing Tests
# =============================================================================


class TestHandlePostRouting:
    """Tests for handle_post() routing to write endpoints."""

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_export(self, handler):
        """handle_post routes /api/v1/dashboard/export correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_export_dashboard_data") as mock_m:
                        mock_m.return_value = {"data": {}}
                        result = await handler.handle_post(
                            "/api/v1/dashboard/export", {}, mock_http
                        )
                        mock_m.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_quick_action(self, handler):
        """handle_post routes quick action execution correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_execute_quick_action") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle_post(
                            "/api/v1/dashboard/quick-actions/archive_read",
                            {},
                            mock_http,
                        )
                        mock_m.assert_called_once_with("archive_read")

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_dismiss_urgent(self, handler):
        """handle_post routes urgent item dismiss correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_dismiss_urgent_item") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle_post(
                            "/api/v1/dashboard/urgent/item1/dismiss", {}, mock_http
                        )
                        mock_m.assert_called_once_with("item1")

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_complete_pending(self, handler):
        """handle_post routes pending action complete correctly."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    with patch.object(handler, "_complete_pending_action") as mock_m:
                        mock_m.return_value = {"data": {}}
                        await handler.handle_post(
                            "/api/v1/dashboard/pending-actions/a1/complete",
                            {},
                            mock_http,
                        )
                        mock_m.assert_called_once_with("a1")

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown(self, handler):
        """handle_post returns None for unrecognized paths."""
        mock_http = MagicMock()
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_post("/api/v1/dashboard/unknown", {}, mock_http)
                    assert result is None


# =============================================================================
# _get_dashboard_debate Tests
# =============================================================================


class TestGetDashboardDebate:
    """Tests for _get_dashboard_debate method."""

    def test_returns_debate_detail(self, handler):
        """Returns debate detail for valid debate_id."""
        result = handler._get_dashboard_debate("debate-123")
        data = _parse_body(result)
        assert data["debate_id"] == "debate-123"

    def test_empty_debate_id_returns_400(self, handler):
        """Returns 400 error for empty debate_id."""
        result = handler._get_dashboard_debate("")
        data = _parse_body(result)
        assert "required" in data.get("error", "").lower()


# =============================================================================
# _execute_quick_action Tests
# =============================================================================


class TestExecuteQuickAction:
    """Tests for _execute_quick_action method."""

    def test_returns_success_for_valid_action(self, handler):
        """Returns success response with action_id and timestamp."""
        result = handler._execute_quick_action("archive_read")
        data = _parse_body(result)
        assert data["success"] is True
        assert data["action_id"] == "archive_read"
        assert "executed_at" in data

    def test_empty_action_id_returns_400(self, handler):
        """Returns 400 error for empty action_id."""
        result = handler._execute_quick_action("")
        data = _parse_body(result)
        assert "required" in data.get("error", "").lower()


# =============================================================================
# _dismiss_urgent_item Tests
# =============================================================================


class TestDismissUrgentItem:
    """Tests for _dismiss_urgent_item method."""

    def test_dismiss_success(self, handler, mock_storage):
        """Dismiss succeeds and returns success response."""
        storage, cursor = mock_storage
        cursor.rowcount = 1
        handler.get_storage = MagicMock(return_value=storage)

        result = handler._dismiss_urgent_item("item-1")
        data = _parse_body(result)
        assert data["success"] is True
        assert data["item_id"] == "item-1"
        assert "dismissed_at" in data

    def test_dismiss_not_found(self, handler, mock_storage):
        """Dismiss returns 404 when item is not found."""
        storage, cursor = mock_storage
        cursor.rowcount = 0
        handler.get_storage = MagicMock(return_value=storage)

        result = handler._dismiss_urgent_item("nonexistent")
        data = _parse_body(result)
        assert "not found" in data.get("error", "").lower()

    def test_dismiss_empty_id_returns_400(self, handler):
        """Dismiss returns 400 for empty item_id."""
        result = handler._dismiss_urgent_item("")
        data = _parse_body(result)
        assert "required" in data.get("error", "").lower()

    def test_dismiss_storage_error(self, handler, mock_storage):
        """Dismiss returns 500 on storage exception."""
        storage, cursor = mock_storage
        storage.connection.return_value.__enter__.side_effect = OSError("db fail")
        handler.get_storage = MagicMock(return_value=storage)

        result = handler._dismiss_urgent_item("item-1")
        data = _parse_body(result)
        assert "failed" in data.get("error", "").lower()

    def test_dismiss_no_storage(self, handler):
        """Dismiss succeeds (no-op) when storage is unavailable."""
        handler.get_storage = MagicMock(return_value=None)

        result = handler._dismiss_urgent_item("item-1")
        data = _parse_body(result)
        assert data["success"] is True


# =============================================================================
# _complete_pending_action Tests
# =============================================================================


class TestCompletePendingAction:
    """Tests for _complete_pending_action method."""

    def test_complete_success(self, handler, mock_storage):
        """Complete succeeds and returns success response."""
        storage, cursor = mock_storage
        cursor.rowcount = 1
        handler.get_storage = MagicMock(return_value=storage)

        result = handler._complete_pending_action("action-1")
        data = _parse_body(result)
        assert data["success"] is True
        assert data["action_id"] == "action-1"
        assert "completed_at" in data

    def test_complete_not_found(self, handler, mock_storage):
        """Complete returns 404 when action not found or already completed."""
        storage, cursor = mock_storage
        cursor.rowcount = 0
        handler.get_storage = MagicMock(return_value=storage)

        result = handler._complete_pending_action("nonexistent")
        data = _parse_body(result)
        assert "not found" in data.get("error", "").lower()

    def test_complete_empty_id_returns_400(self, handler):
        """Complete returns 400 for empty action_id."""
        result = handler._complete_pending_action("")
        data = _parse_body(result)
        assert "required" in data.get("error", "").lower()

    def test_complete_storage_error(self, handler, mock_storage):
        """Complete returns 500 on storage exception."""
        storage, cursor = mock_storage
        storage.connection.return_value.__enter__.side_effect = OSError("db fail")
        handler.get_storage = MagicMock(return_value=storage)

        result = handler._complete_pending_action("action-1")
        data = _parse_body(result)
        assert "failed" in data.get("error", "").lower()

    def test_complete_no_storage(self, handler):
        """Complete succeeds (no-op) when storage is unavailable."""
        handler.get_storage = MagicMock(return_value=None)

        result = handler._complete_pending_action("action-1")
        data = _parse_body(result)
        assert data["success"] is True


# =============================================================================
# _get_labels Tests
# =============================================================================


class TestGetLabels:
    """Tests for _get_labels method."""

    def test_returns_labels_from_storage(self, handler, mock_storage):
        """Returns label counts from debate storage."""
        storage, cursor = mock_storage
        handler.get_storage = MagicMock(return_value=storage)
        cursor.fetchall.return_value = [
            ("technology", 15),
            ("finance", 8),
            (None, 3),
        ]

        result = handler._get_labels()
        data = _parse_body(result)
        assert len(data["labels"]) == 3
        assert data["labels"][0]["name"] == "technology"
        assert data["labels"][0]["count"] == 15
        # Null domain mapped to "general"
        assert data["labels"][2]["name"] == "general"

    def test_returns_empty_without_storage(self, handler):
        """Returns empty labels list when no storage."""
        handler.get_storage = MagicMock(return_value=None)
        result = handler._get_labels()
        data = _parse_body(result)
        assert data["labels"] == []


# =============================================================================
# _get_top_senders Tests
# =============================================================================


class TestGetTopSenders:
    """Tests for _get_top_senders method."""

    def test_returns_senders_from_storage(self, handler, mock_storage):
        """Returns top senders ranked by debate count."""
        storage, cursor = mock_storage
        handler.get_storage = MagicMock(return_value=storage)
        cursor.fetchall.return_value = [
            ("technology", 20),
            (None, 5),
        ]

        result = handler._get_top_senders(10, 0)
        data = _parse_body(result)
        assert len(data["senders"]) == 2
        assert data["senders"][0]["domain"] == "technology"
        assert data["senders"][0]["debate_count"] == 20
        # Null domain mapped to "general"
        assert data["senders"][1]["domain"] == "general"

    def test_returns_empty_without_storage(self, handler):
        """Returns empty senders list when no storage."""
        handler.get_storage = MagicMock(return_value=None)
        result = handler._get_top_senders(10, 0)
        data = _parse_body(result)
        assert data["senders"] == []
        assert data["total"] == 0


# =============================================================================
# _get_quick_actions Tests
# =============================================================================


class TestGetQuickActions:
    """Tests for _get_quick_actions method."""

    def test_returns_all_actions(self, handler):
        """Returns the full list of quick actions."""
        result = handler._get_quick_actions()
        data = _parse_body(result)
        assert data["total"] == 6
        assert len(data["actions"]) == 6
        action_ids = [a["id"] for a in data["actions"]]
        assert "archive_read" in action_ids
        assert "ai_respond" in action_ids
        assert "sync_inbox" in action_ids

    def test_actions_have_required_fields(self, handler):
        """Each quick action has id, name, description, icon, available."""
        result = handler._get_quick_actions()
        data = _parse_body(result)
        for action in data["actions"]:
            assert "id" in action
            assert "name" in action
            assert "description" in action
            assert "icon" in action
            assert "available" in action


# =============================================================================
# _get_connector_type Tests
# =============================================================================


class TestGetConnectorType:
    """Tests for _get_connector_type method."""

    def test_returns_unknown_for_none(self, handler):
        """Returns 'unknown' for None connector."""
        assert handler._get_connector_type(None) == "unknown"

    def test_maps_known_class_name(self, handler):
        """Maps known connector class names correctly."""

        class GithubEnterpriseConnector:
            pass

        conn = GithubEnterpriseConnector()
        assert handler._get_connector_type(conn) == "github"

    def test_strips_connector_suffix(self, handler):
        """Strips 'connector' suffix for unknown class names."""

        class CustomConnector:
            pass

        conn = CustomConnector()
        assert handler._get_connector_type(conn) == "custom"


# =============================================================================
# _call_bypassing_decorators Utility Tests
# =============================================================================


class TestCallBypassingDecorators:
    """Tests for the _call_bypassing_decorators utility."""

    def test_calls_unwrapped_function(self):
        """Unwraps and calls the innermost function."""

        def inner(x):
            return x * 2

        def wrapper(x):
            return inner(x)

        wrapper.__wrapped__ = inner

        result = _call_bypassing_decorators(wrapper, 5)
        assert result == 10

    def test_calls_plain_function(self):
        """Calls a plain function without __wrapped__."""

        def plain(x):
            return x + 1

        result = _call_bypassing_decorators(plain, 3)
        assert result == 4

    def test_unwraps_multiple_layers(self):
        """Unwraps through multiple decorator layers."""

        def original(x):
            return x

        def layer1(x):
            return original(x)

        layer1.__wrapped__ = original

        def layer2(x):
            return layer1(x)

        layer2.__wrapped__ = layer1

        result = _call_bypassing_decorators(layer2, 42)
        assert result == 42


# =============================================================================
# _get_team_performance_detail Tests
# =============================================================================


class TestTeamPerformanceDetail:
    """Tests for _get_team_performance_detail method."""

    def test_returns_detail_for_valid_team(self, handler):
        """Returns team detail with members from ELO."""
        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = [
            MagicMock(
                agent_name="claude-opus",
                elo=1300,
                wins=10,
                losses=2,
                draws=1,
                win_rate=0.77,
                debates_count=13,
            ),
        ]
        handler.get_elo_system = MagicMock(return_value=mock_elo)
        handler.ctx = {}

        result = handler._get_team_performance_detail("claude")
        data = _parse_body(result)
        assert data["team_id"] == "claude"
        assert data["team_name"] == "Claude"
        assert data["member_count"] == 1
        assert data["debates_participated"] == 13

    def test_empty_team_id_returns_400(self, handler):
        """Returns 400 for empty team_id."""
        result = handler._get_team_performance_detail("")
        data = _parse_body(result)
        assert "required" in data.get("error", "").lower()


# =============================================================================
# Quality Metrics Permission Check Test
# =============================================================================


class TestQualityMetricsPermission:
    """Tests for quality-metrics endpoint requiring additional permission."""

    @pytest.mark.asyncio
    async def test_quality_metrics_requires_metrics_permission(self, handler):
        """Quality metrics checks admin:metrics:read in addition to dashboard read."""
        from aragora.server.handlers.secure import ForbiddenError

        mock_http = MagicMock()
        call_count = 0

        def check_perm(ctx, perm):
            nonlocal call_count
            call_count += 1
            # Allow first permission check (dashboard read), deny second (metrics read)
            if call_count == 2:
                raise ForbiddenError("No metrics access")

        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as lim:
            lim.is_allowed.return_value = True
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
                with patch.object(handler, "check_permission", side_effect=check_perm):
                    result = await handler.handle(
                        "/api/v1/dashboard/quality-metrics", {}, mock_http
                    )
                    data = _parse_body(result)
                    assert "denied" in data.get("error", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
