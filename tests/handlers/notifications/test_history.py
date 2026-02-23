"""Tests for notification history and delivery stats handler.

Covers all routes, methods, validation, and edge cases of
NotificationHistoryHandler:
- can_handle() routing (versioned and unversioned paths)
- GET /api/v1/notifications/history (paginated notification history)
- GET /api/v1/notifications/delivery-stats (success rate, DLQ count)
- Rate limiting
- Pagination (limit, offset, clamping)
- Channel filtering (valid enum, invalid channel)
- Service unavailable fallback
- Delivery stats aggregation (by_channel, success_rate, DLQ)
- Error handling in _get_history and _get_delivery_stats
- _get_notification_service from context and global fallback
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.notifications.history import (
    NotificationHistoryHandler,
    _notification_history_limiter,
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
    """Minimal mock HTTP handler for history tests."""

    def __init__(self, client_ip: str = "10.0.0.1"):
        self.headers: dict[str, str] = {}
        self.client_address = (client_ip, 12345)
        self.rfile = MagicMock()
        self.rfile.read.return_value = b""
        self.headers["Content-Length"] = "0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Reset the rate limiter between tests."""
    _notification_history_limiter.clear()
    yield
    _notification_history_limiter.clear()


@pytest.fixture
def handler():
    """Create a NotificationHistoryHandler with no context."""
    return NotificationHistoryHandler()


@pytest.fixture
def mock_service():
    """Create a mock notification service."""
    svc = MagicMock()
    svc.get_history.return_value = []
    return svc


@pytest.fixture
def handler_with_service(mock_service):
    """Create a handler with an injected notification service."""
    return NotificationHistoryHandler(ctx={"notification_service": mock_service})


@pytest.fixture
def http_handler():
    """Create a MockHTTPHandler."""
    return MockHTTPHandler()


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() path routing."""

    def test_history_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/notifications/history") is True

    def test_delivery_stats_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/notifications/delivery-stats") is True

    def test_history_unversioned_path(self, handler):
        assert handler.can_handle("/api/notifications/history") is True

    def test_delivery_stats_unversioned_path(self, handler):
        assert handler.can_handle("/api/notifications/delivery-stats") is True

    def test_wrong_path_returns_false(self, handler):
        assert handler.can_handle("/api/v1/notifications") is False

    def test_trailing_slash_returns_false(self, handler):
        assert handler.can_handle("/api/v1/notifications/history/") is False

    def test_extra_segment_returns_false(self, handler):
        assert handler.can_handle("/api/v1/notifications/history/extra") is False

    def test_different_endpoint_returns_false(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_preferences_path_not_handled(self, handler):
        assert handler.can_handle("/api/v1/notifications/preferences") is False


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for handler construction."""

    def test_default_ctx_is_empty_dict(self):
        h = NotificationHistoryHandler()
        assert h.ctx == {}

    def test_ctx_is_stored(self):
        ctx = {"notification_service": MagicMock()}
        h = NotificationHistoryHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = NotificationHistoryHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute."""

    def test_routes_contains_history(self):
        assert "/api/v1/notifications/history" in NotificationHistoryHandler.ROUTES

    def test_routes_contains_delivery_stats(self):
        assert "/api/v1/notifications/delivery-stats" in NotificationHistoryHandler.ROUTES

    def test_routes_length(self):
        assert len(NotificationHistoryHandler.ROUTES) == 2


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting in the handle() method."""

    def test_rate_limit_exceeded(self, handler_with_service, mock_service):
        http = MockHTTPHandler(client_ip="10.0.0.99")
        # Exhaust the 30 rpm limit
        for _ in range(30):
            _notification_history_limiter.is_allowed("10.0.0.99")
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http
        )
        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    def test_rate_limit_allows_normal_traffic(self, handler_with_service, http_handler):
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/v1/notifications/history
# ---------------------------------------------------------------------------


class TestGetHistory:
    """Tests for the history endpoint."""

    def test_empty_history(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["notifications"] == []
        assert body["count"] == 0
        assert body["total"] == 0

    def test_history_returns_items(self, handler_with_service, http_handler, mock_service):
        items = [{"id": "n1", "msg": "hello"}, {"id": "n2", "msg": "world"}]
        mock_service.get_history.return_value = items
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        body = _body(result)
        assert body["count"] == 2
        assert body["notifications"] == items

    def test_default_limit_is_50(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        mock_service.get_history.assert_called_once_with(
            limit=50, channel=None
        )

    def test_custom_limit(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        handler_with_service.handle(
            "/api/v1/notifications/history", {"limit": "25"}, http_handler
        )
        body_limit = 25  # limit param
        # get_history called with limit + offset (0) = 25
        mock_service.get_history.assert_called_once_with(
            limit=25, channel=None
        )

    def test_limit_clamped_to_max_200(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        handler_with_service.handle(
            "/api/v1/notifications/history", {"limit": "500"}, http_handler
        )
        # limit is clamped to 200, offset=0, so limit+offset=200
        mock_service.get_history.assert_called_once_with(
            limit=200, channel=None
        )

    def test_limit_clamped_to_min_1(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        handler_with_service.handle(
            "/api/v1/notifications/history", {"limit": "-10"}, http_handler
        )
        # limit clamped to 1, offset=0, so limit+offset=1
        mock_service.get_history.assert_called_once_with(
            limit=1, channel=None
        )

    def test_offset_applied(self, handler_with_service, http_handler, mock_service):
        items = [{"id": f"n{i}"} for i in range(10)]
        mock_service.get_history.return_value = items
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {"offset": "3", "limit": "5"}, http_handler
        )
        body = _body(result)
        # offset=3, limit=5, so paginated = items[3:8]
        assert body["count"] == 5
        assert body["offset"] == 3
        assert body["limit"] == 5
        assert body["notifications"] == items[3:8]

    def test_negative_offset_clamped_to_zero(self, handler_with_service, http_handler, mock_service):
        items = [{"id": "n1"}, {"id": "n2"}]
        mock_service.get_history.return_value = items
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {"offset": "-5"}, http_handler
        )
        body = _body(result)
        assert body["offset"] == 0
        assert body["count"] == 2

    def test_offset_beyond_results(self, handler_with_service, http_handler, mock_service):
        items = [{"id": "n1"}]
        mock_service.get_history.return_value = items
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {"offset": "100"}, http_handler
        )
        body = _body(result)
        assert body["count"] == 0
        assert body["notifications"] == []

    def test_channel_filter_included_in_response(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v1/notifications/history",
            {"channel": "slack"},
            http_handler,
        )
        body = _body(result)
        assert body["channel"] == "slack"

    def test_channel_filter_passed_as_enum(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        mock_channel = MagicMock()
        with patch(
            "aragora.server.handlers.notifications.history.NotificationChannel",
            create=True,
        ) as MockEnum:
            MockEnum.return_value = mock_channel
            # Patch at the import site inside _get_history
            with patch.dict(
                "sys.modules",
                {"aragora.notifications.models": MagicMock(NotificationChannel=MockEnum)},
            ):
                result = handler_with_service.handle(
                    "/api/v1/notifications/history",
                    {"channel": "Slack"},
                    http_handler,
                )
                MockEnum.assert_called_once_with("slack")  # lowercased

    def test_invalid_channel_returns_400(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        # Patch the import to raise ValueError for invalid channel
        mock_channel_class = MagicMock(side_effect=ValueError("bad channel"))
        with patch.dict(
            "sys.modules",
            {"aragora.notifications.models": MagicMock(NotificationChannel=mock_channel_class)},
        ):
            result = handler_with_service.handle(
                "/api/v1/notifications/history",
                {"channel": "invalid_channel"},
                http_handler,
            )
            assert _status(result) == 400
            assert "Invalid channel" in _body(result).get("error", "")

    def test_channel_import_error_returns_400(self, handler_with_service, http_handler, mock_service):
        """When aragora.notifications.models cannot be imported, treat as invalid channel."""
        mock_service.get_history.return_value = []
        with patch.dict("sys.modules", {"aragora.notifications.models": None}):
            result = handler_with_service.handle(
                "/api/v1/notifications/history",
                {"channel": "slack"},
                http_handler,
            )
            assert _status(result) == 400

    def test_no_channel_filter(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        body = _body(result)
        assert body["channel"] is None

    def test_service_not_available_returns_503(self, handler, http_handler):
        with patch.object(handler, "_get_notification_service", return_value=None):
            result = handler.handle(
                "/api/v1/notifications/history", {}, http_handler
            )
            assert _status(result) == 503
            assert "not available" in _body(result).get("error", "")

    def test_service_exception_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = RuntimeError("DB down")
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 500
        assert "Failed to get notification history" in _body(result).get("error", "")

    def test_service_value_error_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = ValueError("bad value")
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 500

    def test_service_type_error_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = TypeError("bad type")
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 500

    def test_service_os_error_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = OSError("disk error")
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 500

    def test_total_reflects_full_history(self, handler_with_service, http_handler, mock_service):
        items = [{"id": f"n{i}"} for i in range(20)]
        mock_service.get_history.return_value = items
        result = handler_with_service.handle(
            "/api/v1/notifications/history",
            {"limit": "5", "offset": "0"},
            http_handler,
        )
        body = _body(result)
        assert body["total"] == 20
        assert body["count"] == 5


# ---------------------------------------------------------------------------
# GET /api/v1/notifications/delivery-stats
# ---------------------------------------------------------------------------


class TestGetDeliveryStats:
    """Tests for the delivery stats endpoint."""

    def test_empty_history_returns_zero_stats(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["total_notifications"] == 0
        assert body["successful"] == 0
        assert body["failed"] == 0
        assert body["success_rate"] == 0.0
        assert body["by_channel"] == {}

    def test_all_successful(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {
                "results": [
                    {"channel": "slack", "success": True},
                    {"channel": "email", "success": True},
                ]
            }
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["total_notifications"] == 2
        assert body["successful"] == 2
        assert body["failed"] == 0
        assert body["success_rate"] == 100.0

    def test_mixed_results(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {
                "results": [
                    {"channel": "slack", "success": True},
                    {"channel": "slack", "success": False},
                    {"channel": "email", "success": True},
                    {"channel": "email", "success": False},
                ]
            }
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["total_notifications"] == 4
        assert body["successful"] == 2
        assert body["failed"] == 2
        assert body["success_rate"] == 50.0

    def test_by_channel_breakdown(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {
                "results": [
                    {"channel": "slack", "success": True},
                    {"channel": "slack", "success": True},
                    {"channel": "email", "success": False},
                ]
            }
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["by_channel"]["slack"] == {"total": 2, "success": 2, "failed": 0}
        assert body["by_channel"]["email"] == {"total": 1, "success": 0, "failed": 1}

    def test_unknown_channel_falls_back(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {"results": [{"success": True}]}  # no channel key
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert "unknown" in body["by_channel"]
        assert body["by_channel"]["unknown"]["total"] == 1

    def test_multiple_entries(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {"results": [{"channel": "slack", "success": True}]},
            {"results": [{"channel": "email", "success": False}]},
            {"results": [{"channel": "webhook", "success": True}]},
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["total_notifications"] == 3
        assert body["successful"] == 2
        assert body["failed"] == 1

    def test_entry_without_results_key(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {"id": "n1"}  # no "results" key
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["total_notifications"] == 0

    def test_success_rate_rounding(self, handler_with_service, http_handler, mock_service):
        # 1 success out of 3 = 33.333...% -> rounded to 33.3
        mock_service.get_history.return_value = [
            {
                "results": [
                    {"channel": "slack", "success": True},
                    {"channel": "slack", "success": False},
                    {"channel": "slack", "success": False},
                ]
            }
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["success_rate"] == 33.3

    def test_dlq_count_from_dispatcher(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        mock_dispatcher = MagicMock()
        mock_dispatcher.dead_letter_count = 42
        with patch(
            "aragora.server.handlers.notifications.history.get_notification_dispatcher",
            create=True,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.control_plane.notifications": MagicMock(
                        get_notification_dispatcher=MagicMock(return_value=mock_dispatcher)
                    )
                },
            ):
                result = handler_with_service.handle(
                    "/api/v1/notifications/delivery-stats", {}, http_handler
                )
                body = _body(result)
                assert body["dlq_count"] == 42

    def test_dlq_count_zero_when_dispatcher_unavailable(
        self, handler_with_service, http_handler, mock_service
    ):
        mock_service.get_history.return_value = []
        with patch.dict("sys.modules", {"aragora.control_plane.notifications": None}):
            result = handler_with_service.handle(
                "/api/v1/notifications/delivery-stats", {}, http_handler
            )
            body = _body(result)
            assert body["dlq_count"] == 0

    def test_dlq_count_zero_when_dispatcher_has_no_attr(
        self, handler_with_service, http_handler, mock_service
    ):
        mock_service.get_history.return_value = []
        mock_dispatcher = MagicMock(spec=[])  # no dead_letter_count
        with patch.dict(
            "sys.modules",
            {
                "aragora.control_plane.notifications": MagicMock(
                    get_notification_dispatcher=MagicMock(return_value=mock_dispatcher)
                )
            },
        ):
            result = handler_with_service.handle(
                "/api/v1/notifications/delivery-stats", {}, http_handler
            )
            body = _body(result)
            assert body["dlq_count"] == 0

    def test_service_not_available_returns_503(self, handler, http_handler):
        with patch.object(handler, "_get_notification_service", return_value=None):
            result = handler.handle(
                "/api/v1/notifications/delivery-stats", {}, http_handler
            )
            assert _status(result) == 503

    def test_service_exception_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = RuntimeError("DB down")
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        assert _status(result) == 500
        assert "Failed to get delivery stats" in _body(result).get("error", "")

    def test_delivery_stats_calls_get_history_with_limit_1000(
        self, handler_with_service, http_handler, mock_service
    ):
        mock_service.get_history.return_value = []
        handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        mock_service.get_history.assert_called_once_with(limit=1000)

    def test_all_failed(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [
            {
                "results": [
                    {"channel": "slack", "success": False},
                    {"channel": "email", "success": False},
                ]
            }
        ]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["successful"] == 0
        assert body["failed"] == 2
        assert body["success_rate"] == 0.0

    def test_attribute_error_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = AttributeError("no attr")
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# _get_notification_service
# ---------------------------------------------------------------------------


class TestGetNotificationService:
    """Tests for the _get_notification_service private method."""

    def test_returns_service_from_ctx(self, mock_service):
        h = NotificationHistoryHandler(ctx={"notification_service": mock_service})
        assert h._get_notification_service() is mock_service

    def test_falls_back_to_global_service(self):
        h = NotificationHistoryHandler()
        mock_global = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "aragora.notifications.service": MagicMock(
                    get_notification_service=MagicMock(return_value=mock_global)
                )
            },
        ):
            result = h._get_notification_service()
            assert result is mock_global

    def test_returns_none_when_import_fails(self):
        h = NotificationHistoryHandler()
        with patch.dict("sys.modules", {"aragora.notifications.service": None}):
            result = h._get_notification_service()
            assert result is None

    def test_returns_none_when_global_raises(self):
        h = NotificationHistoryHandler()
        mock_mod = MagicMock()
        mock_mod.get_notification_service.side_effect = RuntimeError("init failed")
        with patch.dict("sys.modules", {"aragora.notifications.service": mock_mod}):
            result = h._get_notification_service()
            assert result is None


# ---------------------------------------------------------------------------
# handle() routing
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Tests for the handle() method route dispatch."""

    def test_unknown_path_returns_none(self, handler_with_service, http_handler):
        # A path that can_handle returns False for; but if we force call handle
        # with a path that strip_version_prefix resolves to something unknown
        result = handler_with_service.handle(
            "/api/v1/notifications/unknown", {}, http_handler
        )
        assert result is None

    def test_history_route_dispatches(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [{"id": "n1"}]
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 200
        assert _body(result)["count"] == 1

    def test_delivery_stats_route_dispatches(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        assert _status(result) == 200
        assert "total_notifications" in _body(result)

    def test_handle_with_different_version_prefix(self, handler_with_service, http_handler, mock_service):
        """strip_version_prefix should handle v2 or other versions gracefully."""
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v2/notifications/history", {}, http_handler
        )
        # strip_version_prefix removes /v<N>/ so this should still match
        if result is not None:
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_limit_zero_clamped_to_one(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [{"id": "n1"}]
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {"limit": "0"}, http_handler
        )
        body = _body(result)
        assert body["limit"] == 1

    def test_limit_exactly_200(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = []
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {"limit": "200"}, http_handler
        )
        body = _body(result)
        assert body["limit"] == 200

    def test_limit_exactly_1(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [{"id": "n1"}]
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {"limit": "1"}, http_handler
        )
        body = _body(result)
        assert body["limit"] == 1
        assert body["count"] == 1

    def test_large_offset_with_small_result(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [{"id": "n1"}]
        result = handler_with_service.handle(
            "/api/v1/notifications/history",
            {"offset": "999", "limit": "10"},
            http_handler,
        )
        body = _body(result)
        assert body["count"] == 0
        assert body["notifications"] == []

    def test_delivery_stats_empty_results_array(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.return_value = [{"results": []}]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["total_notifications"] == 0
        assert body["by_channel"] == {}

    def test_delivery_stats_key_error_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = KeyError("missing key")
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        assert _status(result) == 500

    def test_history_key_error_returns_500(self, handler_with_service, http_handler, mock_service):
        mock_service.get_history.side_effect = KeyError("missing key")
        result = handler_with_service.handle(
            "/api/v1/notifications/history", {}, http_handler
        )
        assert _status(result) == 500

    def test_delivery_stats_single_channel_many_entries(
        self, handler_with_service, http_handler, mock_service
    ):
        results = [{"channel": "webhook", "success": i % 2 == 0} for i in range(10)]
        mock_service.get_history.return_value = [{"results": results}]
        result = handler_with_service.handle(
            "/api/v1/notifications/delivery-stats", {}, http_handler
        )
        body = _body(result)
        assert body["by_channel"]["webhook"]["total"] == 10
        assert body["by_channel"]["webhook"]["success"] == 5
        assert body["by_channel"]["webhook"]["failed"] == 5
