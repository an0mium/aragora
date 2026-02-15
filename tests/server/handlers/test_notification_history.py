"""
Tests for notification history and preferences handlers.

Tests:
- Route handling (can_handle)
- GET /api/v1/notifications/history (pagination, channel filter)
- GET /api/v1/notifications/delivery-stats
- GET /api/v1/notifications/preferences
- PUT /api/v1/notifications/preferences (validation, merge)
- Service unavailable handling
- Rate limiting
- Error handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.notifications.history import (
    NotificationHistoryHandler,
    _notification_history_limiter,
)
from aragora.server.handlers.notifications.preferences import (
    NotificationPreferencesHandler,
    _preferences_limiter,
    _user_preferences,
    _DEFAULT_PREFERENCES,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_notification_service():
    """Create a mock notification service."""
    service = MagicMock()
    service.get_history.return_value = [
        {
            "notification": {"title": "Test", "message": "Hello"},
            "results": [
                {"channel": "slack", "success": True, "recipient": "#general"},
                {"channel": "email", "success": False, "recipient": "user@test.com"},
            ],
        },
        {
            "notification": {"title": "Alert", "message": "Budget warning"},
            "results": [
                {"channel": "slack", "success": True, "recipient": "#alerts"},
            ],
        },
    ]
    return service


@pytest.fixture
def history_handler(mock_notification_service):
    """Create history handler with mocked service."""
    ctx = {"notification_service": mock_notification_service}
    return NotificationHistoryHandler(ctx)


@pytest.fixture
def prefs_handler():
    """Create preferences handler."""
    return NotificationPreferencesHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


def _make_put_handler(body: dict) -> MagicMock:
    """Create a mock handler with JSON body for PUT requests."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    body_bytes = json.dumps(body).encode()
    mock.headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body_bytes)),
    }
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = body_bytes
    return mock


@pytest.fixture(autouse=True)
def clear_state():
    """Clear rate limiters and user preferences between tests."""
    _notification_history_limiter._buckets.clear()
    _preferences_limiter._buckets.clear()
    _user_preferences.clear()
    yield
    _user_preferences.clear()


# ===========================================================================
# History Handler - Route Matching
# ===========================================================================


class TestHistoryRouteMatching:
    def test_can_handle_history(self, history_handler):
        assert history_handler.can_handle("/api/v1/notifications/history") is True

    def test_can_handle_delivery_stats(self, history_handler):
        assert history_handler.can_handle("/api/v1/notifications/delivery-stats") is True

    def test_cannot_handle_unknown(self, history_handler):
        assert history_handler.can_handle("/api/v1/notifications/unknown") is False

    def test_cannot_handle_preferences(self, history_handler):
        assert history_handler.can_handle("/api/v1/notifications/preferences") is False


# ===========================================================================
# History Handler - GET /api/v1/notifications/history
# ===========================================================================


class TestNotificationHistory:
    def test_returns_notifications(self, history_handler, mock_http_handler):
        result = history_handler.handle("/api/v1/notifications/history", {}, mock_http_handler)
        assert result is not None
        body = result[0]
        assert body["count"] == 2
        assert len(body["notifications"]) == 2

    def test_pagination_with_limit(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/history", {"limit": "1"}, mock_http_handler
        )
        body = result[0]
        assert body["count"] == 1
        assert body["limit"] == 1

    def test_pagination_with_offset(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/history", {"offset": "1"}, mock_http_handler
        )
        body = result[0]
        assert body["count"] == 1
        assert body["offset"] == 1

    def test_limit_clamped_to_max(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/history", {"limit": "500"}, mock_http_handler
        )
        body = result[0]
        assert body["limit"] == 200

    def test_channel_filter(self, history_handler, mock_http_handler, mock_notification_service):
        mock_notification_service.get_history.return_value = [
            {
                "notification": {"title": "Slack Only"},
                "results": [{"channel": "slack", "success": True}],
            },
        ]

        result = history_handler.handle(
            "/api/v1/notifications/history", {"channel": "slack"}, mock_http_handler
        )
        body = result[0]
        assert body["channel"] == "slack"

    def test_invalid_channel_returns_400(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/history", {"channel": "invalid_channel"}, mock_http_handler
        )
        assert result[1] == 400

    def test_service_unavailable(self, mock_http_handler):
        handler = NotificationHistoryHandler(ctx={})
        with patch(
            "aragora.server.handlers.notifications.history.NotificationHistoryHandler._get_notification_service",
            return_value=None,
        ):
            result = handler.handle("/api/v1/notifications/history", {}, mock_http_handler)
            assert result[1] == 503

    def test_service_exception(self, history_handler, mock_http_handler, mock_notification_service):
        mock_notification_service.get_history.side_effect = RuntimeError("db error")
        result = history_handler.handle("/api/v1/notifications/history", {}, mock_http_handler)
        assert result[1] == 500


# ===========================================================================
# History Handler - GET /api/v1/notifications/delivery-stats
# ===========================================================================


class TestDeliveryStats:
    def test_returns_stats(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/delivery-stats", {}, mock_http_handler
        )
        assert result is not None
        body = result[0]
        assert body["total_notifications"] == 3
        assert body["successful"] == 2
        assert body["failed"] == 1
        assert body["success_rate"] == 66.7

    def test_by_channel_breakdown(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/delivery-stats", {}, mock_http_handler
        )
        body = result[0]
        assert "slack" in body["by_channel"]
        assert body["by_channel"]["slack"]["success"] == 2
        assert "email" in body["by_channel"]
        assert body["by_channel"]["email"]["failed"] == 1

    def test_includes_dlq_count(self, history_handler, mock_http_handler):
        result = history_handler.handle(
            "/api/v1/notifications/delivery-stats", {}, mock_http_handler
        )
        body = result[0]
        assert "dlq_count" in body

    def test_empty_history_stats(self, mock_http_handler, mock_notification_service):
        mock_notification_service.get_history.return_value = []
        handler = NotificationHistoryHandler(
            ctx={"notification_service": mock_notification_service}
        )

        result = handler.handle("/api/v1/notifications/delivery-stats", {}, mock_http_handler)
        body = result[0]
        assert body["total_notifications"] == 0
        assert body["success_rate"] == 0.0

    def test_stats_service_unavailable(self, mock_http_handler):
        handler = NotificationHistoryHandler(ctx={})
        with patch(
            "aragora.server.handlers.notifications.history.NotificationHistoryHandler._get_notification_service",
            return_value=None,
        ):
            result = handler.handle("/api/v1/notifications/delivery-stats", {}, mock_http_handler)
            assert result[1] == 503


# ===========================================================================
# Preferences Handler - Route Matching
# ===========================================================================


class TestPreferencesRouteMatching:
    def test_can_handle_preferences(self, prefs_handler):
        assert prefs_handler.can_handle("/api/v1/notifications/preferences") is True

    def test_cannot_handle_history(self, prefs_handler):
        assert prefs_handler.can_handle("/api/v1/notifications/history") is False


# ===========================================================================
# Preferences Handler - GET
# ===========================================================================


class TestGetPreferences:
    def test_returns_default_preferences(self, mock_http_handler):
        # Use a fresh handler with a fully clean preferences store
        from aragora.server.handlers.notifications.preferences import (
            _user_preferences as prefs_store,
        )

        prefs_store.clear()
        handler = NotificationPreferencesHandler(ctx={})

        result = handler.handle("/api/v1/notifications/preferences", {}, mock_http_handler)
        assert result is not None
        body = result[0]
        assert "preferences" in body
        prefs = body["preferences"]
        assert prefs["channels"]["slack"] is True
        assert prefs["channels"]["email"] is True
        assert prefs["digest_mode"] is False

    def test_returns_saved_preferences(self, prefs_handler, mock_http_handler):
        # The autouse conftest fixture returns user_id "test-user-001"
        user_id = "test-user-001"
        _user_preferences[user_id] = {"channels": {"slack": False}, "digest_mode": True}

        result = prefs_handler.handle("/api/v1/notifications/preferences", {}, mock_http_handler)
        body = result[0]
        assert body["preferences"]["channels"]["slack"] is False
        assert body["preferences"]["digest_mode"] is True


# ===========================================================================
# Preferences Handler - PUT
# ===========================================================================


class TestUpdatePreferences:
    def test_update_channels(self, prefs_handler):
        handler = _make_put_handler({"channels": {"slack": False}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        assert result is not None
        body = result[0]
        assert body["updated"] is True
        assert body["preferences"]["channels"]["slack"] is False
        assert body["preferences"]["channels"]["email"] is True  # unchanged

    def test_update_event_types(self, prefs_handler):
        handler = _make_put_handler({"event_types": {"budget_alert": False}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        body = result[0]
        assert body["preferences"]["event_types"]["budget_alert"] is False

    def test_update_digest_mode(self, prefs_handler):
        handler = _make_put_handler({"digest_mode": True})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        body = result[0]
        assert body["preferences"]["digest_mode"] is True

    def test_update_quiet_hours(self, prefs_handler):
        handler = _make_put_handler({"quiet_hours": {"enabled": True, "start": "23:00"}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        body = result[0]
        assert body["preferences"]["quiet_hours"]["enabled"] is True
        assert body["preferences"]["quiet_hours"]["start"] == "23:00"

    def test_invalid_channel_name(self, prefs_handler):
        handler = _make_put_handler({"channels": {"pigeon": True}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        assert result[1] == 400

    def test_invalid_channel_value_type(self, prefs_handler):
        handler = _make_put_handler({"channels": {"slack": "yes"}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        assert result[1] == 400

    def test_invalid_event_type_value(self, prefs_handler):
        handler = _make_put_handler({"event_types": {"budget_alert": "maybe"}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        assert result[1] == 400

    def test_invalid_digest_mode_type(self, prefs_handler):
        handler = _make_put_handler({"digest_mode": "yes"})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        assert result[1] == 400

    def test_invalid_json_body(self, prefs_handler):
        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"Content-Length": "5"}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = b"xxxxx"
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        assert result[1] == 400

    def test_preferences_persist_across_gets(self, prefs_handler, mock_http_handler):
        # Update
        put_handler = _make_put_handler({"digest_mode": True})
        prefs_handler.handle_put("/api/v1/notifications/preferences", {}, put_handler)

        # Read back
        result = prefs_handler.handle("/api/v1/notifications/preferences", {}, mock_http_handler)
        body = result[0]
        assert body["preferences"]["digest_mode"] is True

    def test_wrapped_preferences_body(self, prefs_handler):
        """Test that updates wrapped in {'preferences': ...} are accepted."""
        handler = _make_put_handler({"preferences": {"channels": {"webhook": False}}})
        result = prefs_handler.handle_put("/api/v1/notifications/preferences", {}, handler)
        body = result[0]
        assert body["preferences"]["channels"]["webhook"] is False


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestNotificationRateLimiting:
    def test_history_rate_limit(self, mock_http_handler):
        handler = NotificationHistoryHandler(ctx={})
        for _ in range(35):
            _notification_history_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.notifications.history.NotificationHistoryHandler._get_notification_service",
            return_value=MagicMock(),
        ):
            result = handler.handle("/api/v1/notifications/history", {}, mock_http_handler)
            assert result[1] == 429

    def test_preferences_rate_limit(self, prefs_handler, mock_http_handler):
        for _ in range(35):
            _preferences_limiter.is_allowed("127.0.0.1")

        result = prefs_handler.handle("/api/v1/notifications/preferences", {}, mock_http_handler)
        assert result[1] == 429
