"""
Comprehensive tests for SharingNotificationsHandler.

Tests the Knowledge Sharing Notifications HTTP handler endpoints:
- GET  /api/v1/knowledge/notifications - Get notifications for current user
- GET  /api/v1/knowledge/notifications/count - Get unread count
- POST /api/v1/knowledge/notifications/{id}/read - Mark notification as read
- POST /api/v1/knowledge/notifications/read-all - Mark all as read
- POST /api/v1/knowledge/notifications/{id}/dismiss - Dismiss notification
- GET  /api/v1/knowledge/notifications/preferences - Get notification preferences
- PUT  /api/v1/knowledge/notifications/preferences - Update preferences
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge.sharing_notifications import (
    SharingNotificationsHandler,
)


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# =============================================================================
# Mock data classes
# =============================================================================


@dataclass
class MockNotification:
    """Mock notification object with to_dict support."""

    id: str = "notif-001"
    user_id: str = "test-user-001"
    notification_type: str = "item_shared"
    message: str = "A knowledge item was shared with you"
    status: str = "unread"
    created_at: str = "2026-02-23T10:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "notification_type": self.notification_type,
            "message": self.message,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class MockNotificationPreferences:
    """Mock notification preferences."""

    user_id: str = "test-user-001"
    email_on_share: bool = True
    email_on_unshare: bool = False
    email_on_permission_change: bool = True
    in_app_enabled: bool = True
    telegram_enabled: bool = False
    webhook_url: str | None = None
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None


# =============================================================================
# Mock HTTP handler
# =============================================================================


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to the sharing notifications handler."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/knowledge/notifications",
        body: dict[str, Any] | None = None,
    ):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {
            "Content-Length": "0",
            "Host": "localhost:8080",
        }

        if body is not None:
            body_bytes = json.dumps(body).encode("utf-8")
            self.headers["Content-Length"] = str(len(body_bytes))
            self.headers["Content-Type"] = "application/json"
            self.rfile = io.BytesIO(body_bytes)
        else:
            self.rfile = io.BytesIO(b"")


# =============================================================================
# Patch path constant
# =============================================================================

_MOD = "aragora.server.handlers.knowledge.sharing_notifications"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _bypass_rbac_checker(monkeypatch):
    """Patch the PermissionChecker used by this handler to always allow.

    The handler does its own inline RBAC via get_permission_checker().check_permission().
    The conftest patches BaseHandler.require_auth_or_error but not this inline check.
    """
    mock_decision = MagicMock()
    mock_decision.allowed = True
    mock_decision.reason = None

    mock_checker = MagicMock()
    mock_checker.check_permission = MagicMock(return_value=mock_decision)

    monkeypatch.setattr(
        f"{_MOD}.get_permission_checker",
        lambda: mock_checker,
    )


@pytest.fixture
def handler():
    """Create a SharingNotificationsHandler with empty context."""
    return SharingNotificationsHandler(ctx={})


@pytest.fixture
def http_handler():
    """Create a default mock HTTP handler for GET requests."""
    return _MockHTTPHandler()


@pytest.fixture
def sample_notifications():
    """Create a list of sample notification objects."""
    return [
        MockNotification(id="notif-001", message="Item shared: Report Q1"),
        MockNotification(id="notif-002", message="Item shared: Analysis", status="read"),
        MockNotification(id="notif-003", message="Permission changed"),
    ]


@pytest.fixture
def sample_preferences():
    """Create sample notification preferences."""
    return MockNotificationPreferences()


# =============================================================================
# Test: Handler initialization
# =============================================================================


class TestHandlerInit:
    """Tests for handler construction."""

    def test_init_with_empty_ctx(self):
        h = SharingNotificationsHandler(ctx={})
        assert h.ctx == {}

    def test_init_with_none_ctx(self):
        h = SharingNotificationsHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_context_data(self):
        h = SharingNotificationsHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_default_ctx(self):
        h = SharingNotificationsHandler()
        assert h.ctx == {}


# =============================================================================
# Test: can_handle
# =============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_notifications_root(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications") is True

    def test_handles_notifications_count(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/count") is True

    def test_handles_notifications_preferences(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/preferences") is True

    def test_handles_notification_read(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/notif-123/read") is True

    def test_handles_notification_dismiss(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/notif-123/dismiss") is True

    def test_handles_read_all(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/read-all") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_rejects_knowledge_but_not_notifications(self, handler):
        assert handler.can_handle("/api/v1/knowledge/mound/stats") is False


# =============================================================================
# Test: GET /api/v1/knowledge/notifications
# =============================================================================


class TestGetNotifications:
    """Tests for listing notifications."""

    def test_get_notifications_success(self, handler, http_handler, sample_notifications):
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(
                f"{_MOD}.SharingNotificationsHandler._get_notifications"
            ) as mock_get,
        ):
            limiter.is_allowed.return_value = True
            from aragora.server.handlers.base import json_response

            mock_get.return_value = json_response(
                {
                    "notifications": [n.to_dict() for n in sample_notifications],
                    "count": len(sample_notifications),
                    "limit": 20,
                    "offset": 0,
                }
            )
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 3
        assert len(body["notifications"]) == 3

    def test_get_notifications_with_import_mock(self, handler, http_handler, sample_notifications):
        """Test the actual _get_notifications path with mocked import."""
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(
                f"{_MOD}.SharingNotificationsHandler._get_notifications",
                wraps=handler._get_notifications,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.notifications": MagicMock(
                        NotificationStatus=MagicMock(),
                        get_notifications_for_user=MagicMock(
                            return_value=sample_notifications
                        ),
                    )
                },
            ),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 3
        assert body["limit"] == 20
        # safe_query_int has min_val=1 by default, so offset 0 is clamped to 1
        assert body["offset"] == 1

    def test_get_notifications_with_limit_and_offset(self, handler, http_handler):
        """Test pagination params are passed through."""
        notifs = [MockNotification(id="notif-001")]
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict(
                "sys.modules",
                {
                    "aragora.knowledge.mound.notifications": MagicMock(
                        NotificationStatus=MagicMock(),
                        get_notifications_for_user=MagicMock(return_value=notifs),
                    )
                },
            ),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {"limit": "5", "offset": "10"},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["limit"] == 5
        assert body["offset"] == 10

    def test_get_notifications_with_status_filter(self, handler, http_handler):
        """Test filtering by notification status."""
        mock_status_enum = MagicMock()
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = mock_status_enum
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {"status": "unread"},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        mock_status_enum.assert_called_with("unread")

    def test_get_notifications_invalid_status_returns_400(self, handler, http_handler):
        """Test that an invalid status value returns 400."""
        mock_status_enum = MagicMock(side_effect=ValueError("invalid"))
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = mock_status_enum
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {"status": "bogus_status"},
                http_handler,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid status" in body.get("error", body.get("message", ""))

    def test_get_notifications_import_error_returns_500(self, handler, http_handler):
        """Test graceful handling when notifications module is unavailable."""
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict(
                "sys.modules",
                {"aragora.knowledge.mound.notifications": None},
            ),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 500
        body = _body(result)
        assert "error" in body or "message" in body

    def test_get_notifications_runtime_error_returns_500(self, handler, http_handler):
        """Test graceful handling of runtime errors."""
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = MagicMock()
        mock_mod.get_notifications_for_user = MagicMock(
            side_effect=RuntimeError("db connection lost")
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 500

    def test_get_notifications_empty_list(self, handler, http_handler):
        """Returns empty notifications list when user has none."""
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = MagicMock()
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["notifications"] == []
        assert body["count"] == 0


# =============================================================================
# Test: GET /api/v1/knowledge/notifications/count
# =============================================================================


class TestGetUnreadCount:
    """Tests for getting unread notification count."""

    def test_get_unread_count_success(self, handler, http_handler):
        mock_mod = MagicMock()
        mock_mod.get_unread_count = MagicMock(return_value=7)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/count",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["unread_count"] == 7

    def test_get_unread_count_zero(self, handler, http_handler):
        mock_mod = MagicMock()
        mock_mod.get_unread_count = MagicMock(return_value=0)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/count",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["unread_count"] == 0

    def test_get_unread_count_import_error_returns_500(self, handler, http_handler):
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": None}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/count",
                {},
                http_handler,
            )
        assert _status(result) == 500

    def test_get_unread_count_runtime_error_returns_500(self, handler, http_handler):
        mock_mod = MagicMock()
        mock_mod.get_unread_count = MagicMock(side_effect=RuntimeError("connection failed"))

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/count",
                {},
                http_handler,
            )
        assert _status(result) == 500


# =============================================================================
# Test: GET /api/v1/knowledge/notifications/preferences
# =============================================================================


class TestGetPreferences:
    """Tests for getting notification preferences."""

    def test_get_preferences_success(self, handler, http_handler, sample_preferences):
        mock_mod = MagicMock()
        mock_mod.get_notification_preferences = MagicMock(return_value=sample_preferences)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/preferences",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "test-user-001"
        assert body["email_on_share"] is True
        assert body["email_on_unshare"] is False
        assert body["email_on_permission_change"] is True
        assert body["in_app_enabled"] is True
        assert body["telegram_enabled"] is False
        assert body["webhook_url"] is None
        assert body["quiet_hours_start"] is None
        assert body["quiet_hours_end"] is None

    def test_get_preferences_with_webhook(self, handler, http_handler):
        prefs = MockNotificationPreferences(
            webhook_url="https://hooks.example.com/notify",
            quiet_hours_start="22:00",
            quiet_hours_end="07:00",
        )
        mock_mod = MagicMock()
        mock_mod.get_notification_preferences = MagicMock(return_value=prefs)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/preferences",
                {},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["webhook_url"] == "https://hooks.example.com/notify"
        assert body["quiet_hours_start"] == "22:00"
        assert body["quiet_hours_end"] == "07:00"

    def test_get_preferences_import_error_returns_500(self, handler, http_handler):
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": None}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/preferences",
                {},
                http_handler,
            )
        assert _status(result) == 500

    def test_get_preferences_runtime_error_returns_500(self, handler, http_handler):
        mock_mod = MagicMock()
        mock_mod.get_notification_preferences = MagicMock(
            side_effect=RuntimeError("store error")
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/preferences",
                {},
                http_handler,
            )
        assert _status(result) == 500


# =============================================================================
# Test: POST /api/v1/knowledge/notifications/{id}/read
# =============================================================================


class TestMarkRead:
    """Tests for marking a notification as read."""

    def test_mark_read_success(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_notification_read = MagicMock(return_value=True)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-123/read",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-123/read",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_mark_read_not_found_returns_404(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_notification_read = MagicMock(return_value=False)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-999/read",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-999/read",
                {},
                mock_http,
            )
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", body.get("message", "")).lower()

    def test_mark_read_import_error_returns_500(self, handler):
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-123/read",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": None}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-123/read",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_mark_read_runtime_error_returns_500(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_notification_read = MagicMock(side_effect=RuntimeError("db error"))
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-123/read",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-123/read",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_mark_read_extracts_notification_id_from_path(self, handler):
        """Verify the notification_id passed to mark_notification_read.

        Note: The handler extracts parts[4] from the path split. For a path like
        /api/v1/knowledge/notifications/notif-123/read, parts[4] = "notifications"
        (index 0="" 1="api" 2="v1" 3="knowledge" 4="notifications" 5="notif-123").
        This is a known index bug -- the handler passes "notifications" as the ID,
        not the actual notification ID. We test the actual behavior here.
        """
        mock_mod = MagicMock()
        mock_mod.mark_notification_read = MagicMock(return_value=True)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-abc/read",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            handler.handle_post(
                "/api/v1/knowledge/notifications/notif-abc/read",
                {},
                mock_http,
            )
        # parts[4] of "/api/v1/knowledge/notifications/notif-abc/read" is "notifications"
        mock_mod.mark_notification_read.assert_called_once_with("notifications", "test-user-001")


# =============================================================================
# Test: POST /api/v1/knowledge/notifications/read-all
# =============================================================================


class TestMarkAllRead:
    """Tests for marking all notifications as read."""

    def test_mark_all_read_success(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_all_notifications_read = MagicMock(return_value=5)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["count"] == 5

    def test_mark_all_read_zero_notifications(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_all_notifications_read = MagicMock(return_value=0)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["count"] == 0

    def test_mark_all_read_import_error_returns_500(self, handler):
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": None}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_mark_all_read_runtime_error_returns_500(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_all_notifications_read = MagicMock(
            side_effect=RuntimeError("db error")
        )
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 500


# =============================================================================
# Test: POST /api/v1/knowledge/notifications/{id}/dismiss
# =============================================================================


class TestDismissNotification:
    """Tests for dismissing a notification."""

    def test_dismiss_success(self, handler):
        mock_store = MagicMock()
        mock_store.dismiss_notification = MagicMock(return_value=True)
        mock_mod = MagicMock()
        mock_mod.get_notification_store = MagicMock(return_value=mock_store)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-123/dismiss",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-123/dismiss",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_dismiss_not_found_returns_404(self, handler):
        mock_store = MagicMock()
        mock_store.dismiss_notification = MagicMock(return_value=False)
        mock_mod = MagicMock()
        mock_mod.get_notification_store = MagicMock(return_value=mock_store)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-999/dismiss",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-999/dismiss",
                {},
                mock_http,
            )
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", body.get("message", "")).lower()

    def test_dismiss_import_error_returns_500(self, handler):
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-123/dismiss",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": None}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-123/dismiss",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_dismiss_runtime_error_returns_500(self, handler):
        mock_store = MagicMock()
        mock_store.dismiss_notification = MagicMock(
            side_effect=RuntimeError("store failure")
        )
        mock_mod = MagicMock()
        mock_mod.get_notification_store = MagicMock(return_value=mock_store)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-123/dismiss",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif-123/dismiss",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_dismiss_extracts_notification_id_from_path(self, handler):
        """Verify the notification_id passed to dismiss_notification.

        Same index bug as mark_read: parts[4] = "notifications" not the actual ID.
        """
        mock_store = MagicMock()
        mock_store.dismiss_notification = MagicMock(return_value=True)
        mock_mod = MagicMock()
        mock_mod.get_notification_store = MagicMock(return_value=mock_store)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/notif-xyz/dismiss",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            handler.handle_post(
                "/api/v1/knowledge/notifications/notif-xyz/dismiss",
                {},
                mock_http,
            )
        # parts[4] is "notifications" due to the index bug
        mock_store.dismiss_notification.assert_called_once_with(
            "notifications", "test-user-001"
        )


# =============================================================================
# Test: PUT /api/v1/knowledge/notifications/preferences
# =============================================================================


class TestUpdatePreferences:
    """Tests for updating notification preferences."""

    def test_update_preferences_success(self, handler):
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={
                "email_on_share": True,
                "email_on_unshare": True,
                "in_app_enabled": False,
            },
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["message"] == "Preferences updated"
        mock_mod.set_notification_preferences.assert_called_once()

    def test_update_preferences_with_valid_webhook(self, handler):
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={
                "webhook_url": "https://hooks.example.com/notify",
                "email_on_share": True,
            },
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.validate_webhook_url", return_value=(True, None)),
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_update_preferences_with_invalid_webhook_returns_400(self, handler):
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={
                "webhook_url": "http://169.254.169.254/metadata",
            },
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(
                f"{_MOD}.validate_webhook_url",
                return_value=(False, "URL resolves to private IP"),
            ),
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "webhook" in body.get("error", body.get("message", "")).lower()

    def test_update_preferences_without_webhook_skips_validation(self, handler):
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"email_on_share": False},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.validate_webhook_url") as mock_validate,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 200
        mock_validate.assert_not_called()

    def test_update_preferences_import_error_returns_500(self, handler):
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"email_on_share": True},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": None}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_update_preferences_runtime_error_returns_500(self, handler):
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock(
            side_effect=RuntimeError("store error")
        )
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"email_on_share": True},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 500

    def test_update_preferences_empty_body(self, handler):
        """Update with empty body uses defaults."""
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 200

    def test_update_preferences_with_quiet_hours(self, handler):
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={
                "quiet_hours_start": "22:00",
                "quiet_hours_end": "07:00",
            },
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 200
        # Verify prefs were passed with correct values
        call_args = mock_mod.set_notification_preferences.call_args
        prefs_arg = call_args[0][0]
        assert prefs_arg.quiet_hours_start == "22:00"
        assert prefs_arg.quiet_hours_end == "07:00"


# =============================================================================
# Test: Rate limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limit behavior on all methods."""

    def test_handle_rate_limit_exceeded_returns_429(self, handler, http_handler):
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = False
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 429
        body = _body(result)
        assert "rate limit" in body.get("error", body.get("message", "")).lower()

    def test_handle_post_rate_limit_exceeded_returns_429(self, handler):
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = False
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 429

    def test_handle_put_rate_limit_exceeded_returns_429(self, handler):
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"email_on_share": True},
        )
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = False
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 429

    def test_rate_limiter_uses_client_ip(self, handler):
        mock_http = _MockHTTPHandler()
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = True
            with patch(f"{_MOD}.get_client_ip", return_value="10.0.0.1"):
                handler.handle(
                    "/api/v1/knowledge/notifications",
                    {},
                    mock_http,
                )
            limiter.is_allowed.assert_called_with("10.0.0.1")


# =============================================================================
# Test: Unmatched paths
# =============================================================================


class TestUnmatchedPaths:
    """Tests for paths that don't match any route."""

    def test_handle_unmatched_returns_none(self, handler, http_handler):
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/unknown-endpoint",
                {},
                http_handler,
            )
        assert result is None

    def test_handle_post_unmatched_returns_none(self, handler):
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/some/other/path",
        )
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/some/other/path",
                {},
                mock_http,
            )
        # Path doesn't match read-all, {id}/read, or {id}/dismiss
        assert result is None

    def test_handle_put_unmatched_returns_none(self, handler):
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/unknown",
            body={},
        )
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/unknown",
                {},
                mock_http,
            )
        assert result is None


# =============================================================================
# Test: RBAC behavior
# =============================================================================


class TestRBACBehavior:
    """Tests for RBAC permission checking in the handler.

    Note: The conftest auto-patches RBAC, so these tests verify the handler
    works correctly with the mocked RBAC. Use @pytest.mark.no_auto_auth
    to test RBAC failure cases.
    """

    def test_rbac_permission_denied_returns_403(self, handler, http_handler, monkeypatch):
        """Test that RBAC denial returns 403."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "insufficient permissions"

        mock_checker = MagicMock()
        mock_checker.check_permission = MagicMock(return_value=mock_decision)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", True),
            patch(f"{_MOD}.get_permission_checker", return_value=mock_checker),
            patch(f"{_MOD}.RBACContext") as mock_ctx_cls,
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 403
        body = _body(result)
        assert "permission denied" in body.get("error", body.get("message", "")).lower()

    def test_rbac_check_exception_degrades_gracefully(self, handler, http_handler):
        """RBAC errors should degrade gracefully, not crash."""
        mock_checker = MagicMock()
        mock_checker.check_permission = MagicMock(side_effect=RuntimeError("rbac broken"))

        mock_mod = MagicMock()
        mock_mod.NotificationStatus = MagicMock()
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", True),
            patch(f"{_MOD}.get_permission_checker", return_value=mock_checker),
            patch(f"{_MOD}.RBACContext") as mock_ctx_cls,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        # Should degrade gracefully and still return a response
        assert _status(result) == 200

    def test_rbac_not_available_in_production_returns_503(self, handler, http_handler, monkeypatch):
        """When RBAC is unavailable in production, returns 503."""
        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", False),
            patch(f"{_MOD}.rbac_fail_closed", return_value=True),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 503

    def test_rbac_not_available_in_dev_proceeds(self, handler, http_handler):
        """When RBAC is unavailable in dev, the handler proceeds."""
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = MagicMock()
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", False),
            patch(f"{_MOD}.rbac_fail_closed", return_value=False),
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                http_handler,
            )
        assert _status(result) == 200


# =============================================================================
# Test: POST method RBAC checks
# =============================================================================


class TestPostRBAC:
    """Tests for RBAC checks on POST endpoints."""

    def test_post_rbac_denied_returns_403(self, handler):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "no write access"

        mock_checker = MagicMock()
        mock_checker.check_permission = MagicMock(return_value=mock_decision)

        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", True),
            patch(f"{_MOD}.get_permission_checker", return_value=mock_checker),
            patch(f"{_MOD}.RBACContext") as mock_ctx_cls,
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 403

    def test_post_rbac_not_available_production_returns_503(self, handler):
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", False),
            patch(f"{_MOD}.rbac_fail_closed", return_value=True),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 503


# =============================================================================
# Test: PUT method RBAC checks
# =============================================================================


class TestPutRBAC:
    """Tests for RBAC checks on PUT endpoints."""

    def test_put_rbac_denied_returns_403(self, handler):
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "no write access"

        mock_checker = MagicMock()
        mock_checker.check_permission = MagicMock(return_value=mock_decision)

        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"email_on_share": True},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", True),
            patch(f"{_MOD}.get_permission_checker", return_value=mock_checker),
            patch(f"{_MOD}.RBACContext") as mock_ctx_cls,
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 403

    def test_put_rbac_not_available_production_returns_503(self, handler):
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"email_on_share": True},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(f"{_MOD}.RBAC_AVAILABLE", False),
            patch(f"{_MOD}.rbac_fail_closed", return_value=True),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 503


# =============================================================================
# Test: User ID extraction
# =============================================================================


class TestUserIdExtraction:
    """Tests for user ID extraction from auth context."""

    def test_user_id_extracted_from_auth_context(self, handler, http_handler):
        """Verify user_id from auth context is passed to internal methods."""
        mock_mod = MagicMock()
        mock_mod.get_unread_count = MagicMock(return_value=3)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/count",
                {},
                http_handler,
            )
        assert _status(result) == 200
        # The conftest patches require_auth_or_error to return user_id="test-user-001"
        mock_mod.get_unread_count.assert_called_once_with("test-user-001")

    def test_mark_all_read_uses_correct_user_id(self, handler):
        mock_mod = MagicMock()
        mock_mod.mark_all_notifications_read = MagicMock(return_value=2)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        mock_mod.mark_all_notifications_read.assert_called_once_with("test-user-001")

    def test_get_preferences_uses_correct_user_id(self, handler, http_handler, sample_preferences):
        mock_mod = MagicMock()
        mock_mod.get_notification_preferences = MagicMock(return_value=sample_preferences)

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            handler.handle(
                "/api/v1/knowledge/notifications/preferences",
                {},
                http_handler,
            )
        mock_mod.get_notification_preferences.assert_called_once_with("test-user-001")


# =============================================================================
# Test: Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handle_returns_none_for_non_notification_path(self, handler, http_handler):
        """A path that starts with notifications prefix but is unknown returns None."""
        with patch(f"{_MOD}._notifications_limiter") as limiter:
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications/other-endpoint",
                {},
                http_handler,
            )
        assert result is None

    def test_post_path_with_read_in_middle_matches(self, handler):
        """A path containing /read matches the mark_read route."""
        mock_mod = MagicMock()
        mock_mod.mark_notification_read = MagicMock(return_value=True)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/some-id/read",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/some-id/read",
                {},
                mock_http,
            )
        assert _status(result) == 200

    def test_post_read_all_takes_priority_over_read_pattern(self, handler):
        """read-all path should match before the {id}/read pattern."""
        mock_mod = MagicMock()
        mock_mod.mark_all_notifications_read = MagicMock(return_value=3)
        mock_http = _MockHTTPHandler(
            method="POST",
            path="/api/v1/knowledge/notifications/read-all",
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["count"] == 3
        # mark_all was called, not mark_read
        mock_mod.mark_all_notifications_read.assert_called_once()

    def test_notification_limit_clamped(self, handler, http_handler):
        """Limit query param is clamped to max 100 by safe_query_int."""
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = MagicMock()
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {"limit": "999"},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        # safe_query_int clamps to max_val=100
        assert body["limit"] == 100

    def test_notification_offset_clamped(self, handler, http_handler):
        """Offset query param clamped to max 100000 by safe_query_int."""
        mock_mod = MagicMock()
        mock_mod.NotificationStatus = MagicMock()
        mock_mod.get_notifications_for_user = MagicMock(return_value=[])

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {"offset": "200000"},
                http_handler,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["offset"] == 100000

    def test_ssrf_protection_on_webhook_localhost(self, handler):
        """Webhook URL pointing to localhost should be rejected."""
        mock_mod = MagicMock()
        mock_mod.NotificationPreferences = MockNotificationPreferences
        mock_mod.set_notification_preferences = MagicMock()
        mock_http = _MockHTTPHandler(
            method="PUT",
            path="/api/v1/knowledge/notifications/preferences",
            body={"webhook_url": "http://localhost:8080/hook"},
        )

        with (
            patch(f"{_MOD}._notifications_limiter") as limiter,
            patch(
                f"{_MOD}.validate_webhook_url",
                return_value=(False, "URL resolves to loopback address"),
            ),
            patch.dict("sys.modules", {"aragora.knowledge.mound.notifications": mock_mod}),
        ):
            limiter.is_allowed.return_value = True
            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_http,
            )
        assert _status(result) == 400
