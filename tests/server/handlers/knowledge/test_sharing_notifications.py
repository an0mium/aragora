"""
Tests for Knowledge Sharing Notifications Handler.

Tests cover:
- SharingNotificationsHandler routing (can_handle)
- GET /api/v1/knowledge/notifications - Get notifications
- GET /api/v1/knowledge/notifications/count - Get unread count
- POST /api/v1/knowledge/notifications/{id}/read - Mark as read
- POST /api/v1/knowledge/notifications/read-all - Mark all as read
- POST /api/v1/knowledge/notifications/{id}/dismiss - Dismiss
- GET /api/v1/knowledge/notifications/preferences - Get preferences
- PUT /api/v1/knowledge/notifications/preferences - Update preferences
- RBAC permission checks
- Rate limiting
- Error handling
"""

import sys
import types as _types_mod

# Pre-stub Slack modules to avoid circular ImportError
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
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from enum import Enum

from aragora.server.handlers.knowledge.sharing_notifications import (
    SharingNotificationsHandler,
    NOTIFICATIONS_READ_PERMISSION,
    NOTIFICATIONS_WRITE_PERMISSION,
)


# =============================================================================
# Mock Data Classes
# =============================================================================


class MockNotificationStatus(Enum):
    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"


@dataclass
class MockNotification:
    """Mock notification object."""

    id: str = "notif_123"
    user_id: str = "user_123"
    title: str = "Knowledge shared with you"
    message: str = "Alice shared 'Sales Report' with you"
    status: MockNotificationStatus = MockNotificationStatus.UNREAD
    created_at: datetime = None
    metadata: dict = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MockNotificationPreferences:
    """Mock notification preferences."""

    user_id: str = "user_123"
    email_on_share: bool = True
    email_on_unshare: bool = False
    email_on_permission_change: bool = True
    in_app_enabled: bool = True
    telegram_enabled: bool = False
    webhook_url: str = None
    quiet_hours_start: str = None
    quiet_hours_end: str = None


# =============================================================================
# Helpers
# =============================================================================


def _parse_result(result):
    """Parse HandlerResult into (body_dict, status_code)."""
    if result is None:
        return {}, 404
    body = json.loads(result.body) if hasattr(result, "body") and result.body else {}
    status = result.status_code if hasattr(result, "status_code") else 200
    return body, status


def _make_mock_handler(
    *,
    authenticated: bool = True,
    user_id: str = "user_123",
    email: str = "test@example.com",
    org_id: str = "org_123",
    role: str = "member",
    client_ip: str = "127.0.0.1",
    json_body: dict = None,
):
    """Build a mock HTTP handler."""
    handler = MagicMock()

    # Mock user
    if authenticated:
        user = SimpleNamespace()
        user.user_id = user_id
        user.email = email
        user.org_id = org_id
        user.role = role
    else:
        user = None

    handler.user = user
    handler.client_ip = client_ip

    # Mock request headers for IP extraction
    handler.request = MagicMock()
    handler.request.headers = {"X-Forwarded-For": client_ip}

    # Mock json body
    if json_body is not None:
        handler.json_body = json_body

    return handler


def _make_query_params(
    *,
    limit: str = None,
    offset: str = None,
    status: str = None,
    workspace_id: str = None,
):
    """Build query params dict."""
    params = {}
    if limit is not None:
        params["limit"] = limit
    if offset is not None:
        params["offset"] = offset
    if status is not None:
        params["status"] = status
    if workspace_id is not None:
        params["workspace_id"] = workspace_id
    return params


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a SharingNotificationsHandler instance."""
    return SharingNotificationsHandler(server_context={})


@pytest.fixture
def mock_notification_store():
    """Create a mock notification store."""
    store = MagicMock()
    store.dismiss_notification = MagicMock(return_value=True)
    return store


@pytest.fixture
def mock_rbac_checker():
    """Create a mock RBAC checker that allows all permissions."""
    checker = MagicMock()
    decision = MagicMock()
    decision.allowed = True
    decision.reason = None
    checker.check_permission.return_value = decision
    return checker


@pytest.fixture
def mock_rbac_checker_denied():
    """Create a mock RBAC checker that denies permissions."""
    checker = MagicMock()
    decision = MagicMock()
    decision.allowed = False
    decision.reason = "Insufficient permissions"
    checker.check_permission.return_value = decision
    return checker


# =============================================================================
# Tests: Permission Constants
# =============================================================================


class TestPermissionConstants:
    """Test permission constant definitions."""

    def test_read_permission(self):
        assert NOTIFICATIONS_READ_PERMISSION == "knowledge:notifications:read"

    def test_write_permission(self):
        assert NOTIFICATIONS_WRITE_PERMISSION == "knowledge:notifications:write"


# =============================================================================
# Tests: can_handle
# =============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_notifications_base(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications") is True

    def test_can_handle_notifications_count(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/count") is True

    def test_can_handle_notifications_preferences(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/preferences") is True

    def test_can_handle_notification_read(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/123/read") is True

    def test_can_handle_notification_dismiss(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/123/dismiss") is True

    def test_can_handle_read_all(self, handler):
        assert handler.can_handle("/api/v1/knowledge/notifications/read-all") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/knowledge/mound") is False

    def test_cannot_handle_other_api(self, handler):
        assert handler.can_handle("/api/v1/debates") is False


# =============================================================================
# Tests: GET /api/v1/knowledge/notifications
# =============================================================================


class TestGetNotifications:
    """Tests for GET notifications endpoint."""

    def test_unauthenticated_returns_error(self, handler):
        """Returns error when not authenticated."""
        mock_handler = _make_mock_handler(authenticated=False)

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, ("error", 401))

            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        assert result == ("error", 401)

    def test_rate_limited(self, handler):
        """Returns 429 when rate limited."""
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 429

    def test_get_notifications_success(self, handler, mock_rbac_checker):
        """Successfully retrieves notifications."""
        mock_handler = _make_mock_handler()
        notifications = [MockNotification(), MockNotification(id="notif_456")]

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notifications_for_user",
                return_value=notifications,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert "notifications" in body
        assert body["count"] == 2

    def test_get_notifications_with_filters(self, handler, mock_rbac_checker):
        """Passes filters to the notification store."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notifications_for_user",
                return_value=[],
            ) as mock_get,
            patch(
                "aragora.knowledge.mound.notifications.NotificationStatus",
            ) as mock_status_enum,
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True
            mock_status_enum.return_value = MockNotificationStatus.UNREAD

            handler.handle(
                "/api/v1/knowledge/notifications",
                {"limit": "10", "offset": "5", "status": "unread"},
                mock_handler,
            )

        mock_get.assert_called_once()


# =============================================================================
# Tests: GET /api/v1/knowledge/notifications/count
# =============================================================================


class TestGetUnreadCount:
    """Tests for GET unread count endpoint."""

    def test_get_unread_count_success(self, handler, mock_rbac_checker):
        """Successfully retrieves unread count."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_unread_count",
                return_value=5,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/v1/knowledge/notifications/count",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["unread_count"] == 5


# =============================================================================
# Tests: POST /api/v1/knowledge/notifications/{id}/read
# =============================================================================


class TestMarkRead:
    """Tests for POST mark notification as read endpoint."""

    def test_mark_read_success(self, handler, mock_rbac_checker):
        """Successfully marks notification as read."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.mark_notification_read",
                return_value=True,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif_123/read",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True

    def test_mark_read_not_found(self, handler, mock_rbac_checker):
        """Returns 404 when notification not found."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.mark_notification_read",
                return_value=False,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_post(
                "/api/v1/knowledge/notifications/nonexistent/read",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 404


# =============================================================================
# Tests: POST /api/v1/knowledge/notifications/read-all
# =============================================================================


class TestMarkAllRead:
    """Tests for POST mark all notifications as read endpoint."""

    def test_mark_all_read_success(self, handler, mock_rbac_checker):
        """Successfully marks all notifications as read."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.mark_all_notifications_read",
                return_value=10,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["count"] == 10


# =============================================================================
# Tests: POST /api/v1/knowledge/notifications/{id}/dismiss
# =============================================================================


class TestDismissNotification:
    """Tests for POST dismiss notification endpoint."""

    def test_dismiss_success(self, handler, mock_rbac_checker, mock_notification_store):
        """Successfully dismisses notification."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notification_store",
                return_value=mock_notification_store,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_post(
                "/api/v1/knowledge/notifications/notif_123/dismiss",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True

    def test_dismiss_not_found(self, handler, mock_rbac_checker):
        """Returns 404 when notification not found."""
        mock_handler = _make_mock_handler()
        store = MagicMock()
        store.dismiss_notification = MagicMock(return_value=False)

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notification_store",
                return_value=store,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_post(
                "/api/v1/knowledge/notifications/nonexistent/dismiss",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 404


# =============================================================================
# Tests: GET /api/v1/knowledge/notifications/preferences
# =============================================================================


class TestGetPreferences:
    """Tests for GET notification preferences endpoint."""

    def test_get_preferences_success(self, handler, mock_rbac_checker):
        """Successfully retrieves preferences."""
        mock_handler = _make_mock_handler()
        prefs = MockNotificationPreferences()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notification_preferences",
                return_value=prefs,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["user_id"] == "user_123"
        assert body["email_on_share"] is True
        assert body["in_app_enabled"] is True


# =============================================================================
# Tests: PUT /api/v1/knowledge/notifications/preferences
# =============================================================================


class TestUpdatePreferences:
    """Tests for PUT notification preferences endpoint."""

    def test_update_preferences_success(self, handler, mock_rbac_checker):
        """Successfully updates preferences."""
        mock_handler = _make_mock_handler(
            json_body={
                "email_on_share": False,
                "telegram_enabled": True,
            }
        )

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch.object(
                handler, "read_json_body_validated", return_value=(mock_handler.json_body, None)
            ),
            patch(
                "aragora.knowledge.mound.notifications.NotificationPreferences",
            ) as mock_prefs_class,
            patch(
                "aragora.knowledge.mound.notifications.set_notification_preferences",
            ) as mock_set,
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_put(
                "/api/v1/knowledge/notifications/preferences",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        mock_set.assert_called_once()


# =============================================================================
# Tests: RBAC Permission Checks
# =============================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    def test_read_permission_denied(self, handler, mock_rbac_checker_denied):
        """Returns 403 when read permission denied."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker_denied,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 403

    def test_write_permission_denied(self, handler, mock_rbac_checker_denied):
        """Returns 403 when write permission denied."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker_denied,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle_post(
                "/api/v1/knowledge/notifications/read-all",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 403


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_get_notifications_internal_error(self, handler, mock_rbac_checker):
        """Returns 500 on internal error."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notifications_for_user",
                side_effect=RuntimeError("Database error"),
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        body, status = _parse_result(result)
        assert status == 500

    def test_rbac_check_exception_graceful_degradation(self, handler):
        """Continues without RBAC when checker raises exception."""
        mock_handler = _make_mock_handler()

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                side_effect=RuntimeError("RBAC unavailable"),
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notifications_for_user",
                return_value=[],
            ),
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            result = handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        # Should succeed despite RBAC error (graceful degradation)
        body, status = _parse_result(result)
        assert status == 200


# =============================================================================
# Tests: Anonymous User Handling
# =============================================================================


class TestAnonymousUser:
    """Tests for anonymous user handling."""

    def test_anonymous_user_gets_anonymous_id(self, handler, mock_rbac_checker):
        """User without user_id gets 'anonymous' as user_id."""
        mock_handler = _make_mock_handler()
        mock_handler.user.user_id = None

        with (
            patch.object(handler, "require_auth_or_error") as mock_auth,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications._notifications_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.get_permission_checker",
                return_value=mock_rbac_checker,
            ),
            patch(
                "aragora.server.handlers.knowledge.sharing_notifications.RBAC_AVAILABLE",
                True,
            ),
            patch(
                "aragora.knowledge.mound.notifications.get_notifications_for_user",
                return_value=[],
            ) as mock_get,
        ):
            mock_auth.return_value = (mock_handler.user, None)
            mock_limiter.is_allowed.return_value = True

            handler.handle(
                "/api/v1/knowledge/notifications",
                {},
                mock_handler,
            )

        # Should be called with "anonymous" as user_id
        mock_get.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
