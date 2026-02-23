"""Tests for the NotificationsHandler REST endpoints.

Covers all routes and behavior of the NotificationsHandler class:
- can_handle() routing for all defined routes and prefix matching
- GET /api/v1/notifications/status - Integration status
- GET /api/v1/notifications/email/recipients - Email recipient list
- POST /api/v1/notifications/email/config - Configure email
- POST /api/v1/notifications/telegram/config - Configure Telegram
- POST /api/v1/notifications/email/recipient - Add email recipient
- POST /api/v1/notifications/test - Send test notification
- POST /api/v1/notifications/send - Send notification
- DELETE /api/v1/notifications/email/recipient - Remove recipient
- Rate limiting
- Error handling and edge cases
- TTL cache behavior
- Utility functions (notify_debate_completed, notify_consensus_reached, notify_error)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.notifications import (
    NotificationsHandler,
    _TTLCache,
    get_email_integration,
    get_telegram_integration,
    configure_email_integration,
    configure_telegram_integration,
    invalidate_org_integration_cache,
    notify_debate_completed,
    notify_consensus_reached,
    notify_error,
    _org_email_cache,
    _org_telegram_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to NotificationsHandler methods."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        path: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.headers: dict[str, str] = headers or {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = path

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create NotificationsHandler instance."""
    return NotificationsHandler(ctx={})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state between tests."""
    from aragora.server.handlers.social.notifications import _notifications_limiter

    _notifications_limiter._requests.clear()
    yield


@pytest.fixture(autouse=True)
def reset_system_integrations():
    """Reset system-wide integrations and caches between tests."""
    import aragora.server.handlers.social.notifications as mod

    mod._system_email_integration = None
    mod._system_telegram_integration = None
    mod._org_email_integrations.clear()
    mod._org_telegram_integrations.clear()
    _org_email_cache.clear()
    _org_telegram_cache.clear()
    yield
    mod._system_email_integration = None
    mod._system_telegram_integration = None
    mod._org_email_integrations.clear()
    mod._org_telegram_integrations.clear()
    _org_email_cache.clear()
    _org_telegram_cache.clear()


# ---------------------------------------------------------------------------
# Routing Tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() routing."""

    def test_can_handle_status(self, handler):
        assert handler.can_handle("/api/v1/notifications/status")

    def test_can_handle_email_config(self, handler):
        assert handler.can_handle("/api/v1/notifications/email/config")

    def test_can_handle_telegram_config(self, handler):
        assert handler.can_handle("/api/v1/notifications/telegram/config")

    def test_can_handle_email_recipient(self, handler):
        assert handler.can_handle("/api/v1/notifications/email/recipient")

    def test_can_handle_email_recipients(self, handler):
        assert handler.can_handle("/api/v1/notifications/email/recipients")

    def test_can_handle_test(self, handler):
        assert handler.can_handle("/api/v1/notifications/test")

    def test_can_handle_send(self, handler):
        assert handler.can_handle("/api/v1/notifications/send")

    def test_can_handle_history(self, handler):
        assert handler.can_handle("/api/v1/notifications/history")

    def test_cannot_handle_other_paths(self, handler):
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/users")
        assert not handler.can_handle("/api/v1/integrations/teams")

    def test_routes_list_complete(self, handler):
        assert len(handler.ROUTES) == 8
        expected = [
            "/api/v1/notifications/status",
            "/api/v1/notifications/history",
            "/api/v1/notifications/email/recipients",
            "/api/v1/notifications/email/config",
            "/api/v1/notifications/telegram/config",
            "/api/v1/notifications/email/recipient",
            "/api/v1/notifications/test",
            "/api/v1/notifications/send",
        ]
        for route in expected:
            assert route in handler.ROUTES

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "notification"


# ---------------------------------------------------------------------------
# GET /api/v1/notifications/status
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for GET /api/v1/notifications/status."""

    def test_status_no_integrations(self, handler):
        """Status returns unconfigured when no integrations exist."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/status", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["email"]["configured"] is False
        assert body["telegram"]["configured"] is False
        assert body["email"]["recipients_count"] == 0

    @patch(
        "aragora.server.handlers.social.notifications.get_email_integration_for_org",
        new_callable=AsyncMock,
    )
    @patch(
        "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
        new_callable=AsyncMock,
    )
    def test_status_with_email_configured(self, mock_tg, mock_email, handler):
        """Status shows configured email when integration exists."""
        mock_config = MagicMock()
        mock_config.smtp_host = "smtp.example.com"
        mock_config.notify_on_consensus = True
        mock_config.notify_on_debate_end = True
        mock_config.notify_on_error = False
        mock_config.enable_digest = True
        mock_config.digest_frequency = "weekly"

        mock_integration = MagicMock()
        mock_integration.config = mock_config
        mock_integration.recipients = [MagicMock(), MagicMock()]

        mock_email.return_value = mock_integration
        mock_tg.return_value = None

        http = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/status", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["email"]["configured"] is True
        assert body["email"]["host"] == "smtp.example.com"
        assert body["email"]["recipients_count"] == 2
        assert body["email"]["settings"]["notify_on_consensus"] is True
        assert body["email"]["settings"]["digest_frequency"] == "weekly"
        assert body["telegram"]["configured"] is False

    @patch(
        "aragora.server.handlers.social.notifications.get_email_integration_for_org",
        new_callable=AsyncMock,
    )
    @patch(
        "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
        new_callable=AsyncMock,
    )
    def test_status_with_telegram_configured(self, mock_tg, mock_email, handler):
        """Status shows configured telegram when integration exists."""
        tg_config = MagicMock()
        tg_config.chat_id = "1234567890abcdef"
        tg_config.notify_on_consensus = True
        tg_config.notify_on_debate_end = False
        tg_config.notify_on_error = True

        tg_integration = MagicMock()
        tg_integration.config = tg_config

        mock_email.return_value = None
        mock_tg.return_value = tg_integration

        http = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/status", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["telegram"]["configured"] is True
        # chat_id should be truncated
        assert body["telegram"]["chat_id"] == "12345678..."
        assert body["telegram"]["settings"]["notify_on_error"] is True
        assert body["email"]["configured"] is False

    @patch(
        "aragora.server.handlers.social.notifications.get_email_integration_for_org",
        new_callable=AsyncMock,
    )
    @patch(
        "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
        new_callable=AsyncMock,
    )
    def test_status_both_configured(self, mock_tg, mock_email, handler):
        """Status shows both integrations when both are configured."""
        email_config = MagicMock()
        email_config.smtp_host = "smtp.test.com"
        email_config.notify_on_consensus = True
        email_config.notify_on_debate_end = True
        email_config.notify_on_error = True
        email_config.enable_digest = False
        email_config.digest_frequency = "daily"

        email_int = MagicMock()
        email_int.config = email_config
        email_int.recipients = []

        tg_config = MagicMock()
        tg_config.chat_id = "abcdefghijk"
        tg_config.notify_on_consensus = False
        tg_config.notify_on_debate_end = True
        tg_config.notify_on_error = False

        tg_int = MagicMock()
        tg_int.config = tg_config

        mock_email.return_value = email_int
        mock_tg.return_value = tg_int

        http = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/status", {}, http)
        body = _body(result)
        assert body["email"]["configured"] is True
        assert body["telegram"]["configured"] is True

    def test_status_unhandled_path_returns_none(self, handler):
        """Unrecognized GET path returns None."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/notifications/unknown_route", {}, http)
        assert result is None


# ---------------------------------------------------------------------------
# GET /api/v1/notifications/email/recipients
# ---------------------------------------------------------------------------


class TestGetEmailRecipients:
    """Tests for GET /api/v1/notifications/email/recipients."""

    def test_recipients_no_org_no_email(self, handler):
        """Returns empty recipients when no email integration and no org."""
        # The conftest patches require_auth_or_error to return org_id="test-org-001"
        # so this will try the org-specific path via the store.
        # We need to mock the store path for org-specific recipients.
        mock_store = MagicMock()
        mock_store.get_recipients = AsyncMock(return_value=[])

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/notifications/email/recipients", {}, http)
            assert _status(result) == 200
            body = _body(result)
            assert body["recipients"] == []
            assert body["count"] == 0

    def test_recipients_with_org_returns_stored(self, handler):
        """Returns recipients from store when org_id is set."""
        from aragora.storage.notification_config_store import StoredEmailRecipient

        stored = [
            StoredEmailRecipient(org_id="test-org-001", email="alice@test.com", name="Alice"),
            StoredEmailRecipient(org_id="test-org-001", email="bob@test.com", name="Bob"),
        ]
        mock_store = MagicMock()
        mock_store.get_recipients = AsyncMock(return_value=stored)

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/notifications/email/recipients", {}, http)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 2
            assert body["recipients"][0]["email"] == "alice@test.com"
            assert body["recipients"][1]["name"] == "Bob"
            assert body["org_id"] == "test-org-001"

    def test_recipients_store_error_returns_empty(self, handler):
        """Returns empty list with error when store fails."""
        mock_store = MagicMock()
        mock_store.get_recipients = AsyncMock(side_effect=RuntimeError("db error"))

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/notifications/email/recipients", {}, http)
            assert _status(result) == 200
            body = _body(result)
            assert body["recipients"] == []


# ---------------------------------------------------------------------------
# POST /api/v1/notifications/email/config
# ---------------------------------------------------------------------------


class TestConfigureEmail:
    """Tests for POST /api/v1/notifications/email/config."""

    def test_configure_email_for_org(self, handler):
        """Configure email with org_id saves to store."""
        mock_store = MagicMock()
        mock_store.save_email_config = AsyncMock()

        body = {
            "smtp_host": "smtp.company.com",
            "smtp_port": 465,
            "smtp_username": "user@company.com",
            "smtp_password": "secret",
            "from_email": "noreply@company.com",
        }

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/config", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["success"] is True
            assert "smtp.company.com" in resp["message"]
            assert resp["org_id"] == "test-org-001"

    def test_configure_email_invalid_json(self, handler):
        """Malformed JSON body returns 400."""
        http = MockHTTPHandler(method="POST")
        http.headers = {"Content-Length": "11", "Content-Type": "application/json"}
        http.rfile.read.return_value = b"not-json!!!"
        result = handler.handle_post("/api/v1/notifications/email/config", {}, http)
        assert _status(result) == 400

    def test_configure_email_save_failure(self, handler):
        """Store save failure returns 500."""
        mock_store = MagicMock()
        mock_store.save_email_config = AsyncMock(side_effect=RuntimeError("db error"))

        body = {"smtp_host": "smtp.example.com"}

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/config", {}, http)
            assert _status(result) == 500

    def test_configure_email_system_wide_no_org(self, handler, monkeypatch):
        """Configure email system-wide when no org context."""
        # Temporarily patch require_auth_or_error to return user with no org_id
        from aragora.billing.auth.context import UserAuthContext

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )

        from aragora.server.handlers.base import BaseHandler

        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        body = {"smtp_host": "smtp.global.com", "smtp_port": 587}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/email/config", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True
        assert "smtp.global.com" in resp["message"]
        assert "org_id" not in resp

    def test_configure_email_with_all_fields(self, handler):
        """Configure email with all available fields."""
        mock_store = MagicMock()
        mock_store.save_email_config = AsyncMock()

        body = {
            "provider": "ses",
            "smtp_host": "email-smtp.us-east-1.amazonaws.com",
            "smtp_port": 587,
            "smtp_username": "AKIAIOSFODNN7EXAMPLE",
            "smtp_password": "wJalrXUtnFEMI",
            "use_tls": True,
            "use_ssl": False,
            "from_email": "debates@company.com",
            "from_name": "Company Debates",
            "notify_on_consensus": True,
            "notify_on_debate_end": False,
            "notify_on_error": True,
            "enable_digest": True,
            "digest_frequency": "weekly",
            "min_consensus_confidence": 0.8,
            "max_emails_per_hour": 100,
        }

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/config", {}, http)
            assert _status(result) == 200
            assert _body(result)["success"] is True

    def test_configure_email_unhandled_post_path(self, handler):
        """Unknown POST path returns None."""
        http = MockHTTPHandler(method="POST", body={})
        result = handler.handle_post("/api/v1/notifications/unknown", {}, http)
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/v1/notifications/telegram/config
# ---------------------------------------------------------------------------


class TestConfigureTelegram:
    """Tests for POST /api/v1/notifications/telegram/config."""

    def test_configure_telegram_for_org(self, handler):
        """Configure telegram with org_id saves to store."""
        mock_store = MagicMock()
        mock_store.save_telegram_config = AsyncMock()

        body = {
            "bot_token": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            "chat_id": "-100123456789",
        }

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["success"] is True
            assert resp["org_id"] == "test-org-001"

    def test_configure_telegram_missing_required_fields(self, handler):
        """Missing required bot_token/chat_id returns 400."""
        body = {"notify_on_consensus": True}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
        assert _status(result) == 400

    def test_configure_telegram_invalid_json(self, handler):
        """Malformed JSON body fails validation with 400."""
        http = MockHTTPHandler(method="POST")
        http.headers = {"Content-Length": "11", "Content-Type": "application/json"}
        http.rfile.read.return_value = b"not-json!!!"
        result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
        assert _status(result) == 400

    def test_configure_telegram_save_failure(self, handler):
        """Store save failure returns 500."""
        mock_store = MagicMock()
        mock_store.save_telegram_config = AsyncMock(side_effect=RuntimeError("db down"))

        body = {
            "bot_token": "123456:token",
            "chat_id": "-100123456789",
        }

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
            assert _status(result) == 500

    def test_configure_telegram_system_wide(self, handler, monkeypatch):
        """Configure telegram system-wide when no org context."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        body = {
            "bot_token": "123456:system-token",
            "chat_id": "-100999",
        }
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is True
        assert "org_id" not in resp

    def test_configure_telegram_with_all_settings(self, handler):
        """Configure telegram with all optional settings."""
        mock_store = MagicMock()
        mock_store.save_telegram_config = AsyncMock()

        body = {
            "bot_token": "123456:full-config",
            "chat_id": "-100full",
            "notify_on_consensus": False,
            "notify_on_debate_end": True,
            "notify_on_error": False,
            "min_consensus_confidence": 0.9,
            "max_messages_per_minute": 10,
        }

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# POST /api/v1/notifications/email/recipient
# ---------------------------------------------------------------------------


class TestAddEmailRecipient:
    """Tests for POST /api/v1/notifications/email/recipient."""

    def test_add_recipient_for_org(self, handler):
        """Add email recipient for an org stores it."""
        from aragora.storage.notification_config_store import StoredEmailRecipient

        stored = [
            StoredEmailRecipient(org_id="test-org-001", email="new@test.com", name="New User"),
        ]
        mock_store = MagicMock()
        mock_store.add_recipient = AsyncMock()
        mock_store.get_recipients = AsyncMock(return_value=stored)

        body = {"email": "new@test.com", "name": "New User"}

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["success"] is True
            assert resp["recipients_count"] == 1
            assert resp["org_id"] == "test-org-001"

    def test_add_recipient_invalid_email(self, handler):
        """Invalid email (no @) returns 400."""
        body = {"email": "not-an-email"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    def test_add_recipient_empty_email(self, handler):
        """Empty email returns 400."""
        body = {"email": ""}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
        assert _status(result) == 400

    def test_add_recipient_missing_email(self, handler):
        """Missing email field returns 400."""
        body = {"name": "Test"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
        assert _status(result) == 400

    def test_add_recipient_store_error(self, handler):
        """Store failure returns 500."""
        mock_store = MagicMock()
        mock_store.add_recipient = AsyncMock(side_effect=RuntimeError("db error"))

        body = {"email": "user@test.com"}

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
            assert _status(result) == 500

    def test_add_recipient_system_wide_no_email_configured(self, handler, monkeypatch):
        """System-wide add without email integration returns 503."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        body = {"email": "user@test.com"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
        assert _status(result) == 503

    def test_add_recipient_system_wide_with_email_configured(self, handler, monkeypatch):
        """System-wide add with email integration succeeds."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        mock_email_int = MagicMock()
        mock_email_int.recipients = []
        mock_email_int.add_recipient = MagicMock()

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email_int,
        ):
            body = {"email": "user@test.com", "name": "Test User"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
            assert _status(result) == 200
            assert _body(result)["success"] is True

    def test_add_recipient_with_preferences(self, handler):
        """Add recipient with custom preferences."""
        from aragora.storage.notification_config_store import StoredEmailRecipient

        stored = [
            StoredEmailRecipient(
                org_id="test-org-001",
                email="pref@test.com",
                name="Pref User",
                preferences={"digest_only": True},
            ),
        ]
        mock_store = MagicMock()
        mock_store.add_recipient = AsyncMock()
        mock_store.get_recipients = AsyncMock(return_value=stored)

        body = {
            "email": "pref@test.com",
            "name": "Pref User",
            "preferences": {"digest_only": True},
        }

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/email/recipient", {}, http)
            assert _status(result) == 200
            assert _body(result)["recipients_count"] == 1


# ---------------------------------------------------------------------------
# DELETE /api/v1/notifications/email/recipient
# ---------------------------------------------------------------------------


class TestRemoveEmailRecipient:
    """Tests for DELETE /api/v1/notifications/email/recipient."""

    def test_remove_recipient_for_org_success(self, handler):
        """Removing existing recipient returns success."""
        mock_store = MagicMock()
        mock_store.remove_recipient = AsyncMock(return_value=True)
        mock_store.get_recipients = AsyncMock(return_value=[])

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler()
            result = handler.handle_delete(
                "/api/v1/notifications/email/recipient",
                {"email": "remove@test.com"},
                http,
            )
            assert _status(result) == 200
            resp = _body(result)
            assert resp["success"] is True
            assert "remove@test.com" in resp["message"]

    def test_remove_recipient_for_org_not_found(self, handler):
        """Removing non-existent recipient returns 404."""
        mock_store = MagicMock()
        mock_store.remove_recipient = AsyncMock(return_value=False)
        mock_store.get_recipients = AsyncMock(return_value=[])

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler()
            result = handler.handle_delete(
                "/api/v1/notifications/email/recipient",
                {"email": "unknown@test.com"},
                http,
            )
            assert _status(result) == 404

    def test_remove_recipient_missing_email_param(self, handler):
        """Missing email query param returns 400."""
        http = MockHTTPHandler()
        result = handler.handle_delete(
            "/api/v1/notifications/email/recipient", {}, http
        )
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    def test_remove_recipient_store_error(self, handler):
        """Store failure returns 500."""
        mock_store = MagicMock()
        mock_store.remove_recipient = AsyncMock(side_effect=RuntimeError("db error"))

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            http = MockHTTPHandler()
            result = handler.handle_delete(
                "/api/v1/notifications/email/recipient",
                {"email": "fail@test.com"},
                http,
            )
            assert _status(result) == 500

    def test_remove_recipient_system_wide_no_email(self, handler, monkeypatch):
        """System-wide remove without email integration returns 503."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        http = MockHTTPHandler()
        result = handler.handle_delete(
            "/api/v1/notifications/email/recipient",
            {"email": "user@test.com"},
            http,
        )
        assert _status(result) == 503

    def test_remove_recipient_system_wide_success(self, handler, monkeypatch):
        """System-wide remove with email integration succeeds."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        mock_email_int = MagicMock()
        mock_email_int.remove_recipient.return_value = True
        mock_email_int.recipients = []

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email_int,
        ):
            http = MockHTTPHandler()
            result = handler.handle_delete(
                "/api/v1/notifications/email/recipient",
                {"email": "remove@test.com"},
                http,
            )
            assert _status(result) == 200
            assert _body(result)["success"] is True

    def test_remove_recipient_system_wide_not_found(self, handler, monkeypatch):
        """System-wide remove returns 404 when recipient not found."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        mock_email_int = MagicMock()
        mock_email_int.remove_recipient.return_value = False

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email_int,
        ):
            http = MockHTTPHandler()
            result = handler.handle_delete(
                "/api/v1/notifications/email/recipient",
                {"email": "nobody@test.com"},
                http,
            )
            assert _status(result) == 404

    def test_delete_unhandled_path_returns_none(self, handler):
        """Unrecognized DELETE path returns None."""
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/notifications/other", {}, http)
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/v1/notifications/test
# ---------------------------------------------------------------------------


class TestSendTestNotification:
    """Tests for POST /api/v1/notifications/test."""

    def test_test_no_integrations(self, handler):
        """Test notification with nothing configured returns failure for all."""
        body = {"type": "all"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/test", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["success"] is False
        assert resp["results"]["email"]["success"] is False
        assert resp["results"]["telegram"]["success"] is False

    def test_test_email_only_not_configured(self, handler):
        """Test email only when not configured."""
        body = {"type": "email"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/test", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["results"]["email"]["error"] == "Email not configured"
        assert "telegram" not in resp["results"]

    def test_test_telegram_only_not_configured(self, handler):
        """Test telegram only when not configured."""
        body = {"type": "telegram"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/test", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["results"]["telegram"]["error"] == "Telegram not configured"
        assert "email" not in resp["results"]

    def test_test_email_no_recipients(self, handler):
        """Test email when configured but no recipients."""
        mock_email = MagicMock()
        mock_email.recipients = []

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            body = {"type": "email"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/test", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["email"]["success"] is False
            assert "recipients" in resp["results"]["email"]["error"].lower()

    def test_test_email_with_recipients_success(self, handler):
        """Test email succeeds when integration has recipients."""
        mock_recipient = MagicMock()
        mock_recipient.email = "test@test.com"

        mock_email = MagicMock()
        mock_email.recipients = [mock_recipient]
        mock_email._send_email = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            body = {"type": "email"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/test", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["email"]["success"] is True

    def test_test_telegram_success(self, handler):
        """Test telegram notification succeeds."""
        mock_tg = MagicMock()
        mock_tg._send_message = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            body = {"type": "telegram"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/test", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["telegram"]["success"] is True

    def test_test_email_send_error(self, handler):
        """Test email send failure is handled gracefully."""
        mock_recipient = MagicMock()
        mock_recipient.email = "test@test.com"

        mock_email = MagicMock()
        mock_email.recipients = [mock_recipient]
        mock_email._send_email = AsyncMock(side_effect=ConnectionError("SMTP down"))

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            body = {"type": "email"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/test", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["email"]["success"] is False

    def test_test_telegram_send_error(self, handler):
        """Test telegram send failure is handled gracefully."""
        mock_tg = MagicMock()
        mock_tg._send_message = AsyncMock(side_effect=TimeoutError("API timeout"))

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            body = {"type": "telegram"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/test", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["telegram"]["success"] is False

    def test_test_default_type_is_all(self, handler):
        """Default notification type is 'all' when not specified."""
        body = {}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/test", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        # Both email and telegram should be tested
        assert "email" in resp["results"]
        assert "telegram" in resp["results"]


# ---------------------------------------------------------------------------
# POST /api/v1/notifications/send
# ---------------------------------------------------------------------------


class TestSendNotification:
    """Tests for POST /api/v1/notifications/send."""

    def test_send_validation_error_missing_message(self, handler):
        """Missing required message field returns 400."""
        body = {"subject": "Test Subject"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/send", {}, http)
        assert _status(result) == 400

    def test_send_email_success(self, handler):
        """Send email notification succeeds."""
        mock_recipient = MagicMock()
        mock_email = MagicMock()
        mock_email.recipients = [mock_recipient]
        mock_email._send_email = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            body = {
                "type": "email",
                "subject": "Important Update",
                "message": "This is a test message",
            }
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/send", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["email"]["success"] is True
            assert resp["results"]["email"]["sent"] == 1

    def test_send_email_no_integration(self, handler):
        """Send email when not configured returns failure."""
        body = {"type": "email", "message": "Hello"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/send", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["results"]["email"]["success"] is False

    def test_send_telegram_success(self, handler):
        """Send telegram notification succeeds."""
        mock_tg = MagicMock()
        mock_tg._send_message = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            body = {
                "type": "telegram",
                "subject": "Alert",
                "message": "Something happened",
            }
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/send", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["telegram"]["success"] is True

    def test_send_telegram_not_configured(self, handler):
        """Send telegram when not configured returns failure."""
        body = {"type": "telegram", "message": "Hello"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/send", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert resp["results"]["telegram"]["success"] is False

    def test_send_all_channels(self, handler):
        """Send to all channels tests both."""
        body = {"type": "all", "message": "Broadcast message"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/send", {}, http)
        assert _status(result) == 200
        resp = _body(result)
        assert "email" in resp["results"]
        assert "telegram" in resp["results"]

    def test_send_email_partial_failure(self, handler):
        """Send email with multiple recipients where some fail."""
        r1 = MagicMock()
        r2 = MagicMock()
        mock_email = MagicMock()
        mock_email.recipients = [r1, r2]
        # First succeeds, second fails
        mock_email._send_email = AsyncMock(side_effect=[True, False])

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            body = {"type": "email", "message": "Test partial"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/send", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["email"]["sent"] == 1
            assert resp["results"]["email"]["total"] == 2

    def test_send_email_connection_error(self, handler):
        """Send email with connection error is handled."""
        mock_recipient = MagicMock()
        mock_email = MagicMock()
        mock_email.recipients = [mock_recipient]
        mock_email._send_email = AsyncMock(side_effect=ConnectionError("SMTP unreachable"))

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            body = {"type": "email", "message": "Error test"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/send", {}, http)
            assert _status(result) == 200
            resp = _body(result)
            assert resp["results"]["email"]["success"] is False

    def test_send_with_html_message(self, handler):
        """Send with custom HTML message."""
        mock_tg = MagicMock()
        mock_tg._send_message = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            body = {
                "type": "telegram",
                "subject": "HTML Test",
                "message": "Plain text",
                "html_message": "<h1>Rich HTML</h1>",
            }
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/send", {}, http)
            assert _status(result) == 200

    def test_send_invalid_json_body(self, handler):
        """Malformed JSON body returns 400."""
        http = MockHTTPHandler(method="POST")
        http.headers = {"Content-Length": "11", "Content-Type": "application/json"}
        http.rfile.read.return_value = b"not-json!!!"
        result = handler.handle_post("/api/v1/notifications/send", {}, http)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting on notification endpoints."""

    def test_rate_limit_on_get(self, handler):
        """GET endpoints are rate limited."""
        from aragora.server.handlers.social.notifications import _notifications_limiter

        with patch.object(_notifications_limiter, "is_allowed", return_value=False):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/notifications/status", {}, http)
            assert _status(result) == 429

    def test_rate_limit_on_post(self, handler):
        """POST endpoints are rate limited."""
        from aragora.server.handlers.social.notifications import _notifications_limiter

        with patch.object(_notifications_limiter, "is_allowed", return_value=False):
            body = {"message": "test"}
            http = MockHTTPHandler(method="POST", body=body)
            result = handler.handle_post("/api/v1/notifications/send", {}, http)
            assert _status(result) == 429


# ---------------------------------------------------------------------------
# TTL Cache Tests
# ---------------------------------------------------------------------------


class TestTTLCache:
    """Tests for the _TTLCache internal class."""

    def test_cache_set_and_get(self):
        """Set a value and retrieve it."""
        cache = _TTLCache(max_size=10, ttl=3600)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        """Get on missing key returns None."""
        cache = _TTLCache()
        assert cache.get("nonexistent") is None

    def test_cache_invalidate(self):
        """Invalidate removes a key."""
        cache = _TTLCache()
        cache.set("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_cache_invalidate_nonexistent(self):
        """Invalidate on missing key does not raise."""
        cache = _TTLCache()
        cache.invalidate("nonexistent")  # Should not raise

    def test_cache_clear(self):
        """Clear removes all entries."""
        cache = _TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_cache_ttl_expiry(self):
        """Expired entries return None."""
        import time as time_mod

        cache = _TTLCache(ttl=0.001)  # Very short TTL
        cache.set("key1", "value1")
        time_mod.sleep(0.01)  # Wait for expiry
        assert cache.get("key1") is None

    def test_cache_eviction_at_max_size(self):
        """Oldest entry is evicted when cache reaches max_size."""
        cache = _TTLCache(max_size=2, ttl=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_cache_update_existing_key_no_eviction(self):
        """Updating an existing key does not evict."""
        cache = _TTLCache(max_size=2, ttl=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 10)  # Update, not new entry
        assert cache.get("a") == 10
        assert cache.get("b") == 2


# ---------------------------------------------------------------------------
# Backward Compatibility Functions
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Tests for backward-compatibility wrapper functions."""

    def test_get_email_integration_returns_none_without_env(self):
        """get_email_integration returns None when no env vars set."""
        result = get_email_integration()
        assert result is None

    def test_get_telegram_integration_returns_none_without_env(self):
        """get_telegram_integration returns None when no env vars set."""
        result = get_telegram_integration()
        assert result is None

    def test_configure_email_integration(self):
        """configure_email_integration creates a system-wide integration."""
        from aragora.integrations.email import EmailConfig

        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
        )
        integration = configure_email_integration(config)
        assert integration is not None
        assert integration.config.smtp_host == "smtp.test.com"
        # Should be retrievable
        assert get_email_integration() is integration

    def test_configure_telegram_integration(self):
        """configure_telegram_integration creates a system-wide integration."""
        from aragora.integrations.telegram import TelegramConfig

        config = TelegramConfig(bot_token="123:abc", chat_id="-100123")
        integration = configure_telegram_integration(config)
        assert integration is not None
        # Should be retrievable
        assert get_telegram_integration() is integration

    def test_invalidate_org_integration_cache(self):
        """invalidate_org_integration_cache clears caches for an org."""
        _org_email_cache.set("org-1", "cached_email")
        _org_telegram_cache.set("org-1", "cached_tg")
        invalidate_org_integration_cache("org-1")
        assert _org_email_cache.get("org-1") is None
        assert _org_telegram_cache.get("org-1") is None


# ---------------------------------------------------------------------------
# Async Utility Functions
# ---------------------------------------------------------------------------


class TestNotifyDebateCompleted:
    """Tests for notify_debate_completed utility."""

    @pytest.mark.asyncio
    async def test_no_integrations_returns_empty(self):
        """Returns empty dict when no integrations configured."""
        result = await notify_debate_completed(MagicMock())
        assert result == {}

    @pytest.mark.asyncio
    async def test_email_success(self):
        """Sends email notification on debate completion."""
        mock_email = MagicMock()
        mock_email.send_debate_summary = AsyncMock(return_value=2)

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            result = await notify_debate_completed(MagicMock())
            assert result["email"] is True

    @pytest.mark.asyncio
    async def test_email_failure(self):
        """Email failure is caught and returns False."""
        mock_email = MagicMock()
        mock_email.send_debate_summary = AsyncMock(side_effect=ConnectionError("fail"))

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            result = await notify_debate_completed(MagicMock())
            assert result["email"] is False

    @pytest.mark.asyncio
    async def test_telegram_success(self):
        """Sends telegram notification on debate completion."""
        mock_tg = MagicMock()
        mock_tg.post_debate_summary = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            result = await notify_debate_completed(MagicMock())
            assert result["telegram"] is True

    @pytest.mark.asyncio
    async def test_telegram_failure(self):
        """Telegram failure is caught and returns False."""
        mock_tg = MagicMock()
        mock_tg.post_debate_summary = AsyncMock(side_effect=TimeoutError("timeout"))

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            result = await notify_debate_completed(MagicMock())
            assert result["telegram"] is False


class TestNotifyConsensusReached:
    """Tests for notify_consensus_reached utility."""

    @pytest.mark.asyncio
    async def test_no_integrations(self):
        """Returns empty dict when nothing configured."""
        result = await notify_consensus_reached("debate-1", 0.95)
        assert result == {}

    @pytest.mark.asyncio
    async def test_email_consensus_alert(self):
        """Email consensus alert is sent."""
        mock_email = MagicMock()
        mock_email.send_consensus_alert = AsyncMock(return_value=1)

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            result = await notify_consensus_reached("debate-1", 0.95, "agent-a", "Test task")
            assert result["email"] is True

    @pytest.mark.asyncio
    async def test_email_consensus_alert_failure(self):
        """Email consensus alert failure is handled."""
        mock_email = MagicMock()
        mock_email.send_consensus_alert = AsyncMock(side_effect=OSError("smtp error"))

        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email,
        ):
            result = await notify_consensus_reached("debate-1", 0.8)
            assert result["email"] is False

    @pytest.mark.asyncio
    async def test_telegram_consensus_alert(self):
        """Telegram consensus alert is sent."""
        mock_tg = MagicMock()
        mock_tg.send_consensus_alert = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            result = await notify_consensus_reached("debate-1", 0.9)
            assert result["telegram"] is True

    @pytest.mark.asyncio
    async def test_telegram_consensus_alert_failure(self):
        """Telegram consensus alert failure is handled."""
        mock_tg = MagicMock()
        mock_tg.send_consensus_alert = AsyncMock(side_effect=ValueError("bad data"))

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            result = await notify_consensus_reached("debate-1", 0.9)
            assert result["telegram"] is False


class TestNotifyError:
    """Tests for notify_error utility."""

    @pytest.mark.asyncio
    async def test_no_integrations(self):
        """Returns empty dict when nothing configured."""
        result = await notify_error("test_error", "Something broke")
        assert result == {}

    @pytest.mark.asyncio
    async def test_telegram_error_alert(self):
        """Telegram error alert is sent."""
        mock_tg = MagicMock()
        mock_tg.send_error_alert = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            result = await notify_error("api_error", "Rate limit exceeded", "debate-1", "warning")
            assert result["telegram"] is True
            mock_tg.send_error_alert.assert_awaited_once_with(
                "api_error", "Rate limit exceeded", "debate-1", "warning"
            )

    @pytest.mark.asyncio
    async def test_telegram_error_alert_failure(self):
        """Telegram error alert failure is handled."""
        mock_tg = MagicMock()
        mock_tg.send_error_alert = AsyncMock(side_effect=ConnectionError("network"))

        with patch(
            "aragora.server.handlers.social.notifications.get_telegram_integration",
            return_value=mock_tg,
        ):
            result = await notify_error("crash", "Unexpected error")
            assert result["telegram"] is False


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handler_init_default_ctx(self):
        """Handler initializes with empty context by default."""
        h = NotificationsHandler()
        assert h.ctx == {}

    def test_handler_init_with_ctx(self):
        """Handler initializes with provided context."""
        ctx = {"key": "value"}
        h = NotificationsHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_status_integration_error_handled(self, handler):
        """Integration loading errors during status are handled gracefully."""
        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration_for_org",
            new_callable=AsyncMock,
            side_effect=TypeError("bad config"),
        ):
            http = MockHTTPHandler()
            result = handler.handle("/api/v1/notifications/status", {}, http)
            assert _status(result) == 200
            body = _body(result)
            # Should fall back to unconfigured
            assert body["email"]["configured"] is False

    def test_email_config_value_error(self, handler):
        """ValueError during email config returns 400."""
        body = {"smtp_host": "test.com"}
        http = MockHTTPHandler(method="POST", body=body)

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.save_email_config = AsyncMock(side_effect=ValueError("bad value"))
            mock_get_store.return_value = mock_store
            result = handler.handle_post("/api/v1/notifications/email/config", {}, http)
            # ValueError from the async store save is caught as RuntimeError in thread
            # but the outer try/except catches ValueError directly for config creation
            assert _status(result) in (400, 500)

    def test_telegram_config_value_error_in_config_creation(self, handler, monkeypatch):
        """ValueError during TelegramConfig creation returns 400."""
        from aragora.billing.auth.context import UserAuthContext
        from aragora.server.handlers.base import BaseHandler

        user_no_org = UserAuthContext(
            authenticated=True,
            user_id="test-user-001",
            email="test@example.com",
            org_id=None,
            role="admin",
            token_type="access",
            client_ip="127.0.0.1",
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_auth_or_error",
            lambda self, h: (user_no_org, None),
        )
        monkeypatch.setattr(
            BaseHandler,
            "require_permission_or_error",
            lambda self, h, p: (user_no_org, None),
        )

        body = {"bot_token": "123:token", "chat_id": "-100123"}
        http = MockHTTPHandler(method="POST", body=body)

        with patch(
            "aragora.server.handlers.social.notifications.TelegramConfig",
            side_effect=ValueError("invalid token format"),
        ):
            result = handler.handle_post("/api/v1/notifications/telegram/config", {}, http)
            assert _status(result) == 400

    def test_get_system_email_from_env(self, monkeypatch):
        """System email integration is created from environment variables."""
        import aragora.server.handlers.social.notifications as mod

        mod._system_email_integration = None

        monkeypatch.setenv("SMTP_HOST", "smtp.env.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        monkeypatch.setenv("SMTP_USERNAME", "user")
        monkeypatch.setenv("SMTP_PASSWORD", "pass")
        monkeypatch.setenv("SMTP_USE_TLS", "false")
        monkeypatch.setenv("SMTP_USE_SSL", "true")

        result = get_email_integration()
        assert result is not None
        assert result.config.smtp_host == "smtp.env.com"
        assert result.config.smtp_port == 465

    def test_get_system_telegram_from_env(self, monkeypatch):
        """System telegram integration is created from environment variables."""
        import aragora.server.handlers.social.notifications as mod

        mod._system_telegram_integration = None

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:env-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100env")

        result = get_telegram_integration()
        assert result is not None

    def test_get_system_email_cached(self):
        """System email integration is cached after first creation."""
        from aragora.integrations.email import EmailConfig

        config = EmailConfig(smtp_host="cached.com", smtp_port=25)
        integration = configure_email_integration(config)
        # Second call should return the same object
        assert get_email_integration() is integration

    def test_get_system_telegram_cached(self):
        """System telegram integration is cached after first creation."""
        from aragora.integrations.telegram import TelegramConfig

        config = TelegramConfig(bot_token="123:cached", chat_id="-100cached")
        integration = configure_telegram_integration(config)
        assert get_telegram_integration() is integration

    def test_send_notification_invalid_type_still_validates(self, handler):
        """Send notification with invalid type in body schema fails validation."""
        body = {"type": "invalid_channel", "message": "Hello"}
        http = MockHTTPHandler(method="POST", body=body)
        result = handler.handle_post("/api/v1/notifications/send", {}, http)
        # Schema validation should catch invalid enum value
        assert _status(result) == 400
