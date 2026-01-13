"""
Tests for notifications handler (notifications.py).

Tests cover:
- Integration status endpoints
- Email configuration
- Telegram configuration
- Recipient management
- Test notification sending
- Error handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(self):
        self.headers = {"Content-Type": "application/json", "Authorization": "Bearer test-token"}
        self.path = "/api/notifications/status"
        self.command = "GET"
        self._body = b"{}"
        self.rfile = MagicMock()
        self.rfile.read.return_value = self._body

    def set_body(self, data: dict):
        self._body = json.dumps(data).encode()
        self.rfile.read.return_value = self._body


def parse_result(result):
    """Parse HandlerResult to get JSON body and status."""
    body = json.loads(result.body.decode())
    return body, result.status_code


@pytest.fixture
def mock_handler():
    return MockHandler()


@pytest.fixture
def notifications_handler():
    """Create NotificationsHandler instance."""
    # Reset global integrations
    import aragora.server.handlers.notifications as notifications_module

    notifications_module._email_integration = None
    notifications_module._telegram_integration = None

    from aragora.server.handlers.notifications import NotificationsHandler

    # BaseHandler requires a server_context dict
    ctx = {"storage": MagicMock()}
    handler = NotificationsHandler(ctx)
    return handler


@pytest.fixture
def mock_auth():
    """Mock authentication to always succeed."""
    with patch(
        "aragora.server.handlers.notifications.NotificationsHandler.require_auth_or_error"
    ) as mock:
        mock.return_value = ({"user_id": "test-user"}, None)
        yield mock


# ============================================================================
# Status Endpoint Tests
# ============================================================================


class TestNotificationsStatus:
    """Test notification status endpoints."""

    def test_get_status_no_integrations(self, notifications_handler, mock_handler):
        """Test status when no integrations are configured."""
        mock_handler.path = "/api/notifications/status"
        result = notifications_handler.handle(mock_handler.path, {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert body["email"]["configured"] is False
        assert body["telegram"]["configured"] is False

    def test_get_status_email_configured(self, notifications_handler, mock_handler):
        """Test status when email is configured."""
        from aragora.server.handlers.notifications import configure_email_integration
        from aragora.integrations.email import EmailConfig

        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            from_email="test@test.com",
        )
        configure_email_integration(config)

        mock_handler.path = "/api/notifications/status"
        result = notifications_handler.handle(mock_handler.path, {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert body["email"]["configured"] is True
        assert body["email"]["host"] == "smtp.test.com"

    def test_can_handle_notifications_path(self, notifications_handler):
        """Test that handler recognizes notification paths."""
        assert notifications_handler.can_handle("/api/notifications/status") is True
        assert notifications_handler.can_handle("/api/notifications/email/config") is True
        assert notifications_handler.can_handle("/api/other/path") is False


# ============================================================================
# Email Configuration Tests
# ============================================================================


class TestEmailConfiguration:
    """Test email configuration endpoints."""

    def test_configure_email_success(self, notifications_handler, mock_handler, mock_auth):
        """Test successful email configuration."""
        mock_handler.path = "/api/notifications/email/config"
        body_data = {
            "smtp_host": "smtp.test.com",
            "smtp_port": 587,
            "smtp_username": "user",
            "smtp_password": "pass",
            "from_email": "test@test.com",
        }

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (body_data, None)
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 200
            assert body["success"] is True
            assert "smtp.test.com" in body["message"]

    def test_configure_email_json_parse_error(self, notifications_handler, mock_handler, mock_auth):
        """Test email configuration with JSON parse error."""
        mock_handler.path = "/api/notifications/email/config"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            # Simulate JSON validation error
            mock_read.return_value = (
                None,
                MagicMock(body=b'{"error":"Invalid JSON"}', status_code=400),
            )
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            # Should return the error from validation
            assert result.status_code == 400

    def test_configure_email_requires_auth(self, notifications_handler, mock_handler):
        """Test that email configuration requires authentication."""
        with patch.object(notifications_handler, "require_auth_or_error") as mock_auth_method:
            mock_auth_method.return_value = (
                None,
                MagicMock(body=b'{"error":"Unauthorized"}', status_code=401),
            )

            mock_handler.path = "/api/notifications/email/config"
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)

            assert result.status_code == 401


# ============================================================================
# Telegram Configuration Tests
# ============================================================================


class TestTelegramConfiguration:
    """Test Telegram configuration endpoints."""

    def test_configure_telegram_success(self, notifications_handler, mock_handler, mock_auth):
        """Test successful Telegram configuration."""
        mock_handler.path = "/api/notifications/telegram/config"
        mock_handler.set_body(
            {
                "bot_token": "test-bot-token",
                "chat_id": "test-chat-id",
            }
        )

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = (
                {"bot_token": "test-bot-token", "chat_id": "test-chat-id"},
                None,
            )
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 200
            assert body["success"] is True

    def test_configure_telegram_missing_fields(
        self, notifications_handler, mock_handler, mock_auth
    ):
        """Test Telegram configuration with missing required fields."""
        mock_handler.path = "/api/notifications/telegram/config"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"bot_token": "token"}, None)  # Missing chat_id
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 400
            assert "chat_id" in body["error"]


# ============================================================================
# Email Recipient Tests
# ============================================================================


class TestEmailRecipients:
    """Test email recipient management."""

    def test_get_recipients_no_email(self, notifications_handler, mock_handler):
        """Test getting recipients when email not configured."""
        mock_handler.path = "/api/notifications/email/recipients"
        result = notifications_handler.handle(mock_handler.path, {}, mock_handler)
        body, status = parse_result(result)

        assert status == 200
        assert body["recipients"] == []
        assert "error" in body

    def test_add_recipient_success(self, notifications_handler, mock_handler, mock_auth):
        """Test adding email recipient."""
        # First configure email
        from aragora.server.handlers.notifications import configure_email_integration
        from aragora.integrations.email import EmailConfig

        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            from_email="test@test.com",
        )
        configure_email_integration(config)

        mock_handler.path = "/api/notifications/email/recipient"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"email": "user@example.com", "name": "Test User"}, None)
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 200
            assert body["success"] is True
            assert body["recipients_count"] >= 1

    def test_add_recipient_invalid_email(self, notifications_handler, mock_handler, mock_auth):
        """Test adding invalid email recipient."""
        from aragora.server.handlers.notifications import configure_email_integration
        from aragora.integrations.email import EmailConfig

        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            from_email="test@test.com",
        )
        configure_email_integration(config)

        mock_handler.path = "/api/notifications/email/recipient"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"email": "invalid-email"}, None)  # No @ sign
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 400
            assert "email" in body["error"].lower()

    def test_add_recipient_email_not_configured(
        self, notifications_handler, mock_handler, mock_auth
    ):
        """Test adding recipient when email not configured."""
        mock_handler.path = "/api/notifications/email/recipient"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"email": "user@example.com"}, None)
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 503
            assert "not configured" in body["error"]

    def test_remove_recipient_success(self, notifications_handler, mock_handler, mock_auth):
        """Test removing email recipient."""
        from aragora.server.handlers.notifications import (
            configure_email_integration,
            get_email_integration,
        )
        from aragora.integrations.email import EmailConfig, EmailRecipient

        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            from_email="test@test.com",
        )
        integration = configure_email_integration(config)
        integration.add_recipient(EmailRecipient(email="user@example.com", name="Test"))

        mock_handler.path = "/api/notifications/email/recipient"
        result = notifications_handler.handle_delete(
            mock_handler.path, {"email": "user@example.com"}, mock_handler
        )
        body, status = parse_result(result)

        assert status == 200
        assert body["success"] is True

    def test_remove_recipient_not_found(self, notifications_handler, mock_handler, mock_auth):
        """Test removing non-existent recipient."""
        from aragora.server.handlers.notifications import configure_email_integration
        from aragora.integrations.email import EmailConfig

        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            from_email="test@test.com",
        )
        configure_email_integration(config)

        mock_handler.path = "/api/notifications/email/recipient"
        result = notifications_handler.handle_delete(
            mock_handler.path, {"email": "nonexistent@example.com"}, mock_handler
        )
        body, status = parse_result(result)

        assert status == 404


# ============================================================================
# Test Notification Tests
# ============================================================================


class TestNotificationSending:
    """Test notification sending endpoints."""

    def test_send_test_no_integrations(self, notifications_handler, mock_handler, mock_auth):
        """Test sending test notification with no integrations."""
        mock_handler.path = "/api/notifications/test"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"type": "all"}, None)
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 200
            assert body["success"] is False  # No integrations configured
            assert "email" in body["results"]
            assert "telegram" in body["results"]

    def test_send_notification_missing_message(
        self, notifications_handler, mock_handler, mock_auth
    ):
        """Test sending notification without message."""
        mock_handler.path = "/api/notifications/send"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"type": "all"}, None)  # Missing message
            result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
            body, status = parse_result(result)

            assert status == 400
            assert "message" in body["error"].lower()


# ============================================================================
# Exception Handling Tests
# ============================================================================


class TestNotificationExceptionHandling:
    """Test exception handling in notification endpoints."""

    def test_configure_email_exception(self, notifications_handler, mock_handler, mock_auth):
        """Test handling of exceptions during email configuration."""
        mock_handler.path = "/api/notifications/email/config"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"smtp_host": "test.com"}, None)
            # Patch at the module level where it's used
            with patch("aragora.server.handlers.notifications.EmailIntegration") as mock_email:
                mock_email.side_effect = Exception("Connection failed")
                result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
                body, status = parse_result(result)

                assert status == 500
                assert "failed" in body["error"].lower()

    def test_configure_telegram_exception(self, notifications_handler, mock_handler, mock_auth):
        """Test handling of exceptions during telegram configuration."""
        mock_handler.path = "/api/notifications/telegram/config"

        with patch.object(notifications_handler, "read_json_body_validated") as mock_read:
            mock_read.return_value = ({"bot_token": "token", "chat_id": "chat"}, None)
            # Patch at the module level where it's used
            with patch("aragora.server.handlers.notifications.TelegramIntegration") as mock_tg:
                mock_tg.side_effect = Exception("Invalid token")
                result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
                body, status = parse_result(result)

                assert status == 500
                assert "failed" in body["error"].lower()


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_email_integration_from_env(self):
        """Test email integration initialization from environment."""
        import aragora.server.handlers.notifications as mod

        mod._email_integration = None

        with patch.dict("os.environ", {"SMTP_HOST": "smtp.test.com"}):
            # Patch at the module level where it's imported
            with patch("aragora.server.handlers.notifications.EmailIntegration") as mock:
                mock.return_value = MagicMock()
                result = mod.get_email_integration()
                # Should attempt to create integration
                mock.assert_called_once()

    def test_get_telegram_integration_from_env(self):
        """Test telegram integration initialization from environment."""
        import aragora.server.handlers.notifications as mod

        mod._telegram_integration = None

        with patch.dict(
            "os.environ", {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_CHAT_ID": "test-chat"}
        ):
            # Patch at the module level where it's imported
            with patch("aragora.server.handlers.notifications.TelegramIntegration") as mock:
                mock.return_value = MagicMock()
                result = mod.get_telegram_integration()
                mock.assert_called_once()


# ============================================================================
# Path Routing Tests
# ============================================================================


class TestPathRouting:
    """Test endpoint path routing."""

    def test_handle_returns_none_for_unknown_path(self, notifications_handler, mock_handler):
        """Test that handler returns None for unknown paths."""
        mock_handler.path = "/api/notifications/unknown"
        result = notifications_handler.handle(mock_handler.path, {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_unknown_path(
        self, notifications_handler, mock_handler, mock_auth
    ):
        """Test that POST handler returns None for unknown paths."""
        mock_handler.path = "/api/notifications/unknown"
        result = notifications_handler.handle_post(mock_handler.path, {}, mock_handler)
        assert result is None

    def test_handle_delete_returns_none_for_unknown_path(
        self, notifications_handler, mock_handler, mock_auth
    ):
        """Test that DELETE handler returns None for unknown paths."""
        mock_handler.path = "/api/notifications/unknown"
        result = notifications_handler.handle_delete(mock_handler.path, {}, mock_handler)
        assert result is None
