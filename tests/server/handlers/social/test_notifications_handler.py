"""
Tests for aragora.server.handlers.social.notifications - Notifications Handler.

Tests cover:
- Routing and authentication
- Status endpoint (GET /api/v1/notifications/status)
- Email configuration (POST /api/v1/notifications/email/config)
- Telegram configuration (POST /api/v1/notifications/telegram/config)
- Email recipients (POST/DELETE /api/v1/notifications/email/recipient)
- Rate limiting
- Multi-tenant isolation
- Cache invalidation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.notifications import (
    NotificationsHandler,
    get_email_integration,
    get_telegram_integration,
    invalidate_org_integration_cache,
)

from .conftest import (
    MockHandler,
    MockUser,
    get_json,
    get_status_code,
    parse_result,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler(mock_server_context):
    """Create a NotificationsHandler instance."""
    return NotificationsHandler(mock_server_context)


@pytest.fixture
def mock_email_integration():
    """Create a mock email integration."""
    integration = MagicMock()
    integration.config = MagicMock(
        smtp_host="smtp.example.com",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=False,
        enable_digest=True,
        digest_frequency="daily",
    )
    integration.recipients = [
        MagicMock(email="test@example.com", name="Test User"),
    ]
    integration.send = AsyncMock(return_value=True)
    return integration


@pytest.fixture
def mock_telegram_integration():
    """Create a mock Telegram integration."""
    integration = MagicMock()
    integration.config = MagicMock(
        chat_id="123456789",
        notify_on_consensus=True,
        notify_on_debate_end=True,
        notify_on_error=False,
    )
    integration.send = AsyncMock(return_value=True)
    return integration


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/notifications/status") is True

    def test_can_handle_email_config(self, handler):
        """Test handler recognizes email config endpoint."""
        assert handler.can_handle("/api/v1/notifications/email/config") is True

    def test_can_handle_telegram_config(self, handler):
        """Test handler recognizes telegram config endpoint."""
        assert handler.can_handle("/api/v1/notifications/telegram/config") is True

    def test_can_handle_email_recipient(self, handler):
        """Test handler recognizes email recipient endpoint."""
        assert handler.can_handle("/api/v1/notifications/email/recipient") is True

    def test_can_handle_test(self, handler):
        """Test handler recognizes test endpoint."""
        assert handler.can_handle("/api/v1/notifications/test") is True

    def test_can_handle_send(self, handler):
        """Test handler recognizes send endpoint."""
        assert handler.can_handle("/api/v1/notifications/send") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/other/endpoint") is False
        assert handler.can_handle("/api/v1/integrations/slack") is False

    def test_resource_type_defined(self, handler):
        """Handler should define resource type for RBAC."""
        assert hasattr(handler, "RESOURCE_TYPE")
        assert handler.RESOURCE_TYPE == "notification"


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limiter_exists(self):
        """Rate limiter should be configured."""
        from aragora.server.handlers.social.notifications import _notifications_limiter

        assert _notifications_limiter is not None
        assert hasattr(_notifications_limiter, "is_allowed")

    def test_rate_limit_applied(self, handler):
        """Rate limit should be enforced on requests."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/notifications/status",
            method="GET",
            client_address=("192.168.1.100", 12345),
        )

        # Mock rate limiter to reject
        with patch(
            "aragora.server.handlers.social.notifications._notifications_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = handler.handle("/api/v1/notifications/status", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 429


# ===========================================================================
# Status Endpoint Logic Tests
# ===========================================================================


class TestStatusLogic:
    """Tests for status endpoint internal logic."""

    def test_get_status_returns_json(self, handler):
        """Status should return JSON response."""
        with (
            patch(
                "aragora.server.handlers.social.notifications.get_email_integration_for_org",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = handler._get_status(org_id=None)

        assert result is not None
        status_code, body = parse_result(result)
        assert status_code == 200
        assert "email" in body
        assert "telegram" in body

    def test_get_status_email_configured(self, handler, mock_email_integration):
        """Status shows email as configured when integration exists."""
        with (
            patch(
                "aragora.server.handlers.social.notifications.get_email_integration_for_org",
                new_callable=AsyncMock,
                return_value=mock_email_integration,
            ),
            patch(
                "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = handler._get_status(org_id="test-org")

        status_code, body = parse_result(result)
        assert body["email"]["configured"] is True
        assert body["email"]["host"] == "smtp.example.com"
        assert body["email"]["recipients_count"] == 1

    def test_get_status_telegram_configured(self, handler, mock_telegram_integration):
        """Status shows telegram as configured when integration exists."""
        with (
            patch(
                "aragora.server.handlers.social.notifications.get_email_integration_for_org",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
                new_callable=AsyncMock,
                return_value=mock_telegram_integration,
            ),
        ):
            result = handler._get_status(org_id="test-org")

        status_code, body = parse_result(result)
        assert body["telegram"]["configured"] is True
        # Chat ID should be truncated for privacy
        assert "123" in body["telegram"]["chat_id"]
        assert "..." in body["telegram"]["chat_id"]

    def test_get_status_includes_settings(self, handler, mock_email_integration):
        """Status should include notification settings."""
        with (
            patch(
                "aragora.server.handlers.social.notifications.get_email_integration_for_org",
                new_callable=AsyncMock,
                return_value=mock_email_integration,
            ),
            patch(
                "aragora.server.handlers.social.notifications.get_telegram_integration_for_org",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = handler._get_status(org_id="test-org")

        status_code, body = parse_result(result)
        settings = body["email"]["settings"]
        assert settings["notify_on_consensus"] is True
        assert settings["notify_on_debate_end"] is True
        assert settings["enable_digest"] is True


# ===========================================================================
# Email Recipients Logic Tests
# ===========================================================================


class TestEmailRecipientsLogic:
    """Tests for email recipients internal logic."""

    def test_get_recipients_without_org(self, handler, mock_email_integration):
        """Getting recipients without org uses system integration."""
        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=mock_email_integration,
        ):
            result = handler._get_email_recipients(org_id=None)

        status_code, body = parse_result(result)
        assert status_code == 200
        assert body["count"] == 1
        assert body["recipients"][0]["email"] == "test@example.com"

    def test_get_recipients_no_integration(self, handler):
        """Getting recipients without integration shows empty list."""
        with patch(
            "aragora.server.handlers.social.notifications.get_email_integration",
            return_value=None,
        ):
            result = handler._get_email_recipients(org_id=None)

        status_code, body = parse_result(result)
        assert status_code == 200
        assert body["recipients"] == []
        assert "error" in body

    def test_get_recipients_with_org(self, handler):
        """Getting recipients with org queries org-specific store."""
        mock_recipients = [
            MagicMock(email="org-user@example.com", name="Org User"),
        ]

        mock_store = MagicMock()
        mock_store.get_recipients = AsyncMock(return_value=mock_recipients)

        with patch(
            "aragora.server.handlers.social.notifications.get_notification_config_store",
            return_value=mock_store,
        ):
            result = handler._get_email_recipients(org_id="test-org")

        status_code, body = parse_result(result)
        assert status_code == 200
        assert body["count"] == 1
        assert body["org_id"] == "test-org"


# ===========================================================================
# Email Configuration Logic Tests
# ===========================================================================


class TestEmailConfigurationLogic:
    """Tests for email configuration internal logic."""

    def test_configure_email_validates_schema(self, handler):
        """Email configuration should validate against schema."""
        mock_http = MockHandler.with_json_body(
            {"smtp_host": ""},  # Empty host
            path="/api/v1/notifications/email/config",
        )

        with (
            patch.object(
                handler, "read_json_body_validated", return_value=({"smtp_host": ""}, None)
            ),
            patch(
                "aragora.server.handlers.social.notifications.validate_against_schema"
            ) as mock_validate,
        ):
            mock_validate.return_value = MagicMock(is_valid=False, error="smtp_host is required")
            result = handler._configure_email(mock_http, org_id="test-org")

        assert get_status_code(result) == 400

    def test_configure_email_saves_to_store(self, handler):
        """Email configuration should save to org store."""
        mock_http = MockHandler.with_json_body(
            {"smtp_host": "smtp.test.com"},
            path="/api/v1/notifications/email/config",
        )

        mock_store = MagicMock()
        mock_store.save_email_config = AsyncMock()

        with (
            patch.object(
                handler,
                "read_json_body_validated",
                return_value=({"smtp_host": "smtp.test.com"}, None),
            ),
            patch(
                "aragora.server.handlers.social.notifications.validate_against_schema"
            ) as mock_validate,
            patch(
                "aragora.server.handlers.social.notifications.get_notification_config_store",
                return_value=mock_store,
            ),
        ):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler._configure_email(mock_http, org_id="test-org")

        assert get_status_code(result) == 200
        mock_store.save_email_config.assert_called_once()


# ===========================================================================
# Telegram Configuration Logic Tests
# ===========================================================================


class TestTelegramConfigurationLogic:
    """Tests for telegram configuration internal logic."""

    def test_configure_telegram_validates_schema(self, handler):
        """Telegram configuration should validate against schema."""
        mock_http = MockHandler.with_json_body(
            {"bot_token": ""},  # Empty token
            path="/api/v1/notifications/telegram/config",
        )

        with (
            patch.object(
                handler, "read_json_body_validated", return_value=({"bot_token": ""}, None)
            ),
            patch(
                "aragora.server.handlers.social.notifications.validate_against_schema"
            ) as mock_validate,
        ):
            mock_validate.return_value = MagicMock(is_valid=False, error="bot_token is required")
            result = handler._configure_telegram(mock_http, org_id="test-org")

        assert get_status_code(result) == 400


# ===========================================================================
# Cache Invalidation Tests
# ===========================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_cache_clears_email(self):
        """Invalidating cache should clear email entries."""
        from aragora.server.handlers.social import notifications

        # Add cached integration
        notifications._org_email_integrations["org-test"] = MagicMock()

        # Invalidate
        invalidate_org_integration_cache("org-test")

        assert "org-test" not in notifications._org_email_integrations

    def test_invalidate_cache_clears_telegram(self):
        """Invalidating cache should clear telegram entries."""
        from aragora.server.handlers.social import notifications

        # Add cached integration
        notifications._org_telegram_integrations["org-test"] = MagicMock()

        # Invalidate
        invalidate_org_integration_cache("org-test")

        assert "org-test" not in notifications._org_telegram_integrations

    def test_invalidate_nonexistent_org(self):
        """Invalidating nonexistent org should not error."""
        # Should not raise
        invalidate_org_integration_cache("nonexistent-org-12345")


# ===========================================================================
# Backward Compatibility Tests
# ===========================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with system-wide integrations."""

    def test_get_email_integration_returns_system(self):
        """get_email_integration should return system fallback."""
        from aragora.server.handlers.social import notifications

        mock_integration = MagicMock()
        notifications._system_email_integration = mock_integration

        result = get_email_integration()

        assert result is mock_integration

    def test_get_telegram_integration_returns_system(self):
        """get_telegram_integration should return system fallback."""
        from aragora.server.handlers.social import notifications

        mock_integration = MagicMock()
        notifications._system_telegram_integration = mock_integration

        result = get_telegram_integration()

        assert result is mock_integration

    def test_get_email_integration_none_when_not_configured(self):
        """get_email_integration returns None when not configured."""
        from aragora.server.handlers.social import notifications

        notifications._system_email_integration = None

        with patch.dict("os.environ", {}, clear=True):
            result = get_email_integration()

        assert result is None


# ===========================================================================
# Integration Factory Tests
# ===========================================================================


class TestIntegrationFactory:
    """Tests for integration factory functions."""

    @pytest.mark.asyncio
    async def test_get_email_for_org_checks_cache(self):
        """Email integration for org checks cache first."""
        from aragora.server.handlers.social.notifications import (
            get_email_integration_for_org,
        )
        from aragora.server.handlers.social import notifications

        mock_integration = MagicMock()
        notifications._org_email_integrations["cached-org"] = mock_integration

        result = await get_email_integration_for_org("cached-org")

        assert result is mock_integration

    @pytest.mark.asyncio
    async def test_get_telegram_for_org_checks_cache(self):
        """Telegram integration for org checks cache first."""
        from aragora.server.handlers.social.notifications import (
            get_telegram_integration_for_org,
        )
        from aragora.server.handlers.social import notifications

        mock_integration = MagicMock()
        notifications._org_telegram_integrations["cached-org"] = mock_integration

        result = await get_telegram_integration_for_org("cached-org")

        assert result is mock_integration


# ===========================================================================
# Configuration Storage Tests
# ===========================================================================


class TestConfigurationStorage:
    """Tests for configuration storage dataclasses."""

    def test_stored_email_config_fields(self):
        """StoredEmailConfig should have required fields."""
        from aragora.storage.notification_config_store import StoredEmailConfig

        config = StoredEmailConfig(
            org_id="test-org",
            smtp_host="smtp.test.com",
            smtp_port=587,
        )

        assert config.org_id == "test-org"
        assert config.smtp_host == "smtp.test.com"
        assert config.smtp_port == 587

    def test_stored_telegram_config_fields(self):
        """StoredTelegramConfig should have required fields."""
        from aragora.storage.notification_config_store import StoredTelegramConfig

        config = StoredTelegramConfig(
            org_id="test-org",
            bot_token="123:ABC",
            chat_id="-12345",
        )

        assert config.org_id == "test-org"
        assert config.bot_token == "123:ABC"
        assert config.chat_id == "-12345"
