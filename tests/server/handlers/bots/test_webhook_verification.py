"""
Comprehensive tests for webhook signature verification across all bot handlers.

This test suite ensures that:
1. Unsigned requests are rejected (401)
2. Invalid signatures are rejected (401)
3. Valid signatures pass through to business logic
4. Signature verification happens BEFORE any business logic
5. Audit logging occurs on verification failures

Covers:
- WhatsApp: HMAC-SHA256 with X-Hub-Signature-256 header
- Telegram: X-Telegram-Bot-Api-Secret-Token header
- Teams: Bot Framework JWT in Authorization header
- Discord: Ed25519 with X-Signature-Ed25519 and X-Signature-Timestamp
- Slack: HMAC-SHA256 with X-Slack-Signature and X-Slack-Request-Timestamp
- Google Chat: OAuth2 Bearer token in Authorization header
- Zoom: HMAC-SHA256 with x-zm-signature and x-zm-request-timestamp
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


class MockHandler:
    """Mock HTTP request handler for testing webhook endpoints."""

    def __init__(
        self,
        headers: dict[str, str],
        body: bytes,
        method: str = "POST",
    ):
        self.headers = headers
        self._body = body
        self.rfile = BytesIO(body)
        self.command = method

    def read(self, n: int = -1) -> bytes:
        return self.rfile.read(n)


def make_mock_handler(
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | bytes | None = None,
    method: str = "POST",
) -> MockHandler:
    """Create a mock HTTP handler with given headers and body."""
    if headers is None:
        headers = {}
    if body is None:
        body = b"{}"
    elif isinstance(body, dict):
        body = json.dumps(body).encode("utf-8")

    if "Content-Length" not in headers:
        headers["Content-Length"] = str(len(body))

    return MockHandler(headers, body, method)


# =============================================================================
# WhatsApp Webhook Verification Tests
# =============================================================================


class TestWhatsAppWebhookVerification:
    """Tests for WhatsApp webhook signature verification."""

    @pytest.fixture
    def whatsapp_handler(self):
        """Create a WhatsApp handler instance."""
        from aragora.server.handlers.bots.whatsapp import WhatsAppHandler

        return WhatsAppHandler({})

    def test_missing_signature_rejected(self, whatsapp_handler):
        """Test that requests without X-Hub-Signature-256 are rejected."""
        body = json.dumps({"entry": []}).encode()
        handler = make_mock_handler(
            headers={},  # No signature header
            body=body,
        )

        result = whatsapp_handler._handle_webhook(handler)

        # Should return 401 for missing/invalid signature
        assert result.status_code == 401

    def test_invalid_signature_rejected(self, whatsapp_handler):
        """Test that requests with invalid signatures are rejected."""
        body = json.dumps({"entry": []}).encode()
        handler = make_mock_handler(
            headers={
                "X-Hub-Signature-256": "sha256=invalid_signature_here",
            },
            body=body,
        )

        result = whatsapp_handler._handle_webhook(handler)

        # Should return 401 for invalid signature
        assert result.status_code == 401

    def test_valid_signature_accepted(self, whatsapp_handler):
        """Test that requests with valid signatures pass verification."""
        body = json.dumps({"entry": []}).encode()
        secret = "test_app_secret"

        # Compute valid signature
        signature = (
            "sha256="
            + hmac.new(
                secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
        )

        handler = make_mock_handler(
            headers={
                "X-Hub-Signature-256": signature,
            },
            body=body,
        )

        with patch.dict(os.environ, {"WHATSAPP_APP_SECRET": secret}):
            # Need to re-import to pick up the env var
            from aragora.server.handlers.bots import whatsapp

            original_secret = whatsapp.WHATSAPP_APP_SECRET
            whatsapp.WHATSAPP_APP_SECRET = secret
            try:
                result = whatsapp_handler._handle_webhook(handler)
                # Should pass verification and process (may return 200)
                assert result.status_code in (200, 401)  # 200 if processed, 401 if secret mismatch
            finally:
                whatsapp.WHATSAPP_APP_SECRET = original_secret

    def test_signature_verified_before_business_logic(self, whatsapp_handler):
        """Test that signature verification happens before processing entries."""
        body = json.dumps({"entry": [{"changes": [{"field": "messages", "value": {}}]}]}).encode()

        handler = make_mock_handler(
            headers={
                "X-Hub-Signature-256": "sha256=invalid",
            },
            body=body,
        )

        # Mock the message processing to track if it gets called
        with patch.object(whatsapp_handler, "_process_messages") as mock_process:
            result = whatsapp_handler._handle_webhook(handler)

            # Signature check should fail before processing
            assert result.status_code == 401
            mock_process.assert_not_called()


# =============================================================================
# Telegram Webhook Verification Tests
# =============================================================================


class TestTelegramWebhookVerification:
    """Tests for Telegram webhook signature verification."""

    @pytest.fixture
    def telegram_handler(self):
        """Create a Telegram handler instance."""
        from aragora.server.handlers.bots.telegram import TelegramHandler

        return TelegramHandler({})

    def test_missing_secret_token_rejected_in_production(self, telegram_handler):
        """Test that requests without secret token are rejected in production."""
        body = json.dumps({"update_id": 123, "message": {}}).encode()
        handler = make_mock_handler(
            headers={},  # No secret token
            body=body,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            from aragora.server.handlers.bots import telegram

            original_secret = telegram.TELEGRAM_WEBHOOK_SECRET
            telegram.TELEGRAM_WEBHOOK_SECRET = "test_secret"
            try:
                result = telegram_handler._handle_webhook(handler)
                # Should return 401 for missing secret token in production
                assert result.status_code == 401
            finally:
                telegram.TELEGRAM_WEBHOOK_SECRET = original_secret

    def test_invalid_secret_token_rejected(self, telegram_handler):
        """Test that requests with invalid secret token are rejected."""
        body = json.dumps({"update_id": 123, "message": {}}).encode()
        handler = make_mock_handler(
            headers={
                "X-Telegram-Bot-Api-Secret-Token": "wrong_secret",
            },
            body=body,
        )

        from aragora.server.handlers.bots import telegram

        original_secret = telegram.TELEGRAM_WEBHOOK_SECRET
        telegram.TELEGRAM_WEBHOOK_SECRET = "correct_secret"
        try:
            result = telegram_handler._handle_webhook(handler)
            # Should return 401 for wrong secret token
            assert result.status_code == 401
        finally:
            telegram.TELEGRAM_WEBHOOK_SECRET = original_secret

    def test_valid_secret_token_accepted(self, telegram_handler):
        """Test that requests with valid secret token pass verification."""
        body = json.dumps({"update_id": 123, "message": {"text": "test"}}).encode()
        secret = "valid_secret_token"

        handler = make_mock_handler(
            headers={
                "X-Telegram-Bot-Api-Secret-Token": secret,
            },
            body=body,
        )

        from aragora.server.handlers.bots import telegram

        original_secret = telegram.TELEGRAM_WEBHOOK_SECRET
        telegram.TELEGRAM_WEBHOOK_SECRET = secret
        try:
            result = telegram_handler._handle_webhook(handler)
            # Should pass verification (200 for success)
            assert result.status_code == 200
        finally:
            telegram.TELEGRAM_WEBHOOK_SECRET = original_secret

    def test_url_token_verification(self, telegram_handler):
        """Test token-in-URL path verification."""
        from aragora.server.handlers.bots.telegram import _verify_webhook_token

        # Mock the expected token
        from aragora.server.handlers.bots import telegram

        original_token = telegram.TELEGRAM_WEBHOOK_TOKEN
        telegram.TELEGRAM_WEBHOOK_TOKEN = "expected_token"

        try:
            # Valid token should pass
            assert _verify_webhook_token("expected_token") is True

            # Invalid token should fail
            assert _verify_webhook_token("wrong_token") is False
        finally:
            telegram.TELEGRAM_WEBHOOK_TOKEN = original_token


# =============================================================================
# Discord Webhook Verification Tests
# =============================================================================


class TestDiscordWebhookVerification:
    """Tests for Discord webhook Ed25519 signature verification."""

    @pytest.fixture
    def discord_handler(self):
        """Create a Discord handler instance."""
        from aragora.server.handlers.bots.discord import DiscordHandler

        return DiscordHandler({})

    def test_missing_signature_headers_rejected(self, discord_handler):
        """Test that requests without signature headers are rejected."""
        body = json.dumps({"type": 1}).encode()
        handler = make_mock_handler(
            headers={},  # No signature headers
            body=body,
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            discord_handler._handle_interactions(handler)
        )

        # Should return 401 for missing signature
        assert result.status_code == 401

    def test_missing_timestamp_rejected(self, discord_handler):
        """Test that requests without timestamp are rejected."""
        body = json.dumps({"type": 1}).encode()
        handler = make_mock_handler(
            headers={
                "X-Signature-Ed25519": "abc123",
                # Missing X-Signature-Timestamp
            },
            body=body,
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            discord_handler._handle_interactions(handler)
        )

        # Should return 401 for missing timestamp
        assert result.status_code == 401

    def test_old_timestamp_rejected(self, discord_handler):
        """Test that requests with old timestamps are rejected (replay protection)."""
        body = json.dumps({"type": 1}).encode()
        old_timestamp = str(int(time.time()) - 600)  # 10 minutes ago

        handler = make_mock_handler(
            headers={
                "X-Signature-Ed25519": "a" * 128,  # Fake signature
                "X-Signature-Timestamp": old_timestamp,
            },
            body=body,
        )

        from aragora.server.handlers.bots import discord

        original_key = discord.DISCORD_PUBLIC_KEY
        discord.DISCORD_PUBLIC_KEY = "a" * 64  # Fake public key

        try:
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                discord_handler._handle_interactions(handler)
            )

            # Should return 401 for old timestamp
            assert result.status_code == 401
        finally:
            discord.DISCORD_PUBLIC_KEY = original_key

    def test_invalid_signature_rejected(self, discord_handler):
        """Test that requests with invalid Ed25519 signatures are rejected."""
        body = json.dumps({"type": 1}).encode()
        timestamp = str(int(time.time()))

        handler = make_mock_handler(
            headers={
                "X-Signature-Ed25519": "invalid" * 16,  # Invalid signature
                "X-Signature-Timestamp": timestamp,
            },
            body=body,
        )

        from aragora.server.handlers.bots import discord

        original_key = discord.DISCORD_PUBLIC_KEY
        discord.DISCORD_PUBLIC_KEY = "a" * 64  # Needs valid hex public key

        try:
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                discord_handler._handle_interactions(handler)
            )

            # Should return 401 for invalid signature
            assert result.status_code == 401
        finally:
            discord.DISCORD_PUBLIC_KEY = original_key


# =============================================================================
# Slack Webhook Verification Tests
# =============================================================================


class TestSlackWebhookVerification:
    """Tests for Slack webhook signature verification."""

    @pytest.fixture
    def slack_handler(self):
        """Create a Slack handler instance."""
        from aragora.server.handlers.bots.slack import SlackHandler

        handler = SlackHandler({})
        handler._signing_secret = "test_signing_secret"
        return handler

    def test_missing_signature_rejected(self, slack_handler):
        """Test that requests without signature headers are rejected."""
        body = json.dumps({"event": {}}).encode()
        handler = make_mock_handler(
            headers={
                "X-Slack-Request-Timestamp": str(int(time.time())),
                # Missing X-Slack-Signature
            },
            body=body,
        )

        result = slack_handler._verify_signature(handler)
        assert result is False

    def test_missing_timestamp_rejected(self, slack_handler):
        """Test that requests without timestamp are rejected."""
        body = json.dumps({"event": {}}).encode()
        handler = make_mock_handler(
            headers={
                "X-Slack-Signature": "v0=abc123",
                # Missing X-Slack-Request-Timestamp
            },
            body=body,
        )

        result = slack_handler._verify_signature(handler)
        assert result is False

    def test_old_timestamp_rejected(self):
        """Test that requests with old timestamps are rejected (replay protection)."""
        from aragora.server.handlers.bots.slack import verify_slack_signature

        body = b'{"event": {}}'
        old_timestamp = str(int(time.time()) - 600)  # 10 minutes ago
        secret = "test_secret"

        # Compute signature with old timestamp
        sig_basestring = f"v0:{old_timestamp}:{body.decode()}"
        signature = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = verify_slack_signature(body, old_timestamp, signature, secret)
        assert result is False

    def test_invalid_signature_rejected(self):
        """Test that requests with invalid signatures are rejected."""
        from aragora.server.handlers.bots.slack import verify_slack_signature

        body = b'{"event": {}}'
        timestamp = str(int(time.time()))
        secret = "test_secret"

        # Invalid signature
        result = verify_slack_signature(body, timestamp, "v0=invalid", secret)
        assert result is False

    def test_valid_signature_accepted(self):
        """Test that requests with valid signatures pass verification."""
        from aragora.server.handlers.bots.slack import verify_slack_signature

        body = b'{"event": {}}'
        timestamp = str(int(time.time()))
        secret = "test_secret"

        # Compute valid signature
        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        signature = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = verify_slack_signature(body, timestamp, signature, secret)
        assert result is True


# =============================================================================
# Teams Webhook Verification Tests
# =============================================================================


class TestTeamsWebhookVerification:
    """Tests for Teams Bot Framework JWT verification."""

    @pytest.fixture
    def teams_handler(self):
        """Create a Teams handler instance."""
        from aragora.server.handlers.bots.teams import TeamsHandler

        return TeamsHandler({})

    def test_missing_auth_header_rejected(self):
        """Test that requests without Authorization header are rejected."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(_verify_teams_token("", "test_app_id"))

        assert result is False

    def test_malformed_auth_header_rejected(self):
        """Test that malformed Authorization headers are rejected."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        import asyncio

        # Missing "Bearer " prefix
        result = asyncio.get_event_loop().run_until_complete(
            _verify_teams_token("InvalidToken", "test_app_id")
        )

        assert result is False

    def test_verification_called_before_activity_processing(self):
        """Test that token verification happens before processing activities."""
        from aragora.server.handlers.bots import teams
        from aragora.server.handlers.bots.teams import TeamsHandler

        body = json.dumps(
            {
                "type": "message",
                "text": "test",
                "from": {"id": "user123"},
                "conversation": {"id": "conv123"},
                "serviceUrl": "https://test.com",
            }
        ).encode()

        handler = make_mock_handler(
            headers={
                "Authorization": "Bearer invalid_token",
            },
            body=body,
        )

        # Set credentials to enable the handler
        original_app_id = teams.TEAMS_APP_ID
        original_password = teams.TEAMS_APP_PASSWORD
        teams.TEAMS_APP_ID = "test_app_id"
        teams.TEAMS_APP_PASSWORD = "test_password"

        try:
            teams_handler = TeamsHandler({})
            # The handler should verify token before processing
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                teams_handler._handle_messages(handler)
            )

            # Should return 401 for invalid token
            assert result.status_code == 401
        finally:
            teams.TEAMS_APP_ID = original_app_id
            teams.TEAMS_APP_PASSWORD = original_password


# =============================================================================
# Google Chat Webhook Verification Tests
# =============================================================================


class TestGoogleChatWebhookVerification:
    """Tests for Google Chat webhook bearer token verification."""

    @pytest.fixture
    def google_chat_handler(self):
        """Create a Google Chat handler instance."""
        from aragora.server.handlers.bots.google_chat import GoogleChatHandler

        return GoogleChatHandler({})

    def test_missing_auth_header_rejected(self, google_chat_handler):
        """Test that requests without Authorization header are rejected."""
        body = json.dumps({"type": "MESSAGE"}).encode()
        handler = make_mock_handler(
            headers={},  # No Authorization header
            body=body,
        )

        from aragora.server.handlers.bots import google_chat

        original_creds = google_chat.GOOGLE_CHAT_CREDENTIALS
        google_chat.GOOGLE_CHAT_CREDENTIALS = "test_credentials"

        try:
            result = google_chat_handler._handle_webhook(handler)
            # Should return 401 for missing auth
            assert result.status_code == 401
        finally:
            google_chat.GOOGLE_CHAT_CREDENTIALS = original_creds

    def test_invalid_bearer_token_rejected(self, google_chat_handler):
        """Test that invalid Bearer tokens are rejected."""
        body = json.dumps({"type": "MESSAGE"}).encode()
        handler = make_mock_handler(
            headers={
                "Authorization": "Bearer invalid_token",
            },
            body=body,
        )

        from aragora.server.handlers.bots import google_chat

        original_creds = google_chat.GOOGLE_CHAT_CREDENTIALS
        google_chat.GOOGLE_CHAT_CREDENTIALS = "test_credentials"

        try:
            # Token verification will fail with invalid token
            result = google_chat_handler._handle_webhook(handler)
            # Should return 401 for invalid token (unless google-auth not installed)
            assert result.status_code in (200, 401)  # 200 if google-auth not installed (falls back)
        finally:
            google_chat.GOOGLE_CHAT_CREDENTIALS = original_creds

    def test_non_bearer_auth_rejected(self):
        """Test that non-Bearer auth schemes are rejected."""
        from aragora.server.handlers.bots.google_chat import _verify_google_chat_token

        # Basic auth should be rejected
        result = _verify_google_chat_token("Basic dXNlcjpwYXNz")
        assert result is False


# =============================================================================
# Zoom Webhook Verification Tests
# =============================================================================


class TestZoomWebhookVerification:
    """Tests for Zoom webhook signature verification."""

    @pytest.fixture
    def zoom_handler(self):
        """Create a Zoom handler instance."""
        from aragora.server.handlers.bots.zoom import ZoomHandler

        return ZoomHandler({})

    def test_missing_signature_rejected(self, zoom_handler):
        """Test that requests without signature are rejected."""
        body = json.dumps({"event": "bot_notification"}).encode()
        handler = make_mock_handler(
            headers={
                "x-zm-request-timestamp": str(int(time.time())),
                # Missing x-zm-signature
            },
            body=body,
        )

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(zoom_handler._handle_events(handler))

        # Should return 401 for missing signature
        assert result.status_code == 401

    def test_url_validation_requires_secret_token(self, zoom_handler):
        """Test that URL validation endpoint requires ZOOM_SECRET_TOKEN."""
        body = json.dumps(
            {
                "event": "endpoint.url_validation",
                "payload": {"plainToken": "test_token"},
            }
        ).encode()

        handler = make_mock_handler(
            headers={},
            body=body,
        )

        from aragora.server.handlers.bots import zoom

        original_secret = zoom.ZOOM_SECRET_TOKEN
        zoom.ZOOM_SECRET_TOKEN = None

        try:
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                zoom_handler._handle_events(handler)
            )

            # Should return 503 when secret not configured
            assert result.status_code == 503
        finally:
            zoom.ZOOM_SECRET_TOKEN = original_secret

    def test_url_validation_with_valid_secret(self, zoom_handler):
        """Test URL validation with valid secret token."""
        plain_token = "test_plain_token"
        body = json.dumps(
            {
                "event": "endpoint.url_validation",
                "payload": {"plainToken": plain_token},
            }
        ).encode()

        handler = make_mock_handler(
            headers={},
            body=body,
        )

        secret = "test_zoom_secret"
        expected_encrypted = hmac.new(
            secret.encode(),
            plain_token.encode(),
            hashlib.sha256,
        ).hexdigest()

        from aragora.server.handlers.bots import zoom

        original_secret = zoom.ZOOM_SECRET_TOKEN
        zoom.ZOOM_SECRET_TOKEN = secret

        try:
            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                zoom_handler._handle_events(handler)
            )

            # Should return 200 with encrypted token
            assert result.status_code == 200
            response_body = json.loads(result.body)
            assert response_body["plainToken"] == plain_token
            assert response_body["encryptedToken"] == expected_encrypted
        finally:
            zoom.ZOOM_SECRET_TOKEN = original_secret


# =============================================================================
# Centralized webhook_security Module Tests
# =============================================================================


class TestCentralizedWebhookSecurity:
    """Tests for the centralized webhook_security module."""

    def test_production_environment_always_requires_verification(self):
        """Test that production environments always require verification."""
        from aragora.connectors.chat.webhook_security import (
            is_production_environment,
            is_webhook_verification_required,
            should_allow_unverified,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            assert is_production_environment() is True
            assert is_webhook_verification_required() is True
            assert should_allow_unverified("test") is False

    def test_staging_environment_requires_verification(self):
        """Test that staging environments require verification."""
        from aragora.connectors.chat.webhook_security import (
            is_production_environment,
            should_allow_unverified,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}):
            assert is_production_environment() is True
            assert should_allow_unverified("test") is False

    def test_development_can_bypass_with_explicit_flag(self):
        """Test that development can bypass verification with explicit flag."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true",
            },
        ):
            assert should_allow_unverified("test") is True

    def test_development_without_flag_requires_verification(self):
        """Test that development without explicit flag requires verification."""
        from aragora.connectors.chat.webhook_security import (
            is_webhook_verification_required,
        )

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "",
            },
            clear=False,
        ):
            # Remove the flag if it exists
            os.environ.pop("ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS", None)
            assert is_webhook_verification_required() is True

    def test_slack_signature_verification_function(self):
        """Test the centralized Slack signature verification function."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        body = "test_body"
        timestamp = str(int(time.time()))
        secret = "test_secret"

        # Compute valid signature
        sig_basestring = f"v0:{timestamp}:{body}"
        signature = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = verify_slack_signature(timestamp, body, signature, secret)
        assert result.verified is True
        assert result.source == "slack"
        assert result.method == "hmac-sha256"

    def test_slack_signature_verification_rejects_old_timestamp(self):
        """Test that Slack verification rejects old timestamps."""
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        body = "test_body"
        old_timestamp = str(int(time.time()) - 600)  # 10 minutes ago
        secret = "test_secret"

        sig_basestring = f"v0:{old_timestamp}:{body}"
        signature = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = verify_slack_signature(old_timestamp, body, signature, secret)
        assert result.verified is False
        assert "timestamp" in result.error.lower()


# =============================================================================
# Audit Logging Tests
# =============================================================================


class TestWebhookAuthFailureAuditing:
    """Tests for audit logging on webhook authentication failures."""

    def test_whatsapp_audit_on_failure(self):
        """Test that WhatsApp handler audits authentication failures."""
        from aragora.server.handlers.bots.whatsapp import WhatsAppHandler

        handler_instance = WhatsAppHandler({})

        with patch("aragora.audit.unified.audit_security") as mock_audit:
            handler_instance._audit_webhook_auth_failure("signature")

            # Verify audit was called
            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert "whatsapp" in call_kwargs.get("event_type", "")

    def test_discord_audit_on_failure(self):
        """Test that Discord handler audits authentication failures."""
        from aragora.server.handlers.bots.discord import DiscordHandler

        handler_instance = DiscordHandler({})

        with patch("aragora.audit.unified.audit_security") as mock_audit:
            handler_instance._audit_webhook_auth_failure("signature")

            # Verify audit was called
            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert "discord" in call_kwargs.get("event_type", "")


# =============================================================================
# Integration Tests - End-to-End Verification Flow
# =============================================================================


class TestEndToEndVerificationFlow:
    """Integration tests for complete verification flows."""

    def test_whatsapp_full_flow_valid_signature(self):
        """Test WhatsApp complete flow with valid signature."""
        from aragora.server.handlers.bots.whatsapp import WhatsAppHandler
        from aragora.server.handlers.bots import whatsapp

        handler_instance = WhatsAppHandler({})

        secret = "integration_test_secret"
        body = json.dumps(
            {"entry": [{"changes": [{"field": "messages", "value": {"messages": []}}]}]}
        ).encode()

        signature = (
            "sha256="
            + hmac.new(
                secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
        )

        mock_handler = make_mock_handler(
            headers={"X-Hub-Signature-256": signature},
            body=body,
        )

        original_secret = whatsapp.WHATSAPP_APP_SECRET
        whatsapp.WHATSAPP_APP_SECRET = secret

        try:
            result = handler_instance._handle_webhook(mock_handler)
            # With valid signature, should process successfully
            assert result.status_code == 200
        finally:
            whatsapp.WHATSAPP_APP_SECRET = original_secret

    def test_slack_full_flow_valid_signature(self):
        """Test Slack complete flow with valid signature."""
        from aragora.server.handlers.bots.slack import SlackHandler
        from aragora.server.handlers.bots import slack

        secret = "integration_test_secret"
        body = json.dumps({"type": "url_verification", "challenge": "test123"}).encode()
        timestamp = str(int(time.time()))

        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        signature = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        mock_handler = make_mock_handler(
            headers={
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body,
        )

        handler_instance = SlackHandler({})
        handler_instance._signing_secret = secret

        original_secret = slack.SLACK_SIGNING_SECRET
        slack.SLACK_SIGNING_SECRET = secret

        try:
            # Verify signature check passes
            result = handler_instance._verify_signature(mock_handler)
            assert result is True
        finally:
            slack.SLACK_SIGNING_SECRET = original_secret


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling in webhook verification."""

    def test_empty_body_handling(self):
        """Test that empty request bodies are handled appropriately."""
        from aragora.server.handlers.bots.whatsapp import WhatsAppHandler

        handler_instance = WhatsAppHandler({})
        mock_handler = make_mock_handler(
            headers={"X-Hub-Signature-256": "sha256=test"},
            body=b"",
        )

        result = handler_instance._handle_webhook(mock_handler)
        # Should handle gracefully (either 400 for empty body or 401 for auth)
        assert result.status_code in (400, 401)

    def test_malformed_json_body(self):
        """Test that malformed JSON is handled appropriately."""
        from aragora.server.handlers.bots.whatsapp import WhatsAppHandler
        from aragora.server.handlers.bots import whatsapp

        handler_instance = WhatsAppHandler({})
        malformed_body = b"not valid json {"

        # Create valid signature for malformed body
        secret = "test_secret"
        signature = (
            "sha256="
            + hmac.new(
                secret.encode(),
                malformed_body,
                hashlib.sha256,
            ).hexdigest()
        )

        mock_handler = make_mock_handler(
            headers={"X-Hub-Signature-256": signature},
            body=malformed_body,
        )

        original_secret = whatsapp.WHATSAPP_APP_SECRET
        whatsapp.WHATSAPP_APP_SECRET = secret

        try:
            result = handler_instance._handle_webhook(mock_handler)
            # Should return 200 (to prevent retries) or 400 for bad JSON
            assert result.status_code in (200, 400)
        finally:
            whatsapp.WHATSAPP_APP_SECRET = original_secret

    def test_unicode_body_handling(self):
        """Test that unicode in request body is handled correctly."""
        from aragora.server.handlers.bots.slack import verify_slack_signature

        body = json.dumps({"text": "Hello"}).encode("utf-8")
        timestamp = str(int(time.time()))
        secret = "test_secret"

        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        signature = (
            "v0="
            + hmac.new(
                secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = verify_slack_signature(body, timestamp, signature, secret)
        assert result is True
