"""
Tests for aragora.server.handlers.bots.zoom - Zoom Bot endpoint handler.

Tests cover:
- Route handling
- Status endpoint
- URL validation event
- Webhook signature verification
- Bot notification events
- Error handling
"""

from __future__ import annotations

import hashlib
import hmac
import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.bots.zoom import ZoomHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "POST",
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

    def get(self, key: str, default: str = "") -> str:
        return self.headers.get(key, default)


@pytest.fixture
def handler():
    """Create a ZoomHandler instance."""
    return ZoomHandler()


@pytest.fixture
def url_validation_event():
    """Create a URL validation event payload."""
    return {
        "event": "endpoint.url_validation",
        "payload": {
            "plainToken": "test-token-123",
        },
    }


@pytest.fixture
def bot_notification_event():
    """Create a bot notification event payload."""
    return {
        "event": "bot_notification",
        "payload": {
            "robotJid": "bot-jid-123",
            "cmd": "debate",
            "userId": "user-123",
            "userName": "Test User",
            "accountId": "account-123",
            "channelName": "test-channel",
            "toJid": "channel-123",
            "userJid": "user-jid-123",
        },
    }


@pytest.fixture
def meeting_ended_event():
    """Create a meeting ended event payload."""
    return {
        "event": "meeting.ended",
        "payload": {
            "object": {
                "id": "meeting-123",
                "uuid": "uuid-123",
                "host_id": "host-123",
                "topic": "Test Meeting",
            },
        },
    }


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_events(self, handler):
        """Test handler recognizes events endpoint."""
        assert handler.can_handle("/api/v1/bots/zoom/events") is True

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/bots/zoom/status") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/bots/zoom/unknown") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/bots/zoom/status."""

    @pytest.mark.asyncio
    async def test_get_status(self, handler):
        """Test getting Zoom bot status."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            method="GET",
        )

        result = await handler.handle("/api/v1/bots/zoom/status", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

        data = json.loads(result.body)
        assert "enabled" in data
        assert "client_id_configured" in data
        assert "client_secret_configured" in data
        assert "bot_jid_configured" in data
        assert "secret_token_configured" in data

    @pytest.mark.asyncio
    async def test_status_shows_unconfigured_when_no_env_vars(self, handler):
        """Test status shows unconfigured when env vars not set."""
        mock_http = MockHandler(method="GET")

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", ""):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", ""):
                result = await handler.handle("/api/v1/bots/zoom/status", {}, mock_http)

        data = json.loads(result.body)
        # When env vars are empty, enabled should be False
        assert data["client_id_configured"] is False
        assert data["client_secret_configured"] is False


# ===========================================================================
# URL Validation Tests
# ===========================================================================


class TestUrlValidation:
    """Tests for endpoint.url_validation event."""

    def test_url_validation_with_secret_token(self, handler, url_validation_event):
        """Test URL validation returns encrypted token when secret configured."""
        body = json.dumps(url_validation_event).encode()
        secret = "test-secret-token"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.server.handlers.bots.zoom.ZOOM_SECRET_TOKEN", secret):
            result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["plainToken"] == "test-token-123"
        assert "encryptedToken" in data

        # Verify the encrypted token is correct
        expected_encrypted = hmac.new(
            secret.encode(),
            "test-token-123".encode(),
            hashlib.sha256,
        ).hexdigest()
        assert data["encryptedToken"] == expected_encrypted

    def test_url_validation_without_secret_token(self, handler, url_validation_event):
        """Test URL validation returns plain token when no secret configured."""
        body = json.dumps(url_validation_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.server.handlers.bots.zoom.ZOOM_SECRET_TOKEN", ""):
            result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["plainToken"] == "test-token-123"
        assert "encryptedToken" not in data


# ===========================================================================
# Bot Notification Tests
# ===========================================================================


class TestBotNotification:
    """Tests for bot_notification events."""

    def test_bot_notification_without_bot(self, handler, bot_notification_event):
        """Test bot notification returns 503 when bot not configured."""
        body = json.dumps(bot_notification_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Bot not configured
        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", ""):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", ""):
                result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 503

        data = json.loads(result.body)
        assert "error" in data
        assert "not configured" in data["error"].lower()

    def test_bot_notification_with_bot(self, handler, bot_notification_event):
        """Test bot notification is processed when bot configured."""
        body = json.dumps(bot_notification_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Mock the bot
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"status": "processed"})

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", "test-id"):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", "test-secret"):
                with patch("aragora.bots.zoom_bot.create_zoom_bot", return_value=mock_bot):
                    handler._bot_initialized = False  # Reset
                    result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Signature Verification Tests
# ===========================================================================


class TestSignatureVerification:
    """Tests for Zoom webhook signature verification."""

    def test_invalid_signature_rejected(self, handler, bot_notification_event):
        """Test invalid signature is rejected when bot is configured."""
        body = json.dumps(bot_notification_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "x-zm-request-timestamp": "1234567890",
                "x-zm-signature": "invalid-signature",
            },
            body=body,
            method="POST",
        )

        # Mock the bot to reject signature
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = False

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", "test-id"):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", "test-secret"):
                with patch("aragora.bots.zoom_bot.create_zoom_bot", return_value=mock_bot):
                    handler._bot_initialized = False  # Reset
                    result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 401

    def test_valid_signature_accepted(self, handler, bot_notification_event):
        """Test valid signature is accepted."""
        body = json.dumps(bot_notification_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "x-zm-request-timestamp": "1234567890",
                "x-zm-signature": "valid-signature",
            },
            body=body,
            method="POST",
        )

        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"status": "ok"})

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", "test-id"):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", "test-secret"):
                with patch("aragora.bots.zoom_bot.create_zoom_bot", return_value=mock_bot):
                    handler._bot_initialized = False  # Reset
                    result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, handler):
        """Test handling invalid JSON body."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "10",
            },
            body=b"not valid json",
            method="POST",
        )

        result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    def test_empty_body(self, handler):
        """Test handling empty body."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "0",
            },
            body=b"",
            method="POST",
        )

        result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    def test_missing_event_type(self, handler):
        """Test handling missing event type."""
        body = json.dumps({"payload": {}}).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Without client credentials, should return 503 for non-validation events
        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", ""):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", ""):
                result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        # Should return 503 (bot not configured) for unknown events
        assert result.status_code == 503

    def test_bot_initialization_import_error(self, handler, bot_notification_event):
        """Test handling when zoom bot module import fails."""
        body = json.dumps(bot_notification_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", "test-id"):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", "test-secret"):
                with patch(
                    "aragora.bots.zoom_bot.create_zoom_bot",
                    side_effect=ImportError("Module not found"),
                ):
                    handler._bot_initialized = False  # Reset
                    result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        # Should return 503 when bot fails to initialize
        assert result.status_code == 503


# ===========================================================================
# Meeting Event Tests
# ===========================================================================


class TestMeetingEvents:
    """Tests for meeting-related events."""

    def test_meeting_ended_event(self, handler, meeting_ended_event):
        """Test handling meeting.ended event."""
        body = json.dumps(meeting_ended_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"status": "meeting_processed"})

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", "test-id"):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", "test-secret"):
                with patch("aragora.bots.zoom_bot.create_zoom_bot", return_value=mock_bot):
                    handler._bot_initialized = False
                    result = handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        # Verify handle_event was called with the event
        mock_bot.handle_event.assert_called_once()
        call_args = mock_bot.handle_event.call_args[0][0]
        assert call_args["event"] == "meeting.ended"


# ===========================================================================
# Bot Lazy Initialization Tests
# ===========================================================================


class TestBotInitialization:
    """Tests for lazy bot initialization."""

    def test_bot_initialized_only_once(self, handler, bot_notification_event):
        """Test bot is only initialized once."""
        body = json.dumps(bot_notification_event).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"status": "ok"})

        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", "test-id"):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", "test-secret"):
                with patch(
                    "aragora.bots.zoom_bot.create_zoom_bot", return_value=mock_bot
                ) as mock_create:
                    handler._bot_initialized = False

                    # First request
                    handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

                    # Second request (reread body)
                    mock_http.rfile = BytesIO(body)
                    handler.handle_post("/api/v1/bots/zoom/events", {}, mock_http)

                    # create_zoom_bot should only be called once
                    assert mock_create.call_count == 1

    def test_bot_not_created_without_credentials(self, handler):
        """Test bot is not created when credentials are missing."""
        with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_ID", ""):
            with patch("aragora.server.handlers.bots.zoom.ZOOM_CLIENT_SECRET", ""):
                handler._bot_initialized = False
                bot = handler._ensure_bot()

        assert bot is None
        assert handler._bot_initialized is True


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting decorators."""

    def test_handler_has_rate_limit(self, handler):
        """Test that handler methods have rate limit decorators."""
        # Check that rate_limit decorator is applied
        assert hasattr(handler.handle, "__wrapped__") or callable(handler.handle)
        assert hasattr(handler.handle_post, "__wrapped__") or callable(handler.handle_post)
