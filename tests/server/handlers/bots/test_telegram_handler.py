"""Tests for Telegram bot handler."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.telegram import (
    TelegramHandler,
    _verify_telegram_secret,
    _verify_webhook_token,
)


# =============================================================================
# Test Signature Verification
# =============================================================================


class TestTelegramSignatureVerification:
    """Tests for Telegram webhook signature verification."""

    def test_verify_secret_no_config(self):
        """Should pass when no secret is configured."""
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            result = _verify_telegram_secret("any_token")
        assert result is True

    def test_verify_secret_valid(self):
        """Should verify valid secret token."""
        secret = "test_secret_123"
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", secret):
            result = _verify_telegram_secret(secret)
        assert result is True

    def test_verify_secret_invalid(self):
        """Should reject invalid secret token."""
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", "correct"):
            result = _verify_telegram_secret("wrong")
        assert result is False

    def test_verify_webhook_token_no_config(self):
        """Should pass when no webhook token is configured."""
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN", ""):
            result = _verify_webhook_token("any_token")
        assert result is True

    def test_verify_webhook_token_valid(self):
        """Should verify valid webhook token."""
        token = "test_token_abc"
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN", token):
            result = _verify_webhook_token(token)
        assert result is True

    def test_verify_webhook_token_invalid(self):
        """Should reject invalid webhook token."""
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN", "correct"):
            result = _verify_webhook_token("wrong")
        assert result is False


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestTelegramHandlerInit:
    """Tests for Telegram handler initialization."""

    def test_handler_routes(self):
        """Should define correct routes."""
        handler = TelegramHandler({})
        assert "/api/v1/bots/telegram/webhook" in handler.ROUTES
        assert "/api/v1/bots/telegram/status" in handler.ROUTES

    def test_can_handle_webhook_route(self):
        """Should handle webhook route."""
        handler = TelegramHandler({})
        assert handler.can_handle("/api/v1/bots/telegram/webhook") is True

    def test_can_handle_webhook_with_token(self):
        """Should handle webhook route with token."""
        handler = TelegramHandler({})
        assert handler.can_handle("/api/v1/bots/telegram/webhook/token123") is True

    def test_can_handle_status_route(self):
        """Should handle status route."""
        handler = TelegramHandler({})
        assert handler.can_handle("/api/v1/bots/telegram/status") is True


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestTelegramStatus:
    """Tests for Telegram status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Should return status information."""
        handler = TelegramHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots:read"])
            with patch.object(handler, "check_permission"):
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/telegram/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "telegram"
        assert "enabled" in body
        assert "token_configured" in body
        assert "webhook_secret_configured" in body


# =============================================================================
# Test Webhook Handling
# =============================================================================


class TestTelegramWebhook:
    """Tests for Telegram webhook handling."""

    def test_handle_update_message(self):
        """Should handle message update."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "testuser"},
                "text": "Hello bot",
            },
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["ok"] is True

    def test_handle_callback_query(self):
        """Should handle callback query update."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123457,
            "callback_query": {
                "id": "callback123",
                "from": {"id": 67890, "username": "testuser"},
                "data": "vote:debate123:agree",
            },
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = TelegramHandler({})

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "15",
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = b"not valid json"

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    def test_handle_webhook_with_token_path(self):
        """Should handle webhook with token in path."""
        handler = TelegramHandler({})

        update = {"update_id": 123, "message": {"text": "test"}}
        token = "validtoken123"

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(update)))}
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN", token):
            result = handler.handle_post(f"/api/v1/bots/telegram/webhook/{token}", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_webhook_invalid_token_path(self):
        """Should reject webhook with invalid token in path."""
        handler = TelegramHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "2"}
        mock_request.rfile.read.return_value = b"{}"

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN", "correct"):
            result = handler.handle_post(
                "/api/v1/bots/telegram/webhook/wrongtoken", {}, mock_request
            )

        assert result is not None
        assert result.status_code == 401


# =============================================================================
# Test Command Handling
# =============================================================================


class TestTelegramCommands:
    """Tests for Telegram bot commands."""

    def test_handle_start_command(self):
        """Should handle /start command."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                "text": "/start",
                "entities": [{"type": "bot_command", "offset": 0, "length": 6}],
            },
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", ""):
                result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_help_command(self):
        """Should handle /help command."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                "text": "/help",
                "entities": [{"type": "bot_command", "offset": 0, "length": 5}],
            },
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", ""):
                result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
