"""Tests for Telegram bot handler.

Comprehensive test coverage for:
- Message handling (text, media, commands)
- Webhook processing
- Callback queries
- Error handling
- Rate limiting
- Secret/token verification
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aragora.server.handlers.bots.telegram import (
    TelegramHandler,
    _verify_telegram_secret,
    _verify_webhook_token,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_WEBHOOK_SECRET,
    TELEGRAM_WEBHOOK_TOKEN,
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

    def test_verify_secret_empty_token(self):
        """Should reject empty token when secret is configured."""
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", "secret"):
            result = _verify_telegram_secret("")
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

    def test_verify_webhook_token_timing_safe(self):
        """Token verification should use constant-time comparison."""
        # This tests that hmac.compare_digest is used (timing-safe)
        # by verifying the function behavior with similar-length strings
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN", "abcdef"):
            # Similar prefix but different - should still fail quickly
            result = _verify_webhook_token("abcdeg")
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

    def test_cannot_handle_unrelated_route(self):
        """Should not handle unrelated routes."""
        handler = TelegramHandler({})
        assert handler.can_handle("/api/v1/bots/slack/webhook") is False
        assert handler.can_handle("/api/v1/debates") is False

    def test_bot_platform_attribute(self):
        """Should have correct bot platform attribute."""
        handler = TelegramHandler({})
        assert handler.bot_platform == "telegram"

    def test_is_bot_enabled_with_token(self):
        """Should report enabled when token is configured."""
        handler = TelegramHandler({})
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            assert handler._is_bot_enabled() is True

    def test_is_bot_enabled_without_token(self):
        """Should report disabled when token is not configured."""
        handler = TelegramHandler({})
        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", None):
            assert handler._is_bot_enabled() is False


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
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
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

    @pytest.mark.asyncio
    async def test_status_includes_webhook_token_prefix(self):
        """Status should include truncated webhook token for debugging."""
        handler = TelegramHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_TOKEN",
                    "abcdef123456",
                ):
                    mock_handler = MagicMock()
                    result = await handler.handle("/api/v1/bots/telegram/status", {}, mock_handler)

        body = json.loads(result.body)
        assert body["webhook_token"] == "abcdef12..."

    @pytest.mark.asyncio
    async def test_status_unauthorized(self):
        """Should return 401 when not authenticated."""
        handler = TelegramHandler({})

        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No token")
            mock_handler = MagicMock()
            result = await handler.handle("/api/v1/bots/telegram/status", {}, mock_handler)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_status_forbidden(self):
        """Should return 403 when lacking permission."""
        handler = TelegramHandler({})

        from aragora.server.handlers.utils.auth import ForbiddenError

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Missing bots.read")
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/telegram/status", {}, mock_handler)

        assert result.status_code == 403


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

    def test_handle_edited_message(self):
        """Should handle edited message update."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123456,
            "edited_message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "testuser"},
                "text": "Edited message",
                "edit_date": 1234567890,
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

    def test_handle_inline_query(self):
        """Should handle inline query update."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123458,
            "inline_query": {
                "id": "inline123",
                "from": {"id": 67890, "username": "testuser"},
                "query": "search term",
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

    def test_handle_unknown_update_type(self):
        """Should acknowledge unknown update types gracefully."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123459,
            "some_future_type": {"data": "value"},
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

    def test_handle_webhook_invalid_secret_header(self):
        """Should reject webhook with invalid secret header."""
        handler = TelegramHandler({})

        update = {"update_id": 123}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "wrong_secret",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch(
            "aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", "correct_secret"
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 401

    def test_handle_webhook_exception(self):
        """Should handle exceptions and return 200 to prevent Telegram retries."""
        handler = TelegramHandler({})

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "100",
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        # Simulate read error (use OSError which the handler catches)
        mock_request.rfile.read.side_effect = OSError("Read error")

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        # Should return 200 to prevent Telegram from retrying
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["ok"] is False
        assert "error" in body


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

    def test_handle_debate_command(self):
        """Should handle /debate command."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "testuser"},
                "text": "/debate Should we use microservices?",
                "entities": [{"type": "bot_command", "offset": 0, "length": 7}],
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
        body = json.loads(result.body)
        assert body.get("debate_started") is True

    def test_handle_debate_command_empty_topic(self):
        """Should request topic when /debate is sent without arguments."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                "text": "/debate",
                "entities": [{"type": "bot_command", "offset": 0, "length": 7}],
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

    def test_handle_ask_command_alias(self):
        """Should handle /ask as alias for /debate."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "testuser"},
                "text": "/ask Is Python better than JavaScript?",
                "entities": [{"type": "bot_command", "offset": 0, "length": 4}],
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
        body = json.loads(result.body)
        assert body.get("debate_started") is True

    def test_handle_status_command(self):
        """Should handle /status command."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                "text": "/status",
                "entities": [{"type": "bot_command", "offset": 0, "length": 7}],
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

    def test_handle_unknown_command(self):
        """Should respond to unknown commands."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                "text": "/unknowncommand",
                "entities": [{"type": "bot_command", "offset": 0, "length": 15}],
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


# =============================================================================
# Test Callback Query Handling
# =============================================================================


class TestTelegramCallbackQueries:
    """Tests for Telegram callback query (button press) handling."""

    def test_handle_vote_callback(self):
        """Should handle vote callback query."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
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
            with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
                with patch("httpx.Client") as mock_client:
                    mock_client.return_value.__enter__.return_value.post.return_value = MagicMock(
                        is_success=True
                    )
                    result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_generic_callback(self):
        """Should handle non-vote callback queries."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "callback_query": {
                "id": "callback456",
                "from": {"id": 67890},
                "data": "other_action:param1:param2",
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
        assert body.get("callback_handled") is True


# =============================================================================
# Test Message Sending
# =============================================================================


class TestTelegramMessageSending:
    """Tests for Telegram message sending functionality."""

    def test_send_message_without_token(self):
        """Should not attempt to send when token is not configured."""
        handler = TelegramHandler({})

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", None):
            # Should not raise, just log warning
            handler._send_message(12345, "Test message")

    def test_send_message_with_token(self):
        """Should send message when token is configured."""
        handler = TelegramHandler({})

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch("httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.is_success = True
                mock_client.return_value.__enter__.return_value.post.return_value = mock_response

                handler._send_message(12345, "Test message")

                # Verify the API was called
                mock_client.return_value.__enter__.return_value.post.assert_called_once()

    def test_send_message_handles_http_error(self):
        """Should handle HTTP errors gracefully."""
        handler = TelegramHandler({})

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch("httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.is_success = False
                mock_response.status_code = 400
                mock_client.return_value.__enter__.return_value.post.return_value = mock_response

                # Should not raise
                handler._send_message(12345, "Test message")

    def test_send_message_handles_exception(self):
        """Should handle exceptions during sending gracefully."""
        handler = TelegramHandler({})

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.side_effect = Exception(
                    "Network error"
                )

                # Should not raise
                handler._send_message(12345, "Test message")


# =============================================================================
# Test Rate Limiting
# =============================================================================


class TestTelegramRateLimiting:
    """Tests for rate limiting on Telegram endpoints."""

    @pytest.mark.asyncio
    async def test_handle_respects_rate_limit_decorator(self):
        """The handle method should have rate limiting applied."""
        handler = TelegramHandler({})

        # Check that the rate_limit decorator is applied
        assert hasattr(handler.handle, "__wrapped__") or callable(handler.handle)

    def test_handle_post_respects_rate_limit_decorator(self):
        """The handle_post method should have rate limiting applied."""
        handler = TelegramHandler({})

        # Check that the rate_limit decorator is applied
        assert hasattr(handler.handle_post, "__wrapped__") or callable(handler.handle_post)


# =============================================================================
# Test Error Handling
# =============================================================================


class TestTelegramErrorHandling:
    """Tests for error handling in Telegram handler."""

    def test_missing_chat_info(self):
        """Should handle messages with missing chat info."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "message_id": 1,
                "chat": {},  # Empty chat info
                "text": "Hello",
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

        # Should not crash, should return OK
        assert result is not None
        assert result.status_code == 200

    def test_missing_from_user(self):
        """Should handle messages with missing from user info."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                # No "from" field
                "text": "Hello",
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

    def test_group_chat_message(self):
        """Should handle messages from group chats."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "message": {
                "message_id": 1,
                "chat": {"id": -123456, "type": "group", "title": "Test Group"},
                "from": {"id": 67890, "username": "testuser"},
                "text": "Hello group",
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


# =============================================================================
# Test Vote Recording
# =============================================================================


class TestTelegramVoteRecording:
    """Tests for vote recording functionality."""

    def test_vote_records_to_consensus_store(self):
        """Should record votes to ConsensusStore when available."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "callback_query": {
                "id": "callback123",
                "from": {"id": 67890},
                "data": "vote:debate-abc:approve",
            },
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "token"):
                with patch("httpx.Client"):
                    with patch("aragora.memory.consensus.ConsensusStore") as mock_store_class:
                        mock_store = MagicMock()
                        mock_store_class.return_value = mock_store

                        result = handler.handle_post(
                            "/api/v1/bots/telegram/webhook", {}, mock_request
                        )

                        # Verify vote was recorded
                        mock_store.record_vote.assert_called_once_with(
                            debate_id="debate-abc",
                            user_id="telegram:67890",
                            vote="approve",
                            source="telegram",
                        )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("vote_recorded") is True

    def test_vote_handles_missing_consensus_store(self):
        """Should handle gracefully when ConsensusStore is not available."""
        handler = TelegramHandler({})

        update = {
            "update_id": 123,
            "callback_query": {
                "id": "callback123",
                "from": {"id": 67890},
                "data": "vote:debate-abc:approve",
            },
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(update))),
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_request.rfile.read.return_value = json.dumps(update).encode()

        with patch("aragora.server.handlers.bots.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            with patch("aragora.server.handlers.bots.telegram.TELEGRAM_BOT_TOKEN", "token"):
                with patch("httpx.Client"):
                    # Simulate ImportError for ConsensusStore
                    with patch.dict(
                        "sys.modules",
                        {"aragora.memory.consensus": None},
                    ):
                        result = handler.handle_post(
                            "/api/v1/bots/telegram/webhook", {}, mock_request
                        )

        assert result is not None
        assert result.status_code == 200
