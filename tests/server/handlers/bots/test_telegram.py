"""Comprehensive tests for Telegram bot webhook handler.

Tests cover:
- Initialization and routing
- Webhook secret/token verification (including production fail-closed behavior)
- Message handling (text, edited, group chat, missing fields)
- Command processing (/start, /help, /debate, /ask, /status, /aragora, unknown)
- Callback queries (vote, generic)
- Inline queries
- Vote recording (ConsensusStore success, ImportError fallback, runtime errors)
- Message sending (_send_message, _answer_callback_query)
- Status endpoint (RBAC auth, unauthorized, forbidden)
- Error handling (invalid JSON, exceptions, missing data)
- Debate async start (_start_debate_async, _fallback_queue_debate, _run_debate_direct)
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots import telegram as telegram_module
from aragora.server.handlers.bots.telegram import (
    TelegramHandler,
    _verify_telegram_secret,
    _verify_webhook_token,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_mock_request(
    update: dict[str, Any],
    secret_token: str = "",
) -> MagicMock:
    """Create a mock HTTP handler with Telegram update payload."""
    body = json.dumps(update).encode("utf-8")
    mock = MagicMock()
    mock.headers = {
        "Content-Length": str(len(body)),
        "X-Telegram-Bot-Api-Secret-Token": secret_token,
    }
    mock.rfile = BytesIO(body)
    return mock


def _make_handler() -> TelegramHandler:
    """Create a TelegramHandler with empty server context."""
    return TelegramHandler({})


# Helper: dispatch an update through the webhook endpoint
def _dispatch_update(
    handler: TelegramHandler,
    update: dict[str, Any],
    *,
    path: str = "/api/v1/bots/telegram/webhook",
    secret_token: str = "",
    mock_debate: bool = False,
):
    """Send an update through handle_post with secret verification disabled.

    Args:
        handler: The TelegramHandler instance.
        update: The Telegram update dict.
        path: The URL path.
        secret_token: Value for X-Telegram-Bot-Api-Secret-Token header.
        mock_debate: If True, mock _start_debate_async to avoid timeouts.
    """
    mock_request = _make_mock_request(update, secret_token=secret_token)
    with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", ""):
            if mock_debate:
                with patch.object(
                    handler,
                    "_start_debate_async",
                    return_value="mock-debate-id-12345678",
                ):
                    return handler.handle_post(path, {}, mock_request)
            return handler.handle_post(path, {}, mock_request)


# =============================================================================
# Test Webhook Secret Verification
# =============================================================================


class TestVerifyTelegramSecret:
    """Tests for _verify_telegram_secret function."""

    def test_returns_true_when_secret_matches(self):
        """Should return True when secret_token matches configured secret."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", "my_secret"):
            assert _verify_telegram_secret("my_secret") is True

    def test_returns_false_when_secret_mismatches(self):
        """Should return False when secret_token does not match."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", "correct"):
            assert _verify_telegram_secret("wrong") is False

    def test_returns_false_for_empty_token_when_secret_configured(self):
        """Should return False when token is empty but secret is configured."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", "configured"):
            assert _verify_telegram_secret("") is False

    def test_skips_verification_in_development_when_no_secret(self):
        """Should return True in dev mode when no secret is configured."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                assert _verify_telegram_secret("anything") is True

    def test_fails_closed_in_production_when_no_secret(self):
        """Should return False in production when no secret is configured."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
                assert _verify_telegram_secret("anything") is False

    def test_fails_closed_in_staging_when_no_secret(self):
        """Should return False in staging (non-dev) when no secret is configured."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "staging"}):
                assert _verify_telegram_secret("anything") is False

    def test_passes_in_test_env_when_no_secret(self):
        """Should return True in 'test' environment when no secret is configured."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "test"}):
                assert _verify_telegram_secret("anything") is True


# =============================================================================
# Test Webhook Token Verification
# =============================================================================


class TestVerifyWebhookToken:
    """Tests for _verify_webhook_token function."""

    def test_returns_true_when_token_matches(self):
        """Should return True when token matches the derived webhook token."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", "abc123"):
            assert _verify_webhook_token("abc123") is True

    def test_returns_false_when_token_mismatches(self):
        """Should return False when token does not match."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", "correct"):
            assert _verify_webhook_token("wrong") is False

    def test_skips_verification_in_dev_when_no_token(self):
        """Should return True in development when no webhook token is set."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                assert _verify_webhook_token("any") is True

    def test_fails_closed_in_production_when_no_token(self):
        """Should return False in production when no webhook token is set."""
        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
                assert _verify_webhook_token("any") is False


# =============================================================================
# Test Handler Initialization and Routing
# =============================================================================


class TestTelegramHandlerInit:
    """Tests for TelegramHandler class setup."""

    def test_bot_platform_is_telegram(self):
        """Should identify as 'telegram' platform."""
        handler = _make_handler()
        assert handler.bot_platform == "telegram"

    def test_routes_include_webhook_and_status(self):
        """Should include webhook and status in ROUTES."""
        handler = _make_handler()
        assert "/api/v1/bots/telegram/webhook" in handler.ROUTES
        assert "/api/v1/bots/telegram/status" in handler.ROUTES

    def test_can_handle_webhook(self):
        """Should handle the base webhook path."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/telegram/webhook", "POST") is True

    def test_can_handle_webhook_with_token_suffix(self):
        """Should handle webhook paths with appended token."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/telegram/webhook/sometoken123") is True

    def test_can_handle_status(self):
        """Should handle the status path."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/telegram/status") is True

    def test_cannot_handle_unknown_path(self):
        """Should not handle unrelated paths."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/slack/webhook") is False
        assert handler.can_handle("/api/v1/debates") is False

    def test_is_bot_enabled_true_when_token_set(self):
        """Should report enabled when TELEGRAM_BOT_TOKEN is set."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            assert handler._is_bot_enabled() is True

    def test_is_bot_enabled_false_when_no_token(self):
        """Should report disabled when TELEGRAM_BOT_TOKEN is not set."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", None):
            assert handler._is_bot_enabled() is False


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestTelegramStatusEndpoint:
    """Tests for the GET /status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_platform_info(self):
        """Should return status JSON with platform details."""
        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch.object(handler, "check_permission"):
                result = await handler.handle("/api/v1/bots/telegram/status", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "telegram"
        assert "enabled" in body
        assert "token_configured" in body
        assert "webhook_secret_configured" in body

    @pytest.mark.asyncio
    async def test_status_shows_truncated_webhook_token(self):
        """Should show first 8 chars of webhook token with ellipsis."""
        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
            with patch.object(handler, "check_permission"):
                with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", "abcdef1234567890"):
                    result = await handler.handle("/api/v1/bots/telegram/status", {}, MagicMock())

        body = json.loads(result.body)
        assert body["webhook_token"] == "abcdef12..."

    @pytest.mark.asyncio
    async def test_status_webhook_token_none_when_not_configured(self):
        """Should return null webhook_token when not configured."""
        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
            with patch.object(handler, "check_permission"):
                with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", ""):
                    result = await handler.handle("/api/v1/bots/telegram/status", {}, MagicMock())

        body = json.loads(result.body)
        assert body["webhook_token"] is None

    @pytest.mark.asyncio
    async def test_status_returns_401_when_unauthenticated(self):
        """Should return 401 when authentication fails."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No token")
            result = await handler.handle("/api/v1/bots/telegram/status", {}, MagicMock())

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_status_returns_403_when_forbidden(self):
        """Should return 403 when lacking bots.read permission."""
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Missing permission")
                result = await handler.handle("/api/v1/bots/telegram/status", {}, MagicMock())

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_get_path(self):
        """Should return None for GET requests to unknown paths."""
        handler = _make_handler()
        result = await handler.handle("/api/v1/bots/telegram/webhook", {}, MagicMock())
        assert result is None

    def test_build_status_response_with_extra(self):
        """Should merge extra_status fields into status response."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", "sec"):
                result = handler._build_status_response({"custom_field": 42})

        body = json.loads(result.body)
        assert body["custom_field"] == 42
        assert body["platform"] == "telegram"


# =============================================================================
# Test Webhook POST Handling
# =============================================================================


class TestWebhookPostRouting:
    """Tests for handle_post routing logic."""

    def test_returns_none_for_unmatched_path(self):
        """Should return None for paths not matching webhook patterns."""
        handler = _make_handler()
        result = handler.handle_post("/api/v1/bots/slack/webhook", {}, MagicMock())
        assert result is None

    def test_webhook_with_valid_token_path(self):
        """Should accept webhook when URL token matches derived token."""
        handler = _make_handler()
        token = "validtok123"
        update = {"update_id": 1, "message": {"chat": {"id": 1}, "text": "hi"}}
        mock_req = _make_mock_request(update)

        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", token):
            result = handler.handle_post(f"/api/v1/bots/telegram/webhook/{token}", {}, mock_req)

        assert result is not None
        assert result.status_code == 200

    def test_webhook_with_invalid_token_path_returns_401(self):
        """Should reject webhook when URL token does not match."""
        handler = _make_handler()
        update = {"update_id": 1}
        mock_req = _make_mock_request(update)

        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_TOKEN", "correct"):
            result = handler.handle_post("/api/v1/bots/telegram/webhook/wrong", {}, mock_req)

        assert result is not None
        assert result.status_code == 401

    def test_webhook_rejects_invalid_secret_header(self):
        """Should reject webhook when secret header does not match."""
        handler = _make_handler()
        update = {"update_id": 1}
        mock_req = _make_mock_request(update, secret_token="bad_secret")

        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", "correct"):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_req)

        assert result is not None
        assert result.status_code == 401

    def test_webhook_invalid_json_returns_400(self):
        """Should return 400 for invalid JSON body."""
        handler = _make_handler()
        mock_req = MagicMock()
        mock_req.headers = {
            "Content-Length": "10",
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_req.rfile = BytesIO(b"not json!!")

        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_req)

        assert result is not None
        assert result.status_code == 400

    def test_webhook_exception_returns_200_to_prevent_retries(self):
        """Should return 200 even on error to prevent Telegram from retrying."""
        handler = _make_handler()
        mock_req = MagicMock()
        mock_req.headers = {
            "Content-Length": "100",
            "X-Telegram-Bot-Api-Secret-Token": "",
        }
        mock_req.rfile.read.side_effect = OSError("Disk failure")

        with patch.object(telegram_module, "TELEGRAM_WEBHOOK_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, mock_req)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["ok"] is False


# =============================================================================
# Test Message Handling
# =============================================================================


class TestMessageHandling:
    """Tests for incoming Telegram message processing."""

    def test_regular_text_message(self):
        """Should handle a plain text message and return ok."""
        handler = _make_handler()
        update = {
            "update_id": 100,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "alice"},
                "text": "Hello, bot!",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["handled"] == "message"

    def test_edited_message(self):
        """Should handle edited message updates."""
        handler = _make_handler()
        update = {
            "update_id": 101,
            "edited_message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "bob"},
                "text": "Corrected text",
                "edit_date": 1700000000,
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_message_missing_from_field(self):
        """Should handle messages where 'from' user is missing."""
        handler = _make_handler()
        update = {
            "update_id": 102,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "text": "Anonymous message",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_message_empty_chat_object(self):
        """Should handle messages with empty chat dict."""
        handler = _make_handler()
        update = {
            "update_id": 103,
            "message": {
                "chat": {},
                "text": "No chat info",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_group_chat_message(self):
        """Should handle messages from group chats (negative chat id)."""
        handler = _make_handler()
        update = {
            "update_id": 104,
            "message": {
                "chat": {"id": -100123456, "type": "supergroup", "title": "Dev Team"},
                "from": {"id": 67890, "first_name": "Charlie"},
                "text": "Group discussion",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_message_without_text(self):
        """Should handle messages without text (e.g., photo, sticker)."""
        handler = _make_handler()
        update = {
            "update_id": 105,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                # No "text" field
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_unknown_update_type_acknowledged(self):
        """Should acknowledge unknown update types with ok: true."""
        handler = _make_handler()
        update = {
            "update_id": 106,
            "channel_post": {"chat": {"id": -1001}, "text": "Channel post"},
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["ok"] is True


# =============================================================================
# Test Command Processing
# =============================================================================


class TestCommandProcessing:
    """Tests for Telegram bot command handling."""

    def _make_command_update(self, command_text: str, command_length: int) -> dict[str, Any]:
        """Create an update with a bot command entity."""
        return {
            "update_id": 200,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "username": "testuser"},
                "text": command_text,
                "entities": [{"type": "bot_command", "offset": 0, "length": command_length}],
            },
        }

    def test_start_command(self):
        """Should handle /start command and return ok."""
        handler = _make_handler()
        update = self._make_command_update("/start", 6)
        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_help_command(self):
        """Should handle /help command and return ok."""
        handler = _make_handler()
        update = self._make_command_update("/help", 5)
        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_status_command(self):
        """Should handle /status command and return ok."""
        handler = _make_handler()
        update = self._make_command_update("/status", 7)
        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_debate_command_with_topic(self):
        """Should start a debate and include debate_id in response."""
        handler = _make_handler()
        update = self._make_command_update("/debate Should we use Rust?", 7)
        result = _dispatch_update(handler, update, mock_debate=True)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("debate_started") is True
        assert "debate_id" in body

    def test_debate_command_empty_topic_prompts_user(self):
        """Should prompt for topic when /debate has no arguments."""
        handler = _make_handler()
        update = self._make_command_update("/debate", 7)
        result = _dispatch_update(handler, update, mock_debate=True)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should not start debate
        assert body.get("debate_started") is not True

    def test_ask_command_is_alias_for_debate(self):
        """Should treat /ask as an alias for /debate."""
        handler = _make_handler()
        update = self._make_command_update("/ask Is TypeScript worth it?", 4)
        result = _dispatch_update(handler, update, mock_debate=True)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("debate_started") is True

    def test_aragora_command_is_alias_for_debate(self):
        """Should treat /aragora as an alias for /debate."""
        handler = _make_handler()
        update = self._make_command_update("/aragora What is best practice?", 8)
        result = _dispatch_update(handler, update, mock_debate=True)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("debate_started") is True

    def test_unknown_command(self):
        """Should acknowledge unknown commands gracefully."""
        handler = _make_handler()
        update = self._make_command_update("/foobar", 7)
        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_command_is_case_insensitive(self):
        """Commands should be lowercased before routing."""
        handler = _make_handler()
        update = self._make_command_update("/HELP", 5)
        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_non_leading_entity_is_not_command(self):
        """Entity with offset > 0 should not be treated as a command."""
        handler = _make_handler()
        update = {
            "update_id": 210,
            "message": {
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890},
                "text": "Please run /help now",
                "entities": [{"type": "bot_command", "offset": 11, "length": 5}],
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should be treated as regular message, not a command
        assert body.get("handled") == "message"


# =============================================================================
# Test Callback Query Handling
# =============================================================================


class TestCallbackQueryHandling:
    """Tests for inline button callback queries."""

    def test_vote_callback(self):
        """Should handle vote callback data format vote:debate_id:option."""
        handler = _make_handler()
        update = {
            "update_id": 300,
            "callback_query": {
                "id": "cb_123",
                "from": {"id": 67890, "username": "voter"},
                "data": "vote:debate-xyz:approve",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200

    def test_non_vote_callback_acknowledged(self):
        """Should acknowledge non-vote callback queries."""
        handler = _make_handler()
        update = {
            "update_id": 301,
            "callback_query": {
                "id": "cb_456",
                "from": {"id": 67890},
                "data": "other:action:data",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("callback_handled") is True

    def test_callback_with_empty_data(self):
        """Should handle callback with empty data string."""
        handler = _make_handler()
        update = {
            "update_id": 302,
            "callback_query": {
                "id": "cb_789",
                "from": {"id": 67890},
                "data": "",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Inline Query
# =============================================================================


class TestInlineQueryHandling:
    """Tests for @bot inline query processing."""

    def test_inline_query_returns_ok(self):
        """Should acknowledge inline queries."""
        handler = _make_handler()
        update = {
            "update_id": 400,
            "inline_query": {
                "id": "iq_001",
                "from": {"id": 67890},
                "query": "search term",
            },
        }

        result = _dispatch_update(handler, update)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["ok"] is True


# =============================================================================
# Test Vote Recording
# =============================================================================


class TestVoteRecording:
    """Tests for the _handle_vote method and ConsensusStore integration."""

    def test_vote_recorded_successfully(self):
        """Should record vote via ConsensusStore and return vote_recorded: true."""
        handler = _make_handler()

        mock_store = MagicMock()
        mock_store_cls = MagicMock(return_value=mock_store)

        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch("httpx.Client"):
                with patch.dict(
                    "sys.modules",
                    {"aragora.memory.consensus": MagicMock(ConsensusStore=mock_store_cls)},
                ):
                    result = handler._handle_vote("cb1", 67890, "debate-1", "agree")

        mock_store.record_vote.assert_called_once_with(
            debate_id="debate-1",
            user_id="telegram:67890",
            vote="agree",
            source="telegram",
        )
        body = json.loads(result.body)
        assert body["vote_recorded"] is True

    def test_vote_fallback_when_consensus_store_unavailable(self):
        """Should return vote_recorded: false when ConsensusStore cannot be imported."""
        handler = _make_handler()

        with patch.dict("sys.modules", {"aragora.memory.consensus": None}):
            with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
                with patch("httpx.Client"):
                    result = handler._handle_vote("cb2", 67890, "debate-2", "disagree")

        body = json.loads(result.body)
        assert body["vote_recorded"] is False

    def test_vote_handles_runtime_error(self):
        """Should return error on runtime failure during vote recording."""
        handler = _make_handler()

        mock_store = MagicMock()
        mock_store.record_vote.side_effect = RuntimeError("DB down")
        mock_store_cls = MagicMock(return_value=mock_store)

        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch("httpx.Client"):
                with patch.dict(
                    "sys.modules",
                    {"aragora.memory.consensus": MagicMock(ConsensusStore=mock_store_cls)},
                ):
                    result = handler._handle_vote("cb3", 67890, "debate-3", "neutral")

        body = json.loads(result.body)
        assert body["ok"] is False
        assert "error" in body


# =============================================================================
# Test Message Sending
# =============================================================================


class TestMessageSending:
    """Tests for _send_message and _answer_callback_query methods."""

    def test_send_message_does_nothing_without_token(self):
        """Should skip sending when TELEGRAM_BOT_TOKEN is not set."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", None):
            # Should not raise
            handler._send_message(12345, "Test")

    def test_send_message_calls_telegram_api(self):
        """Should POST to sendMessage endpoint when token is set."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "test_tok"):
            with patch("httpx.Client") as mock_client:
                mock_resp = MagicMock(is_success=True)
                mock_client.return_value.__enter__.return_value.post.return_value = mock_resp

                handler._send_message(12345, "Hello")

                call_args = mock_client.return_value.__enter__.return_value.post.call_args
                assert "sendMessage" in call_args[0][0]
                assert call_args[1]["json"]["chat_id"] == 12345
                assert call_args[1]["json"]["text"] == "Hello"

    def test_send_message_handles_http_failure(self):
        """Should not raise on HTTP failure."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch("httpx.Client") as mock_client:
                mock_resp = MagicMock(is_success=False, status_code=500)
                mock_client.return_value.__enter__.return_value.post.return_value = mock_resp

                # Should not raise
                handler._send_message(12345, "Fail gracefully")

    def test_send_message_handles_connection_error(self):
        """Should not raise on network error."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.side_effect = ConnectionError(
                    "Network down"
                )

                # Should not raise
                handler._send_message(12345, "Network issue")

    def test_answer_callback_query_does_nothing_without_token(self):
        """Should skip answering callback when TELEGRAM_BOT_TOKEN is not set."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", None):
            # Should not raise
            handler._answer_callback_query("cb1", "Acknowledged")

    def test_answer_callback_query_calls_api(self):
        """Should POST to answerCallbackQuery when token is set."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = MagicMock()

                handler._answer_callback_query("cb_id", "Vote received")

                call_args = mock_client.return_value.__enter__.return_value.post.call_args
                assert "answerCallbackQuery" in call_args[0][0]

    def test_answer_callback_query_handles_exception(self):
        """Should not raise when answerCallbackQuery fails."""
        handler = _make_handler()
        with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", "tok"):
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.side_effect = OSError(
                    "Timeout"
                )

                # Should not raise
                handler._answer_callback_query("cb_id", "Error")


# =============================================================================
# Test Debate Async Start
# =============================================================================


class TestDebateAsyncStart:
    """Tests for _start_debate_async and fallback paths."""

    def test_start_debate_returns_uuid(self):
        """Should return a debate ID string."""
        handler = _make_handler()

        # Mock the internal async routing to avoid actual asyncio execution
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = "mocked-debate-id-1234"
            with patch.object(telegram_module, "TELEGRAM_BOT_TOKEN", ""):
                debate_id = handler._start_debate_async(12345, 67890, "Test topic")

        assert isinstance(debate_id, str)
        assert len(debate_id) > 0

    def test_cmd_debate_sends_confirmation_message(self):
        """Should send a confirmation message with debate ID."""
        handler = _make_handler()

        with patch.object(handler, "_send_message") as mock_send:
            with patch.object(handler, "_start_debate_async", return_value="abc-123"):
                result = handler._cmd_debate(12345, 67890, "Test topic")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_started"] is True
        assert body["debate_id"] == "abc-123"
        mock_send.assert_called_once()
        # Verify the sent text mentions the topic
        sent_text = mock_send.call_args[0][1]
        assert "Test topic" in sent_text

    def test_cmd_debate_empty_topic_does_not_start_debate(self):
        """Should not start debate when topic is whitespace-only."""
        handler = _make_handler()

        with patch.object(handler, "_send_message") as mock_send:
            result = handler._cmd_debate(12345, 67890, "   ")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("debate_started") is not True
        # Should send a prompt message
        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Please provide a topic" in sent_text


class TestAttachmentExtraction:
    """Tests for Telegram attachment extraction."""

    def test_extracts_document_attachment(self):
        handler = _make_handler()
        message = {
            "document": {
                "file_id": "file123",
                "file_name": "spec.txt",
                "mime_type": "text/plain",
                "file_size": 123,
            },
            "caption": "Spec preview",
        }

        attachments = handler._extract_attachments(message)

        assert len(attachments) == 1
        assert attachments[0]["type"] == "document"
        assert attachments[0]["file_id"] == "file123"
        assert attachments[0]["filename"] == "spec.txt"
        assert attachments[0]["content_type"] == "text/plain"
        assert attachments[0]["text"] == "Spec preview"

    def test_extracts_largest_photo(self):
        handler = _make_handler()
        message = {
            "photo": [
                {"file_id": "small", "file_size": 10},
                {"file_id": "large", "file_size": 20},
            ]
        }

        attachments = handler._extract_attachments(message)

        assert len(attachments) == 1
        assert attachments[0]["type"] == "photo"
        assert attachments[0]["file_id"] == "large"

    def test_cmd_debate_passes_attachments(self):
        handler = _make_handler()
        message = {
            "document": {"file_id": "file123", "file_name": "spec.txt"},
            "caption": "Spec preview",
        }

        with patch.object(handler, "_send_message"):
            with patch.object(handler, "_start_debate_async") as mock_start:
                handler._cmd_debate(
                    12345, 67890, "Test topic", handler._extract_attachments(message)
                )

        assert mock_start.called is True
        _, _, _, attachments = mock_start.call_args[0]
        assert isinstance(attachments, list)
        assert attachments[0]["file_id"] == "file123"
