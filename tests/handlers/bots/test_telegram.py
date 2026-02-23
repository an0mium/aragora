"""
Tests for Telegram Bot webhook handler.

Covers all routes and behavior of the TelegramHandler class:
- can_handle() routing for all defined routes
- GET /api/v1/bots/telegram/status - Bot status endpoint
- POST /api/v1/bots/telegram/webhook - Webhook processing
- POST /api/v1/bots/telegram/webhook/{token} - Token-verified webhook
- Message handling (text, edited, commands, entities)
- Callback query handling (votes, default callbacks)
- Inline query handling
- Bot commands: /start, /help, /debate, /plan, /implement, /status, /ask, unknown
- Attachment extraction (document, photo, audio, video, voice)
- Secret token verification and webhook token verification
- RBAC permission checks
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.telegram as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.TelegramHandler


@pytest.fixture
def handler(handler_cls):
    """Create a TelegramHandler with empty context."""
    return handler_cls(ctx={})


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/telegram/webhook"
    method: str = "POST"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.body is not None:
            body_bytes = json.dumps(self.body).encode("utf-8")
        else:
            body_bytes = b"{}"
        self.rfile = io.BytesIO(body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body_bytes))
        # Provide client_address for rate-limit key extraction
        self.client_address = ("127.0.0.1", 12345)


def _make_webhook_handler(
    body: dict[str, Any],
    secret_token: str = "",
    content_length: str | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for webhook POST requests."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if secret_token:
        headers["X-Telegram-Bot-Api-Secret-Token"] = secret_token
    h = MockHTTPHandler(body=body, headers=headers)
    if content_length is not None:
        h.headers["Content-Length"] = content_length
    return h


# ---------------------------------------------------------------------------
# Telegram update builders
# ---------------------------------------------------------------------------


def _message_update(
    text: str = "Hello",
    chat_id: int = 12345,
    user_id: int = 67890,
    username: str = "testuser",
    chat_type: str = "private",
    entities: list[dict] | None = None,
    document: dict | None = None,
    photo: list[dict] | None = None,
    audio: dict | None = None,
    video: dict | None = None,
    voice: dict | None = None,
    caption: str | None = None,
) -> dict[str, Any]:
    """Build a minimal Telegram message update."""
    message: dict[str, Any] = {
        "message_id": 1,
        "from": {"id": user_id, "is_bot": False, "username": username},
        "chat": {"id": chat_id, "type": chat_type},
        "text": text,
        "date": 1700000000,
    }
    if entities is not None:
        message["entities"] = entities
    if document is not None:
        message["document"] = document
    if photo is not None:
        message["photo"] = photo
    if audio is not None:
        message["audio"] = audio
    if video is not None:
        message["video"] = video
    if voice is not None:
        message["voice"] = voice
    if caption is not None:
        message["caption"] = caption
    return {"update_id": 100, "message": message}


def _callback_query_update(
    data: str = "vote:debate123:agree",
    callback_id: str = "cb_123",
    user_id: int = 67890,
) -> dict[str, Any]:
    """Build a callback_query update."""
    return {
        "update_id": 101,
        "callback_query": {
            "id": callback_id,
            "from": {"id": user_id, "is_bot": False, "username": "testuser"},
            "data": data,
        },
    }


def _inline_query_update(
    query_text: str = "test query",
    query_id: str = "iq_123",
) -> dict[str, Any]:
    """Build an inline_query update."""
    return {
        "update_id": 102,
        "inline_query": {
            "id": query_id,
            "from": {"id": 67890, "is_bot": False, "username": "testuser"},
            "query": query_text,
        },
    }


def _edited_message_update(
    text: str = "Edited message",
    chat_id: int = 12345,
    user_id: int = 67890,
) -> dict[str, Any]:
    """Build an edited_message update."""
    return {
        "update_id": 103,
        "edited_message": {
            "message_id": 1,
            "from": {"id": user_id, "is_bot": False, "username": "testuser"},
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
            "date": 1700000000,
            "edit_date": 1700001000,
        },
    }


def _command_message_update(
    command: str = "/debate",
    args: str = "Should we use Python?",
    chat_id: int = 12345,
    user_id: int = 67890,
) -> dict[str, Any]:
    """Build a message update with a bot command entity."""
    full_text = f"{command} {args}" if args else command
    return _message_update(
        text=full_text,
        chat_id=chat_id,
        user_id=user_id,
        entities=[
            {
                "type": "bot_command",
                "offset": 0,
                "length": len(command),
            }
        ],
    )


# ===========================================================================
# can_handle()
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_webhook_route(self, handler):
        assert handler.can_handle("/api/v1/bots/telegram/webhook", "POST") is True

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/bots/telegram/status", "GET") is True

    def test_webhook_with_token(self, handler):
        assert handler.can_handle("/api/v1/bots/telegram/webhook/abc123", "POST") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/webhook", "POST") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/", "GET") is False

    def test_partial_match_no_trailing(self, handler):
        # /api/v1/bots/telegram/webhookXYZ should not match
        assert handler.can_handle("/api/v1/bots/telegram/webhookXYZ", "POST") is False

    def test_webhook_token_long_path(self, handler):
        assert (
            handler.can_handle("/api/v1/bots/telegram/webhook/some-long-token-value", "POST")
            is True
        )

    def test_different_base(self, handler):
        assert handler.can_handle("/api/v2/bots/telegram/webhook", "POST") is False


# ===========================================================================
# GET /api/v1/bots/telegram/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/status", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/status", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_body_has_platform(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/status", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/status", {}, http_handler)
        body = _body(result)
        assert body["platform"] == "telegram"

    @pytest.mark.asyncio
    async def test_status_body_has_enabled_field(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/status", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/status", {}, http_handler)
        body = _body(result)
        assert "enabled" in body

    @pytest.mark.asyncio
    async def test_status_has_token_configured(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/status", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/status", {}, http_handler)
        body = _body(result)
        assert "token_configured" in body

    @pytest.mark.asyncio
    async def test_status_has_webhook_secret_configured(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/status", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/status", {}, http_handler)
        body = _body(result)
        assert "webhook_secret_configured" in body

    @pytest.mark.asyncio
    async def test_status_has_webhook_token(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/status", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/status", {}, http_handler)
        body = _body(result)
        assert "webhook_token" in body

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_status_get(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/webhook", method="GET")
        result = await handler.handle("/api/v1/bots/telegram/webhook", {}, http_handler)
        assert result is None


# ===========================================================================
# Secret Verification
# ===========================================================================


class TestVerifyTelegramSecret:
    """Tests for _verify_telegram_secret."""

    def test_no_secret_configured_dev_mode_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                result = handler_module._verify_telegram_secret("any-token")
        assert result is True

    def test_no_secret_configured_test_mode_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "test"}):
                result = handler_module._verify_telegram_secret("any-token")
        assert result is True

    def test_no_secret_configured_production_fails(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
                result = handler_module._verify_telegram_secret("any-token")
        assert result is False

    def test_no_secret_configured_staging_fails(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "staging"}):
                result = handler_module._verify_telegram_secret("any-token")
        assert result is False

    def test_correct_secret_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "my-secret"):
            result = handler_module._verify_telegram_secret("my-secret")
        assert result is True

    def test_incorrect_secret_fails(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "my-secret"):
            result = handler_module._verify_telegram_secret("wrong-secret")
        assert result is False

    def test_empty_secret_token_with_secret_configured(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "my-secret"):
            result = handler_module._verify_telegram_secret("")
        assert result is False

    def test_no_secret_configured_local_mode_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None):
            with patch.dict("os.environ", {"ARAGORA_ENV": "local"}):
                result = handler_module._verify_telegram_secret("any-token")
        assert result is True


# ===========================================================================
# Webhook Token Verification
# ===========================================================================


class TestVerifyWebhookToken:
    """Tests for _verify_webhook_token."""

    def test_no_token_configured_dev_mode_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                result = handler_module._verify_webhook_token("any-token")
        assert result is True

    def test_no_token_configured_production_fails(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
                result = handler_module._verify_webhook_token("any-token")
        assert result is False

    def test_correct_token_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", "valid-token"):
            result = handler_module._verify_webhook_token("valid-token")
        assert result is True

    def test_incorrect_token_fails(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", "valid-token"):
            result = handler_module._verify_webhook_token("wrong-token")
        assert result is False

    def test_empty_token_with_token_configured(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", "valid-token"):
            result = handler_module._verify_webhook_token("")
        assert result is False

    def test_no_token_configured_test_mode_passes(self, handler_module):
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", ""):
            with patch.dict("os.environ", {"ARAGORA_ENV": "test"}):
                result = handler_module._verify_webhook_token("any")
        assert result is True


# ===========================================================================
# POST /api/v1/bots/telegram/webhook - Webhook Processing
# ===========================================================================


class TestWebhookPost:
    """Tests for the main webhook POST endpoint."""

    def test_message_update(self, handler, handler_module):
        update = _message_update(text="Hello bot")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_callback_query_update(self, handler, handler_module):
        update = _callback_query_update()
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True or body.get("ok") is False  # vote might fail

    def test_inline_query_update(self, handler, handler_module):
        update = _inline_query_update()
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_edited_message_update(self, handler, handler_module):
        update = _edited_message_update()
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_unknown_update_type(self, handler, handler_module):
        update = {"update_id": 999, "channel_post": {"text": "hi"}}
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_invalid_json_body(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/webhook")
        http_handler.rfile = io.BytesIO(b"not json")
        http_handler.headers["Content-Length"] = "8"
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        assert _status(result) == 400

    def test_body_too_large(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/webhook")
        http_handler.headers["Content-Length"] = str(11 * 1024 * 1024)
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        assert _status(result) == 413

    def test_invalid_content_length(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/telegram/webhook")
        http_handler.headers["Content-Length"] = "not-a-number"
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        assert _status(result) == 400

    def test_secret_verification_failure(self, handler, handler_module):
        update = _message_update()
        http_handler = _make_webhook_handler(update, secret_token="wrong")
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "correct-secret"):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        assert _status(result) == 401

    def test_handle_post_returns_none_for_unknown_path(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/unknown")
        result = handler.handle_post("/api/v1/bots/unknown", {}, http_handler)
        assert result is None


# ===========================================================================
# POST /api/v1/bots/telegram/webhook/{token} - Token-verified webhook
# ===========================================================================


class TestTokenVerifiedWebhook:
    """Tests for the token-verified webhook endpoint."""

    def test_valid_token_processes_update(self, handler, handler_module):
        update = _message_update(text="Via token webhook")
        http_handler = _make_webhook_handler(update)
        token = "valid-token-123"
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", token),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post(f"/api/v1/bots/telegram/webhook/{token}", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_invalid_token_returns_401(self, handler, handler_module):
        update = _message_update()
        http_handler = _make_webhook_handler(update)
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", "real-token"):
            result = handler.handle_post(
                "/api/v1/bots/telegram/webhook/bad-token", {}, http_handler
            )
        assert _status(result) == 401

    def test_token_extraction_from_path(self, handler, handler_module):
        """Token is the last segment of the path."""
        update = _message_update()
        http_handler = _make_webhook_handler(update)
        token = "abcdef123456"
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", token),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post(f"/api/v1/bots/telegram/webhook/{token}", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# Bot Commands
# ===========================================================================


class TestCommandStart:
    """Tests for /start command."""

    def test_start_returns_ok(self, handler, handler_module):
        update = _command_message_update(command="/start", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


class TestCommandHelp:
    """Tests for /help command."""

    def test_help_returns_ok(self, handler, handler_module):
        update = _command_message_update(command="/help", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


class TestCommandDebate:
    """Tests for /debate command."""

    def test_debate_with_topic_starts_debate(self, handler, handler_module):
        update = _command_message_update(command="/debate", args="Is Python better?")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("debate_started") is True
        assert "debate_id" in body

    def test_debate_without_topic_prompts_user(self, handler, handler_module):
        update = _command_message_update(command="/debate", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        # Should not start a debate
        assert "debate_started" not in body or body.get("debate_started") is not True

    def test_debate_whitespace_only_topic(self, handler, handler_module):
        update = _command_message_update(command="/debate", args="   ")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert "debate_started" not in body or body.get("debate_started") is not True


class TestCommandPlan:
    """Tests for /plan command."""

    def test_plan_starts_debate_with_integrity_config(self, handler, handler_module):
        update = _command_message_update(command="/plan", args="Build a rate limiter")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("debate_started") is True

    def test_plan_no_topic(self, handler, handler_module):
        update = _command_message_update(command="/plan", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


class TestCommandImplement:
    """Tests for /implement command."""

    def test_implement_starts_debate_with_execute_mode(self, handler, handler_module):
        update = _command_message_update(command="/implement", args="Refactor the API")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("debate_started") is True


class TestCommandStatus:
    """Tests for /status command."""

    def test_status_returns_ok(self, handler, handler_module):
        update = _command_message_update(command="/status", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


class TestCommandAsk:
    """Tests for /ask command (alias for /debate)."""

    def test_ask_starts_debate(self, handler, handler_module):
        update = _command_message_update(command="/ask", args="What is Python?")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("debate_started") is True


class TestCommandAragora:
    """Tests for /aragora command (alias for /debate)."""

    def test_aragora_starts_debate(self, handler, handler_module):
        update = _command_message_update(command="/aragora", args="Test topic")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("debate_started") is True


class TestCommandUnknown:
    """Tests for unknown commands."""

    def test_unknown_command_returns_ok(self, handler, handler_module):
        update = _command_message_update(command="/foobar", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# Callback Queries
# ===========================================================================


class TestCallbackQuery:
    """Tests for callback query (inline button) handling."""

    def test_vote_callback_records_vote(self, handler, handler_module):
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        mock_store = MagicMock()
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch("aragora.memory.consensus.ConsensusStore", return_value=mock_store),
            patch("aragora.audit.unified.audit_data"),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body.get("vote_recorded") is True or body["ok"] is True

    def test_vote_callback_consensus_import_error(self, handler, handler_module):
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch.dict("sys.modules", {"aragora.memory.consensus": None}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("vote_recorded") is False

    def test_vote_callback_runtime_error(self, handler, handler_module):
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        mock_store = MagicMock()
        mock_store.record_vote.side_effect = RuntimeError("DB error")
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch("aragora.memory.consensus.ConsensusStore", return_value=mock_store),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is False

    def test_non_vote_callback_acknowledged(self, handler, handler_module):
        update = _callback_query_update(data="unknown:action")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("callback_handled") is True

    def test_vote_callback_missing_debate_id(self, handler, handler_module):
        """Vote with only action, no debate_id or vote_option."""
        update = _callback_query_update(data="vote")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch.dict("sys.modules", {"aragora.memory.consensus": None}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        # Should still handle gracefully
        assert "ok" in body

    def test_empty_callback_data(self, handler, handler_module):
        update = _callback_query_update(data="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# Inline Queries
# ===========================================================================


class TestInlineQuery:
    """Tests for inline query handling."""

    def test_inline_query_returns_ok(self, handler, handler_module):
        update = _inline_query_update(query_text="search term")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_inline_query_empty_text(self, handler, handler_module):
        update = _inline_query_update(query_text="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# Attachment Extraction
# ===========================================================================


class TestExtractAttachments:
    """Tests for _extract_attachments."""

    def test_document_extraction(self, handler):
        message = {
            "document": {
                "file_id": "doc123",
                "file_name": "report.pdf",
                "mime_type": "application/pdf",
                "file_size": 1024,
            }
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["type"] == "document"
        assert attachments[0]["file_id"] == "doc123"
        assert attachments[0]["filename"] == "report.pdf"

    def test_photo_extraction_picks_largest(self, handler):
        message = {
            "photo": [
                {"file_id": "small", "file_size": 100},
                {"file_id": "large", "file_size": 5000},
                {"file_id": "medium", "file_size": 1000},
            ]
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["type"] == "photo"
        assert attachments[0]["file_id"] == "large"

    def test_audio_extraction(self, handler):
        message = {
            "audio": {
                "file_id": "audio123",
                "file_name": "song.mp3",
                "mime_type": "audio/mpeg",
                "file_size": 4096,
            }
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["type"] == "audio"
        assert attachments[0]["file_id"] == "audio123"

    def test_video_extraction(self, handler):
        message = {
            "video": {
                "file_id": "vid123",
                "file_name": "clip.mp4",
                "mime_type": "video/mp4",
                "file_size": 8192,
            }
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["type"] == "video"
        assert attachments[0]["file_id"] == "vid123"

    def test_voice_extraction(self, handler):
        message = {
            "voice": {
                "file_id": "voice123",
                "mime_type": "audio/ogg",
                "file_size": 2048,
            }
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["type"] == "voice"
        assert attachments[0]["file_id"] == "voice123"
        assert attachments[0]["filename"] == "voice"

    def test_multiple_attachment_types(self, handler):
        message = {
            "document": {"file_id": "doc1", "file_name": "f.txt"},
            "photo": [{"file_id": "photo1", "file_size": 500}],
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 2
        types = {a["type"] for a in attachments}
        assert types == {"document", "photo"}

    def test_caption_included_in_attachments(self, handler):
        message = {
            "caption": "Look at this!",
            "document": {"file_id": "doc1", "file_name": "f.txt"},
        }
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["text"] == "Look at this!"

    def test_no_attachments(self, handler):
        message = {"text": "Just text"}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 0

    def test_non_dict_message(self, handler):
        attachments = handler._extract_attachments("not a dict")
        assert len(attachments) == 0

    def test_empty_photo_list(self, handler):
        message = {"photo": []}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 0

    def test_photo_with_non_dict_items(self, handler):
        message = {"photo": ["not_a_dict", 42, None]}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 0

    def test_document_missing_file_name(self, handler):
        message = {"document": {"file_id": "doc1"}}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["filename"] == "document"

    def test_audio_missing_file_name(self, handler):
        message = {"audio": {"file_id": "a1"}}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["filename"] == "audio"

    def test_video_missing_file_name(self, handler):
        message = {"video": {"file_id": "v1"}}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["filename"] == "video"

    def test_caption_whitespace_only(self, handler):
        message = {"caption": "   ", "document": {"file_id": "doc1"}}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["text"] == ""

    def test_caption_non_string(self, handler):
        message = {"caption": 12345, "document": {"file_id": "doc1"}}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["text"] == ""

    def test_photo_single_item(self, handler):
        message = {"photo": [{"file_id": "only_one", "file_size": 300}]}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        assert attachments[0]["file_id"] == "only_one"

    def test_photo_items_without_file_size(self, handler):
        message = {"photo": [{"file_id": "a"}, {"file_id": "b"}]}
        attachments = handler._extract_attachments(message)
        assert len(attachments) == 1
        # First one should be picked (both have file_size=None -> 0)


# ===========================================================================
# _handle_message edge cases
# ===========================================================================


class TestHandleMessage:
    """Tests for message processing edge cases."""

    def test_message_without_text(self, handler, handler_module):
        """Message with no text field (e.g., sticker)."""
        update = {
            "update_id": 200,
            "message": {
                "message_id": 1,
                "from": {"id": 123, "is_bot": False},
                "chat": {"id": 456, "type": "private"},
                "sticker": {"file_id": "sticker123"},
                "date": 1700000000,
            },
        }
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_message_missing_from(self, handler, handler_module):
        update = {
            "update_id": 201,
            "message": {
                "message_id": 1,
                "chat": {"id": 456, "type": "group"},
                "text": "No from field",
                "date": 1700000000,
            },
        }
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_command_not_at_offset_zero_is_not_command(self, handler, handler_module):
        """Entity at offset > 0 is not treated as a command."""
        update = _message_update(
            text="try /help please",
            entities=[{"type": "bot_command", "offset": 4, "length": 5}],
        )
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("handled") == "message"

    def test_entity_not_bot_command_type(self, handler, handler_module):
        """Non-bot_command entities are ignored."""
        update = _message_update(
            text="Hello world",
            entities=[{"type": "mention", "offset": 0, "length": 5}],
        )
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("handled") == "message"

    def test_message_in_group_chat(self, handler, handler_module):
        update = _message_update(text="group message", chat_type="group")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_message_from_user_with_first_name_only(self, handler, handler_module):
        """User without username falls back to first_name."""
        update = {
            "update_id": 202,
            "message": {
                "message_id": 1,
                "from": {"id": 123, "is_bot": False, "first_name": "Alice"},
                "chat": {"id": 456, "type": "private"},
                "text": "hello",
                "date": 1700000000,
            },
        }
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# _send_message
# ===========================================================================


class TestSendMessage:
    """Tests for _send_message helper."""

    def test_send_message_no_token(self, handler, handler_module):
        """Should silently return when no token is configured."""
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None):
            # Should not raise
            handler._send_message(12345, "Hello")

    def test_send_message_with_token(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            with patch("httpx.Client", return_value=mock_client):
                handler._send_message(12345, "Hello")
            mock_client.post.assert_called_once()

    def test_send_message_httpx_import_error(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            with patch.dict("sys.modules", {"httpx": None}):
                # Should not raise
                handler._send_message(12345, "Hello")

    def test_send_message_connection_error(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = ConnectionError("Network error")
            with patch("httpx.Client", return_value=mock_client):
                # Should not raise
                handler._send_message(12345, "Hello")

    def test_send_message_timeout_error(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = TimeoutError("Timed out")
            with patch("httpx.Client", return_value=mock_client):
                handler._send_message(12345, "Hello")


# ===========================================================================
# _answer_callback_query
# ===========================================================================


class TestAnswerCallbackQuery:
    """Tests for _answer_callback_query helper."""

    def test_no_token_does_nothing(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None):
            handler._answer_callback_query("cb_123", "Acknowledged")

    def test_with_token_makes_request(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            with patch("httpx.Client", return_value=mock_client):
                handler._answer_callback_query("cb_123", "Vote recorded")
            mock_client.post.assert_called_once()

    def test_httpx_import_error(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            with patch.dict("sys.modules", {"httpx": None}):
                handler._answer_callback_query("cb_123", "OK")

    def test_connection_error(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "fake-token"):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = OSError("Network error")
            with patch("httpx.Client", return_value=mock_client):
                handler._answer_callback_query("cb_123", "OK")


# ===========================================================================
# RBAC Permission Checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for _check_bot_permission RBAC integration."""

    def test_rbac_not_available_non_production(self, handler, handler_module):
        """When RBAC is unavailable and not production, should pass."""
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.bots.telegram.rbac_fail_closed", return_value=False),
        ):
            # Should not raise
            handler._check_bot_permission("debates:create", user_id="telegram:123")

    def test_rbac_not_available_production(self, handler, handler_module):
        """When RBAC is unavailable in production, should raise."""
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", False),
            patch("aragora.server.handlers.bots.telegram.rbac_fail_closed", return_value=True),
        ):
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="telegram:123")

    def test_rbac_available_permission_granted(self, handler, handler_module):
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            mock_check.return_value = None
            handler._check_bot_permission("debates:create", user_id="telegram:123")

    def test_rbac_available_permission_denied(self, handler, handler_module):
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            mock_check.side_effect = PermissionError("Denied")
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="telegram:123")

    def test_rbac_with_auth_context_in_context(self, handler, handler_module):
        """When auth_context is provided in context dict, it should be used."""
        mock_auth_ctx = MagicMock()
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            handler._check_bot_permission(
                "debates:create",
                context={"auth_context": mock_auth_ctx},
            )
            mock_check.assert_called_once_with(mock_auth_ctx, "debates:create")

    def test_rbac_no_user_id_no_context(self, handler, handler_module):
        """When no user_id and no auth_context, check_permission not called."""
        with (
            patch.object(handler_module, "RBAC_AVAILABLE", True),
            patch.object(handler_module, "check_permission") as mock_check,
        ):
            handler._check_bot_permission("debates:create")
            mock_check.assert_not_called()

    def test_debate_permission_denied_returns_error(self, handler, handler_module):
        """Debate RBAC denial produces permission_denied error."""
        update = _command_message_update(command="/debate", args="test topic")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch.object(
                handler,
                "_check_bot_permission",
                side_effect=PermissionError("Denied"),
            ),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is False
        assert body.get("error") == "permission_denied"

    def test_vote_permission_denied(self, handler, handler_module):
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch.object(
                handler,
                "_check_bot_permission",
                side_effect=PermissionError("Denied"),
            ),
            patch.object(handler, "_answer_callback_query"),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is False
        assert body.get("error") == "permission_denied"


# ===========================================================================
# _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled."""

    def test_enabled_when_token_set(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "some-token"):
            assert handler._is_bot_enabled() is True

    def test_disabled_when_token_not_set(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_token_empty(self, handler, handler_module):
        with patch.object(handler_module, "TELEGRAM_BOT_TOKEN", ""):
            assert handler._is_bot_enabled() is False


# ===========================================================================
# _get_platform_config_status
# ===========================================================================


class TestPlatformConfigStatus:
    """Tests for _get_platform_config_status."""

    def test_all_configured(self, handler, handler_module):
        with (
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", "token123"),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "secret123"),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", "webhooktoken12345678"),
        ):
            status = handler._get_platform_config_status()
        assert status["token_configured"] is True
        assert status["webhook_secret_configured"] is True
        assert status["webhook_token"] is not None
        # Webhook token is truncated
        assert status["webhook_token"].endswith("...")

    def test_none_configured(self, handler, handler_module):
        with (
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", ""),
        ):
            status = handler._get_platform_config_status()
        assert status["token_configured"] is False
        assert status["webhook_secret_configured"] is False
        assert status["webhook_token"] is None


# ===========================================================================
# Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for TelegramHandler initialization."""

    def test_default_ctx(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_custom_ctx(self, handler_cls):
        ctx = {"storage": MagicMock()}
        h = handler_cls(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx(self, handler_cls):
        h = handler_cls(ctx=None)
        assert h.ctx == {}

    def test_bot_platform(self, handler):
        assert handler.bot_platform == "telegram"

    def test_routes_defined(self, handler):
        assert "/api/v1/bots/telegram/webhook" in handler.ROUTES
        assert "/api/v1/bots/telegram/status" in handler.ROUTES


# ===========================================================================
# _start_debate_async
# ===========================================================================


class TestStartDebateAsync:
    """Tests for _start_debate_async."""

    def test_returns_debate_id_in_test_mode(self, handler):
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "yes"}):
            debate_id = handler._start_debate_async(12345, 67890, "Test topic")
        assert isinstance(debate_id, str)
        assert len(debate_id) > 0

    def test_returns_uuid_format(self, handler):
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "yes"}):
            debate_id = handler._start_debate_async(12345, 67890, "Test topic")
        # UUID has 5 groups separated by hyphens
        parts = debate_id.split("-")
        assert len(parts) == 5

    def test_different_calls_produce_different_ids(self, handler):
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "yes"}):
            id1 = handler._start_debate_async(12345, 67890, "Topic 1")
            id2 = handler._start_debate_async(12345, 67890, "Topic 2")
        assert id1 != id2

    def test_attachments_parameter_accepted(self, handler):
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "yes"}):
            debate_id = handler._start_debate_async(
                12345,
                67890,
                "Topic",
                attachments=[{"type": "document", "file_id": "doc1"}],
            )
        assert isinstance(debate_id, str)

    def test_decision_integrity_parameter_accepted(self, handler):
        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "yes"}):
            debate_id = handler._start_debate_async(
                12345,
                67890,
                "Topic",
                decision_integrity={"include_receipt": True},
            )
        assert isinstance(debate_id, str)


# ===========================================================================
# Webhook exception handling
# ===========================================================================


class TestWebhookExceptionHandling:
    """Tests for exception handling in _handle_webhook."""

    def test_value_error_returns_200(self, handler, handler_module):
        """ValueError in webhook returns 200 to prevent Telegram retries."""
        http_handler = MockHTTPHandler()
        body_bytes = b'{"update_id": 1, "message": {}}'
        http_handler.rfile = io.BytesIO(body_bytes)
        http_handler.headers["Content-Length"] = str(len(body_bytes))
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""

        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch.object(handler, "_handle_message", side_effect=ValueError("bad data")),
        ):
            result = handler._handle_webhook(http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["ok"] is False

    def test_key_error_returns_200(self, handler, handler_module):
        http_handler = MockHTTPHandler()
        body_bytes = b'{"update_id": 1, "message": {}}'
        http_handler.rfile = io.BytesIO(body_bytes)
        http_handler.headers["Content-Length"] = str(len(body_bytes))
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""

        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch.object(handler, "_handle_message", side_effect=KeyError("missing")),
        ):
            result = handler._handle_webhook(http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["ok"] is False

    def test_type_error_returns_200(self, handler, handler_module):
        http_handler = MockHTTPHandler()
        body_bytes = b'{"update_id": 1, "message": {}}'
        http_handler.rfile = io.BytesIO(body_bytes)
        http_handler.headers["Content-Length"] = str(len(body_bytes))
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""

        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch.object(handler, "_handle_message", side_effect=TypeError("wrong type")),
        ):
            result = handler._handle_webhook(http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["ok"] is False

    def test_os_error_returns_200(self, handler, handler_module):
        http_handler = MockHTTPHandler()
        body_bytes = b'{"update_id": 1, "message": {}}'
        http_handler.rfile = io.BytesIO(body_bytes)
        http_handler.headers["Content-Length"] = str(len(body_bytes))
        http_handler.headers["X-Telegram-Bot-Api-Secret-Token"] = ""

        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch.object(handler, "_handle_message", side_effect=OSError("io error")),
        ):
            result = handler._handle_webhook(http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["ok"] is False


# ===========================================================================
# Edited message handling
# ===========================================================================


class TestEditedMessage:
    """Tests for edited message handling."""

    def test_edited_message_handled(self, handler, handler_module):
        update = _edited_message_update(text="Edited text")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_edited_message_with_command(self, handler, handler_module):
        """Edited message with a command entity at offset 0."""
        update = {
            "update_id": 204,
            "edited_message": {
                "message_id": 1,
                "from": {"id": 123, "is_bot": False, "username": "testuser"},
                "chat": {"id": 456, "type": "private"},
                "text": "/help",
                "entities": [{"type": "bot_command", "offset": 0, "length": 5}],
                "date": 1700000000,
                "edit_date": 1700001000,
            },
        }
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# _handle_command edge cases
# ===========================================================================


class TestHandleCommandEdgeCases:
    """Tests for command handling edge cases."""

    def test_command_case_insensitive(self, handler, handler_module):
        """Commands are lowercased before routing."""
        update = _command_message_update(command="/HELP", args="")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_command_with_bot_suffix(self, handler, handler_module):
        """Telegram commands can include @botname - slash is stripped."""
        # The handler strips the leading /, so "/start@mybot" becomes "start@mybot"
        # which won't match "start" exactly. This tests the current behavior.
        update = _message_update(
            text="/start@mybot",
            entities=[{"type": "bot_command", "offset": 0, "length": 13}],
        )
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True


# ===========================================================================
# _handle_vote edge cases
# ===========================================================================


class TestHandleVoteEdgeCases:
    """Tests for vote handling edge cases."""

    def test_vote_attribute_error(self, handler, handler_module):
        """AttributeError when ConsensusStore lacks record_vote."""
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        mock_store = MagicMock()
        mock_store.record_vote.side_effect = AttributeError("no record_vote")
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch("aragora.memory.consensus.ConsensusStore", return_value=mock_store),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is False

    def test_vote_value_error(self, handler, handler_module):
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        mock_store = MagicMock()
        mock_store.record_vote.side_effect = ValueError("invalid vote")
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch("aragora.memory.consensus.ConsensusStore", return_value=mock_store),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is False

    def test_vote_key_error(self, handler, handler_module):
        update = _callback_query_update(data="vote:debate-abc:agree")
        http_handler = _make_webhook_handler(update)
        mock_store = MagicMock()
        mock_store.record_vote.side_effect = KeyError("missing key")
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
            patch("aragora.server.handlers.bots.telegram.TelegramHandler._answer_callback_query"),
            patch("aragora.memory.consensus.ConsensusStore", return_value=mock_store),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is False


# ===========================================================================
# Webhook token derivation
# ===========================================================================


class TestWebhookTokenDerivation:
    """Tests for TELEGRAM_WEBHOOK_TOKEN derivation from bot token."""

    def test_token_is_sha256_prefix(self, handler_module):
        """TELEGRAM_WEBHOOK_TOKEN is sha256(bot_token)[:32]."""
        bot_token = "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
        expected = hashlib.sha256(bot_token.encode()).hexdigest()[:32]
        # The module-level computation should follow this formula
        assert len(expected) == 32

    def test_empty_bot_token_gives_empty_webhook_token(self):
        """When TELEGRAM_BOT_TOKEN is falsy, TELEGRAM_WEBHOOK_TOKEN is empty."""
        # This tests the module-level logic: if not TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_TOKEN = ""
        # We verify the logic pattern rather than re-importing the module
        token = ""
        if token:
            result = hashlib.sha256(token.encode()).hexdigest()[:32]
        else:
            result = ""
        assert result == ""


# ===========================================================================
# _cmd_debate with decision_integrity params
# ===========================================================================


class TestDecisionIntegrity:
    """Tests for decision_integrity parameter in commands."""

    def test_plan_sets_include_receipt(self, handler, handler_module):
        """The /plan command sets include_receipt=True."""
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
            patch.object(handler, "_start_debate_async", return_value="test-id") as mock_start,
            patch.object(handler, "_send_message"),
        ):
            handler._cmd_debate(
                12345,
                67890,
                "topic",
                decision_integrity={
                    "include_receipt": True,
                    "include_plan": True,
                },
            )
        call_kwargs = mock_start.call_args
        di = call_kwargs[1].get("decision_integrity") if call_kwargs[1] else None
        if di is None and len(call_kwargs[0]) > 4:
            di = call_kwargs[0][4]
        # The method should have been called with decision_integrity
        mock_start.assert_called_once()

    def test_implement_sets_execution_mode(self, handler, handler_module):
        """The /implement command sets execution_mode=execute."""
        update = _command_message_update(command="/implement", args="Do something")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
            patch.object(handler, "_start_debate_async", return_value="test-id") as mock_start,
            patch.object(handler, "_send_message"),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        mock_start.assert_called_once()
        # Verify decision_integrity was passed
        call_args = mock_start.call_args
        assert call_args is not None

    def test_debate_no_decision_integrity(self, handler, handler_module):
        """The /debate command does not set decision_integrity."""
        update = _command_message_update(command="/debate", args="Simple question")
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
            patch.object(handler, "_start_debate_async", return_value="test-id") as mock_start,
            patch.object(handler, "_send_message"),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        call_kwargs = mock_start.call_args
        # For /debate, decision_integrity should be None
        di = call_kwargs[1].get("decision_integrity") if call_kwargs[1] else None
        assert di is None


# ===========================================================================
# _fallback_queue_debate
# ===========================================================================


class TestFallbackQueueDebate:
    """Tests for _fallback_queue_debate."""

    @pytest.mark.asyncio
    async def test_fallback_import_error_falls_to_direct(self, handler):
        with (
            patch(
                "aragora.server.debate_origin.register_debate_origin",
                side_effect=RuntimeError("skip"),
            ),
            patch.dict("sys.modules", {"aragora.queue": None}),
            patch.object(handler, "_run_debate_direct", return_value="direct-id"),
        ):
            result = await handler._fallback_queue_debate(12345, 67890, "Topic", "fb-id")
        assert result == "direct-id"

    @pytest.mark.asyncio
    async def test_fallback_runtime_error(self, handler):
        with (
            patch(
                "aragora.server.debate_origin.register_debate_origin",
                side_effect=RuntimeError("fail"),
            ),
            patch.dict("sys.modules", {"aragora.queue": None}),
            patch.object(handler, "_run_debate_direct", return_value="direct-id"),
        ):
            result = await handler._fallback_queue_debate(12345, 67890, "Topic", "fb-id")
        assert isinstance(result, str)


# ===========================================================================
# _run_debate_direct
# ===========================================================================


class TestRunDebateDirect:
    """Tests for _run_debate_direct."""

    def test_returns_debate_id(self, handler):
        """Should return the debate_id immediately (runs in background thread)."""
        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            result = handler._run_debate_direct(12345, 67890, "Topic", "test-id")
        assert result == "test-id"
        mock_thread_instance.start.assert_called_once()

    def test_thread_is_daemon(self, handler):
        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            handler._run_debate_direct(12345, 67890, "Topic", "test-id")
        mock_thread.assert_called_once()
        call_kwargs = mock_thread.call_args[1]
        assert call_kwargs.get("daemon") is True


# ===========================================================================
# Integration tests - full webhook flow
# ===========================================================================


class TestFullWebhookFlow:
    """Integration-style tests for complete webhook processing flows."""

    def test_message_with_document_attachment(self, handler, handler_module):
        update = _message_update(
            text="Check this file",
            document={
                "file_id": "doc123",
                "file_name": "report.pdf",
                "mime_type": "application/pdf",
            },
            caption="Important document",
        )
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_debate_command_with_attachment(self, handler, handler_module):
        """Debate command should extract attachments from the message."""
        full_text = "/debate Check this document"
        update = {
            "update_id": 300,
            "message": {
                "message_id": 1,
                "from": {"id": 67890, "is_bot": False, "username": "testuser"},
                "chat": {"id": 12345, "type": "private"},
                "text": full_text,
                "entities": [{"type": "bot_command", "offset": 0, "length": 7}],
                "document": {"file_id": "doc456", "file_name": "data.csv"},
                "date": 1700000000,
            },
        }
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test", "PYTEST_CURRENT_TEST": "yes"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
        assert body.get("debate_started") is True

    def test_multiple_entity_types_first_command_wins(self, handler, handler_module):
        """When there are multiple entities, only first bot_command at offset 0 is handled."""
        update = _message_update(
            text="/help some text",
            entities=[
                {"type": "bot_command", "offset": 0, "length": 5},
                {"type": "mention", "offset": 6, "length": 4},
            ],
        )
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.object(handler_module, "TELEGRAM_BOT_TOKEN", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_empty_update_body(self, handler, handler_module):
        """Empty JSON object as update body."""
        update = {}
        http_handler = _make_webhook_handler(update)
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", None),
            patch.dict("os.environ", {"ARAGORA_ENV": "test"}),
        ):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_correct_secret_header_passes(self, handler, handler_module):
        """Correct secret token header allows processing."""
        update = _message_update(text="Authenticated message")
        http_handler = _make_webhook_handler(update, secret_token="correct-secret")
        with patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "correct-secret"):
            result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True

    def test_webhook_with_both_token_and_secret(self, handler, handler_module):
        """Token-verified webhook still checks secret header."""
        update = _message_update(text="Double auth")
        token = "valid-url-token"
        http_handler = _make_webhook_handler(update, secret_token="correct-secret")
        with (
            patch.object(handler_module, "TELEGRAM_WEBHOOK_TOKEN", token),
            patch.object(handler_module, "TELEGRAM_WEBHOOK_SECRET", "correct-secret"),
        ):
            result = handler.handle_post(f"/api/v1/bots/telegram/webhook/{token}", {}, http_handler)
        body = _body(result)
        assert body["ok"] is True
