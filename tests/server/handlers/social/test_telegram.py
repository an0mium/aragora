"""
Tests for aragora.server.handlers.social.telegram - Telegram Bot Handler.

Comprehensive test coverage including:
- Webhook processing (valid, invalid, signature verification)
- Message routing (commands, text, callbacks, inline queries)
- Debate initiation flow (validation, execution, result delivery)
- Gauntlet initiation flow (validation, execution, result delivery)
- Error handling (invalid JSON, missing fields, API failures)
- Edge cases (rate limits, empty messages, malformed data)
- RBAC permission checks
- Async task handling
- TTS voice summary integration
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from aragora.server.handlers.social.telegram import (
    TELEGRAM_API_BASE,
    TelegramHandler,
    create_tracked_task,
    get_telegram_handler,
    _handle_task_exception,
)


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
        method: str = "GET",
        client_address: tuple[str, int] | None = None,
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = client_address or ("127.0.0.1", 12345)
        self.response_code: int | None = None
        self._response_headers: dict[str, str] = {}

    def send_response(self, code):
        self.response_code = code

    def send_header(self, key, value):
        self._response_headers[key] = value

    def end_headers(self):
        pass

    @classmethod
    def with_json_body(
        cls,
        data: dict[str, Any],
        path: str = "/",
        method: str = "POST",
        headers: dict[str, str] | None = None,
    ) -> "MockHandler":
        """Create a MockHandler with JSON body."""
        body = json.dumps(data).encode("utf-8")
        all_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if headers:
            all_headers.update(headers)
        return cls(headers=all_headers, body=body, path=path, method=method)


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
    }


@pytest.fixture
def handler(mock_server_context):
    """Create a TelegramHandler instance."""
    return TelegramHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json", "Content-Length": "0"},
        path="/api/v1/integrations/telegram/status",
    )


@pytest.fixture(autouse=True)
def reset_telegram_globals():
    """Reset Telegram global state before each test."""
    import aragora.server.handlers.social.telegram as tg

    tg._telegram_handler = None
    yield


# ===========================================================================
# Helper Functions
# ===========================================================================


def get_body(result) -> bytes:
    """Extract body from handler result (dict or HandlerResult dataclass)."""
    if hasattr(result, "body"):
        return result.body
    return result.get("body", b"")


def get_status_code(result) -> int:
    """Extract status code from handler result."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result.get("status", result.get("status_code", 200))


def get_json(result) -> dict[str, Any]:
    """Parse JSON body from handler result."""
    body = get_body(result)
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def create_telegram_update(
    update_id: int = 12345,
    message: dict | None = None,
    callback_query: dict | None = None,
    inline_query: dict | None = None,
    edited_message: dict | None = None,
) -> dict[str, Any]:
    """Create a Telegram update payload."""
    update: dict[str, Any] = {"update_id": update_id}
    if message is not None:
        update["message"] = message
    if callback_query is not None:
        update["callback_query"] = callback_query
    if inline_query is not None:
        update["inline_query"] = inline_query
    if edited_message is not None:
        update["edited_message"] = edited_message
    return update


def create_telegram_message(
    chat_id: int = 123456,
    user_id: int = 789,
    username: str = "testuser",
    text: str = "Hello",
    message_id: int = 1,
) -> dict[str, Any]:
    """Create a Telegram message payload."""
    return {
        "message_id": message_id,
        "from": {"id": user_id, "username": username, "first_name": "Test"},
        "chat": {"id": chat_id, "type": "private"},
        "date": int(time.time()),
        "text": text,
    }


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/integrations/telegram/status") is True

    def test_can_handle_webhook(self, handler):
        """Test handler recognizes webhook endpoint."""
        assert handler.can_handle("/api/v1/integrations/telegram/webhook") is True

    def test_can_handle_set_webhook(self, handler):
        """Test handler recognizes set-webhook endpoint."""
        assert handler.can_handle("/api/v1/integrations/telegram/set-webhook") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/integrations/telegram/unknown") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False
        assert handler.can_handle("/api/integrations/telegram/webhook") is False

    def test_routes_defined(self, handler):
        """Test handler has ROUTES defined with correct paths."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) == 3
        assert "/api/v1/integrations/telegram/webhook" in handler.ROUTES
        assert "/api/v1/integrations/telegram/status" in handler.ROUTES
        assert "/api/v1/integrations/telegram/set-webhook" in handler.ROUTES

    def test_handle_returns_404_for_unknown_path(self, handler):
        """Test handle returns 404 for unregistered path."""
        mock_http = MockHandler(path="/api/v1/unknown", method="GET")
        result = handler.handle("/api/v1/unknown", {}, mock_http)
        assert get_status_code(result) == 404


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/integrations/telegram/status."""

    def test_get_status(self, handler, mock_http_handler):
        """Test getting status."""
        result = handler.handle("/api/v1/integrations/telegram/status", {}, mock_http_handler)

        assert result is not None
        data = get_json(result)

        # Status should include config flags
        assert "enabled" in data
        assert "bot_token_configured" in data
        assert "webhook_secret_configured" in data

    def test_status_fields_are_booleans(self, handler, mock_http_handler):
        """Test status fields are booleans."""
        result = handler.handle("/api/v1/integrations/telegram/status", {}, mock_http_handler)
        data = get_json(result)

        assert isinstance(data["enabled"], bool)
        assert isinstance(data["bot_token_configured"], bool)
        assert isinstance(data["webhook_secret_configured"], bool)

    def test_status_reflects_env_config(self, handler):
        """Test status reflects environment configuration."""
        mock_http = MockHandler(method="GET")

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET",
                "test_secret",
            ):
                # Need to call _get_status directly since module-level vars are cached
                result = handler._get_status()
                # Result is based on module-level cached values, so test just that it works
                assert result is not None


# ===========================================================================
# Webhook Endpoint Tests
# ===========================================================================


class TestWebhookEndpoint:
    """Tests for POST /api/integrations/telegram/webhook."""

    def test_webhook_requires_post(self, handler):
        """Test webhook endpoint rejects GET."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/telegram/webhook",
            method="GET",
        )
        result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        # Should return method not allowed
        assert result is not None
        assert get_status_code(result) == 405

    def test_webhook_requires_secret_verification(self, handler):
        """Test webhook rejects request when secret verification fails."""
        body = json.dumps({"update_id": 12345}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=False):
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        assert get_status_code(result) == 401

    def test_webhook_handles_empty_update(self, handler):
        """Test webhook handles empty/minimal update."""
        body = json.dumps({"update_id": 12345}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_handles_message(self, handler):
        """Test webhook handles message update."""
        message = create_telegram_message(text="/help")
        update = create_telegram_update(message=message)
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
                result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_handles_callback_query(self, handler):
        """Test webhook handles callback query update."""
        update = create_telegram_update(
            callback_query={
                "id": "callback123",
                "data": "vote:debate123:agree",
                "from": {"id": 456, "username": "voter"},
                "message": {"chat": {"id": 789}},
            }
        )
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_handles_edited_message(self, handler):
        """Test webhook handles edited message update."""
        update = create_telegram_update(edited_message=create_telegram_message(text="Edited text"))
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        # Edited messages are ignored but acknowledged
        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_handles_inline_query(self, handler):
        """Test webhook handles inline query update."""
        update = create_telegram_update(
            inline_query={
                "id": "query123",
                "query": "test query text for debate",
                "from": {"id": 456, "username": "querier"},
            }
        )
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_invalid_json(self, handler):
        """Test webhook handles invalid JSON gracefully."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "10",
            },
            body=b"not json!!",
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        # Should still return ok to acknowledge receipt
        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True


# ===========================================================================
# Secret Verification Tests
# ===========================================================================


class TestSecretVerification:
    """Tests for webhook secret verification."""

    def test_verify_secret_success(self, handler):
        """Test successful secret verification."""
        mock_http = MockHandler(headers={"X-Telegram-Bot-Api-Secret-Token": "test_secret"})

        with patch(
            "aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET",
            "test_secret",
        ):
            result = handler._verify_secret(mock_http)

        assert result is True

    def test_verify_secret_failure(self, handler):
        """Test failed secret verification."""
        mock_http = MockHandler(headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"})

        with patch(
            "aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET",
            "test_secret",
        ):
            result = handler._verify_secret(mock_http)

        assert result is False

    def test_verify_secret_missing_header(self, handler):
        """Test secret verification with missing header."""
        mock_http = MockHandler(headers={})

        with patch(
            "aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET",
            "test_secret",
        ):
            result = handler._verify_secret(mock_http)

        assert result is False

    def test_verify_secret_no_secret_configured_dev(self, handler):
        """Test secret verification in development when no secret configured."""
        mock_http = MockHandler(headers={})

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
                result = handler._verify_secret(mock_http)

        # Should pass in development mode
        assert result is True

    def test_verify_secret_no_secret_configured_production(self, handler):
        """Test secret verification in production when no secret configured."""
        mock_http = MockHandler(headers={})

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
            with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
                result = handler._verify_secret(mock_http)

        # Should fail closed in production
        assert result is False

    def test_verify_secret_handles_exception(self, handler):
        """Test secret verification handles exceptions gracefully."""
        mock_http = MagicMock()
        mock_http.headers.get.side_effect = Exception("Header access error")

        with patch(
            "aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET",
            "test_secret",
        ):
            result = handler._verify_secret(mock_http)

        assert result is False


# ===========================================================================
# Command Tests
# ===========================================================================


class TestCommands:
    """Tests for bot command handling."""

    def test_command_start(self, handler):
        """Test /start command."""
        response = handler._command_start("testuser")
        assert "Welcome" in response
        assert "testuser" in response
        assert "/help" in response
        assert "/debate" in response
        assert "/plan" in response
        assert "/implement" in response

    def test_command_help(self, handler):
        """Test /help command."""
        response = handler._command_help()
        assert "/debate" in response
        assert "/plan" in response
        assert "/implement" in response
        assert "/gauntlet" in response
        assert "/status" in response
        assert "/agents" in response
        assert "Examples" in response

    def test_command_status(self, handler):
        """Test /status command."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = [
                MagicMock(name="agent1", elo=1600),
                MagicMock(name="agent2", elo=1500),
            ]
            response = handler._command_status()

        assert "Status" in response or "Online" in response

    def test_command_status_error(self, handler):
        """Test /status command with error."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.side_effect = Exception("DB error")
            response = handler._command_status()

        assert "Online" in response

    def test_command_agents_empty(self, handler):
        """Test /agents command with no agents."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = []
            response = handler._command_agents()

        assert "No agents" in response

    def test_command_agents_with_agents(self, handler):
        """Test /agents command with agents."""
        mock_agents = [
            MagicMock(name="claude", elo=1650, wins=10),
            MagicMock(name="gpt-4", elo=1600, wins=8),
            MagicMock(name="gemini", elo=1550, wins=5),
        ]
        for i, agent in enumerate(mock_agents):
            agent.name = ["claude", "gpt-4", "gemini"][i]

        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = mock_agents
            response = handler._command_agents()

        assert "Top Agents" in response or "ELO" in response

    def test_command_agents_error(self, handler):
        """Test /agents command with error."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.side_effect = Exception("DB error")
            response = handler._command_agents()

        assert "Could not fetch" in response

    def test_handle_command_routes_correctly(self, handler):
        """Test _handle_command routes to correct handler."""
        with patch.object(handler, "_command_help", return_value="Help text") as mock_help:
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                handler._handle_command(123, 456, "user", "/help")
                mock_help.assert_called_once()

    def test_handle_command_with_bot_suffix(self, handler):
        """Test command handling strips @botname suffix."""
        with patch.object(handler, "_command_help", return_value="Help text") as mock_help:
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                handler._handle_command(123, 456, "user", "/help@aragora_bot")
                mock_help.assert_called_once()

    def test_handle_command_unknown(self, handler):
        """Test unknown command handling."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            handler._handle_command(123, 456, "user", "/unknown")
            mock_task.assert_called_once()

    def test_handle_command_case_insensitive(self, handler):
        """Test commands are case insensitive."""
        with patch.object(handler, "_command_help", return_value="Help text") as mock_help:
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                handler._handle_command(123, 456, "user", "/HELP")
                mock_help.assert_called_once()

    def test_handle_command_plan(self, handler):
        """Test /plan command routes to debate with decision integrity."""
        with patch.object(handler, "_command_debate", return_value={"ok": True}) as mock_cmd:
            handler._handle_command(123, 456, "user", "/plan Improve onboarding")
        assert mock_cmd.call_args is not None
        kwargs = mock_cmd.call_args.kwargs
        assert kwargs["decision_integrity"]["include_plan"] is True
        assert kwargs["decision_integrity"]["requested_by"] == "telegram:456"
        assert kwargs["mode_label"] == "plan"

    def test_handle_command_implement(self, handler):
        """Test /implement command routes to debate with context snapshot."""
        with patch.object(handler, "_command_debate", return_value={"ok": True}) as mock_cmd:
            handler._handle_command(123, 456, "user", "/implement Automate reporting")
        kwargs = mock_cmd.call_args.kwargs
        assert kwargs["decision_integrity"]["include_context"] is True
        assert kwargs["decision_integrity"]["requested_by"] == "telegram:456"
        assert kwargs["mode_label"] == "implementation plan"


# ===========================================================================
# Debate Command Tests
# ===========================================================================


class TestDebateCommand:
    """Tests for /debate command."""

    def test_debate_no_topic(self, handler):
        """Test debate command without topic."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(123, 456, "user", "")

        assert result is not None
        mock_task.assert_called_once()

    def test_debate_topic_too_short(self, handler):
        """Test debate command with short topic."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(123, 456, "user", "test")

        mock_task.assert_called_once()
        data = get_json(result)
        assert data.get("ok") is True

    def test_debate_topic_too_long(self, handler):
        """Test debate command with long topic."""
        long_topic = "x" * 600
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(123, 456, "user", long_topic)

        mock_task.assert_called_once()
        data = get_json(result)
        assert data.get("ok") is True

    def test_debate_valid_topic(self, handler):
        """Test debate command with valid topic."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(
                123, 456, "user", "Should artificial intelligence be regulated by governments?"
            )

        # Should send acknowledgment and queue debate
        assert mock_task.call_count >= 2
        data = get_json(result)
        assert data.get("ok") is True

    def test_debate_strips_quotes(self, handler):
        """Test debate command strips surrounding quotes."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(
                123, 456, "user", '"Should AI be regulated by governments?"'
            )

        # Should create acknowledgment and queue debate
        assert mock_task.call_count >= 2

    def test_debate_boundary_length(self, handler):
        """Test debate command at exactly 10 characters (minimum)."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(123, 456, "user", "A" * 10)

        # 10 chars should trigger the debate, but 9 should be rejected
        assert mock_task.call_count >= 2

    def test_debate_boundary_length_short(self, handler):
        """Test debate command at exactly 9 characters (too short)."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(123, 456, "user", "A" * 9)

        # 9 chars should be rejected
        assert mock_task.call_count == 1  # Only error message sent


# ===========================================================================
# Gauntlet Command Tests
# ===========================================================================


class TestGauntletCommand:
    """Tests for /gauntlet command."""

    def test_gauntlet_no_statement(self, handler):
        """Test gauntlet command without statement."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(123, 456, "user", "")

        mock_task.assert_called_once()
        data = get_json(result)
        assert data.get("ok") is True

    def test_gauntlet_statement_too_short(self, handler):
        """Test gauntlet command with short statement."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(123, 456, "user", "test")

        mock_task.assert_called_once()

    def test_gauntlet_statement_too_long(self, handler):
        """Test gauntlet command with statement over 1000 characters."""
        long_statement = "x" * 1100
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(123, 456, "user", long_statement)

        mock_task.assert_called_once()

    def test_gauntlet_valid_statement(self, handler):
        """Test gauntlet command with valid statement."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(
                123,
                456,
                "user",
                "We should migrate our monolith to microservices architecture",
            )

        # Should send acknowledgment and queue gauntlet
        assert mock_task.call_count >= 2

    def test_gauntlet_strips_quotes(self, handler):
        """Test gauntlet command strips surrounding quotes."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(123, 456, "user", "'We should use microservices'")

        # Should create acknowledgment and queue gauntlet (if long enough after strip)
        assert mock_task.call_count >= 1


# ===========================================================================
# Callback Query Tests
# ===========================================================================


class TestCallbackQueries:
    """Tests for inline keyboard callback handling."""

    def test_handle_callback_query_vote_agree(self, handler):
        """Test vote agree callback handling."""
        callback = {
            "id": "callback123",
            "data": "vote:debate123:agree",
            "from": {"id": 456, "username": "voter"},
            "message": {"chat": {"id": 789}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch.object(handler, "_handle_vote") as mock_vote:
                mock_vote.return_value = MagicMock(status_code=200, body=b'{"ok":true}')
                result = handler._handle_callback_query(callback)

                mock_vote.assert_called_once_with(
                    "callback123", 789, 456, "voter", "debate123", "agree"
                )

    def test_handle_callback_query_vote_disagree(self, handler):
        """Test vote disagree callback handling."""
        callback = {
            "id": "callback456",
            "data": "vote:debate789:disagree",
            "from": {"id": 123, "username": "dissenter"},
            "message": {"chat": {"id": 456}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch.object(handler, "_handle_vote") as mock_vote:
                mock_vote.return_value = MagicMock(status_code=200, body=b'{"ok":true}')
                result = handler._handle_callback_query(callback)

                mock_vote.assert_called_once_with(
                    "callback456", 456, 123, "dissenter", "debate789", "disagree"
                )

    def test_handle_callback_query_details(self, handler):
        """Test view details callback handling."""
        callback = {
            "id": "callback123",
            "data": "details:debate123",
            "from": {"id": 456, "username": "viewer"},
            "message": {"chat": {"id": 789}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch.object(handler, "_handle_view_details") as mock_details:
                mock_details.return_value = MagicMock(status_code=200, body=b'{"ok":true}')
                result = handler._handle_callback_query(callback)

                # _handle_view_details now takes (callback_id, chat_id, user_id, username, debate_id)
                mock_details.assert_called_once_with("callback123", 789, 456, "viewer", "debate123")

    def test_handle_callback_query_unknown_action(self, handler):
        """Test callback query with unknown action."""
        callback = {
            "id": "callback123",
            "data": "unknown_action:data",
            "from": {"id": 456, "username": "user"},
            "message": {"chat": {"id": 789}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._handle_callback_query(callback)

        # Should acknowledge unknown callbacks
        assert result is not None
        mock_task.assert_called_once()

    def test_handle_callback_query_malformed_data(self, handler):
        """Test callback query with malformed data."""
        callback = {
            "id": "callback123",
            "data": "malformed_without_colons",
            "from": {"id": 456, "username": "user"},
            "message": {"chat": {"id": 789}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._handle_callback_query(callback)

        assert result is not None
        mock_task.assert_called_once()


# ===========================================================================
# Vote Handling Tests
# ===========================================================================


class TestVoteHandling:
    """Tests for vote callback handling."""

    def test_handle_vote_records_vote(self, handler):
        """Test vote is recorded in storage."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch(
                "aragora.server.handlers.social.telegram.callbacks.emit_vote_received"
            ) as mock_emit:
                with patch("aragora.server.storage.get_debates_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    mock_db.return_value.record_vote = MagicMock()

                    handler._handle_vote("callback123", 789, 456, "voter", "debate123", "agree")

                    mock_emit.assert_called_once()

    def test_handle_vote_handles_storage_error(self, handler):
        """Test vote handling gracefully handles storage errors."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch("aragora.server.storage.get_debates_db") as mock_db:
                mock_db.return_value.record_vote.side_effect = Exception("DB error")

                # Should not raise
                result = handler._handle_vote(
                    "callback123", 789, 456, "voter", "debate123", "agree"
                )

                assert result is not None


# ===========================================================================
# View Details Tests
# ===========================================================================


class TestViewDetails:
    """Tests for view details callback handling."""

    def test_handle_view_details_found(self, handler):
        """Test view details when debate is found."""
        mock_debate = {
            "task": "Test debate topic",
            "final_answer": "The conclusion is...",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
            "agents": ["claude", "gpt-4"],
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch("aragora.server.storage.get_debates_db") as mock_db:
                mock_db.return_value.get.return_value = mock_debate

                # _handle_view_details now takes (callback_id, chat_id, user_id, username, debate_id)
                result = handler._handle_view_details(
                    "callback123", 789, 456, "viewer", "debate123"
                )

                assert result is not None

    def test_handle_view_details_not_found(self, handler):
        """Test view details when debate is not found."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            with patch("aragora.server.storage.get_debates_db") as mock_db:
                mock_db.return_value.get.return_value = None

                # _handle_view_details now takes (callback_id, chat_id, user_id, username, debate_id)
                result = handler._handle_view_details(
                    "callback123", 789, 456, "viewer", "debate123"
                )

                assert result is not None
                # Should send "not found" callback answer
                mock_task.assert_called_once()

    def test_handle_view_details_storage_error(self, handler):
        """Test view details handles storage errors gracefully."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            with patch("aragora.server.storage.get_debates_db") as mock_db:
                mock_db.return_value.get.side_effect = Exception("DB error")

                # _handle_view_details now takes (callback_id, chat_id, user_id, username, debate_id)
                result = handler._handle_view_details(
                    "callback123", 789, 456, "viewer", "debate123"
                )

                assert result is not None


# ===========================================================================
# Message Handling Tests
# ===========================================================================


class TestMessageHandling:
    """Tests for general message handling."""

    def test_handle_message_empty(self, handler):
        """Test handling empty message."""
        message = {
            "chat": {"id": 123},
            "from": {"id": 456},
            "text": "",
        }

        result = handler._handle_message(message)
        assert result is not None

    def test_handle_message_no_chat_id(self, handler):
        """Test handling message without chat id."""
        message = {"from": {"id": 456}, "text": "Hello"}

        result = handler._handle_message(message)
        data = get_json(result)
        assert data.get("ok") is True

    def test_handle_message_short(self, handler):
        """Test handling short non-command message."""
        message = {
            "chat": {"id": 123},
            "from": {"id": 456, "username": "user"},
            "text": "hi",
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            result = handler._handle_message(message)

        assert result is not None

    def test_handle_message_long_suggests_debate(self, handler):
        """Test handling longer message prompts debate suggestion."""
        message = {
            "chat": {"id": 123},
            "from": {"id": 456, "username": "user"},
            "text": "This is a longer message that could be a debate topic",
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._handle_message(message)

        mock_task.assert_called_once()

    def test_handle_message_command_routed(self, handler):
        """Test command messages are routed to command handler."""
        message = {
            "chat": {"id": 123},
            "from": {"id": 456, "username": "user"},
            "text": "/status",
        }

        with patch.object(handler, "_handle_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(status_code=200, body=b'{"ok":true}')
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                result = handler._handle_message(message)

            mock_cmd.assert_called_once_with(123, 456, "user", "/status")


# ===========================================================================
# Inline Query Tests
# ===========================================================================


class TestInlineQueries:
    """Tests for inline query handling."""

    def test_handle_inline_query_short(self, handler):
        """Test inline query with short text returns empty results."""
        query = {
            "id": "query123",
            "query": "hi",
            "from": {"id": 456, "username": "user"},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._handle_inline_query(query)

        assert result is not None
        data = get_json(result)
        assert data.get("ok") is True

    def test_handle_inline_query_valid(self, handler):
        """Test inline query with valid text returns results."""
        query = {
            "id": "query123",
            "query": "Should AI be regulated by governments?",
            "from": {"id": 456, "username": "user"},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._handle_inline_query(query)

        assert result is not None
        mock_task.assert_called_once()


# ===========================================================================
# Set Webhook Tests
# ===========================================================================


class TestSetWebhook:
    """Tests for POST /api/integrations/telegram/set-webhook."""

    def test_set_webhook_missing_url(self, handler):
        """Test set webhook with missing URL (when token configured)."""
        body = json.dumps({}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Mock the token being configured
        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            result = handler._set_webhook(mock_http)
            assert get_status_code(result) == 400

    def test_set_webhook_no_token(self, handler):
        """Test set webhook returns 500 when token not configured."""
        body = json.dumps({"url": "https://example.com/webhook"}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Token not configured should return 500
        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", ""):
            result = handler._set_webhook(mock_http)
            assert get_status_code(result) == 500

    def test_set_webhook_invalid_json(self, handler):
        """Test set webhook with invalid JSON."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "10",
            },
            body=b"not json!!",
            method="POST",
        )

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            result = handler._set_webhook(mock_http)
            assert get_status_code(result) == 400

    def test_set_webhook_valid(self, handler):
        """Test set webhook with valid URL."""
        body = json.dumps({"url": "https://example.com/webhook"}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
                result = handler._set_webhook(mock_http)

                assert get_status_code(result) == 200
                mock_task.assert_called_once()

    def test_set_webhook_requires_post(self, handler):
        """Test set webhook rejects non-POST methods."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/telegram/set-webhook",
            method="GET",
        )

        result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http)
        assert get_status_code(result) == 405


# ===========================================================================
# Factory Tests
# ===========================================================================


class TestFactory:
    """Tests for handler factory function."""

    def test_get_telegram_handler_singleton(self):
        """Test get_telegram_handler returns consistent instance."""
        import aragora.server.handlers.social.telegram as tg

        tg._telegram_handler = None

        handler1 = get_telegram_handler({})
        handler2 = get_telegram_handler({})

        assert handler1 is handler2

    def test_get_telegram_handler_creates_instance(self):
        """Test get_telegram_handler creates instance."""
        import aragora.server.handlers.social.telegram as tg

        tg._telegram_handler = None

        handler = get_telegram_handler({})
        assert isinstance(handler, TelegramHandler)

    def test_get_telegram_handler_default_context(self):
        """Test get_telegram_handler with no context."""
        import aragora.server.handlers.social.telegram as tg

        tg._telegram_handler = None

        handler = get_telegram_handler(None)
        assert isinstance(handler, TelegramHandler)
        assert handler.ctx == {}


# ===========================================================================
# Async Task Handling Tests
# ===========================================================================


class TestAsyncTaskHandling:
    """Tests for async task creation and error handling."""

    def test_create_tracked_task(self):
        """Test create_tracked_task creates task with callback."""

        async def test_coro():
            return "result"

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            result = create_tracked_task(test_coro(), "test-task")

            mock_create_task.assert_called_once()
            mock_task.add_done_callback.assert_called_once()

    def test_handle_task_exception_cancelled(self):
        """Test exception handler for cancelled task."""
        mock_task = MagicMock()
        mock_task.cancelled.return_value = True

        # Should not raise
        _handle_task_exception(mock_task, "test-task")

    def test_handle_task_exception_with_error(self):
        """Test exception handler for task with exception."""
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = ValueError("Test error")

        # Should not raise
        _handle_task_exception(mock_task, "test-task")

    def test_handle_task_exception_success(self):
        """Test exception handler for successful task."""
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None

        # Should not raise
        _handle_task_exception(mock_task, "test-task")


# ===========================================================================
# Async Method Tests
# ===========================================================================


class TestAsyncMethods:
    """Tests for async methods."""

    @pytest.mark.asyncio
    async def test_send_message_async_no_token(self, handler):
        """Test send_message_async returns early without token."""
        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", ""):
            # Should return early without error
            await handler._send_message_async(123, "Test message")

    @pytest.mark.asyncio
    async def test_send_message_async_success(self, handler):
        """Test send_message_async with successful response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                await handler._send_message_async(123, "Test message")

                mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_async_with_parse_mode(self, handler):
        """Test send_message_async with parse mode."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                await handler._send_message_async(123, "*Bold* message", parse_mode="Markdown")

                call_kwargs = mock_client.post.call_args[1]
                assert call_kwargs["json"]["parse_mode"] == "Markdown"

    @pytest.mark.asyncio
    async def test_send_message_async_with_reply_markup(self, handler):
        """Test send_message_async with inline keyboard."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        keyboard = {"inline_keyboard": [[{"text": "Test", "callback_data": "test"}]]}

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                await handler._send_message_async(123, "Test message", reply_markup=keyboard)

                call_kwargs = mock_client.post.call_args[1]
                assert "reply_markup" in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_send_message_async_api_error(self, handler):
        """Test send_message_async handles API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "Bad Request"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                # Should not raise
                await handler._send_message_async(123, "Test message")

    @pytest.mark.asyncio
    async def test_send_message_async_exception(self, handler):
        """Test send_message_async handles exception."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                # Should not raise
                await handler._send_message_async(123, "Test message")

    @pytest.mark.asyncio
    async def test_answer_callback_async_no_token(self, handler):
        """Test answer_callback_async returns early without token."""
        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", ""):
            # Should return early without error
            await handler._answer_callback_async("callback123", "Text")

    @pytest.mark.asyncio
    async def test_answer_inline_query_async_no_token(self, handler):
        """Test answer_inline_query_async returns early without token."""
        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", ""):
            # Should return early without error
            await handler._answer_inline_query_async("query123", [])


# ===========================================================================
# Debate Async Tests
# ===========================================================================


class TestDebateAsync:
    """Tests for async debate execution."""

    @pytest.mark.asyncio
    async def test_run_debate_async_no_agents(self, handler):
        """Test debate fails gracefully when no agents available."""
        with patch("aragora.agents.get_agents_by_names", return_value=[]):
            with patch.object(handler, "_send_message_async") as mock_send:
                await handler._run_debate_async(123, 456, "user", "Test debate topic here")

                mock_send.assert_called()
                # Check the message indicates failure
                call_args = mock_send.call_args_list[-1]
                assert "No agents" in call_args[0][1] or "failed" in call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_run_debate_async_exception(self, handler):
        """Test debate handles exceptions gracefully."""
        with patch("aragora.agents.get_agents_by_names") as mock_agents:
            mock_agents.side_effect = Exception("Agent initialization failed")

            with patch.object(handler, "_send_message_async") as mock_send:
                await handler._run_debate_async(123, 456, "user", "Test debate topic here")

                # Should send error message
                mock_send.assert_called()


# ===========================================================================
# Gauntlet Async Tests
# ===========================================================================


class TestGauntletAsync:
    """Tests for async gauntlet execution."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_success(self, handler):
        """Test successful gauntlet execution."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "gauntlet123",
            "score": 0.85,
            "passed": True,
            "vulnerabilities": [],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
        ):
            with patch.object(handler, "_send_message_async") as mock_send:
                await handler._run_gauntlet_async(123, 456, "user", "Test statement for gauntlet")

                mock_send.assert_called()

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_failure(self, handler):
        """Test gauntlet execution with API failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal error"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
        ):
            with patch.object(handler, "_send_message_async") as mock_send:
                await handler._run_gauntlet_async(123, 456, "user", "Test statement for gauntlet")

                mock_send.assert_called()

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_with_vulnerabilities(self, handler):
        """Test gauntlet execution with vulnerabilities found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "gauntlet123",
            "score": 0.45,
            "passed": False,
            "vulnerabilities": [
                {"description": "Vulnerability 1", "critical": True},
                {"description": "Vulnerability 2", "critical": False},
                {"description": "Vulnerability 3", "critical": False},
                {"description": "Vulnerability 4", "critical": False},
                {"description": "Vulnerability 5", "critical": False},
                {"description": "Vulnerability 6", "critical": False},
            ],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
        ):
            with patch.object(handler, "_send_message_async") as mock_send:
                await handler._run_gauntlet_async(123, 456, "user", "Test statement for gauntlet")

                mock_send.assert_called()
                # Verify truncation message for >5 vulnerabilities
                call_args = mock_send.call_args[0]
                assert "more" in call_args[1] or "Issues Found" in call_args[1]


# ===========================================================================
# RBAC Tests
# ===========================================================================


class TestRBAC:
    """Tests for RBAC permission checking."""

    def test_get_auth_context_rbac_not_available(self, handler):
        """Test auth context extraction when RBAC not available."""
        with patch("aragora.server.handlers.social.telegram.RBAC_AVAILABLE", False):
            result = handler._get_auth_context(MagicMock())
            assert result is None

    def test_check_permission_rbac_not_available(self, handler):
        """Test permission check when RBAC not available."""
        with patch("aragora.server.handlers.social.telegram.RBAC_AVAILABLE", False):
            result = handler._check_permission(MagicMock(), "messaging:write")
            assert result is None

    def test_check_permission_no_context(self, handler):
        """Test permission check when no auth context available."""
        with patch.object(handler, "_get_auth_context", return_value=None):
            result = handler._check_permission(MagicMock(), "messaging:write")
            assert result is None

    def test_set_webhook_checks_permission(self, handler):
        """Test set-webhook endpoint checks RBAC permission."""
        from aragora.server.handlers.social.telegram import PERM_TELEGRAM_ADMIN

        mock_http = MockHandler.with_json_body(
            {"url": "https://example.com/webhook"},
            path="/api/v1/integrations/telegram/set-webhook",
            method="POST",
        )

        with patch.object(
            handler, "_check_permission", return_value=MagicMock(status_code=403)
        ) as mock_check:
            result = handler.handle("/api/v1/integrations/telegram/set-webhook", {}, mock_http)

            # Set-webhook now requires telegram:admin permission
            mock_check.assert_called_once_with(mock_http, PERM_TELEGRAM_ADMIN)
            assert get_status_code(result) == 403


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests for handler registration."""

    def test_handler_has_required_methods(self, handler):
        """Test handler has required methods."""
        assert hasattr(handler, "handle")
        assert hasattr(handler, "handle_post")
        assert hasattr(handler, "can_handle")
        assert callable(handler.handle)
        assert callable(handler.handle_post)
        assert callable(handler.can_handle)

    def test_handle_post_delegates_to_handle(self, handler):
        """Test handle_post delegates to handle."""
        mock_http = MockHandler(method="POST")

        with patch.object(
            handler, "handle", return_value=MagicMock(status_code=200, body=b"{}")
        ) as mock_handle:
            handler.handle_post("/api/v1/integrations/telegram/status", {}, mock_http)
            mock_handle.assert_called_once()

    def test_handler_context_initialization(self, mock_server_context):
        """Test handler initializes with provided context."""
        handler = TelegramHandler(mock_server_context)
        assert handler.ctx == mock_server_context

    def test_handler_default_context(self):
        """Test handler initializes with empty context when none provided."""
        handler = TelegramHandler(None)
        assert handler.ctx == {}


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_webhook_with_empty_message_text(self, handler):
        """Test webhook handles message with empty text."""
        message = create_telegram_message(text="")
        update = create_telegram_update(message=message)
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_with_unicode_message(self, handler):
        """Test webhook handles message with unicode characters."""
        message = create_telegram_message(text="Hello \U0001f600 World! 你好世界")
        update = create_telegram_update(message=message)
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        data = get_json(result)
        assert data.get("ok") is True

    def test_webhook_with_very_long_message(self, handler):
        """Test webhook handles very long message."""
        message = create_telegram_message(text="x" * 5000)
        update = create_telegram_update(message=message)
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                result = handler.handle("/api/v1/integrations/telegram/webhook", {}, mock_http)

        data = get_json(result)
        assert data.get("ok") is True

    def test_callback_query_with_missing_message(self, handler):
        """Test callback query handling with missing message field."""
        callback = {
            "id": "callback123",
            "data": "vote:debate123:agree",
            "from": {"id": 456, "username": "voter"},
            # Missing "message" field
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            result = handler._handle_callback_query(callback)

        assert result is not None

    def test_command_with_multiple_arguments(self, handler):
        """Test command parsing with multiple space-separated arguments."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            result = handler._command_debate(
                123, 456, "user", "First argument second argument third argument"
            )

        assert result is not None

    def test_message_from_user_without_username(self, handler):
        """Test message handling from user without username."""
        message = {
            "message_id": 1,
            "from": {"id": 456, "first_name": "Test"},  # No username
            "chat": {"id": 123, "type": "private"},
            "date": int(time.time()),
            "text": "/help",
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            result = handler._handle_message(message)

        data = get_json(result)
        assert data.get("ok") is True


# ===========================================================================
# TTS Voice Summary Tests
# ===========================================================================


class TestTTSVoiceSummary:
    """Tests for TTS voice summary functionality."""

    @pytest.mark.asyncio
    async def test_send_voice_summary_tts_not_available(self, handler):
        """Test voice summary when TTS helper not available."""
        mock_helper = MagicMock()
        mock_helper.is_available = False

        with patch("aragora.server.handlers.social.telegram.TTS_VOICE_ENABLED", True):
            with patch(
                "aragora.server.handlers.social.tts_helper.get_tts_helper",
                return_value=mock_helper,
            ):
                # Should return early without error
                await handler._send_voice_summary(123, "topic", "answer", True, 0.85, 3)

    @pytest.mark.asyncio
    async def test_send_voice_async_no_token(self, handler):
        """Test send_voice_async returns early without token."""
        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", ""):
            # Should return early without error
            await handler._send_voice_async(123, b"audio_data", 10.0)

    @pytest.mark.asyncio
    async def test_send_voice_async_success(self, handler):
        """Test successful voice message sending."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                await handler._send_voice_async(123, b"audio_data", 10.0)

                mock_client.post.assert_called_once()


# ===========================================================================
# Set Webhook Async Tests
# ===========================================================================


class TestSetWebhookAsync:
    """Tests for async webhook configuration."""

    @pytest.mark.asyncio
    async def test_set_webhook_async_success(self, handler):
        """Test successful webhook configuration."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET",
                "test_secret",
            ):
                with patch(
                    "aragora.server.http_client_pool.get_http_pool",
                    return_value=mock_pool,
                ):
                    await handler._set_webhook_async("https://example.com/webhook")

                    # Verify secret_token is included
                    call_kwargs = mock_client.post.call_args[1]
                    assert "secret_token" in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_set_webhook_async_without_secret(self, handler):
        """Test webhook configuration without secret token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch("aragora.server.handlers.social.telegram.TELEGRAM_WEBHOOK_SECRET", ""):
                with patch(
                    "aragora.server.http_client_pool.get_http_pool",
                    return_value=mock_pool,
                ):
                    await handler._set_webhook_async("https://example.com/webhook")

                    # Verify secret_token is NOT included
                    call_kwargs = mock_client.post.call_args[1]
                    assert "secret_token" not in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_set_webhook_async_api_error(self, handler):
        """Test webhook configuration with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "description": "Invalid URL"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                # Should not raise
                await handler._set_webhook_async("https://example.com/webhook")

    @pytest.mark.asyncio
    async def test_set_webhook_async_exception(self, handler):
        """Test webhook configuration with network exception."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                # Should not raise
                await handler._set_webhook_async("https://example.com/webhook")


# ===========================================================================
# Telemetry and Events Tests
# ===========================================================================


class TestTelemetryAndEvents:
    """Tests for telemetry recording and event emission."""

    def test_webhook_records_telemetry(self, handler):
        """Test webhook processing records telemetry metrics."""
        message = create_telegram_message(text="/help")
        update = create_telegram_update(message=message)
        body = json.dumps(update).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch.object(handler, "_verify_secret", return_value=True):
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch(
                    "aragora.server.handlers.social.telegram.webhooks.record_webhook_request"
                ) as mock_record:
                    with patch(
                        "aragora.server.handlers.social.telegram.webhooks.record_webhook_latency"
                    ):
                        result = handler.handle(
                            "/api/v1/integrations/telegram/webhook", {}, mock_http
                        )

                        mock_record.assert_called_once_with("telegram", "success")

    def test_command_records_metrics(self, handler):
        """Test command handling records command metrics."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch(
                "aragora.server.handlers.social.telegram.commands.record_command"
            ) as mock_record:
                handler._handle_command(123, 456, "user", "/status")

                mock_record.assert_called_once_with("telegram", "status")

    def test_message_emits_event(self, handler):
        """Test message handling emits webhook event."""
        message = create_telegram_message(
            text="This is a longer message for testing event emission"
        )

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch(
                "aragora.server.handlers.social.telegram.callbacks.emit_message_received"
            ) as mock_emit:
                handler._handle_message(message)

                mock_emit.assert_called_once()
                call_kwargs = mock_emit.call_args[1]
                assert call_kwargs["platform"] == "telegram"
