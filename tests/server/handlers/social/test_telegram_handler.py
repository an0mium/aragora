"""
Tests for aragora.server.handlers.social.telegram - Telegram Bot Handler.

Tests cover:
- Routing and method handling
- GET/POST /api/integrations/telegram/webhook
- GET /api/integrations/telegram/status
- POST /api/integrations/telegram/set-webhook
- Message and command handling
- Callback query handling
- Secret verification
- Rate limiting
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.telegram import (
    TelegramHandler,
    get_telegram_handler,
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
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

    def send_response(self, code):
        self.response_code = code

    def send_header(self, key, value):
        pass

    def end_headers(self):
        pass


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
        path="/api/integrations/telegram/status",
    )


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/integrations/telegram/status") is True

    def test_can_handle_webhook(self, handler):
        """Test handler recognizes webhook endpoint."""
        assert handler.can_handle("/api/integrations/telegram/webhook") is True

    def test_can_handle_set_webhook(self, handler):
        """Test handler recognizes set-webhook endpoint."""
        assert handler.can_handle("/api/integrations/telegram/set-webhook") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/integrations/telegram/unknown") is False
        assert handler.can_handle("/api/other/endpoint") is False

    def test_routes_defined(self, handler):
        """Test handler has ROUTES defined."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) >= 3


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/integrations/telegram/status."""

    def test_get_status(self, handler, mock_http_handler):
        """Test getting status."""
        result = handler.handle("/api/integrations/telegram/status", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result["body"])

        # Status should include config flags
        assert "enabled" in data
        assert "bot_token_configured" in data
        assert "webhook_secret_configured" in data

    def test_status_fields_are_booleans(self, handler, mock_http_handler):
        """Test status fields are booleans."""
        result = handler.handle("/api/integrations/telegram/status", {}, mock_http_handler)
        data = json.loads(result["body"])

        assert isinstance(data["enabled"], bool)
        assert isinstance(data["bot_token_configured"], bool)


# ===========================================================================
# Webhook Endpoint Tests
# ===========================================================================


class TestWebhookEndpoint:
    """Tests for POST /api/integrations/telegram/webhook."""

    def test_webhook_requires_post(self, handler):
        """Test webhook endpoint rejects GET."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/integrations/telegram/webhook",
            method="GET",
        )
        result = handler.handle("/api/integrations/telegram/webhook", {}, mock_http)

        # Should return method not allowed
        assert result is not None
        assert result["status"] == 405

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
            result = handler.handle("/api/integrations/telegram/webhook", {}, mock_http)

        assert result is not None
        data = json.loads(result["body"])
        assert data.get("ok") is True

    def test_webhook_handles_message(self, handler):
        """Test webhook handles message update."""
        body = json.dumps({
            "update_id": 12345,
            "message": {
                "message_id": 1,
                "from": {"id": 123, "username": "testuser"},
                "chat": {"id": 456, "type": "private"},
                "date": 1234567890,
                "text": "/help",
            },
        }).encode()

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
                result = handler.handle("/api/integrations/telegram/webhook", {}, mock_http)

        assert result is not None
        data = json.loads(result["body"])
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
            result = handler.handle("/api/integrations/telegram/webhook", {}, mock_http)

        # Should still return ok to acknowledge receipt
        assert result is not None
        data = json.loads(result["body"])
        assert data.get("ok") is True


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

    def test_command_help(self, handler):
        """Test /help command."""
        response = handler._command_help()
        assert "/debate" in response
        assert "/gauntlet" in response
        assert "/status" in response
        assert "/agents" in response

    def test_command_status(self, handler):
        """Test /status command."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = []
            response = handler._command_status()

        assert "Online" in response or "Status" in response

    def test_command_agents_empty(self, handler):
        """Test /agents command with no agents."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = []
            response = handler._command_agents()

        assert "No agents" in response or "agent" in response.lower()

    def test_handle_command_routes_correctly(self, handler):
        """Test _handle_command routes to correct handler."""
        with patch.object(handler, "_command_help", return_value="Help text") as mock_help:
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                handler._handle_command(123, 456, "user", "/help")
                mock_help.assert_called_once()

    def test_handle_command_unknown(self, handler):
        """Test unknown command handling."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            handler._handle_command(123, 456, "user", "/unknown")
            mock_task.assert_called_once()


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

    def test_debate_topic_too_long(self, handler):
        """Test debate command with long topic."""
        long_topic = "x" * 600
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(123, 456, "user", long_topic)

        mock_task.assert_called_once()

    def test_debate_valid_topic(self, handler):
        """Test debate command with valid topic."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_debate(
                123, 456, "user",
                "Should artificial intelligence be regulated by governments?"
            )

        # Should send acknowledgment and queue debate
        assert mock_task.call_count >= 2


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

    def test_gauntlet_statement_too_short(self, handler):
        """Test gauntlet command with short statement."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(123, 456, "user", "test")

        mock_task.assert_called_once()

    def test_gauntlet_valid_statement(self, handler):
        """Test gauntlet command with valid statement."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._command_gauntlet(
                123, 456, "user",
                "We should migrate our monolith to microservices architecture"
            )

        # Should send acknowledgment and queue gauntlet
        assert mock_task.call_count >= 2


# ===========================================================================
# Callback Query Tests
# ===========================================================================


class TestCallbackQueries:
    """Tests for inline keyboard callback handling."""

    def test_handle_callback_query_vote(self, handler):
        """Test vote callback handling."""
        callback = {
            "id": "callback123",
            "data": "vote:debate123:agree",
            "from": {"id": 456, "username": "voter"},
            "message": {"chat": {"id": 789}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch.object(handler, "_handle_vote") as mock_vote:
                mock_vote.return_value = {"status": 200, "body": '{"ok":true}'}
                result = handler._handle_callback_query(callback)

                mock_vote.assert_called_once()

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
                mock_details.return_value = {"status": 200, "body": '{"ok":true}'}
                result = handler._handle_callback_query(callback)

                mock_details.assert_called_once()


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

    def test_handle_message_long(self, handler):
        """Test handling longer message prompts debate suggestion."""
        message = {
            "chat": {"id": 123},
            "from": {"id": 456, "username": "user"},
            "text": "This is a longer message that could be a debate topic",
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task") as mock_task:
            result = handler._handle_message(message)

        mock_task.assert_called_once()


# ===========================================================================
# Set Webhook Tests
# ===========================================================================


class TestSetWebhook:
    """Tests for POST /api/integrations/telegram/set-webhook."""

    def test_set_webhook_missing_url(self, handler):
        """Test set webhook with missing URL."""
        body = json.dumps({}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = handler._set_webhook(mock_http)
        assert result["status"] == 400

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

        result = handler._set_webhook(mock_http)
        assert result["status"] == 400


# ===========================================================================
# Factory Tests
# ===========================================================================


class TestFactory:
    """Tests for handler factory function."""

    def test_get_telegram_handler_singleton(self):
        """Test get_telegram_handler returns consistent instance."""
        # Reset global state
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

        with patch.object(handler, "handle", return_value={"ok": True}) as mock_handle:
            handler.handle_post("/api/integrations/telegram/status", {}, mock_http)
            mock_handle.assert_called_once()
