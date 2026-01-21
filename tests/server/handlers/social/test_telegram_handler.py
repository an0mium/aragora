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


def get_body(result):
    """Extract body from handler result (dict or HandlerResult dataclass)."""
    if hasattr(result, "body"):
        return result.body
    return result.get("body", b"")


def get_status_code(result):
    """Extract status code from handler result."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result.get("status", result.get("status_code", 200))


class TestStatusEndpoint:
    """Tests for GET /api/integrations/telegram/status."""

    def test_get_status(self, handler, mock_http_handler):
        """Test getting status."""
        result = handler.handle("/api/integrations/telegram/status", {}, mock_http_handler)

        assert result is not None
        data = json.loads(get_body(result))

        # Status should include config flags
        assert "enabled" in data
        assert "bot_token_configured" in data
        assert "webhook_secret_configured" in data

    def test_status_fields_are_booleans(self, handler, mock_http_handler):
        """Test status fields are booleans."""
        result = handler.handle("/api/integrations/telegram/status", {}, mock_http_handler)
        data = json.loads(get_body(result))

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
        assert get_status_code(result) == 405

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
        data = json.loads(get_body(result))
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
        data = json.loads(get_body(result))
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
        data = json.loads(get_body(result))
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


# ===========================================================================
# Voice Message Tests
# ===========================================================================


class TestVoiceMessages:
    """Tests for voice message functionality."""

    @pytest.mark.asyncio
    async def test_send_voice_async_success(self, handler):
        """Test successful voice message sending."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"ok": True})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
                await handler._send_voice_async(123, b"audio_data")

            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_voice_async_api_error(self, handler):
        """Test voice message sending handles API errors."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={"ok": False, "description": "Bad Request"})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
                # Should not raise, just log error
                await handler._send_voice_async(123, b"audio_data")

    @pytest.mark.asyncio
    async def test_send_voice_async_network_error(self, handler):
        """Test voice message sending handles network errors."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=Exception("Network error"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
                # Should not raise, just log error
                await handler._send_voice_async(123, b"audio_data")

    @pytest.mark.asyncio
    async def test_send_voice_summary_with_tts(self, handler):
        """Test voice summary sends audio when TTS available."""
        mock_tts = MagicMock()
        mock_tts.is_available = True
        mock_result = MagicMock()
        mock_result.audio_bytes = b"synthesized_audio"
        mock_tts.synthesize_debate_result = AsyncMock(return_value=mock_result)

        with patch("aragora.server.handlers.social.telegram.get_tts_helper", return_value=mock_tts):
            with patch.object(handler, "_send_voice_async", new_callable=AsyncMock) as mock_send:
                await handler._send_voice_summary(
                    chat_id=123,
                    task="Test debate",
                    final_answer="The answer",
                    consensus_reached=True,
                    confidence=0.85,
                    rounds_used=3,
                )

                mock_tts.synthesize_debate_result.assert_called_once()
                mock_send.assert_called_once_with(123, b"synthesized_audio")

    @pytest.mark.asyncio
    async def test_send_voice_summary_tts_unavailable(self, handler):
        """Test voice summary skips when TTS unavailable."""
        mock_tts = MagicMock()
        mock_tts.is_available = False

        with patch("aragora.server.handlers.social.telegram.get_tts_helper", return_value=mock_tts):
            with patch.object(handler, "_send_voice_async", new_callable=AsyncMock) as mock_send:
                await handler._send_voice_summary(
                    chat_id=123,
                    task="Test debate",
                    final_answer="The answer",
                    consensus_reached=True,
                    confidence=0.85,
                    rounds_used=3,
                )

                mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_voice_summary_synthesis_fails(self, handler):
        """Test voice summary handles synthesis failure."""
        mock_tts = MagicMock()
        mock_tts.is_available = True
        mock_tts.synthesize_debate_result = AsyncMock(return_value=None)

        with patch("aragora.server.handlers.social.telegram.get_tts_helper", return_value=mock_tts):
            with patch.object(handler, "_send_voice_async", new_callable=AsyncMock) as mock_send:
                await handler._send_voice_summary(
                    chat_id=123,
                    task="Test debate",
                    final_answer="The answer",
                    consensus_reached=True,
                    confidence=0.85,
                    rounds_used=3,
                )

                mock_send.assert_not_called()


# ===========================================================================
# Debate Async Tests
# ===========================================================================


class TestDebateAsync:
    """Tests for async debate execution."""

    @pytest.mark.asyncio
    async def test_run_debate_async_emits_start_event(self, handler):
        """Test debate emits start event."""
        with patch("aragora.server.handlers.social.telegram.Arena") as mock_arena_class:
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=MagicMock(
                final_answer="Test answer",
                consensus_reached=True,
                confidence=0.9,
                rounds_used=2,
            ))
            mock_arena_class.return_value = mock_arena

            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_debate_started") as mock_emit:
                    with patch.object(handler, "_send_message_async", new_callable=AsyncMock):
                        with patch.object(handler, "_send_voice_summary", new_callable=AsyncMock):
                            await handler._run_debate_async(
                                chat_id=123,
                                user_id=456,
                                username="testuser",
                                topic="Test topic",
                            )

                            mock_emit.assert_called_once()
                            call_kwargs = mock_emit.call_args[1]
                            assert call_kwargs["platform"] == "telegram"
                            assert call_kwargs["chat_id"] == "123"
                            assert call_kwargs["topic"] == "Test topic"

    @pytest.mark.asyncio
    async def test_run_debate_async_emits_complete_event(self, handler):
        """Test debate emits completion event."""
        with patch("aragora.server.handlers.social.telegram.Arena") as mock_arena_class:
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=MagicMock(
                final_answer="Test answer",
                consensus_reached=True,
                confidence=0.9,
                rounds_used=2,
            ))
            mock_arena_class.return_value = mock_arena

            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_debate_completed") as mock_emit:
                    with patch.object(handler, "_send_message_async", new_callable=AsyncMock):
                        with patch.object(handler, "_send_voice_summary", new_callable=AsyncMock):
                            await handler._run_debate_async(
                                chat_id=123,
                                user_id=456,
                                username="testuser",
                                topic="Test topic",
                            )

                            mock_emit.assert_called_once()
                            call_kwargs = mock_emit.call_args[1]
                            assert call_kwargs["consensus_reached"] is True
                            assert call_kwargs["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_run_debate_async_handles_arena_error(self, handler):
        """Test debate handles Arena errors gracefully."""
        with patch("aragora.server.handlers.social.telegram.Arena") as mock_arena_class:
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(side_effect=Exception("Arena error"))
            mock_arena_class.return_value = mock_arena

            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch.object(handler, "_send_message_async", new_callable=AsyncMock) as mock_send:
                    await handler._run_debate_async(
                        chat_id=123,
                        user_id=456,
                        username="testuser",
                        topic="Test topic",
                    )

                    # Should send error message
                    error_call = [c for c in mock_send.call_args_list if "error" in str(c).lower() or "failed" in str(c).lower()]
                    assert len(error_call) >= 1 or mock_send.call_count >= 1


# ===========================================================================
# Gauntlet Async Tests
# ===========================================================================


class TestGauntletAsync:
    """Tests for async gauntlet execution."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_emits_start_event(self, handler):
        """Test gauntlet emits start event."""
        with patch("aragora.server.handlers.social.telegram.Gauntlet") as mock_gauntlet_class:
            mock_gauntlet = MagicMock()
            mock_gauntlet.run = AsyncMock(return_value=MagicMock(
                passed=True,
                score=0.85,
                vulnerabilities=[],
            ))
            mock_gauntlet_class.return_value = mock_gauntlet

            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_gauntlet_started") as mock_emit:
                    with patch.object(handler, "_send_message_async", new_callable=AsyncMock):
                        await handler._run_gauntlet_async(
                            chat_id=123,
                            user_id=456,
                            username="testuser",
                            statement="Test statement",
                        )

                        mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_emits_complete_event(self, handler):
        """Test gauntlet emits completion event."""
        with patch("aragora.server.handlers.social.telegram.Gauntlet") as mock_gauntlet_class:
            mock_gauntlet = MagicMock()
            mock_gauntlet.run = AsyncMock(return_value=MagicMock(
                passed=False,
                score=0.45,
                vulnerabilities=["v1", "v2"],
            ))
            mock_gauntlet_class.return_value = mock_gauntlet

            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_gauntlet_completed") as mock_emit:
                    with patch.object(handler, "_send_message_async", new_callable=AsyncMock):
                        await handler._run_gauntlet_async(
                            chat_id=123,
                            user_id=456,
                            username="testuser",
                            statement="Test statement",
                        )

                        mock_emit.assert_called_once()
                        call_kwargs = mock_emit.call_args[1]
                        assert call_kwargs["passed"] is False
                        assert call_kwargs["vulnerability_count"] == 2


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_send_message_async_timeout(self, handler):
        """Test message sending handles timeout."""
        import asyncio

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
                # Should not raise
                await handler._send_message_async(123, "Test message")

    @pytest.mark.asyncio
    async def test_send_message_async_rate_limit(self, handler):
        """Test message sending handles rate limiting."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.json = AsyncMock(return_value={
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests",
                "parameters": {"retry_after": 30},
            })

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.telegram.TELEGRAM_BOT_TOKEN", "test_token"):
                # Should not raise
                await handler._send_message_async(123, "Test message")

    def test_webhook_handles_malformed_callback_data(self, handler):
        """Test callback query handles malformed data."""
        callback = {
            "id": "callback123",
            "data": "malformed_data_without_proper_format",
            "from": {"id": 456, "username": "user"},
            "message": {"chat": {"id": 789}},
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            # Should not raise
            result = handler._handle_callback_query(callback)
            assert result is not None

    def test_handle_message_missing_fields(self, handler):
        """Test message handling with missing fields."""
        message = {"chat": {"id": 123}}  # Missing 'from' and 'text'

        # Should not raise
        result = handler._handle_message(message)
        assert result is not None


# ===========================================================================
# Chat Event Integration Tests
# ===========================================================================


class TestChatEvents:
    """Tests for chat event emissions."""

    def test_handle_message_emits_event(self, handler):
        """Test message handling emits event."""
        message = {
            "chat": {"id": 123},
            "from": {"id": 456, "username": "testuser"},
            "text": "Hello there!",
        }

        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch("aragora.server.handlers.social.chat_events.emit_message_received") as mock_emit:
                handler._handle_message(message)

                mock_emit.assert_called_once()
                call_kwargs = mock_emit.call_args[1]
                assert call_kwargs["platform"] == "telegram"
                assert call_kwargs["text"] == "Hello there!"

    def test_handle_command_emits_event(self, handler):
        """Test command handling emits event."""
        with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
            with patch("aragora.server.handlers.social.chat_events.emit_command_received") as mock_emit:
                handler._handle_command(123, 456, "testuser", "/help")

                mock_emit.assert_called_once()
                call_kwargs = mock_emit.call_args[1]
                assert call_kwargs["command"] == "/help"

    def test_handle_vote_emits_event(self, handler):
        """Test vote handling emits event."""
        with patch("aragora.server.storage.get_debates_db") as mock_db:
            mock_db.return_value = MagicMock()
            with patch("aragora.server.handlers.social.telegram.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_vote_received") as mock_emit:
                    handler._handle_vote(123, 456, "testuser", "debate123", "agree")

                    mock_emit.assert_called_once()
                    call_kwargs = mock_emit.call_args[1]
                    assert call_kwargs["debate_id"] == "debate123"
                    assert call_kwargs["vote"] == "agree"
