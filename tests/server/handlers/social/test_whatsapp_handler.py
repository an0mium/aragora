"""
Tests for aragora.server.handlers.social.whatsapp - WhatsApp Business API Handler.

Tests cover:
- Routing and method handling
- GET /api/integrations/whatsapp/webhook (verification)
- POST /api/integrations/whatsapp/webhook (message handling)
- GET /api/integrations/whatsapp/status
- Message handling (text, interactive, button)
- Command handling
- Signature verification
- Rate limiting
"""

from __future__ import annotations

import hashlib
import hmac
import json
from io import BytesIO
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.whatsapp import (
    WHATSAPP_VERIFY_TOKEN,
    WhatsAppHandler,
    get_whatsapp_handler,
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
    """Create a WhatsAppHandler instance."""
    return WhatsAppHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json", "Content-Length": "0"},
        path="/api/integrations/whatsapp/status",
    )


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/integrations/whatsapp/status") is True

    def test_can_handle_webhook(self, handler):
        """Test handler recognizes webhook endpoint."""
        assert handler.can_handle("/api/integrations/whatsapp/webhook") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/integrations/whatsapp/unknown") is False
        assert handler.can_handle("/api/other/endpoint") is False

    def test_routes_defined(self, handler):
        """Test handler has ROUTES defined."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) >= 2


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
    """Tests for GET /api/integrations/whatsapp/status."""

    def test_get_status(self, handler, mock_http_handler):
        """Test getting status."""
        result = handler.handle("/api/integrations/whatsapp/status", {}, mock_http_handler)

        assert result is not None
        data = json.loads(get_body(result))

        # Status should include config flags
        assert "enabled" in data
        assert "access_token_configured" in data
        assert "phone_number_id_configured" in data
        assert "verify_token_configured" in data

    def test_status_fields_are_booleans(self, handler, mock_http_handler):
        """Test status fields are booleans."""
        result = handler.handle("/api/integrations/whatsapp/status", {}, mock_http_handler)
        data = json.loads(get_body(result))

        assert isinstance(data["enabled"], bool)
        assert isinstance(data["access_token_configured"], bool)
        assert isinstance(data["verify_token_configured"], bool)


# ===========================================================================
# Webhook Verification Tests
# ===========================================================================


class TestWebhookVerification:
    """Tests for GET /api/integrations/whatsapp/webhook (Meta verification)."""

    def test_verify_webhook_success(self, handler):
        """Test successful webhook verification."""
        mock_http = MockHandler(
            headers={"Content-Type": "text/plain"},
            path="/api/integrations/whatsapp/webhook",
            method="GET",
        )

        query_params = {
            "hub.mode": "subscribe",
            "hub.verify_token": WHATSAPP_VERIFY_TOKEN,
            "hub.challenge": "challenge_string_123",
        }

        result = handler.handle("/api/integrations/whatsapp/webhook", query_params, mock_http)

        assert result is not None
        assert get_status_code(result) == 200
        body = get_body(result)
        assert body == "challenge_string_123" or body == b"challenge_string_123"

    def test_verify_webhook_wrong_token(self, handler):
        """Test webhook verification with wrong token."""
        mock_http = MockHandler(
            headers={"Content-Type": "text/plain"},
            path="/api/integrations/whatsapp/webhook",
            method="GET",
        )

        query_params = {
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "challenge_string_123",
        }

        result = handler.handle("/api/integrations/whatsapp/webhook", query_params, mock_http)

        assert result is not None
        assert get_status_code(result) in (400, 403)

    def test_verify_webhook_wrong_mode(self, handler):
        """Test webhook verification with wrong mode."""
        mock_http = MockHandler(
            headers={"Content-Type": "text/plain"},
            path="/api/integrations/whatsapp/webhook",
            method="GET",
        )

        query_params = {
            "hub.mode": "unsubscribe",
            "hub.verify_token": WHATSAPP_VERIFY_TOKEN,
            "hub.challenge": "challenge_string_123",
        }

        result = handler.handle("/api/integrations/whatsapp/webhook", query_params, mock_http)

        assert result is not None
        assert get_status_code(result) in (400, 403)


# ===========================================================================
# Webhook POST Tests
# ===========================================================================


class TestWebhookPost:
    """Tests for POST /api/integrations/whatsapp/webhook."""

    def test_webhook_handles_empty_object(self, handler):
        """Test webhook handles empty payload."""
        body = json.dumps({"object": "other"}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = handler.handle("/api/integrations/whatsapp/webhook", {}, mock_http)

        assert result is not None
        data = json.loads(get_body(result))
        assert data.get("status") == "ok"

    def test_webhook_handles_message(self, handler):
        """Test webhook handles incoming message."""
        body = json.dumps({
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "contacts": [{"wa_id": "15551234567", "profile": {"name": "Test User"}}],
                        "messages": [{
                            "from": "15551234567",
                            "type": "text",
                            "text": {"body": "help"},
                        }],
                    },
                    "field": "messages",
                }],
            }],
        }).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
            result = handler.handle("/api/integrations/whatsapp/webhook", {}, mock_http)

        assert result is not None
        data = json.loads(get_body(result))
        assert data.get("status") == "ok"

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

        result = handler.handle("/api/integrations/whatsapp/webhook", {}, mock_http)

        # Should still return ok to acknowledge receipt
        assert result is not None
        data = json.loads(get_body(result))
        assert data.get("status") == "ok"


# ===========================================================================
# Text Message Handling Tests
# ===========================================================================


class TestTextMessageHandling:
    """Tests for text message handling."""

    def test_handle_text_message_help(self, handler):
        """Test handling 'help' command."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._handle_text_message("15551234567", "Test User", "help")

        mock_task.assert_called_once()

    def test_handle_text_message_status(self, handler):
        """Test handling 'status' command."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            with patch("aragora.ranking.elo.EloSystem") as mock_elo:
                mock_elo.return_value.get_all_ratings.return_value = []
                handler._handle_text_message("15551234567", "Test User", "status")

        mock_task.assert_called_once()

    def test_handle_text_message_debate(self, handler):
        """Test handling 'debate' command."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._handle_text_message(
                "15551234567",
                "Test User",
                "debate Should AI be regulated?"
            )

        # Should not send immediate response (debate handler manages it)
        # The _command_debate is called which handles its own messaging

    def test_handle_text_message_short(self, handler):
        """Test handling short message."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._handle_text_message("15551234567", "Test User", "hi")

        mock_task.assert_called_once()

    def test_handle_text_message_long(self, handler):
        """Test handling longer message suggests debate."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._handle_text_message(
                "15551234567",
                "Test User",
                "This is a longer message that could potentially be a debate topic"
            )

        mock_task.assert_called_once()


# ===========================================================================
# Command Tests
# ===========================================================================


class TestCommands:
    """Tests for command handling."""

    def test_command_help(self, handler):
        """Test help command."""
        response = handler._command_help()
        assert "help" in response.lower()
        assert "debate" in response.lower()
        assert "gauntlet" in response.lower()
        assert "status" in response.lower()

    def test_command_status(self, handler):
        """Test status command."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = []
            response = handler._command_status()

        assert "Status" in response or "Online" in response

    def test_command_agents_empty(self, handler):
        """Test agents command with no agents."""
        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = []
            response = handler._command_agents()

        assert "No agents" in response or "agent" in response.lower()


# ===========================================================================
# Debate Command Tests
# ===========================================================================


class TestDebateCommand:
    """Tests for debate command."""

    def test_debate_topic_too_short(self, handler):
        """Test debate with short topic."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._command_debate("15551234567", "User", "test")

        mock_task.assert_called_once()

    def test_debate_topic_too_long(self, handler):
        """Test debate with long topic."""
        long_topic = "x" * 600
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._command_debate("15551234567", "User", long_topic)

        mock_task.assert_called_once()

    def test_debate_valid_topic(self, handler):
        """Test debate with valid topic."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._command_debate(
                "15551234567",
                "User",
                "Should artificial intelligence be regulated by governments?"
            )

        # Should send acknowledgment and queue debate
        assert mock_task.call_count >= 2


# ===========================================================================
# Gauntlet Command Tests
# ===========================================================================


class TestGauntletCommand:
    """Tests for gauntlet command."""

    def test_gauntlet_statement_too_short(self, handler):
        """Test gauntlet with short statement."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._command_gauntlet("15551234567", "User", "test")

        mock_task.assert_called_once()

    def test_gauntlet_statement_too_long(self, handler):
        """Test gauntlet with long statement."""
        long_statement = "x" * 1100
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._command_gauntlet("15551234567", "User", long_statement)

        mock_task.assert_called_once()

    def test_gauntlet_valid_statement(self, handler):
        """Test gauntlet with valid statement."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task") as mock_task:
            handler._command_gauntlet(
                "15551234567",
                "User",
                "We should migrate our monolith to microservices architecture"
            )

        # Should send acknowledgment and queue gauntlet
        assert mock_task.call_count >= 2


# ===========================================================================
# Interactive Reply Tests
# ===========================================================================


class TestInteractiveReplies:
    """Tests for interactive message replies."""

    def test_handle_interactive_reply_button(self, handler):
        """Test button reply handling."""
        message = {
            "interactive": {
                "type": "button_reply",
                "button_reply": {
                    "id": "vote_agree_debate123",
                    "title": "Agree",
                },
            },
        }

        with patch.object(handler, "_process_button_click") as mock_process:
            handler._handle_interactive_reply("15551234567", "User", message)
            mock_process.assert_called_once_with(
                "15551234567", "User", "vote_agree_debate123"
            )

    def test_handle_interactive_reply_list(self, handler):
        """Test list reply handling."""
        message = {
            "interactive": {
                "type": "list_reply",
                "list_reply": {
                    "id": "details_debate123",
                    "title": "View Details",
                },
            },
        }

        with patch.object(handler, "_process_button_click") as mock_process:
            handler._handle_interactive_reply("15551234567", "User", message)
            mock_process.assert_called_once_with(
                "15551234567", "User", "details_debate123"
            )


# ===========================================================================
# Button Click Processing Tests
# ===========================================================================


class TestButtonClickProcessing:
    """Tests for button click processing."""

    def test_process_vote_agree(self, handler):
        """Test processing agree vote."""
        with patch.object(handler, "_record_vote") as mock_vote:
            handler._process_button_click("15551234567", "User", "vote_agree_debate123")
            mock_vote.assert_called_once_with("15551234567", "User", "debate123", "agree")

    def test_process_vote_disagree(self, handler):
        """Test processing disagree vote."""
        with patch.object(handler, "_record_vote") as mock_vote:
            handler._process_button_click("15551234567", "User", "vote_disagree_debate123")
            mock_vote.assert_called_once_with("15551234567", "User", "debate123", "disagree")

    def test_process_details(self, handler):
        """Test processing view details."""
        with patch.object(handler, "_send_debate_details") as mock_details:
            handler._process_button_click("15551234567", "User", "details_debate123")
            mock_details.assert_called_once_with("15551234567", "debate123")


# ===========================================================================
# Vote Recording Tests
# ===========================================================================


class TestVoteRecording:
    """Tests for vote recording."""

    def test_record_vote_agree(self, handler):
        """Test recording agree vote."""
        with patch("aragora.server.storage.get_debates_db") as mock_db:
            mock_db.return_value = MagicMock()
            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                handler._record_vote("15551234567", "User", "debate123", "agree")

    def test_record_vote_disagree(self, handler):
        """Test recording disagree vote."""
        with patch("aragora.server.storage.get_debates_db") as mock_db:
            mock_db.return_value = MagicMock()
            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                handler._record_vote("15551234567", "User", "debate123", "disagree")


# ===========================================================================
# Signature Verification Tests
# ===========================================================================


class TestSignatureVerification:
    """Tests for webhook signature verification."""

    def test_verify_signature_no_secret(self, handler):
        """Test verification when no secret configured."""
        mock_http = MockHandler()

        # When WHATSAPP_APP_SECRET is empty, verification should pass
        with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_APP_SECRET", ""):
            result = handler._verify_signature(mock_http)
            assert result is True

    def test_verify_signature_missing_header(self, handler):
        """Test verification with missing signature header."""
        mock_http = MockHandler(headers={})

        with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_APP_SECRET", "secret"):
            result = handler._verify_signature(mock_http)
            assert result is False

    def test_verify_signature_wrong_format(self, handler):
        """Test verification with wrong signature format."""
        mock_http = MockHandler(
            headers={"X-Hub-Signature-256": "wrong_format"},
        )

        with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_APP_SECRET", "secret"):
            result = handler._verify_signature(mock_http)
            assert result is False


# ===========================================================================
# Factory Tests
# ===========================================================================


class TestFactory:
    """Tests for handler factory function."""

    def test_get_whatsapp_handler_singleton(self):
        """Test get_whatsapp_handler returns consistent instance."""
        # Reset global state
        import aragora.server.handlers.social.whatsapp as wa
        wa._whatsapp_handler = None

        handler1 = get_whatsapp_handler({})
        handler2 = get_whatsapp_handler({})

        assert handler1 is handler2

    def test_get_whatsapp_handler_creates_instance(self):
        """Test get_whatsapp_handler creates instance."""
        import aragora.server.handlers.social.whatsapp as wa
        wa._whatsapp_handler = None

        handler = get_whatsapp_handler({})
        assert isinstance(handler, WhatsAppHandler)


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
            handler.handle_post("/api/integrations/whatsapp/status", {}, mock_http)
            mock_handle.assert_called_once()

    def test_full_webhook_flow(self, handler):
        """Test full webhook message flow."""
        # Step 1: Verify webhook
        verify_result = handler._verify_webhook({
            "hub.mode": "subscribe",
            "hub.verify_token": WHATSAPP_VERIFY_TOKEN,
            "hub.challenge": "test_challenge",
        })
        assert verify_result["status"] == 200
        assert verify_result["body"] == "test_challenge"


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
            mock_response.json = AsyncMock(return_value={"messages": [{"id": "wamid123"}]})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    await handler._send_voice_async("15551234567", b"audio_data")

            assert mock_session.post.call_count >= 1

    @pytest.mark.asyncio
    async def test_send_voice_async_api_error(self, handler):
        """Test voice message sending handles API errors."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={"error": {"message": "Bad Request"}})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    # Should not raise, just log error
                    await handler._send_voice_async("15551234567", b"audio_data")

    @pytest.mark.asyncio
    async def test_send_voice_async_network_error(self, handler):
        """Test voice message sending handles network errors."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=Exception("Network error"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    # Should not raise, just log error
                    await handler._send_voice_async("15551234567", b"audio_data")

    @pytest.mark.asyncio
    async def test_send_voice_summary_with_tts(self, handler):
        """Test voice summary sends audio when TTS available."""
        mock_tts = MagicMock()
        mock_tts.is_available = True
        mock_result = MagicMock()
        mock_result.audio_bytes = b"synthesized_audio"
        mock_tts.synthesize_debate_result = AsyncMock(return_value=mock_result)

        with patch("aragora.server.handlers.social.whatsapp.get_tts_helper", return_value=mock_tts):
            with patch.object(handler, "_send_voice_async", new_callable=AsyncMock) as mock_send:
                await handler._send_voice_summary(
                    phone_number="15551234567",
                    task="Test debate",
                    final_answer="The answer",
                    consensus_reached=True,
                    confidence=0.85,
                    rounds_used=3,
                )

                mock_tts.synthesize_debate_result.assert_called_once()
                mock_send.assert_called_once_with("15551234567", b"synthesized_audio")

    @pytest.mark.asyncio
    async def test_send_voice_summary_tts_unavailable(self, handler):
        """Test voice summary skips when TTS unavailable."""
        mock_tts = MagicMock()
        mock_tts.is_available = False

        with patch("aragora.server.handlers.social.whatsapp.get_tts_helper", return_value=mock_tts):
            with patch.object(handler, "_send_voice_async", new_callable=AsyncMock) as mock_send:
                await handler._send_voice_summary(
                    phone_number="15551234567",
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
        with patch("aragora.server.handlers.social.whatsapp.Arena") as mock_arena_class:
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=MagicMock(
                final_answer="Test answer",
                consensus_reached=True,
                confidence=0.9,
                rounds_used=2,
            ))
            mock_arena_class.return_value = mock_arena

            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_debate_started") as mock_emit:
                    with patch.object(handler, "_send_text_message_async", new_callable=AsyncMock):
                        with patch.object(handler, "_send_interactive_buttons_async", new_callable=AsyncMock):
                            with patch.object(handler, "_send_voice_summary", new_callable=AsyncMock):
                                await handler._run_debate_async(
                                    phone_number="15551234567",
                                    username="Test User",
                                    topic="Test topic",
                                )

                                mock_emit.assert_called_once()
                                call_kwargs = mock_emit.call_args[1]
                                assert call_kwargs["platform"] == "whatsapp"
                                assert call_kwargs["topic"] == "Test topic"

    @pytest.mark.asyncio
    async def test_run_debate_async_emits_complete_event(self, handler):
        """Test debate emits completion event."""
        with patch("aragora.server.handlers.social.whatsapp.Arena") as mock_arena_class:
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=MagicMock(
                final_answer="Test answer",
                consensus_reached=True,
                confidence=0.9,
                rounds_used=2,
            ))
            mock_arena_class.return_value = mock_arena

            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_debate_completed") as mock_emit:
                    with patch.object(handler, "_send_text_message_async", new_callable=AsyncMock):
                        with patch.object(handler, "_send_interactive_buttons_async", new_callable=AsyncMock):
                            with patch.object(handler, "_send_voice_summary", new_callable=AsyncMock):
                                await handler._run_debate_async(
                                    phone_number="15551234567",
                                    username="Test User",
                                    topic="Test topic",
                                )

                                mock_emit.assert_called_once()
                                call_kwargs = mock_emit.call_args[1]
                                assert call_kwargs["consensus_reached"] is True

    @pytest.mark.asyncio
    async def test_run_debate_async_handles_arena_error(self, handler):
        """Test debate handles Arena errors gracefully."""
        with patch("aragora.server.handlers.social.whatsapp.Arena") as mock_arena_class:
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(side_effect=Exception("Arena error"))
            mock_arena_class.return_value = mock_arena

            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                with patch.object(handler, "_send_text_message_async", new_callable=AsyncMock) as mock_send:
                    await handler._run_debate_async(
                        phone_number="15551234567",
                        username="Test User",
                        topic="Test topic",
                    )

                    # Should send error message
                    assert mock_send.call_count >= 1


# ===========================================================================
# Gauntlet Async Tests
# ===========================================================================


class TestGauntletAsync:
    """Tests for async gauntlet execution."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_emits_start_event(self, handler):
        """Test gauntlet emits start event."""
        with patch("aragora.server.handlers.social.whatsapp.Gauntlet") as mock_gauntlet_class:
            mock_gauntlet = MagicMock()
            mock_gauntlet.run = AsyncMock(return_value=MagicMock(
                passed=True,
                score=0.85,
                vulnerabilities=[],
            ))
            mock_gauntlet_class.return_value = mock_gauntlet

            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_gauntlet_started") as mock_emit:
                    with patch.object(handler, "_send_text_message_async", new_callable=AsyncMock):
                        await handler._run_gauntlet_async(
                            phone_number="15551234567",
                            username="Test User",
                            statement="Test statement",
                        )

                        mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_gauntlet_async_emits_complete_event(self, handler):
        """Test gauntlet emits completion event."""
        with patch("aragora.server.handlers.social.whatsapp.Gauntlet") as mock_gauntlet_class:
            mock_gauntlet = MagicMock()
            mock_gauntlet.run = AsyncMock(return_value=MagicMock(
                passed=False,
                score=0.45,
                vulnerabilities=["v1", "v2"],
            ))
            mock_gauntlet_class.return_value = mock_gauntlet

            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_gauntlet_completed") as mock_emit:
                    with patch.object(handler, "_send_text_message_async", new_callable=AsyncMock):
                        await handler._run_gauntlet_async(
                            phone_number="15551234567",
                            username="Test User",
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

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    # Should not raise
                    await handler._send_text_message_async("15551234567", "Test message")

    @pytest.mark.asyncio
    async def test_send_message_async_rate_limit(self, handler):
        """Test message sending handles rate limiting."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.json = AsyncMock(return_value={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "OAuthException",
                    "code": 4,
                },
            })

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    # Should not raise
                    await handler._send_text_message_async("15551234567", "Test message")

    def test_handle_message_missing_text(self, handler):
        """Test message handling with missing text field."""
        # Should not raise when text is missing
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
            handler._handle_text_message("15551234567", "User", "")

    def test_handle_interactive_missing_reply(self, handler):
        """Test interactive handling with missing reply data."""
        message = {
            "interactive": {
                "type": "button_reply",
                # Missing button_reply
            },
        }

        # Should not raise
        with patch.object(handler, "_process_button_click"):
            handler._handle_interactive_reply("15551234567", "User", message)


# ===========================================================================
# Interactive Buttons Extended Tests
# ===========================================================================


class TestInteractiveButtonsExtended:
    """Extended tests for interactive button handling."""

    @pytest.mark.asyncio
    async def test_send_interactive_buttons_async_success(self, handler):
        """Test sending interactive buttons."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"messages": [{"id": "wamid123"}]})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    await handler._send_interactive_buttons_async(
                        phone_number="15551234567",
                        body_text="Choose an option:",
                        buttons=[
                            {"id": "btn1", "title": "Option 1"},
                            {"id": "btn2", "title": "Option 2"},
                        ],
                    )

            mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_interactive_buttons_async_error(self, handler):
        """Test interactive buttons handles API error."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={"error": {"message": "Invalid buttons"}})

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_ACCESS_TOKEN", "test_token"):
                with patch("aragora.server.handlers.social.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123456"):
                    # Should not raise
                    await handler._send_interactive_buttons_async(
                        phone_number="15551234567",
                        body_text="Choose:",
                        buttons=[{"id": "btn1", "title": "Option"}],
                    )


# ===========================================================================
# Chat Event Integration Tests
# ===========================================================================


class TestChatEvents:
    """Tests for chat event emissions."""

    def test_handle_text_message_emits_event(self, handler):
        """Test text message handling emits event."""
        with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
            with patch("aragora.server.handlers.social.chat_events.emit_message_received") as mock_emit:
                handler._handle_text_message("15551234567", "Test User", "Hello there!")

                mock_emit.assert_called_once()
                call_kwargs = mock_emit.call_args[1]
                assert call_kwargs["platform"] == "whatsapp"
                assert call_kwargs["text"] == "Hello there!"

    def test_record_vote_emits_event(self, handler):
        """Test vote recording emits event."""
        with patch("aragora.server.storage.get_debates_db") as mock_db:
            mock_db.return_value = MagicMock()
            with patch("aragora.server.handlers.social.whatsapp.create_tracked_task"):
                with patch("aragora.server.handlers.social.chat_events.emit_vote_received") as mock_emit:
                    handler._record_vote("15551234567", "User", "debate123", "agree")

                    mock_emit.assert_called_once()
                    call_kwargs = mock_emit.call_args[1]
                    assert call_kwargs["debate_id"] == "debate123"
                    assert call_kwargs["vote"] == "agree"
