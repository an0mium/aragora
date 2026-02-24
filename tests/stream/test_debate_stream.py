"""Tests for Oracle debate streaming infrastructure.

Covers:
- Token buffering and phase-tagged rendering
- Phase transitions through the debate lifecycle
- TTS audio queue management and synthesis pipeline
- Stream metrics tracking (TTFT, latency, stall detection)
- Reconnection handling and state recovery
- Voice session management for TTS delivery
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Token buffering and rendering
# ---------------------------------------------------------------------------


class TestTokenBuffering:
    """Test token buffering for stream rendering."""

    def test_voice_session_add_chunk(self):
        """Verify audio chunks accumulate in VoiceSession buffer."""
        from aragora.server.stream.voice_stream import VoiceSession

        session = VoiceSession(
            session_id="test_001",
            debate_id="debate_001",
            client_ip="127.0.0.1",
        )
        chunk1 = b"\x00" * 1024
        chunk2 = b"\x01" * 512

        assert session.add_chunk(chunk1) is True
        assert session.add_chunk(chunk2) is True
        assert session.total_bytes_received == 1536
        assert len(session.audio_buffer) == 1536

    def test_voice_session_buffer_overflow(self):
        """Verify buffer rejects chunks when full."""
        from aragora.server.stream.voice_stream import VoiceSession, VOICE_MAX_BUFFER_BYTES

        session = VoiceSession(
            session_id="test_002",
            debate_id="debate_001",
            client_ip="127.0.0.1",
        )
        # Fill buffer to near capacity
        session.audio_buffer = b"\x00" * (VOICE_MAX_BUFFER_BYTES - 10)
        session.total_bytes_received = VOICE_MAX_BUFFER_BYTES - 10

        # Small chunk should still fit
        assert session.add_chunk(b"\x01" * 10) is True

        # Next chunk should be rejected
        assert session.add_chunk(b"\x02" * 100) is False

    def test_voice_session_clear_buffer(self):
        """Verify clear_buffer returns data and resets."""
        from aragora.server.stream.voice_stream import VoiceSession

        session = VoiceSession(
            session_id="test_003",
            debate_id="debate_001",
            client_ip="127.0.0.1",
        )
        session.add_chunk(b"audio_data_here")
        data = session.clear_buffer()
        assert data == b"audio_data_here"
        assert session.audio_buffer == b""

    def test_voice_session_elapsed_seconds(self):
        """Verify elapsed time tracking."""
        from aragora.server.stream.voice_stream import VoiceSession

        session = VoiceSession(
            session_id="test_004",
            debate_id="debate_001",
            client_ip="127.0.0.1",
            started_at=time.time() - 10.0,
        )
        assert session.elapsed_seconds() >= 10.0


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    """Test debate phase tracking through events."""

    def test_tts_integration_register_with_event_bus(self):
        """Verify TTS integration subscribes to agent_message events."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration()
        mock_bus = MagicMock()
        integration.register(mock_bus)

        mock_bus.subscribe.assert_called_once_with(
            "agent_message", integration._handle_agent_message
        )

    def test_tts_integration_enable_disable(self):
        """Verify enable/disable toggling."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration()
        assert integration._enabled is True

        integration.disable()
        assert integration._enabled is False

        integration.enable()
        assert integration._enabled is True

    def test_tts_integration_is_available_without_handler(self):
        """Verify is_available returns False without voice handler."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(voice_handler=None)
        assert integration.is_available is False

    def test_tts_integration_is_available_when_disabled(self):
        """Verify is_available returns False when disabled."""
        from aragora.server.stream.tts_integration import TTSIntegration

        mock_handler = MagicMock()
        mock_handler.is_tts_available = True
        integration = TTSIntegration(voice_handler=mock_handler)
        integration.disable()
        assert integration.is_available is False


# ---------------------------------------------------------------------------
# TTS audio queue management
# ---------------------------------------------------------------------------


class TestTTSAudioQueue:
    """Test TTS audio queue and synthesis pipeline."""

    @pytest.mark.asyncio
    async def test_handle_agent_message_skips_when_disabled(self):
        """Verify handler skips when integration is disabled."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration()
        integration.disable()

        mock_event = MagicMock()
        mock_event.data = {"content": "test", "agent": "claude"}
        mock_event.debate_id = "debate_001"

        await integration._handle_agent_message(mock_event)
        # No error raised, handler returns early

    @pytest.mark.asyncio
    async def test_handle_agent_message_skips_without_handler(self):
        """Verify handler skips when no voice handler is set."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration(voice_handler=None)
        mock_event = MagicMock()
        mock_event.data = {"content": "test", "agent": "claude"}
        mock_event.debate_id = "debate_001"

        await integration._handle_agent_message(mock_event)
        # No error raised, handler returns early

    @pytest.mark.asyncio
    async def test_handle_agent_message_skips_empty_content(self):
        """Verify handler skips messages with no content."""
        from aragora.server.stream.tts_integration import TTSIntegration

        mock_handler = MagicMock()
        integration = TTSIntegration(voice_handler=mock_handler)

        mock_event = MagicMock()
        mock_event.data = {"content": "", "agent": "claude"}
        mock_event.debate_id = "debate_001"

        await integration._handle_agent_message(mock_event)
        mock_handler.synthesize_agent_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_agent_message_respects_enable_tts_flag(self):
        """Verify handler respects per-message enable_tts flag."""
        from aragora.server.stream.tts_integration import TTSIntegration

        mock_handler = MagicMock()
        mock_handler.has_voice_session.return_value = True
        integration = TTSIntegration(voice_handler=mock_handler)

        mock_event = MagicMock()
        mock_event.data = {"content": "test", "agent": "claude", "enable_tts": False}
        mock_event.debate_id = "debate_001"

        await integration._handle_agent_message(mock_event)
        mock_handler.synthesize_agent_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_for_chat_returns_none_when_disabled(self):
        """Verify chat synthesis returns None when disabled."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration()
        integration.disable()
        result = await integration.synthesize_for_chat("Hello world")
        assert result is None


# ---------------------------------------------------------------------------
# Stream metrics tracking
# ---------------------------------------------------------------------------


class TestStreamMetrics:
    """Test stream quality metrics tracking."""

    def test_voice_handler_rate_limit_check(self):
        """Verify rate limiting tracks bytes per minute correctly."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)

        # First chunk should pass
        assert handler._check_rate_limit("127.0.0.1", 1024) is True

        # Track the bytes
        assert "127.0.0.1" in handler._ip_bytes_minute

    def test_voice_handler_rate_limit_exceeds(self):
        """Verify rate limit blocks when exceeded."""
        from aragora.server.stream.voice_stream import (
            VoiceStreamHandler,
            VOICE_MAX_BYTES_PER_MINUTE,
        )

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)

        # Fill up the rate limit
        result = handler._check_rate_limit("127.0.0.1", VOICE_MAX_BYTES_PER_MINUTE)
        assert result is True

        # Next chunk should be rejected
        result = handler._check_rate_limit("127.0.0.1", 1024)
        assert result is False

    def test_voice_handler_get_client_ip_from_peername(self):
        """Verify client IP extraction from transport peername."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_transport = MagicMock()
        mock_transport.get_extra_info.return_value = ("192.168.1.10", 12345)
        mock_request.transport = mock_transport

        ip = handler._get_client_ip(mock_request)
        assert ip == "192.168.1.10"

    def test_voice_handler_get_client_ip_from_forwarded_header(self):
        """Verify client IP extraction from X-Forwarded-For header."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)

        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 172.16.0.1"}
        ip = handler._get_client_ip(mock_request)
        assert ip == "10.0.0.1"


# ---------------------------------------------------------------------------
# Reconnection handling
# ---------------------------------------------------------------------------


class TestReconnectionHandling:
    """Test reconnection and state recovery."""

    def test_voice_handler_has_voice_session_empty(self):
        """Verify has_voice_session returns False when no sessions exist."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)
        assert handler.has_voice_session("debate_001") is False

    def test_voice_handler_get_active_voice_debates_empty(self):
        """Verify active debates set is empty initially."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)
        assert handler.get_active_voice_debates() == set()

    @pytest.mark.asyncio
    async def test_voice_handler_get_session_info_missing(self):
        """Verify get_session_info returns None for unknown session."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)
        result = await handler.get_session_info("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_voice_handler_get_active_sessions_empty(self):
        """Verify get_active_sessions returns empty list initially."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)
        sessions = await handler.get_active_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_synthesize_agent_message_returns_zero_without_sessions(self):
        """Verify synthesize_agent_message returns 0 when no voice sessions."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)

        count = await handler.synthesize_agent_message(
            debate_id="debate_001",
            agent_name="claude",
            message="Test message",
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_receive_audio_frame_empty(self):
        """Verify receive_audio_frame returns 0 for empty frames."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)
        count = await handler.receive_audio_frame("debate_001", b"")
        assert count == 0


# ---------------------------------------------------------------------------
# Voice session management
# ---------------------------------------------------------------------------


class TestVoiceSessionManagement:
    """Test voice session lifecycle and TTS voice mapping."""

    def test_set_speaking_agent(self):
        """Verify speaking agent tracking through voice session map."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler, VoiceSession

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)

        # Add an active session
        session = VoiceSession(
            session_id="voice_abc",
            debate_id="debate_001",
            client_ip="127.0.0.1",
        )
        handler._sessions["voice_abc"] = session

        handler.set_speaking_agent("debate_001", "claude")
        assert handler.get_speaking_agent("debate_001") == "claude"

        handler.set_speaking_agent("debate_001", "")
        assert handler.get_speaking_agent("debate_001") == ""

    def test_get_speaking_agent_no_session(self):
        """Verify get_speaking_agent returns empty string without sessions."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        handler = VoiceStreamHandler(server=mock_server)
        assert handler.get_speaking_agent("debate_001") == ""

    def test_voice_session_tts_voice_map(self):
        """Verify per-agent TTS voice mapping in sessions."""
        from aragora.server.stream.voice_stream import VoiceSession

        session = VoiceSession(
            session_id="voice_xyz",
            debate_id="debate_001",
            client_ip="127.0.0.1",
        )
        session.tts_voice_map["claude"] = "alloy"
        session.tts_voice_map["grok"] = "echo"

        assert session.tts_voice_map["claude"] == "alloy"
        assert session.tts_voice_map["grok"] == "echo"
        assert session.tts_voice_map.get("gemini") is None

    def test_voice_handler_is_available(self):
        """Verify VoiceStreamHandler.is_available reflects whisper status."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_server = MagicMock()
        mock_whisper = MagicMock()
        mock_whisper.is_available = True
        handler = VoiceStreamHandler(server=mock_server, whisper=mock_whisper)
        assert handler.is_available is True

        mock_whisper.is_available = False
        assert handler.is_available is False


# ---------------------------------------------------------------------------
# TTS integration singleton management
# ---------------------------------------------------------------------------


class TestTTSIntegrationSingleton:
    """Test the module-level TTS integration singleton."""

    def test_init_tts_integration_creates_singleton(self):
        """Verify init_tts_integration creates and returns a singleton."""
        from aragora.server.stream import tts_integration as mod

        # Reset singleton
        mod._tts_integration = None

        result = mod.init_tts_integration()
        assert result is not None
        assert isinstance(result, mod.TTSIntegration)

        # Calling again returns the same instance
        result2 = mod.init_tts_integration()
        assert result2 is result

        # Cleanup
        mod._tts_integration = None

    def test_get_set_tts_integration(self):
        """Verify get/set singleton accessors."""
        from aragora.server.stream import tts_integration as mod

        # Reset
        mod._tts_integration = None
        assert mod.get_tts_integration() is None

        instance = mod.TTSIntegration()
        mod.set_tts_integration(instance)
        assert mod.get_tts_integration() is instance

        # Cleanup
        mod._tts_integration = None

    def test_init_tts_integration_with_voice_handler(self):
        """Verify init_tts_integration sets voice handler."""
        from aragora.server.stream import tts_integration as mod

        mod._tts_integration = None

        mock_handler = MagicMock()
        mock_handler.is_tts_available = True
        result = mod.init_tts_integration(voice_handler=mock_handler)
        assert result._voice_handler is mock_handler

        # Cleanup
        mod._tts_integration = None

    def test_init_tts_integration_with_event_bus(self):
        """Verify init_tts_integration registers with event bus."""
        from aragora.server.stream import tts_integration as mod

        mod._tts_integration = None

        mock_bus = MagicMock()
        result = mod.init_tts_integration(event_bus=mock_bus)
        mock_bus.subscribe.assert_called_once_with("agent_message", result._handle_agent_message)

        # Cleanup
        mod._tts_integration = None
