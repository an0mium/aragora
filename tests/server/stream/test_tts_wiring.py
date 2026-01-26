"""
Tests for TTS integration wiring in servers.

Verifies that the TTS integration is properly wired to the voice handler
when servers are created.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTTSWiring:
    """Tests for TTS integration wiring."""

    def test_aiohttp_server_wires_tts_integration(self):
        """Test that AiohttpUnifiedServer wires TTS to voice handler."""
        with (
            patch("aragora.server.stream.servers.VoiceStreamHandler") as mock_voice_handler_class,
            patch("aragora.server.stream.tts_integration.get_tts_integration") as mock_get_tts,
            patch("aragora.server.stream.tts_integration.set_tts_integration") as mock_set_tts,
            patch("aragora.server.stream.tts_integration.TTSIntegration") as mock_tts_class,
        ):
            # Configure mocks
            mock_voice_handler = MagicMock()
            mock_voice_handler_class.return_value = mock_voice_handler

            mock_get_tts.return_value = None  # No existing integration
            mock_tts_integration = MagicMock()
            mock_tts_class.return_value = mock_tts_integration

            # Create server
            from aragora.server.stream.servers import AiohttpUnifiedServer

            server = AiohttpUnifiedServer(port=8080)

            # Verify voice handler was created
            mock_voice_handler_class.assert_called_once()

            # Verify TTS integration was created with voice handler
            mock_tts_class.assert_called_once_with(mock_voice_handler)

            # Verify TTS integration was set as singleton
            mock_set_tts.assert_called_once_with(mock_tts_integration)

    def test_aiohttp_server_wires_to_existing_tts_integration(self):
        """Test that server wires to existing TTS integration if available."""
        with (
            patch("aragora.server.stream.servers.VoiceStreamHandler") as mock_voice_handler_class,
            patch("aragora.server.stream.tts_integration.get_tts_integration") as mock_get_tts,
            patch("aragora.server.stream.tts_integration.TTSIntegration"),
        ):
            # Configure mocks
            mock_voice_handler = MagicMock()
            mock_voice_handler_class.return_value = mock_voice_handler

            # Existing TTS integration
            existing_integration = MagicMock()
            mock_get_tts.return_value = existing_integration

            # Create server
            from aragora.server.stream.servers import AiohttpUnifiedServer

            server = AiohttpUnifiedServer(port=8080)

            # Verify voice handler was wired to existing integration
            existing_integration.set_voice_handler.assert_called_once_with(mock_voice_handler)


class TestTTSIntegrationFlow:
    """Tests for the complete TTS integration flow."""

    @pytest.mark.asyncio
    async def test_tts_integration_handles_agent_message(self):
        """Test that TTS integration handles agent messages correctly."""
        from aragora.server.stream.tts_integration import TTSIntegration

        # Create mock voice handler
        mock_voice_handler = MagicMock()
        mock_voice_handler.is_tts_available = True
        mock_voice_handler.has_voice_session.return_value = True
        mock_voice_handler.synthesize_agent_message = MagicMock(return_value=1)

        integration = TTSIntegration(mock_voice_handler)
        integration.enable()

        # Verify integration is available
        assert integration.is_available is True

        # Create mock event
        mock_event = MagicMock()
        mock_event.debate_id = "debate-123"
        mock_event.data = {
            "agent": "Claude",
            "content": "This is a test message.",
            "enable_tts": True,
        }

        # Handle message should call voice handler
        await integration._handle_agent_message(mock_event)

        # Voice handler should be checked for session
        mock_voice_handler.has_voice_session.assert_called_once_with("debate-123")

    @pytest.mark.asyncio
    async def test_tts_integration_respects_enable_tts_flag(self):
        """Test that TTS integration respects the enable_tts flag."""
        from aragora.server.stream.tts_integration import TTSIntegration

        mock_voice_handler = MagicMock()
        mock_voice_handler.is_tts_available = True
        mock_voice_handler.has_voice_session.return_value = True

        integration = TTSIntegration(mock_voice_handler)

        # Event with enable_tts=False
        mock_event = MagicMock()
        mock_event.debate_id = "debate-123"
        mock_event.data = {
            "agent": "Claude",
            "content": "This should not be spoken.",
            "enable_tts": False,
        }

        await integration._handle_agent_message(mock_event)

        # Voice handler should not be called
        mock_voice_handler.has_voice_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_integration_disabled_does_nothing(self):
        """Test that disabled TTS integration does nothing."""
        from aragora.server.stream.tts_integration import TTSIntegration

        mock_voice_handler = MagicMock()
        mock_voice_handler.is_tts_available = True

        integration = TTSIntegration(mock_voice_handler)
        integration.disable()

        assert integration.is_available is False

        mock_event = MagicMock()
        mock_event.debate_id = "debate-123"
        mock_event.data = {"agent": "Claude", "content": "Test"}

        await integration._handle_agent_message(mock_event)

        # Voice handler should not be called
        mock_voice_handler.has_voice_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_synthesize_for_chat(self):
        """Test TTS synthesis for chat channels."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration()
        integration.enable()

        with patch("aragora.broadcast.tts_backends.get_fallback_backend") as mock_get_backend:
            # Configure mock backend
            mock_backend = MagicMock()
            mock_backend.is_available.return_value = True

            mock_audio_path = MagicMock()
            mock_audio_path.exists.return_value = True
            mock_audio_path.read_bytes.return_value = b"fake_audio_data"
            mock_audio_path.unlink.return_value = None

            mock_backend.synthesize = AsyncMock(return_value=mock_audio_path)
            mock_get_backend.return_value = mock_backend

            # Call synthesize_for_chat
            result = await integration.synthesize_for_chat(
                text="Hello, this is a test.",
                channel_type="telegram",
                channel_id="12345",
            )

            # Should return audio bytes
            assert result == b"fake_audio_data"
            mock_backend.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_tts_synthesize_for_chat_no_backend(self):
        """Test TTS synthesis gracefully handles missing backend."""
        from aragora.server.stream.tts_integration import TTSIntegration

        integration = TTSIntegration()
        integration.enable()

        with patch("aragora.broadcast.tts_backends.get_fallback_backend") as mock_get_backend:
            mock_get_backend.return_value = None

            result = await integration.synthesize_for_chat(
                text="Hello, this is a test.",
                channel_type="telegram",
            )

            assert result is None


class TestInitTTSIntegration:
    """Tests for init_tts_integration function."""

    def test_init_creates_singleton(self):
        """Test that init_tts_integration creates a singleton."""
        import aragora.server.stream.tts_integration as tts_module

        # Reset singleton
        tts_module._tts_integration = None

        from aragora.server.stream.tts_integration import (
            init_tts_integration,
            get_tts_integration,
            TTSIntegration,
        )

        # First call creates
        integration1 = init_tts_integration()
        assert integration1 is not None
        assert isinstance(integration1, TTSIntegration)

        # Second call returns same instance
        integration2 = init_tts_integration()
        assert integration2 is integration1

        # Get returns same instance
        integration3 = get_tts_integration()
        assert integration3 is integration1

        # Clean up
        tts_module._tts_integration = None

    def test_init_with_voice_handler(self):
        """Test init with voice handler wires correctly."""
        import aragora.server.stream.tts_integration as tts_module

        # Reset singleton
        tts_module._tts_integration = None

        from aragora.server.stream.tts_integration import init_tts_integration

        mock_handler = MagicMock()
        mock_handler.is_tts_available = True

        integration = init_tts_integration(voice_handler=mock_handler)

        assert integration._voice_handler is mock_handler
        assert integration.is_available is True

        # Clean up
        tts_module._tts_integration = None
