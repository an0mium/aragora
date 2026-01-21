"""Tests for TTSBridge - Text-to-Speech integration for chat platforms."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from aragora.connectors.chat.tts_bridge import (
    TTSBridge,
    TTSConfig,
    get_tts_bridge,
    clear_tts_bridge,
)


@pytest.fixture
def tts_config():
    """Create a test TTS config."""
    return TTSConfig(
        default_voice="narrator",
        max_text_length=1000,
        cache_enabled=True,
    )


@pytest.fixture
def tts_bridge(tts_config):
    """Create a TTS bridge with test config."""
    clear_tts_bridge()
    return TTSBridge(config=tts_config)


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TTSConfig()
        assert config.default_voice == "narrator"
        assert config.max_text_length == 4000
        assert config.cache_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TTSConfig(
            default_voice="moderator",
            max_text_length=2000,
            cache_enabled=False,
        )
        assert config.default_voice == "moderator"
        assert config.max_text_length == 2000
        assert config.cache_enabled is False

    def test_voice_map_defaults(self):
        """Test default voice mappings."""
        config = TTSConfig()
        assert "narrator" in config.voice_map
        assert "moderator" in config.voice_map
        assert "claude" in config.voice_map
        assert "consensus" in config.voice_map


class TestTTSBridge:
    """Tests for TTSBridge class."""

    def test_init_with_config(self, tts_config):
        """Test initialization with custom config."""
        bridge = TTSBridge(config=tts_config)
        assert bridge.config == tts_config

    def test_init_default_config(self):
        """Test initialization with default config."""
        bridge = TTSBridge()
        assert bridge.config is not None
        assert bridge.config.default_voice == "narrator"

    def test_resolve_voice_explicit(self, tts_bridge):
        """Test voice resolution with explicit voice."""
        voice = tts_bridge._resolve_voice("moderator")
        assert voice == "moderator"

    def test_resolve_voice_from_context(self, tts_bridge):
        """Test voice resolution from context."""
        voice = tts_bridge._resolve_voice(None, context="claude")
        assert voice == "analyst"  # claude maps to analyst

    def test_resolve_voice_default(self, tts_bridge):
        """Test voice resolution falls back to default."""
        voice = tts_bridge._resolve_voice(None, context=None)
        assert voice == "narrator"

    @pytest.mark.asyncio
    async def test_synthesize_truncates_long_text(self, tts_bridge):
        """Test that long text is truncated."""
        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value="/tmp/test.mp3")
        tts_bridge._backend = mock_backend

        long_text = "x" * 2000  # Longer than max_text_length (1000)
        await tts_bridge.synthesize(long_text)

        # Check that the text was truncated
        call_args = mock_backend.synthesize.call_args
        synthesized_text = call_args.kwargs.get("text") or call_args.args[0]
        assert len(synthesized_text) <= 1000
        assert synthesized_text.endswith("...")

    @pytest.mark.asyncio
    async def test_synthesize_debate_summary_consensus(self, tts_bridge):
        """Test synthesizing debate summary with consensus."""
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value="/tmp/summary.mp3")
        tts_bridge._backend = mock_backend

        await tts_bridge.synthesize_debate_summary(
            task="What is the meaning of life?",
            final_answer="42",
            consensus_reached=True,
            confidence=0.95,
            rounds_used=3,
        )

        # Verify the call was made
        assert mock_backend.synthesize.called
        call_args = mock_backend.synthesize.call_args
        text = call_args.kwargs.get("text") or call_args.args[0]
        assert "consensus" in text.lower()
        assert "95%" in text

    @pytest.mark.asyncio
    async def test_synthesize_debate_summary_no_consensus(self, tts_bridge):
        """Test synthesizing debate summary without consensus."""
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value="/tmp/summary.mp3")
        tts_bridge._backend = mock_backend

        await tts_bridge.synthesize_debate_summary(
            task="Complex question",
            final_answer=None,
            consensus_reached=False,
            confidence=0.4,
            rounds_used=5,
        )

        call_args = mock_backend.synthesize.call_args
        text = call_args.kwargs.get("text") or call_args.args[0]
        assert "no consensus" in text.lower()

    @pytest.mark.asyncio
    async def test_synthesize_consensus_alert(self, tts_bridge):
        """Test synthesizing consensus alert."""
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value="/tmp/alert.mp3")
        tts_bridge._backend = mock_backend

        await tts_bridge.synthesize_consensus_alert(
            answer="The answer is 42",
            confidence=0.9,
        )

        call_args = mock_backend.synthesize.call_args
        text = call_args.kwargs.get("text") or call_args.args[0]
        assert "90%" in text
        assert "42" in text

    @pytest.mark.asyncio
    async def test_synthesize_error_alert(self, tts_bridge):
        """Test synthesizing error alert."""
        mock_backend = MagicMock()
        mock_backend.synthesize = AsyncMock(return_value="/tmp/error.mp3")
        tts_bridge._backend = mock_backend

        await tts_bridge.synthesize_error_alert(
            error_type="Timeout",
            error_message="Agent did not respond in time",
        )

        call_args = mock_backend.synthesize.call_args
        text = call_args.kwargs.get("text") or call_args.args[0]
        assert "timeout" in text.lower()


class TestTTSBridgeSingleton:
    """Tests for TTS bridge singleton functions."""

    def test_get_tts_bridge_returns_singleton(self):
        """Test that get_tts_bridge returns the same instance."""
        clear_tts_bridge()
        bridge1 = get_tts_bridge()
        bridge2 = get_tts_bridge()
        assert bridge1 is bridge2

    def test_clear_tts_bridge_resets_singleton(self):
        """Test that clear_tts_bridge resets the singleton."""
        clear_tts_bridge()
        bridge1 = get_tts_bridge()
        clear_tts_bridge()
        bridge2 = get_tts_bridge()
        assert bridge1 is not bridge2


class TestTTSBridgeIntegration:
    """Integration tests for TTS bridge."""

    def test_is_available_without_backend(self):
        """Test is_available when backend unavailable."""
        from aragora.exceptions import ConfigurationError

        clear_tts_bridge()
        try:
            bridge = TTSBridge()
            # Don't mock - let it try to load the real backend
            # This tests graceful handling when TTS backends aren't installed
            # The result depends on whether broadcast module is available
            result = bridge.is_available
            assert isinstance(result, bool)
        except ConfigurationError:
            # ConfigurationError is raised when no TTS backends are available
            # This is expected behavior in CI environments without TTS packages
            pass
