"""Tests for debate origin voice synthesis.

Tests cover:
1. Voice synthesis from debate results
2. Handling TTS bridge unavailability
3. Error handling during synthesis
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.debate_origin.voice import _synthesize_voice
from aragora.server.debate_origin.models import DebateOrigin


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-voice-123",
        platform="telegram",
        channel_id="chat-456",
        user_id="user-789",
        metadata={"topic": "Voice Test"},
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "consensus_reached": True,
        "final_answer": "The team agrees on the recommended approach.",
        "confidence": 0.90,
        "participants": ["claude", "gpt-4"],
        "task": "Test voice synthesis",
    }


# =============================================================================
# Test: Voice Synthesis
# =============================================================================


class TestSynthesizeVoice:
    """Tests for _synthesize_voice function."""

    @pytest.mark.asyncio
    async def test_synthesizes_voice_successfully(self, sample_origin, sample_result):
        """_synthesize_voice returns audio path on success."""
        mock_bridge = MagicMock()
        mock_bridge.synthesize_response = AsyncMock(return_value="/tmp/audio.mp3")

        # Patch at the import location inside the function
        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": MagicMock(get_tts_bridge=lambda: mock_bridge)},
        ):
            result = await _synthesize_voice(sample_result, sample_origin)

        assert result == "/tmp/audio.mp3"
        mock_bridge.synthesize_response.assert_called_once()

        # Verify the voice text includes key information
        call_args = mock_bridge.synthesize_response.call_args
        voice_text = call_args[0][0]
        assert "Debate complete" in voice_text
        assert "Consensus was reached" in voice_text
        assert "90%" in voice_text
        assert "team agrees" in voice_text

    @pytest.mark.asyncio
    async def test_includes_no_consensus_in_voice(self, sample_origin, sample_result):
        """_synthesize_voice indicates when consensus not reached."""
        sample_result["consensus_reached"] = False
        mock_bridge = MagicMock()
        mock_bridge.synthesize_response = AsyncMock(return_value="/tmp/no_consensus.mp3")

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": MagicMock(get_tts_bridge=lambda: mock_bridge)},
        ):
            await _synthesize_voice(sample_result, sample_origin)

        call_args = mock_bridge.synthesize_response.call_args
        voice_text = call_args[0][0]
        assert "Consensus was not reached" in voice_text

    @pytest.mark.asyncio
    async def test_truncates_long_answer_for_voice(self, sample_origin, sample_result):
        """_synthesize_voice truncates answers over 300 chars."""
        sample_result["final_answer"] = "A" * 500
        mock_bridge = MagicMock()
        mock_bridge.synthesize_response = AsyncMock(return_value="/tmp/truncated.mp3")

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": MagicMock(get_tts_bridge=lambda: mock_bridge)},
        ):
            await _synthesize_voice(sample_result, sample_origin)

        call_args = mock_bridge.synthesize_response.call_args
        voice_text = call_args[0][0]

        # Should have truncated and added suffix
        assert "A" * 300 in voice_text
        assert "A" * 301 not in voice_text
        assert "See full text for details" in voice_text

    @pytest.mark.asyncio
    async def test_uses_consensus_voice(self, sample_origin, sample_result):
        """_synthesize_voice uses 'consensus' voice setting."""
        mock_bridge = MagicMock()
        mock_bridge.synthesize_response = AsyncMock(return_value="/tmp/voice.mp3")

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": MagicMock(get_tts_bridge=lambda: mock_bridge)},
        ):
            await _synthesize_voice(sample_result, sample_origin)

        call_args = mock_bridge.synthesize_response.call_args
        assert call_args[1]["voice"] == "consensus"

    @pytest.mark.asyncio
    async def test_returns_none_when_bridge_unavailable(self, sample_origin, sample_result):
        """_synthesize_voice returns None when TTS bridge not available."""
        # Remove the module from sys.modules to trigger ImportError
        import sys

        # Ensure the module isn't cached
        modules_to_remove = [k for k in sys.modules if "tts_bridge" in k]
        original_modules = {k: sys.modules[k] for k in modules_to_remove if k in sys.modules}

        for k in modules_to_remove:
            sys.modules.pop(k, None)

        # Create a mock that raises ImportError
        def raise_import_error():
            raise ImportError("No TTS bridge")

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": None},
        ):
            # Patch builtins.__import__ to raise ImportError for tts_bridge
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if "tts_bridge" in name:
                    raise ImportError("No TTS bridge")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = await _synthesize_voice(sample_result, sample_origin)

        # Restore modules
        sys.modules.update(original_modules)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_synthesis_error(self, sample_origin, sample_result):
        """_synthesize_voice returns None when synthesis fails."""
        mock_bridge = MagicMock()
        mock_bridge.synthesize_response = AsyncMock(
            side_effect=RuntimeError("TTS service unavailable")
        )

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": MagicMock(get_tts_bridge=lambda: mock_bridge)},
        ):
            result = await _synthesize_voice(sample_result, sample_origin)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_missing_result_fields(self, sample_origin):
        """_synthesize_voice handles missing result fields gracefully."""
        minimal_result: dict[str, Any] = {}
        mock_bridge = MagicMock()
        mock_bridge.synthesize_response = AsyncMock(return_value="/tmp/minimal.mp3")

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.tts_bridge": MagicMock(get_tts_bridge=lambda: mock_bridge)},
        ):
            result = await _synthesize_voice(minimal_result, sample_origin)

        assert result == "/tmp/minimal.mp3"

        # Verify it still generates reasonable voice text
        call_args = mock_bridge.synthesize_response.call_args
        voice_text = call_args[0][0]
        assert "Debate complete" in voice_text
        assert "0%" in voice_text  # Default confidence
        assert "No conclusion available" in voice_text
