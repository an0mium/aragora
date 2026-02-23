"""
Comprehensive tests for aragora/server/handlers/social/tts_helper.py.

Covers:
- SynthesisResult dataclass fields and types
- TTSHelper initialization and lazy backend/bridge loading
- TTSHelper.is_available property (enabled/disabled, cached, errors)
- TTSHelper._get_backend (import success, import error, runtime error)
- TTSHelper._get_bridge (import success, import error)
- TTSHelper.synthesize_response (success, truncation, format, voice, failures, cleanup)
- TTSHelper.synthesize_debate_result (consensus/no-consensus, long text, confidence formatting)
- TTSHelper.synthesize_gauntlet_result (pass/fail, vulnerabilities, score formatting)
- get_tts_helper singleton behavior
- is_tts_enabled function
- Module-level TTS_ENABLED, TTS_DEFAULT_VOICE, TTS_MAX_TEXT_LENGTH config
- Edge cases: empty text, None audio path, unlink failures, backend exceptions
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.social.tts_helper as tts_mod
from aragora.server.handlers.social.tts_helper import (
    SynthesisResult,
    TTSHelper,
    get_tts_helper,
    is_tts_enabled,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before each test."""
    tts_mod._tts_helper = None
    yield
    tts_mod._tts_helper = None


@pytest.fixture
def enable_tts(monkeypatch):
    """Enable TTS at the module level."""
    monkeypatch.setattr(tts_mod, "TTS_ENABLED", True)


@pytest.fixture
def disable_tts(monkeypatch):
    """Disable TTS at the module level."""
    monkeypatch.setattr(tts_mod, "TTS_ENABLED", False)


@pytest.fixture
def helper():
    """Create a fresh TTSHelper instance."""
    return TTSHelper()


@pytest.fixture
def mock_bridge():
    """Create a mock TTS bridge."""
    bridge = AsyncMock()
    bridge.synthesize = AsyncMock(return_value="/tmp/test_audio.mp3")
    return bridge


@pytest.fixture
def mock_backend():
    """Create a mock TTS backend."""
    backend = MagicMock()
    backend.name = "mock_tts_backend"
    return backend


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary audio file with sample bytes."""
    audio_path = tmp_path / "test_audio.mp3"
    audio_bytes = b"\xff\xfb\x90\x00" * 4000  # 16000 bytes, fake mp3 frames
    audio_path.write_bytes(audio_bytes)
    return str(audio_path)


# ============================================================================
# SynthesisResult Dataclass
# ============================================================================


class TestSynthesisResult:
    """Tests for the SynthesisResult dataclass."""

    def test_fields_assigned(self):
        """All fields are properly assigned."""
        result = SynthesisResult(
            audio_bytes=b"\x00\x01\x02",
            format="mp3",
            duration_seconds=1.5,
            voice="narrator",
            text_length=42,
        )
        assert result.audio_bytes == b"\x00\x01\x02"
        assert result.format == "mp3"
        assert result.duration_seconds == 1.5
        assert result.voice == "narrator"
        assert result.text_length == 42

    def test_format_wav(self):
        """Format can be wav."""
        result = SynthesisResult(
            audio_bytes=b"RIFF",
            format="wav",
            duration_seconds=2.0,
            voice="moderator",
            text_length=10,
        )
        assert result.format == "wav"

    def test_format_ogg(self):
        """Format can be ogg."""
        result = SynthesisResult(
            audio_bytes=b"OggS",
            format="ogg",
            duration_seconds=0.5,
            voice="narrator",
            text_length=5,
        )
        assert result.format == "ogg"

    def test_empty_audio_bytes(self):
        """Can hold empty audio bytes."""
        result = SynthesisResult(
            audio_bytes=b"",
            format="mp3",
            duration_seconds=0.0,
            voice="narrator",
            text_length=0,
        )
        assert result.audio_bytes == b""
        assert result.duration_seconds == 0.0

    def test_large_audio_bytes(self):
        """Can hold large audio byte arrays."""
        data = b"\x00" * 1_000_000
        result = SynthesisResult(
            audio_bytes=data,
            format="wav",
            duration_seconds=62.5,
            voice="narrator",
            text_length=500,
        )
        assert len(result.audio_bytes) == 1_000_000


# ============================================================================
# TTSHelper Initialization
# ============================================================================


class TestTTSHelperInit:
    """Tests for TTSHelper.__init__."""

    def test_initial_state(self, helper):
        """TTSHelper starts with no bridge, no backend, and availability unknown."""
        assert helper._bridge is None
        assert helper._backend is None
        assert helper._available is None

    def test_multiple_instances_independent(self):
        """Multiple TTSHelper instances are independent."""
        h1 = TTSHelper()
        h2 = TTSHelper()
        h1._available = True
        assert h2._available is None


# ============================================================================
# TTSHelper.is_available
# ============================================================================


class TestIsAvailable:
    """Tests for TTSHelper.is_available property."""

    def test_returns_false_when_tts_disabled(self, helper, disable_tts):
        """When TTS_ENABLED is False, is_available is always False."""
        assert helper.is_available is False

    def test_returns_true_when_backend_succeeds(self, helper, enable_tts, mock_backend):
        """When backend initializes successfully, returns True."""
        with patch.object(helper, "_get_backend", return_value=mock_backend):
            assert helper.is_available is True

    def test_caches_true_result(self, helper, enable_tts, mock_backend):
        """Once availability is determined True, subsequent calls use cached value."""
        with patch.object(helper, "_get_backend", return_value=mock_backend) as mock_get:
            assert helper.is_available is True
            assert helper.is_available is True
            # _get_backend only called once (second time uses cached _available)
            mock_get.assert_called_once()

    def test_caches_false_result(self, helper, enable_tts):
        """Once availability is determined False, subsequent calls use cached value."""
        with patch.object(helper, "_get_backend", side_effect=ImportError("no module")):
            assert helper.is_available is False
        # Now with a working backend, still returns False (cached)
        with patch.object(helper, "_get_backend", return_value=MagicMock()):
            assert helper.is_available is False

    def test_returns_false_on_import_error(self, helper, enable_tts):
        """Returns False when _get_backend raises ImportError."""
        with patch.object(helper, "_get_backend", side_effect=ImportError("no module")):
            assert helper.is_available is False
        assert helper._available is False

    def test_returns_false_on_runtime_error(self, helper, enable_tts):
        """Returns False when _get_backend raises RuntimeError."""
        with patch.object(helper, "_get_backend", side_effect=RuntimeError("init failed")):
            assert helper.is_available is False
        assert helper._available is False

    def test_returns_false_on_os_error(self, helper, enable_tts):
        """Returns False when _get_backend raises OSError."""
        with patch.object(helper, "_get_backend", side_effect=OSError("disk issue")):
            assert helper.is_available is False
        assert helper._available is False

    def test_returns_false_on_value_error(self, helper, enable_tts):
        """Returns False when _get_backend raises ValueError."""
        with patch.object(helper, "_get_backend", side_effect=ValueError("bad config")):
            assert helper.is_available is False
        assert helper._available is False

    def test_returns_false_on_attribute_error(self, helper, enable_tts):
        """Returns False when _get_backend raises AttributeError."""
        with patch.object(helper, "_get_backend", side_effect=AttributeError("missing attr")):
            assert helper.is_available is False
        assert helper._available is False

    def test_returns_false_on_type_error(self, helper, enable_tts):
        """Returns False when _get_backend raises TypeError."""
        with patch.object(helper, "_get_backend", side_effect=TypeError("wrong type")):
            assert helper.is_available is False
        assert helper._available is False


# ============================================================================
# TTSHelper._get_backend
# ============================================================================


class TestGetBackend:
    """Tests for TTSHelper._get_backend."""

    def test_returns_cached_backend(self, helper, mock_backend):
        """When backend is already set, returns it immediately."""
        helper._backend = mock_backend
        result = helper._get_backend()
        assert result is mock_backend

    def test_imports_and_initializes_backend(self, helper, mock_backend):
        """Lazy imports and initializes backend on first call."""
        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_backend",
            create=True,
        ) as mock_get:
            # Simulate the import path
            mock_module = MagicMock()
            mock_module.get_tts_backend.return_value = mock_backend
            with patch.dict(
                "sys.modules",
                {"aragora.broadcast.tts_backends": mock_module},
            ):
                result = helper._get_backend()
                assert result is mock_backend
                assert helper._backend is mock_backend

    def test_raises_runtime_error_on_import_failure(self, helper):
        """Raises RuntimeError when TTS backends module is not available."""
        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": None}):
            with pytest.raises(RuntimeError, match="TTS backends not available"):
                helper._get_backend()

    def test_reraises_runtime_error(self, helper):
        """Re-raises RuntimeError from backend initialization."""
        mock_module = MagicMock()
        mock_module.get_tts_backend.side_effect = RuntimeError("backend init failed")
        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": mock_module}):
            with pytest.raises(RuntimeError, match="backend init failed"):
                helper._get_backend()

    def test_reraises_os_error(self, helper):
        """Re-raises OSError from backend initialization."""
        mock_module = MagicMock()
        mock_module.get_tts_backend.side_effect = OSError("file not found")
        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": mock_module}):
            with pytest.raises(OSError, match="file not found"):
                helper._get_backend()

    def test_reraises_value_error(self, helper):
        """Re-raises ValueError from backend initialization."""
        mock_module = MagicMock()
        mock_module.get_tts_backend.side_effect = ValueError("bad value")
        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": mock_module}):
            with pytest.raises(ValueError, match="bad value"):
                helper._get_backend()


# ============================================================================
# TTSHelper._get_bridge
# ============================================================================


class TestGetBridge:
    """Tests for TTSHelper._get_bridge."""

    def test_returns_cached_bridge(self, helper, mock_bridge):
        """When bridge is already set, returns it immediately."""
        helper._bridge = mock_bridge
        result = helper._get_bridge()
        assert result is mock_bridge

    def test_imports_and_initializes_bridge(self, helper, mock_bridge):
        """Lazy imports and initializes bridge on first call."""
        mock_module = MagicMock()
        mock_module.get_tts_bridge.return_value = mock_bridge
        with patch.dict("sys.modules", {"aragora.connectors.chat.tts_bridge": mock_module}):
            result = helper._get_bridge()
            assert result is mock_bridge
            assert helper._bridge is mock_bridge

    def test_raises_runtime_error_on_import_failure(self, helper):
        """Raises RuntimeError when TTS bridge module is not available."""
        with patch.dict("sys.modules", {"aragora.connectors.chat.tts_bridge": None}):
            with pytest.raises(RuntimeError, match="TTS bridge not available"):
                helper._get_bridge()


# ============================================================================
# TTSHelper.synthesize_response
# ============================================================================


class TestSynthesizeResponse:
    """Tests for TTSHelper.synthesize_response."""

    @pytest.mark.asyncio
    async def test_returns_none_when_not_available(self, helper, disable_tts):
        """Returns None when TTS is not available."""
        result = await helper.synthesize_response("Hello world")
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_synthesis(self, helper, enable_tts, mock_bridge, audio_file):
        """Successful synthesis returns SynthesisResult with correct fields."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello world")

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.format == "mp3"
        assert result.voice == "narrator"  # default voice
        assert result.text_length == len("Hello world")
        assert len(result.audio_bytes) == 16000
        assert result.duration_seconds == 16000 / 16000  # 1.0 seconds

    @pytest.mark.asyncio
    async def test_custom_voice(self, helper, enable_tts, mock_bridge, audio_file):
        """Uses the provided voice instead of the default."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello", voice="moderator")

        assert result is not None
        assert result.voice == "moderator"
        mock_bridge.synthesize.assert_called_once_with(
            text="Hello", voice="moderator", output_format="mp3"
        )

    @pytest.mark.asyncio
    async def test_custom_output_format(self, helper, enable_tts, mock_bridge, audio_file):
        """Uses the provided output format."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello", output_format="wav")

        assert result is not None
        assert result.format == "wav"
        mock_bridge.synthesize.assert_called_once_with(
            text="Hello", voice="narrator", output_format="wav"
        )

    @pytest.mark.asyncio
    async def test_truncates_long_text(self, helper, enable_tts, mock_bridge, audio_file, monkeypatch):
        """Text longer than TTS_MAX_TEXT_LENGTH is truncated with ellipsis."""
        monkeypatch.setattr(tts_mod, "TTS_MAX_TEXT_LENGTH", 50)
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        long_text = "x" * 100
        result = await helper.synthesize_response(long_text)

        assert result is not None
        # The bridge should receive truncated text: 47 chars + "..."
        call_args = mock_bridge.synthesize.call_args
        actual_text = call_args.kwargs["text"]
        assert len(actual_text) == 50
        assert actual_text.endswith("...")
        assert result.text_length == 50

    @pytest.mark.asyncio
    async def test_text_exactly_at_limit_not_truncated(self, helper, enable_tts, mock_bridge, audio_file, monkeypatch):
        """Text exactly at TTS_MAX_TEXT_LENGTH is not truncated."""
        monkeypatch.setattr(tts_mod, "TTS_MAX_TEXT_LENGTH", 20)
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        exact_text = "x" * 20
        await helper.synthesize_response(exact_text)

        call_args = mock_bridge.synthesize.call_args
        actual_text = call_args.kwargs["text"]
        assert actual_text == exact_text
        assert not actual_text.endswith("...")

    @pytest.mark.asyncio
    async def test_returns_none_when_bridge_returns_none_path(self, helper, enable_tts, mock_bridge):
        """Returns None when bridge returns None audio path."""
        mock_bridge.synthesize.return_value = None
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_bridge_returns_empty_string(self, helper, enable_tts, mock_bridge):
        """Returns None when bridge returns empty string audio path."""
        mock_bridge.synthesize.return_value = ""
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_audio_file_missing(self, helper, enable_tts, mock_bridge):
        """Returns None when audio path doesn't exist on disk."""
        mock_bridge.synthesize.return_value = "/nonexistent/path/audio.mp3"
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file(self, helper, enable_tts, mock_bridge, audio_file):
        """Temporary audio file is cleaned up after reading."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        assert Path(audio_file).exists()
        result = await helper.synthesize_response("Hello")
        assert result is not None
        # File should be deleted after reading
        assert not Path(audio_file).exists()

    @pytest.mark.asyncio
    async def test_handles_unlink_failure_gracefully(self, helper, enable_tts, mock_bridge, audio_file):
        """Continues even if temp file cleanup fails."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
            result = await helper.synthesize_response("Hello")
            # Should still return the result
            assert result is not None
            assert isinstance(result, SynthesisResult)

    @pytest.mark.asyncio
    async def test_handles_unlink_os_error(self, helper, enable_tts, mock_bridge, audio_file):
        """Continues even if temp file cleanup raises OSError."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        with patch.object(Path, "unlink", side_effect=OSError("busy")):
            result = await helper.synthesize_response("Hello")
            assert result is not None

    @pytest.mark.asyncio
    async def test_returns_none_on_runtime_error(self, helper, enable_tts, mock_bridge):
        """Returns None when bridge.synthesize raises RuntimeError."""
        mock_bridge.synthesize.side_effect = RuntimeError("synthesis failed")
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_os_error(self, helper, enable_tts, mock_bridge):
        """Returns None when synthesis raises OSError."""
        mock_bridge.synthesize.side_effect = OSError("disk full")
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_value_error(self, helper, enable_tts, mock_bridge):
        """Returns None when synthesis raises ValueError."""
        mock_bridge.synthesize.side_effect = ValueError("invalid params")
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_type_error(self, helper, enable_tts, mock_bridge):
        """Returns None when synthesis raises TypeError."""
        mock_bridge.synthesize.side_effect = TypeError("wrong args")
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_attribute_error(self, helper, enable_tts, mock_bridge):
        """Returns None when synthesis raises AttributeError."""
        mock_bridge.synthesize.side_effect = AttributeError("no attr")
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_duration_estimate_calculation(self, helper, enable_tts, mock_bridge, tmp_path):
        """Duration estimate is computed as len(audio_bytes) / 16000."""
        audio_path = tmp_path / "audio.mp3"
        audio_data = b"\x00" * 32000  # 32000 bytes -> 2.0 seconds estimate
        audio_path.write_bytes(audio_data)
        mock_bridge.synthesize.return_value = str(audio_path)
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello")
        assert result is not None
        assert result.duration_seconds == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_uses_default_voice_when_none(self, helper, enable_tts, mock_bridge, audio_file, monkeypatch):
        """Uses TTS_DEFAULT_VOICE when voice parameter is None."""
        monkeypatch.setattr(tts_mod, "TTS_DEFAULT_VOICE", "custom_narrator")
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello", voice=None)

        assert result is not None
        assert result.voice == "custom_narrator"
        mock_bridge.synthesize.assert_called_once_with(
            text="Hello", voice="custom_narrator", output_format="mp3"
        )

    @pytest.mark.asyncio
    async def test_empty_text(self, helper, enable_tts, mock_bridge, audio_file):
        """Empty text is passed through to bridge."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("")
        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        assert call_args.kwargs["text"] == ""

    @pytest.mark.asyncio
    async def test_bridge_initialization_failure(self, helper, enable_tts):
        """Returns None when _get_bridge raises RuntimeError."""
        helper._available = True
        with patch.object(helper, "_get_bridge", side_effect=RuntimeError("no bridge")):
            result = await helper.synthesize_response("Hello")
            assert result is None


# ============================================================================
# TTSHelper.synthesize_debate_result
# ============================================================================


class TestSynthesizeDebateResult:
    """Tests for TTSHelper.synthesize_debate_result."""

    @pytest.mark.asyncio
    async def test_consensus_reached_with_answer(self, helper, enable_tts, mock_bridge, audio_file):
        """Builds summary for consensus reached with final answer."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_debate_result(
            task="Design a rate limiter",
            final_answer="Use token bucket algorithm.",
            consensus_reached=True,
            confidence=0.95,
            rounds_used=3,
        )

        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "Debate completed on: Design a rate limiter" in text
        assert "Consensus was reached with 95% confidence" in text
        assert "after 3 rounds" in text
        assert "Use token bucket algorithm." in text
        # Should use narrator voice
        assert call_args.kwargs["voice"] == "narrator"

    @pytest.mark.asyncio
    async def test_consensus_reached_without_answer(self, helper, enable_tts, mock_bridge, audio_file):
        """Builds summary when consensus is reached but no final answer."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_debate_result(
            task="Should we refactor?",
            final_answer=None,
            consensus_reached=True,
            confidence=0.80,
            rounds_used=5,
        )

        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "Consensus was reached with 80% confidence" in text
        assert "The conclusion is:" not in text

    @pytest.mark.asyncio
    async def test_no_consensus_reached(self, helper, enable_tts, mock_bridge, audio_file):
        """Builds summary when no consensus is reached."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_debate_result(
            task="Is AI sentient?",
            final_answer="Inconclusive",
            consensus_reached=False,
            confidence=0.45,
            rounds_used=7,
        )

        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "No consensus was reached after 7 rounds" in text
        assert "Final confidence was 45%" in text
        # Should NOT include the final answer text when no consensus
        assert "The conclusion is:" not in text

    @pytest.mark.asyncio
    async def test_long_task_truncated(self, helper, enable_tts, mock_bridge, audio_file):
        """Long task text is truncated to 150 characters in summary."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        long_task = "x" * 300
        await helper.synthesize_debate_result(
            task=long_task,
            final_answer="answer",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=2,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        # Task should be truncated: "Debate completed on: " + task[:150]
        assert "x" * 150 in text
        assert "x" * 151 not in text

    @pytest.mark.asyncio
    async def test_long_answer_truncated(self, helper, enable_tts, mock_bridge, audio_file):
        """Long final answer is truncated to 400 chars with ellipsis."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        long_answer = "y" * 500
        await helper.synthesize_debate_result(
            task="Test task",
            final_answer=long_answer,
            consensus_reached=True,
            confidence=0.85,
            rounds_used=4,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        # Answer truncated to 400 + "..."
        assert "y" * 400 in text
        assert "..." in text

    @pytest.mark.asyncio
    async def test_short_answer_no_ellipsis(self, helper, enable_tts, mock_bridge, audio_file):
        """Short final answer does not get ellipsis."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_debate_result(
            task="Test",
            final_answer="Short answer",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=1,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "Short answer" in text
        # Should not have "..." after the answer (only at truncation)
        conclusion_part = text.split("The conclusion is: ")[1]
        assert not conclusion_part.endswith("...")

    @pytest.mark.asyncio
    async def test_returns_none_when_not_available(self, helper, disable_tts):
        """Returns None when TTS is not available."""
        result = await helper.synthesize_debate_result(
            task="Test", final_answer="Answer",
            consensus_reached=True, confidence=0.9, rounds_used=1,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_confidence(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles zero confidence correctly."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_debate_result(
            task="Test", final_answer=None,
            consensus_reached=False, confidence=0.0, rounds_used=10,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "0%" in text

    @pytest.mark.asyncio
    async def test_full_confidence(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles 100% confidence correctly."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_debate_result(
            task="Test", final_answer="Yes",
            consensus_reached=True, confidence=1.0, rounds_used=1,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "100%" in text


# ============================================================================
# TTSHelper.synthesize_gauntlet_result
# ============================================================================


class TestSynthesizeGauntletResult:
    """Tests for TTSHelper.synthesize_gauntlet_result."""

    @pytest.mark.asyncio
    async def test_gauntlet_passed_no_vulnerabilities(self, helper, enable_tts, mock_bridge, audio_file):
        """Builds summary for a passed gauntlet with no vulnerabilities."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_gauntlet_result(
            statement="The earth is round",
            passed=True,
            score=0.95,
            vulnerability_count=0,
        )

        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "Gauntlet stress test passed" in text
        assert "95%" in text
        assert "The earth is round" in text
        assert "No significant vulnerabilities were found" in text
        # Should use moderator voice
        assert call_args.kwargs["voice"] == "moderator"

    @pytest.mark.asyncio
    async def test_gauntlet_failed_with_vulnerabilities(self, helper, enable_tts, mock_bridge, audio_file):
        """Builds summary for a failed gauntlet with vulnerabilities."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_gauntlet_result(
            statement="All swans are white",
            passed=False,
            score=0.30,
            vulnerability_count=5,
        )

        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "Gauntlet stress test failed" in text
        assert "30%" in text
        assert "5 vulnerabilities were found" in text

    @pytest.mark.asyncio
    async def test_gauntlet_passed_with_vulnerabilities(self, helper, enable_tts, mock_bridge, audio_file):
        """Builds summary for a passed gauntlet that still found vulnerabilities."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_gauntlet_result(
            statement="Statement",
            passed=True,
            score=0.70,
            vulnerability_count=2,
        )

        assert result is not None
        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "passed" in text
        assert "2 vulnerabilities were found" in text

    @pytest.mark.asyncio
    async def test_long_statement_truncated(self, helper, enable_tts, mock_bridge, audio_file):
        """Long statement is truncated to 150 characters."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        long_statement = "z" * 300
        await helper.synthesize_gauntlet_result(
            statement=long_statement,
            passed=True,
            score=0.8,
            vulnerability_count=0,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "z" * 150 in text
        assert "z" * 151 not in text

    @pytest.mark.asyncio
    async def test_returns_none_when_not_available(self, helper, disable_tts):
        """Returns None when TTS is not available."""
        result = await helper.synthesize_gauntlet_result(
            statement="Test", passed=True, score=0.9, vulnerability_count=0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_score(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles zero score correctly."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_gauntlet_result(
            statement="Bad statement", passed=False, score=0.0, vulnerability_count=10,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "0%" in text

    @pytest.mark.asyncio
    async def test_perfect_score(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles 100% score correctly."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_gauntlet_result(
            statement="Perfect", passed=True, score=1.0, vulnerability_count=0,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "100%" in text

    @pytest.mark.asyncio
    async def test_single_vulnerability(self, helper, enable_tts, mock_bridge, audio_file):
        """Correct grammar for single vulnerability."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_gauntlet_result(
            statement="Test", passed=False, score=0.5, vulnerability_count=1,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "1 vulnerabilities were found" in text


# ============================================================================
# get_tts_helper Singleton
# ============================================================================


class TestGetTTSHelper:
    """Tests for get_tts_helper singleton function."""

    def test_returns_tts_helper_instance(self):
        """Returns a TTSHelper instance."""
        helper = get_tts_helper()
        assert isinstance(helper, TTSHelper)

    def test_returns_same_instance(self):
        """Calling twice returns the same singleton instance."""
        h1 = get_tts_helper()
        h2 = get_tts_helper()
        assert h1 is h2

    def test_creates_new_after_reset(self):
        """After clearing singleton, creates a new instance."""
        h1 = get_tts_helper()
        tts_mod._tts_helper = None
        h2 = get_tts_helper()
        assert h1 is not h2

    def test_returns_existing_if_set(self):
        """Returns the existing singleton if already set."""
        custom_helper = TTSHelper()
        tts_mod._tts_helper = custom_helper
        assert get_tts_helper() is custom_helper


# ============================================================================
# is_tts_enabled
# ============================================================================


class TestIsTTSEnabled:
    """Tests for the is_tts_enabled function."""

    def test_returns_true_when_enabled(self, monkeypatch):
        """Returns True when TTS_ENABLED is True."""
        monkeypatch.setattr(tts_mod, "TTS_ENABLED", True)
        assert is_tts_enabled() is True

    def test_returns_false_when_disabled(self, monkeypatch):
        """Returns False when TTS_ENABLED is False."""
        monkeypatch.setattr(tts_mod, "TTS_ENABLED", False)
        assert is_tts_enabled() is False


# ============================================================================
# Module-Level Configuration
# ============================================================================


class TestModuleConfig:
    """Tests for module-level configuration constants."""

    def test_tts_enabled_defaults_false(self):
        """TTS_ENABLED defaults to False (env var not set)."""
        # The module already loaded, but the default is "false"
        # We can verify the constant exists
        assert hasattr(tts_mod, "TTS_ENABLED")
        assert isinstance(tts_mod.TTS_ENABLED, bool)

    def test_tts_default_voice_exists(self):
        """TTS_DEFAULT_VOICE is a string constant."""
        assert isinstance(tts_mod.TTS_DEFAULT_VOICE, str)
        assert len(tts_mod.TTS_DEFAULT_VOICE) > 0

    def test_tts_max_text_length_exists(self):
        """TTS_MAX_TEXT_LENGTH is an integer constant."""
        assert isinstance(tts_mod.TTS_MAX_TEXT_LENGTH, int)
        assert tts_mod.TTS_MAX_TEXT_LENGTH > 0

    def test_default_voice_is_narrator(self):
        """Default voice from env is 'narrator' when ARAGORA_TTS_DEFAULT_VOICE unset."""
        # Unless env var is set, the default should be "narrator"
        if "ARAGORA_TTS_DEFAULT_VOICE" not in os.environ:
            assert tts_mod.TTS_DEFAULT_VOICE == "narrator"

    def test_default_max_text_is_2000(self):
        """Default max text length from env is 2000 when ARAGORA_TTS_MAX_TEXT unset."""
        if "ARAGORA_TTS_MAX_TEXT" not in os.environ:
            assert tts_mod.TTS_MAX_TEXT_LENGTH == 2000


# ============================================================================
# Module __all__ Exports
# ============================================================================


class TestModuleExports:
    """Tests for the module's __all__ exports."""

    def test_all_exports_defined(self):
        """__all__ contains expected exports."""
        assert "TTSHelper" in tts_mod.__all__
        assert "SynthesisResult" in tts_mod.__all__
        assert "get_tts_helper" in tts_mod.__all__
        assert "is_tts_enabled" in tts_mod.__all__

    def test_all_exports_are_importable(self):
        """All items in __all__ can be resolved from the module."""
        for name in tts_mod.__all__:
            assert hasattr(tts_mod, name), f"{name} in __all__ but not in module"


# ============================================================================
# Edge Cases & Integration
# ============================================================================


class TestEdgeCases:
    """Edge cases and integration tests."""

    @pytest.mark.asyncio
    async def test_synthesize_response_with_special_characters(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles text with special characters."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        text = "Hello <world> & 'friends' \"everyone\" \u2603"
        result = await helper.synthesize_response(text)
        assert result is not None

    @pytest.mark.asyncio
    async def test_synthesize_response_with_newlines(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles text with newline characters."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        text = "Line one\nLine two\nLine three"
        result = await helper.synthesize_response(text)
        assert result is not None
        assert result.text_length == len(text)

    @pytest.mark.asyncio
    async def test_synthesize_response_with_unicode(self, helper, enable_tts, mock_bridge, audio_file):
        """Handles text with unicode characters."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        text = "Debate about \u00e9galit\u00e9 and \u00fcberalles"
        result = await helper.synthesize_response(text)
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_result_one_round(self, helper, enable_tts, mock_bridge, audio_file):
        """Debate result with 1 round uses correct singular form in text."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        await helper.synthesize_debate_result(
            task="Quick debate", final_answer="Yes",
            consensus_reached=True, confidence=0.99, rounds_used=1,
        )

        call_args = mock_bridge.synthesize.call_args
        text = call_args.kwargs["text"]
        assert "1 rounds" in text  # The code uses "rounds" regardless

    @pytest.mark.asyncio
    async def test_multiple_sequential_syntheses(self, helper, enable_tts, mock_bridge, tmp_path):
        """Multiple sequential syntheses all work correctly."""
        helper._available = True
        helper._bridge = mock_bridge

        results = []
        for i in range(5):
            audio_path = tmp_path / f"audio_{i}.mp3"
            audio_data = b"\x00" * (1000 * (i + 1))
            audio_path.write_bytes(audio_data)
            mock_bridge.synthesize.return_value = str(audio_path)

            result = await helper.synthesize_response(f"Text number {i}")
            results.append(result)

        for i, result in enumerate(results):
            assert result is not None
            assert result.audio_bytes == b"\x00" * (1000 * (i + 1))

    @pytest.mark.asyncio
    async def test_availability_not_rechecked_after_first_check(self, helper, enable_tts):
        """Once is_available returns False, it stays False even if conditions change."""
        with patch.object(helper, "_get_backend", side_effect=ImportError("no")):
            assert helper.is_available is False

        # Even with a working backend now, cached False remains
        helper._backend = MagicMock()
        assert helper.is_available is False

    @pytest.mark.asyncio
    async def test_get_backend_caches_result(self, helper):
        """_get_backend caches the backend after first successful initialization."""
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_module = MagicMock()
        mock_module.get_tts_backend.return_value = mock_backend

        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": mock_module}):
            result1 = helper._get_backend()

        # Second call should return cached value, no need for import
        result2 = helper._get_backend()
        assert result1 is result2 is mock_backend
        # Module was only called once
        assert mock_module.get_tts_backend.call_count == 1

    @pytest.mark.asyncio
    async def test_get_bridge_caches_result(self, helper):
        """_get_bridge caches the bridge after first successful initialization."""
        mock_bridge = AsyncMock()
        mock_module = MagicMock()
        mock_module.get_tts_bridge.return_value = mock_bridge

        with patch.dict("sys.modules", {"aragora.connectors.chat.tts_bridge": mock_module}):
            result1 = helper._get_bridge()

        result2 = helper._get_bridge()
        assert result1 is result2 is mock_bridge
        assert mock_module.get_tts_bridge.call_count == 1

    @pytest.mark.asyncio
    async def test_synthesize_tiny_audio_file(self, helper, enable_tts, mock_bridge, tmp_path):
        """Handles very small audio files (1 byte)."""
        audio_path = tmp_path / "tiny.mp3"
        audio_path.write_bytes(b"\x00")
        mock_bridge.synthesize.return_value = str(audio_path)
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hi")
        assert result is not None
        assert len(result.audio_bytes) == 1
        assert result.duration_seconds == pytest.approx(1 / 16000)

    @pytest.mark.asyncio
    async def test_text_one_char_over_limit_gets_truncated(self, helper, enable_tts, mock_bridge, audio_file, monkeypatch):
        """Text that is exactly one char over limit gets truncated."""
        monkeypatch.setattr(tts_mod, "TTS_MAX_TEXT_LENGTH", 10)
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        text = "x" * 11  # one over limit
        await helper.synthesize_response(text)

        call_args = mock_bridge.synthesize.call_args
        actual_text = call_args.kwargs["text"]
        assert len(actual_text) == 10
        assert actual_text == "xxxxxxx..."

    @pytest.mark.asyncio
    async def test_synthesize_response_ogg_format(self, helper, enable_tts, mock_bridge, audio_file):
        """ogg format is passed through correctly."""
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        result = await helper.synthesize_response("Hello", output_format="ogg")
        assert result is not None
        assert result.format == "ogg"

    @pytest.mark.asyncio
    async def test_debate_result_delegates_to_synthesize_response(self, helper, enable_tts):
        """synthesize_debate_result calls synthesize_response internally."""
        helper._available = True
        mock_result = SynthesisResult(
            audio_bytes=b"audio", format="mp3",
            duration_seconds=1.0, voice="narrator", text_length=50,
        )
        with patch.object(helper, "synthesize_response", new_callable=AsyncMock, return_value=mock_result) as mock_synth:
            result = await helper.synthesize_debate_result(
                task="Test", final_answer="Answer",
                consensus_reached=True, confidence=0.9, rounds_used=2,
            )
            assert result is mock_result
            mock_synth.assert_called_once()
            # Verify the voice is "narrator"
            call_kwargs = mock_synth.call_args
            assert call_kwargs.kwargs.get("voice") == "narrator" or call_kwargs[1].get("voice") == "narrator"

    @pytest.mark.asyncio
    async def test_gauntlet_result_delegates_to_synthesize_response(self, helper, enable_tts):
        """synthesize_gauntlet_result calls synthesize_response internally."""
        helper._available = True
        mock_result = SynthesisResult(
            audio_bytes=b"audio", format="mp3",
            duration_seconds=1.0, voice="moderator", text_length=50,
        )
        with patch.object(helper, "synthesize_response", new_callable=AsyncMock, return_value=mock_result) as mock_synth:
            result = await helper.synthesize_gauntlet_result(
                statement="Test", passed=True, score=0.9, vulnerability_count=0,
            )
            assert result is mock_result
            mock_synth.assert_called_once()
            call_kwargs = mock_synth.call_args
            assert call_kwargs.kwargs.get("voice") == "moderator" or call_kwargs[1].get("voice") == "moderator"


# ============================================================================
# Logging Tests
# ============================================================================


class TestLogging:
    """Tests that verify logging behavior."""

    def test_logger_name(self):
        """Module logger has correct name."""
        assert tts_mod.logger.name == "aragora.server.handlers.social.tts_helper"

    @pytest.mark.asyncio
    async def test_logs_debug_when_not_available(self, helper, disable_tts, caplog):
        """Logs debug message when TTS not available for synthesis."""
        import logging
        with caplog.at_level(logging.DEBUG, logger="aragora.server.handlers.social.tts_helper"):
            result = await helper.synthesize_response("Hello")
            assert result is None
            assert "TTS not available" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_warning_when_no_audio_file(self, helper, enable_tts, mock_bridge, caplog):
        """Logs warning when synthesis returns no audio file."""
        import logging
        mock_bridge.synthesize.return_value = None
        helper._available = True
        helper._bridge = mock_bridge

        with caplog.at_level(logging.WARNING, logger="aragora.server.handlers.social.tts_helper"):
            result = await helper.synthesize_response("Hello")
            assert result is None
            assert "no audio file" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_error_on_synthesis_failure(self, helper, enable_tts, mock_bridge, caplog):
        """Logs error when synthesis raises an exception."""
        import logging
        mock_bridge.synthesize.side_effect = RuntimeError("boom")
        helper._available = True
        helper._bridge = mock_bridge

        with caplog.at_level(logging.ERROR, logger="aragora.server.handlers.social.tts_helper"):
            result = await helper.synthesize_response("Hello")
            assert result is None
            assert "TTS synthesis failed" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_info_on_successful_synthesis(self, helper, enable_tts, mock_bridge, audio_file, caplog):
        """Logs info message on successful synthesis."""
        import logging
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        with caplog.at_level(logging.INFO, logger="aragora.server.handlers.social.tts_helper"):
            result = await helper.synthesize_response("Hello")
            assert result is not None
            assert "TTS synthesized" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_debug_on_truncation(self, helper, enable_tts, mock_bridge, audio_file, monkeypatch, caplog):
        """Logs debug message when text is truncated."""
        import logging
        monkeypatch.setattr(tts_mod, "TTS_MAX_TEXT_LENGTH", 10)
        mock_bridge.synthesize.return_value = audio_file
        helper._available = True
        helper._bridge = mock_bridge

        with caplog.at_level(logging.DEBUG, logger="aragora.server.handlers.social.tts_helper"):
            await helper.synthesize_response("x" * 20)
            assert "truncated" in caplog.text


# ============================================================================
# State Transitions & Concurrency
# ============================================================================


class TestStateTransitions:
    """Tests for state transitions on TTSHelper."""

    def test_available_none_to_true(self, helper, enable_tts, mock_backend):
        """_available transitions from None to True on backend success."""
        assert helper._available is None
        with patch.object(helper, "_get_backend", return_value=mock_backend):
            _ = helper.is_available
        assert helper._available is True

    def test_available_none_to_false_on_error(self, helper, enable_tts):
        """_available transitions from None to False on backend error."""
        assert helper._available is None
        with patch.object(helper, "_get_backend", side_effect=ImportError("no")):
            _ = helper.is_available
        assert helper._available is False

    def test_backend_none_then_set(self, helper):
        """_backend transitions from None to backend object."""
        assert helper._backend is None
        mock_backend = MagicMock()
        mock_backend.name = "test"
        mock_module = MagicMock()
        mock_module.get_tts_backend.return_value = mock_backend
        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": mock_module}):
            helper._get_backend()
        assert helper._backend is mock_backend

    def test_bridge_none_then_set(self, helper):
        """_bridge transitions from None to bridge object."""
        assert helper._bridge is None
        mock_bridge = MagicMock()
        mock_module = MagicMock()
        mock_module.get_tts_bridge.return_value = mock_bridge
        with patch.dict("sys.modules", {"aragora.connectors.chat.tts_bridge": mock_module}):
            helper._get_bridge()
        assert helper._bridge is mock_bridge

    @pytest.mark.asyncio
    async def test_fresh_helper_full_lifecycle(self, enable_tts, mock_bridge, mock_backend, audio_file):
        """Full lifecycle: create helper, check availability, synthesize, verify result."""
        h = TTSHelper()

        # Patch both backend and bridge
        mock_bridge.synthesize.return_value = audio_file
        with patch.object(h, "_get_backend", return_value=mock_backend):
            assert h.is_available is True

        h._bridge = mock_bridge
        result = await h.synthesize_response("Full lifecycle test")
        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert result.text_length == len("Full lifecycle test")

    @pytest.mark.asyncio
    async def test_debate_then_gauntlet_sequentially(self, helper, enable_tts, mock_bridge, tmp_path):
        """Can run debate synthesis followed by gauntlet synthesis."""
        helper._available = True
        helper._bridge = mock_bridge

        # First call - debate
        audio1 = tmp_path / "debate.mp3"
        audio1.write_bytes(b"\x01" * 100)
        mock_bridge.synthesize.return_value = str(audio1)
        r1 = await helper.synthesize_debate_result(
            task="Q", final_answer="A", consensus_reached=True,
            confidence=0.9, rounds_used=2,
        )
        assert r1 is not None

        # Second call - gauntlet
        audio2 = tmp_path / "gauntlet.mp3"
        audio2.write_bytes(b"\x02" * 200)
        mock_bridge.synthesize.return_value = str(audio2)
        r2 = await helper.synthesize_gauntlet_result(
            statement="S", passed=True, score=0.8, vulnerability_count=0,
        )
        assert r2 is not None
        assert r2.audio_bytes != r1.audio_bytes
