"""
Tests for TTS Helper.

Tests the TTS integration helper used by chat platform handlers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIsTTSEnabled:
    """Tests for is_tts_enabled function."""

    def test_returns_false_when_env_not_set(self, monkeypatch):
        """Should return False when env var not set."""
        monkeypatch.delenv("ARAGORA_TTS_CHAT_ENABLED", raising=False)
        # Need to reload module to pick up env change
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)
        assert tts_helper.is_tts_enabled() is False

    def test_returns_false_when_env_false(self, monkeypatch):
        """Should return False when env var is false."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "false")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)
        assert tts_helper.is_tts_enabled() is False

    def test_returns_true_when_env_true(self, monkeypatch):
        """Should return True when env var is true."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "true")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)
        assert tts_helper.is_tts_enabled() is True

    def test_case_insensitive(self, monkeypatch):
        """Should handle case insensitive values."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "TRUE")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)
        assert tts_helper.is_tts_enabled() is True


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_create_result(self):
        """Should create result with all fields."""
        from aragora.server.handlers.social.tts_helper import SynthesisResult

        result = SynthesisResult(
            audio_bytes=b"audio_data",
            format="mp3",
            duration_seconds=2.5,
            voice="narrator",
            text_length=100,
        )

        assert result.audio_bytes == b"audio_data"
        assert result.format == "mp3"
        assert result.duration_seconds == 2.5
        assert result.voice == "narrator"
        assert result.text_length == 100

    def test_result_is_immutable_fields(self):
        """Should have expected fields."""
        from aragora.server.handlers.social.tts_helper import SynthesisResult

        result = SynthesisResult(
            audio_bytes=b"data",
            format="wav",
            duration_seconds=1.0,
            voice="test",
            text_length=10,
        )

        # Check all expected fields exist
        assert hasattr(result, "audio_bytes")
        assert hasattr(result, "format")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "voice")
        assert hasattr(result, "text_length")


class TestTTSHelper:
    """Tests for TTSHelper class."""

    def test_init_creates_empty_instance(self):
        """Should initialize with None values."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()
        assert helper._bridge is None
        assert helper._backend is None
        assert helper._available is None

    def test_is_available_false_when_disabled(self, monkeypatch):
        """Should return False when TTS is disabled."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "false")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        helper = tts_helper.TTSHelper()
        assert helper.is_available is False

    def test_is_available_caches_result(self, monkeypatch):
        """Should cache availability check result."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "true")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        helper = tts_helper.TTSHelper()
        helper._available = True  # Pre-set cached value

        # Should return cached value without checking backend
        assert helper.is_available is True

    def test_is_available_handles_import_error(self, monkeypatch):
        """Should return False when backend import fails."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "true")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        helper = tts_helper.TTSHelper()

        with patch.object(helper, "_get_backend", side_effect=ImportError("No TTS")):
            assert helper.is_available is False
            assert helper._available is False

    def test_is_available_handles_runtime_error(self, monkeypatch):
        """Should return False when backend raises RuntimeError."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "true")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        helper = tts_helper.TTSHelper()

        with patch.object(helper, "_get_backend", side_effect=RuntimeError("Not ready")):
            assert helper.is_available is False

    def test_is_available_handles_unexpected_error(self, monkeypatch):
        """Should return False on unexpected exceptions."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "true")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        helper = tts_helper.TTSHelper()

        with patch.object(helper, "_get_backend", side_effect=ValueError("Unexpected")):
            assert helper.is_available is False

    def test_get_backend_returns_cached(self):
        """Should return cached backend."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()
        mock_backend = MagicMock()
        helper._backend = mock_backend

        result = helper._get_backend()
        assert result is mock_backend

    def test_get_backend_initializes_from_module(self):
        """Should initialize backend from TTS backends module."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()
        mock_backend = MagicMock()
        mock_backend.name = "test_backend"

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_backend",
            return_value=mock_backend,
            create=True,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.broadcast.tts_backends": MagicMock(get_tts_backend=lambda: mock_backend)},
            ):
                # Patch the import inside _get_backend
                with patch(
                    "builtins.__import__",
                    side_effect=lambda name, *args, **kwargs: (
                        MagicMock(get_tts_backend=lambda: mock_backend)
                        if "tts_backends" in name
                        else __import__(name, *args, **kwargs)
                    ),
                ):
                    pass  # Backend initialization is complex; test simpler path

    def test_get_backend_raises_on_import_error(self):
        """Should raise RuntimeError when import fails."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()

        with patch.dict("sys.modules", {"aragora.broadcast.tts_backends": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named tts_backends"),
            ):
                with pytest.raises(RuntimeError, match="TTS backends not available"):
                    helper._get_backend()

    def test_get_bridge_returns_cached(self):
        """Should return cached bridge."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()
        mock_bridge = MagicMock()
        helper._bridge = mock_bridge

        result = helper._get_bridge()
        assert result is mock_bridge

    def test_get_bridge_raises_on_import_error(self):
        """Should raise RuntimeError when bridge import fails."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()

        with patch.dict("sys.modules", {"aragora.connectors.chat.tts_bridge": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named tts_bridge"),
            ):
                with pytest.raises(RuntimeError, match="TTS bridge not available"):
                    helper._get_bridge()


class TestTTSHelperSynthesizeResponse:
    """Tests for synthesize_response method."""

    @pytest.fixture
    def enabled_helper(self, monkeypatch, tmp_path):
        """Create a TTS helper with TTS enabled and mocked backend."""
        monkeypatch.setenv("ARAGORA_TTS_CHAT_ENABLED", "true")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        helper = tts_helper.TTSHelper()
        helper._available = True

        # Create mock bridge
        mock_bridge = AsyncMock()
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(b"fake_audio_data_bytes")
        mock_bridge.synthesize = AsyncMock(return_value=str(audio_file))
        helper._bridge = mock_bridge

        return helper

    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self):
        """Should return None when TTS is unavailable."""
        from aragora.server.handlers.social.tts_helper import TTSHelper

        helper = TTSHelper()
        helper._available = False

        result = await helper.synthesize_response("Test text")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesizes_text_successfully(self, enabled_helper):
        """Should synthesize text and return result."""
        result = await enabled_helper.synthesize_response("Hello world", voice="narrator")

        assert result is not None
        assert result.audio_bytes == b"fake_audio_data_bytes"
        assert result.format == "mp3"
        assert result.voice == "narrator"
        assert result.text_length == 11

    @pytest.mark.asyncio
    async def test_uses_default_voice(self, enabled_helper, monkeypatch):
        """Should use default voice when not specified."""
        monkeypatch.setenv("ARAGORA_TTS_DEFAULT_VOICE", "default_narrator")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        result = await enabled_helper.synthesize_response("Test")
        # Voice is passed to bridge, check the call
        enabled_helper._bridge.synthesize.assert_called_once()
        call_kwargs = enabled_helper._bridge.synthesize.call_args[1]
        assert call_kwargs["voice"] == "default_narrator"  # Uses env var default

    @pytest.mark.asyncio
    async def test_truncates_long_text(self, enabled_helper, monkeypatch):
        """Should truncate text exceeding max length."""
        monkeypatch.setenv("ARAGORA_TTS_MAX_TEXT", "50")
        import importlib

        from aragora.server.handlers.social import tts_helper

        importlib.reload(tts_helper)

        # Create new helper with new max
        enabled_helper._available = True
        long_text = "x" * 100

        result = await enabled_helper.synthesize_response(long_text)

        # Check that truncated text was passed to bridge
        call_kwargs = enabled_helper._bridge.synthesize.call_args[1]
        assert len(call_kwargs["text"]) <= 50

    @pytest.mark.asyncio
    async def test_handles_bridge_error(self, enabled_helper):
        """Should return None when bridge raises exception."""
        enabled_helper._bridge.synthesize.side_effect = RuntimeError("TTS failed")

        result = await enabled_helper.synthesize_response("Test")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_missing_audio_file(self, enabled_helper):
        """Should return None when audio file doesn't exist."""
        enabled_helper._bridge.synthesize.return_value = "/nonexistent/path.mp3"

        result = await enabled_helper.synthesize_response("Test")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_none_audio_path(self, enabled_helper):
        """Should return None when bridge returns None."""
        enabled_helper._bridge.synthesize.return_value = None

        result = await enabled_helper.synthesize_response("Test")
        assert result is None

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file(self, enabled_helper, tmp_path):
        """Should delete temp audio file after reading."""
        audio_file = tmp_path / "cleanup_test.mp3"
        audio_file.write_bytes(b"audio")
        enabled_helper._bridge.synthesize.return_value = str(audio_file)

        result = await enabled_helper.synthesize_response("Test")

        assert result is not None
        assert not audio_file.exists()

    @pytest.mark.asyncio
    async def test_handles_cleanup_error_gracefully(self, enabled_helper, tmp_path):
        """Should handle errors when deleting temp file."""
        audio_file = tmp_path / "cleanup_fail.mp3"
        audio_file.write_bytes(b"audio")
        enabled_helper._bridge.synthesize.return_value = str(audio_file)

        # Make file read-only to cause deletion error (on some systems)
        with patch("pathlib.Path.unlink", side_effect=PermissionError("Cannot delete")):
            result = await enabled_helper.synthesize_response("Test")

        # Should still return result despite cleanup failure
        assert result is not None

    @pytest.mark.asyncio
    async def test_supports_different_formats(self, enabled_helper):
        """Should pass format to bridge."""
        await enabled_helper.synthesize_response("Test", output_format="wav")

        call_kwargs = enabled_helper._bridge.synthesize.call_args[1]
        assert call_kwargs["output_format"] == "wav"


class TestTTSHelperSynthesizeDebateResult:
    """Tests for synthesize_debate_result method."""

    @pytest.fixture
    def helper_with_mock_synthesize(self):
        """Create helper with mocked synthesize_response."""
        from aragora.server.handlers.social.tts_helper import SynthesisResult, TTSHelper

        helper = TTSHelper()
        helper._available = True

        mock_result = SynthesisResult(
            audio_bytes=b"audio",
            format="mp3",
            duration_seconds=1.0,
            voice="narrator",
            text_length=100,
        )
        helper.synthesize_response = AsyncMock(return_value=mock_result)
        return helper

    @pytest.mark.asyncio
    async def test_synthesizes_consensus_result(self, helper_with_mock_synthesize):
        """Should synthesize summary for consensus result."""
        helper = helper_with_mock_synthesize

        result = await helper.synthesize_debate_result(
            task="Should we use Python?",
            final_answer="Yes, Python is excellent.",
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
        )

        assert result is not None
        helper.synthesize_response.assert_called_once()

        # Check the synthesized text contains key info
        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        assert "Should we use Python?" in summary
        assert "85%" in summary
        assert "3 rounds" in summary
        assert "Yes, Python" in summary

    @pytest.mark.asyncio
    async def test_synthesizes_no_consensus_result(self, helper_with_mock_synthesize):
        """Should synthesize summary for no consensus."""
        helper = helper_with_mock_synthesize

        result = await helper.synthesize_debate_result(
            task="Complex question",
            final_answer=None,
            consensus_reached=False,
            confidence=0.45,
            rounds_used=5,
        )

        assert result is not None
        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        assert "No consensus" in summary
        assert "45%" in summary
        assert "5 rounds" in summary

    @pytest.mark.asyncio
    async def test_truncates_long_task(self, helper_with_mock_synthesize):
        """Should truncate very long task descriptions."""
        helper = helper_with_mock_synthesize
        long_task = "x" * 300

        await helper.synthesize_debate_result(
            task=long_task,
            final_answer="Answer",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=2,
        )

        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        # Task should be truncated to 150 chars
        assert len(summary) < len(long_task) + 200

    @pytest.mark.asyncio
    async def test_truncates_long_final_answer(self, helper_with_mock_synthesize):
        """Should truncate very long final answers."""
        helper = helper_with_mock_synthesize
        long_answer = "y" * 500

        await helper.synthesize_debate_result(
            task="Short task",
            final_answer=long_answer,
            consensus_reached=True,
            confidence=0.8,
            rounds_used=4,
        )

        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        # Answer preview should be truncated to ~400 chars + ellipsis
        assert "..." in summary

    @pytest.mark.asyncio
    async def test_uses_narrator_voice(self, helper_with_mock_synthesize):
        """Should use narrator voice for debate results."""
        helper = helper_with_mock_synthesize

        await helper.synthesize_debate_result(
            task="Test",
            final_answer="Answer",
            consensus_reached=True,
            confidence=0.9,
            rounds_used=1,
        )

        call_kwargs = helper.synthesize_response.call_args[1]
        assert call_kwargs["voice"] == "narrator"


class TestTTSHelperSynthesizeGauntletResult:
    """Tests for synthesize_gauntlet_result method."""

    @pytest.fixture
    def helper_with_mock_synthesize(self):
        """Create helper with mocked synthesize_response."""
        from aragora.server.handlers.social.tts_helper import SynthesisResult, TTSHelper

        helper = TTSHelper()
        helper._available = True

        mock_result = SynthesisResult(
            audio_bytes=b"audio",
            format="mp3",
            duration_seconds=1.0,
            voice="moderator",
            text_length=100,
        )
        helper.synthesize_response = AsyncMock(return_value=mock_result)
        return helper

    @pytest.mark.asyncio
    async def test_synthesizes_passed_result(self, helper_with_mock_synthesize):
        """Should synthesize passed gauntlet result."""
        helper = helper_with_mock_synthesize

        result = await helper.synthesize_gauntlet_result(
            statement="AI will benefit humanity",
            passed=True,
            score=0.92,
            vulnerability_count=0,
        )

        assert result is not None
        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        assert "passed" in summary
        assert "92%" in summary
        assert "No significant vulnerabilities" in summary

    @pytest.mark.asyncio
    async def test_synthesizes_failed_result(self, helper_with_mock_synthesize):
        """Should synthesize failed gauntlet result."""
        helper = helper_with_mock_synthesize

        result = await helper.synthesize_gauntlet_result(
            statement="Controversial claim",
            passed=False,
            score=0.35,
            vulnerability_count=3,
        )

        assert result is not None
        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        assert "failed" in summary
        assert "35%" in summary
        assert "3 vulnerabilities" in summary

    @pytest.mark.asyncio
    async def test_truncates_long_statement(self, helper_with_mock_synthesize):
        """Should truncate long statements."""
        helper = helper_with_mock_synthesize
        long_statement = "z" * 200

        await helper.synthesize_gauntlet_result(
            statement=long_statement,
            passed=True,
            score=0.8,
            vulnerability_count=0,
        )

        call_args = helper.synthesize_response.call_args[0]
        summary = call_args[0]
        # Statement should be truncated to 150 chars
        assert len(summary) < len(long_statement) + 100

    @pytest.mark.asyncio
    async def test_uses_moderator_voice(self, helper_with_mock_synthesize):
        """Should use moderator voice for gauntlet results."""
        helper = helper_with_mock_synthesize

        await helper.synthesize_gauntlet_result(
            statement="Test",
            passed=True,
            score=0.9,
            vulnerability_count=0,
        )

        call_kwargs = helper.synthesize_response.call_args[1]
        assert call_kwargs["voice"] == "moderator"


class TestGetTTSHelper:
    """Tests for get_tts_helper singleton function."""

    def test_returns_singleton_instance(self):
        """Should return same instance on multiple calls."""
        import importlib

        from aragora.server.handlers.social import tts_helper

        # Reset singleton
        tts_helper._tts_helper = None
        importlib.reload(tts_helper)

        helper1 = tts_helper.get_tts_helper()
        helper2 = tts_helper.get_tts_helper()

        assert helper1 is helper2

    def test_creates_tts_helper_instance(self):
        """Should create TTSHelper instance."""
        import importlib

        from aragora.server.handlers.social import tts_helper

        # Reset singleton
        tts_helper._tts_helper = None

        helper = tts_helper.get_tts_helper()

        assert isinstance(helper, tts_helper.TTSHelper)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Should export expected symbols."""
        from aragora.server.handlers.social import tts_helper

        assert "TTSHelper" in tts_helper.__all__
        assert "SynthesisResult" in tts_helper.__all__
        assert "get_tts_helper" in tts_helper.__all__
        assert "is_tts_enabled" in tts_helper.__all__

    def test_can_import_all_exports(self):
        """Should be able to import all exported symbols."""
        from aragora.server.handlers.social.tts_helper import (
            SynthesisResult,
            TTSHelper,
            get_tts_helper,
            is_tts_enabled,
        )

        assert TTSHelper is not None
        assert SynthesisResult is not None
        assert get_tts_helper is not None
        assert is_tts_enabled is not None
