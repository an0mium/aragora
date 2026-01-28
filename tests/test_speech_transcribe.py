"""Tests for speech.transcribe module.

Tests cover:
- get_provider() factory function
- transcribe_audio() async function
- transcribe_audio_file() convenience function
- Error handling for invalid inputs
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.speech.transcribe import (
    get_provider,
    transcribe_audio,
    transcribe_audio_file,
)


class TestGetProvider:
    """Tests for get_provider factory function."""

    def test_default_provider(self):
        """Should return OpenAI Whisper by default."""
        provider = get_provider()
        assert provider.name == "openai_whisper"

    def test_whisper_alias(self):
        """Should accept 'whisper' as alias for openai_whisper."""
        provider = get_provider("whisper")
        assert provider.name == "openai_whisper"

    def test_openai_whisper_explicit(self):
        """Should accept 'openai_whisper' explicitly."""
        provider = get_provider("openai_whisper")
        assert provider.name == "openai_whisper"

    def test_case_insensitive(self):
        """Provider names should be case-insensitive."""
        provider = get_provider("WHISPER")
        assert provider.name == "openai_whisper"

        provider = get_provider("OpenAI_Whisper")
        assert provider.name == "openai_whisper"

    def test_invalid_provider_raises_error(self):
        """Should raise ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown STT provider"):
            get_provider("invalid_provider")

    def test_invalid_provider_lists_available(self):
        """Error message should list available providers."""
        with pytest.raises(ValueError, match="Available providers"):
            get_provider("nonexistent")

    def test_provider_from_env_var(self, monkeypatch):
        """Should use ARAGORA_STT_PROVIDER env var when no name given."""
        monkeypatch.setenv("ARAGORA_STT_PROVIDER", "whisper")
        provider = get_provider()
        assert provider.name == "openai_whisper"

    def test_explicit_name_overrides_env_var(self, monkeypatch):
        """Explicit provider_name should override env var."""
        monkeypatch.setenv("ARAGORA_STT_PROVIDER", "some_other")
        provider = get_provider("whisper")
        assert provider.name == "openai_whisper"

    def test_accepts_config(self):
        """Should accept optional config parameter."""
        from aragora.speech.providers.base import STTProviderConfig

        config = STTProviderConfig(default_language="en", model="whisper-1")
        provider = get_provider("whisper", config=config)
        assert provider is not None


@pytest.mark.asyncio
class TestTranscribeAudio:
    """Tests for transcribe_audio async function."""

    async def test_raises_when_provider_unavailable(self):
        """Should raise RuntimeError if provider is unavailable."""
        with patch("aragora.speech.transcribe.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.is_available = AsyncMock(return_value=False)
            mock_provider.name = "test_provider"
            mock_get.return_value = mock_provider

            with pytest.raises(RuntimeError, match="not available"):
                await transcribe_audio(Path("test.mp3"))

    async def test_calls_provider_transcribe(self):
        """Should call provider.transcribe with audio file."""
        from aragora.speech.providers.base import TranscriptionResult

        with patch("aragora.speech.transcribe.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.is_available = AsyncMock(return_value=True)
            mock_provider.transcribe = AsyncMock(
                return_value=TranscriptionResult(text="Hello world", language="en", duration=2.5)
            )
            mock_get.return_value = mock_provider

            result = await transcribe_audio(Path("test.mp3"))

            assert result.text == "Hello world"
            assert result.duration == 2.5
            mock_provider.transcribe.assert_called_once()

    async def test_passes_language_to_provider(self):
        """Should pass language parameter to provider."""
        from aragora.speech.providers.base import TranscriptionResult

        with patch("aragora.speech.transcribe.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.is_available = AsyncMock(return_value=True)
            mock_provider.transcribe = AsyncMock(
                return_value=TranscriptionResult(text="Hola", language="es", duration=1.0)
            )
            mock_get.return_value = mock_provider

            await transcribe_audio(Path("test.mp3"), language="es")

            # Check language was passed to transcribe
            call_args = mock_provider.transcribe.call_args
            assert call_args[0][1] == "es"  # Second positional arg is language

    async def test_passes_prompt_to_provider(self):
        """Should pass prompt parameter to provider."""
        from aragora.speech.providers.base import TranscriptionResult

        with patch("aragora.speech.transcribe.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.is_available = AsyncMock(return_value=True)
            mock_provider.transcribe = AsyncMock(
                return_value=TranscriptionResult(text="Test", language="en", duration=1.0)
            )
            mock_get.return_value = mock_provider

            await transcribe_audio(Path("test.mp3"), prompt="Technical terms")

            # Check prompt was passed to transcribe
            call_args = mock_provider.transcribe.call_args
            assert call_args[0][2] == "Technical terms"  # Third positional arg is prompt

    async def test_uses_specified_provider(self):
        """Should use the specified provider_name."""
        from aragora.speech.providers.base import TranscriptionResult

        with patch("aragora.speech.transcribe.get_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.is_available = AsyncMock(return_value=True)
            mock_provider.transcribe = AsyncMock(
                return_value=TranscriptionResult(text="Test", language="en", duration=1.0)
            )
            mock_get.return_value = mock_provider

            await transcribe_audio(Path("test.mp3"), provider_name="whisper")

            mock_get.assert_called_once_with("whisper", None)


@pytest.mark.asyncio
class TestTranscribeAudioFile:
    """Tests for transcribe_audio_file convenience function."""

    async def test_nonexistent_file_raises_error(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            await transcribe_audio_file("/nonexistent/path/file.mp3")

    async def test_directory_path_raises_error(self, tmp_path):
        """Should raise ValueError for directory paths."""
        with pytest.raises(ValueError, match="not a file"):
            await transcribe_audio_file(tmp_path)

    async def test_accepts_string_path(self, tmp_path):
        """Should accept string path argument."""
        from aragora.speech.providers.base import TranscriptionResult

        # Create a temporary file
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with patch("aragora.speech.transcribe.transcribe_audio") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="Hello", language="en", duration=1.0
            )

            result = await transcribe_audio_file(str(test_file))

            assert result.text == "Hello"

    async def test_accepts_path_object(self, tmp_path):
        """Should accept Path object argument."""
        from aragora.speech.providers.base import TranscriptionResult

        # Create a temporary file
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with patch("aragora.speech.transcribe.transcribe_audio") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="Hello", language="en", duration=1.0
            )

            result = await transcribe_audio_file(test_file)

            assert result.text == "Hello"

    async def test_passes_language_parameter(self, tmp_path):
        """Should pass language to transcribe_audio."""
        from aragora.speech.providers.base import TranscriptionResult

        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with patch("aragora.speech.transcribe.transcribe_audio") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(text="", language="en", duration=0)

            await transcribe_audio_file(str(test_file), language="fr")

            mock_transcribe.assert_called_once()
            call_kwargs = mock_transcribe.call_args[1]
            assert call_kwargs["language"] == "fr"

    async def test_passes_prompt_parameter(self, tmp_path):
        """Should pass prompt to transcribe_audio."""
        from aragora.speech.providers.base import TranscriptionResult

        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with patch("aragora.speech.transcribe.transcribe_audio") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(text="", language="en", duration=0)

            await transcribe_audio_file(str(test_file), prompt="Medical terms")

            mock_transcribe.assert_called_once()
            call_kwargs = mock_transcribe.call_args[1]
            assert call_kwargs["prompt"] == "Medical terms"

    async def test_passes_provider_name_parameter(self, tmp_path):
        """Should pass provider_name to transcribe_audio."""
        from aragora.speech.providers.base import TranscriptionResult

        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        with patch("aragora.speech.transcribe.transcribe_audio") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(text="", language="en", duration=0)

            await transcribe_audio_file(str(test_file), provider_name="whisper")

            mock_transcribe.assert_called_once()
            call_kwargs = mock_transcribe.call_args[1]
            assert call_kwargs["provider_name"] == "whisper"
