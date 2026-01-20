"""
Tests for the speech-to-text module.

Tests cover:
- Base provider classes and dataclasses
- OpenAI Whisper provider
- High-level transcription functions
- Speech API handlers
"""

import io
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from aragora.speech.providers.base import (
    STTProvider,
    STTProviderConfig,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestSTTProviderConfig:
    """Tests for STTProviderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = STTProviderConfig()

        assert config.provider_name == "base"
        assert config.default_language is None
        assert config.model == "default"
        assert config.include_timestamps is True
        assert config.include_word_timestamps is False
        assert config.extra_options == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = STTProviderConfig(
            provider_name="whisper",
            default_language="en",
            model="whisper-1",
            include_timestamps=False,
            include_word_timestamps=True,
            extra_options={"temperature": 0.0},
        )

        assert config.provider_name == "whisper"
        assert config.default_language == "en"
        assert config.model == "whisper-1"
        assert config.include_timestamps is False
        assert config.include_word_timestamps is True
        assert config.extra_options == {"temperature": 0.0}


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_basic_segment(self):
        """Test basic segment creation."""
        segment = TranscriptionSegment(
            text="Hello, world!",
            start=0.0,
            end=1.5,
        )

        assert segment.text == "Hello, world!"
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.confidence is None
        assert segment.words is None

    def test_segment_with_confidence(self):
        """Test segment with confidence score."""
        segment = TranscriptionSegment(
            text="Test",
            start=0.0,
            end=0.5,
            confidence=-0.25,
        )

        assert segment.confidence == -0.25

    def test_segment_with_words(self):
        """Test segment with word-level timestamps."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.4, "end": 0.8},
        ]
        segment = TranscriptionSegment(
            text="Hello world",
            start=0.0,
            end=0.8,
            words=words,
        )

        assert segment.words == words
        assert len(segment.words) == 2


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = TranscriptionResult(
            text="Hello, world!",
            language="en",
            duration=5.0,
        )

        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.duration == 5.0
        assert result.segments == []
        assert result.provider == "unknown"
        assert result.model == "unknown"

    def test_result_with_segments(self):
        """Test result with segments."""
        segments = [
            TranscriptionSegment(text="Hello", start=0.0, end=0.5),
            TranscriptionSegment(text="world", start=0.6, end=1.0),
        ]
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=1.0,
            segments=segments,
            provider="openai_whisper",
            model="whisper-1",
        )

        assert len(result.segments) == 2
        assert result.provider == "openai_whisper"
        assert result.model == "whisper-1"

    def test_to_dict(self):
        """Test JSON serialization."""
        segments = [
            TranscriptionSegment(
                text="Hello",
                start=0.0,
                end=0.5,
                confidence=-0.1,
                words=[{"word": "Hello", "start": 0.0, "end": 0.5}],
            ),
        ]
        result = TranscriptionResult(
            text="Hello",
            language="en",
            duration=0.5,
            segments=segments,
            provider="whisper",
            model="whisper-1",
        )

        d = result.to_dict()

        assert d["text"] == "Hello"
        assert d["language"] == "en"
        assert d["duration"] == 0.5
        assert d["provider"] == "whisper"
        assert d["model"] == "whisper-1"
        assert len(d["segments"]) == 1
        assert d["segments"][0]["text"] == "Hello"
        assert d["segments"][0]["start"] == 0.0
        assert d["segments"][0]["end"] == 0.5
        assert d["segments"][0]["confidence"] == -0.1
        assert d["segments"][0]["words"] is not None


class TestSTTProviderBase:
    """Tests for STTProvider base class."""

    def test_abstract_methods(self):
        """Test that STTProvider is abstract."""
        with pytest.raises(TypeError):
            STTProvider()

    def test_concrete_implementation(self):
        """Test concrete STTProvider implementation."""

        class ConcreteProvider(STTProvider):
            @property
            def name(self) -> str:
                return "concrete"

            async def transcribe(self, audio_file, language=None, prompt=None):
                return TranscriptionResult(
                    text="test",
                    language="en",
                    duration=1.0,
                )

            async def is_available(self) -> bool:
                return True

        provider = ConcreteProvider()
        assert provider.name == "concrete"
        assert provider.supported_formats() == [
            "mp3",
            "mp4",
            "mpeg",
            "mpga",
            "m4a",
            "wav",
            "webm",
            "ogg",
            "flac",
        ]
        assert provider.max_file_size_mb() == 25

    def test_default_config(self):
        """Test default config is created."""

        class ConcreteProvider(STTProvider):
            @property
            def name(self) -> str:
                return "test"

            async def transcribe(self, audio_file, language=None, prompt=None):
                pass

            async def is_available(self) -> bool:
                return True

        provider = ConcreteProvider()
        assert isinstance(provider.config, STTProviderConfig)

    def test_custom_config(self):
        """Test custom config is used."""

        class ConcreteProvider(STTProvider):
            @property
            def name(self) -> str:
                return "test"

            async def transcribe(self, audio_file, language=None, prompt=None):
                pass

            async def is_available(self) -> bool:
                return True

        config = STTProviderConfig(model="custom-model")
        provider = ConcreteProvider(config=config)
        assert provider.config.model == "custom-model"


class TestOpenAIWhisperProvider:
    """Tests for OpenAI Whisper provider."""

    def test_provider_name(self):
        """Test provider name."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIWhisperProvider()

        assert provider.name == "openai_whisper"

    def test_api_key_from_env(self):
        """Test API key is read from environment."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIWhisperProvider()

        assert provider._api_key == "env-key"

    def test_api_key_explicit(self):
        """Test explicit API key takes precedence."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIWhisperProvider(api_key="explicit-key")

        assert provider._api_key == "explicit-key"

    @pytest.mark.asyncio
    async def test_is_available_with_key(self):
        """Test availability with API key."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIWhisperProvider()

            # Mock the OpenAI client
            with patch("aragora.speech.providers.openai_whisper.OpenAIWhisperProvider._get_client"):
                available = await provider.is_available()

        assert available is True

    @pytest.mark.asyncio
    async def test_is_available_without_key(self):
        """Test availability without API key."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            provider = OpenAIWhisperProvider()
            available = await provider.is_available()

        assert available is False

    def test_supported_formats(self):
        """Test supported audio formats."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        provider = OpenAIWhisperProvider(api_key="test")
        formats = provider.supported_formats()

        assert "mp3" in formats
        assert "wav" in formats
        assert "webm" in formats
        assert "m4a" in formats

    def test_max_file_size(self):
        """Test max file size."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        provider = OpenAIWhisperProvider(api_key="test")
        assert provider.max_file_size_mb() == 25

    @pytest.mark.asyncio
    async def test_transcribe_with_mock(self):
        """Test transcription with mocked OpenAI client."""
        from aragora.speech.providers.openai_whisper import OpenAIWhisperProvider

        # Create mock response
        mock_response = MagicMock()
        mock_response.text = "Hello, world!"
        mock_response.language = "en"
        mock_response.duration = 2.5
        mock_response.segments = [
            {"text": "Hello, world!", "start": 0.0, "end": 2.5, "avg_logprob": -0.2}
        ]

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        provider = OpenAIWhisperProvider(api_key="test")
        provider._client = mock_client

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = Path(f.name)

        try:
            result = await provider.transcribe(temp_path)

            assert result.text == "Hello, world!"
            assert result.language == "en"
            assert result.duration == 2.5
            assert len(result.segments) == 1
            assert result.segments[0].text == "Hello, world!"
            assert result.provider == "openai_whisper"
        finally:
            temp_path.unlink()


class TestTranscribeFunctions:
    """Tests for high-level transcription functions."""

    def test_get_provider_default(self):
        """Test get_provider with default settings."""
        from aragora.speech.transcribe import get_provider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            provider = get_provider()

        assert provider.name == "openai_whisper"

    def test_get_provider_by_name(self):
        """Test get_provider with explicit name."""
        from aragora.speech.transcribe import get_provider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            provider = get_provider("whisper")

        assert provider.name == "openai_whisper"

    def test_get_provider_from_env(self):
        """Test get_provider reads from environment."""
        from aragora.speech.transcribe import get_provider

        with patch.dict(
            "os.environ",
            {
                "ARAGORA_STT_PROVIDER": "whisper",
                "OPENAI_API_KEY": "test",
            },
        ):
            provider = get_provider()

        assert provider.name == "openai_whisper"

    def test_get_provider_unknown(self):
        """Test get_provider raises for unknown provider."""
        from aragora.speech.transcribe import get_provider

        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown_provider")

        assert "Unknown STT provider" in str(exc_info.value)

    def test_get_provider_with_config(self):
        """Test get_provider with custom config."""
        from aragora.speech.transcribe import get_provider

        config = STTProviderConfig(include_timestamps=False)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            provider = get_provider(config=config)

        assert provider.config.include_timestamps is False

    @pytest.mark.asyncio
    async def test_transcribe_audio_not_available(self):
        """Test transcribe_audio raises when provider unavailable."""
        from aragora.speech.transcribe import transcribe_audio

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("OPENAI_API_KEY", None)

            with pytest.raises(RuntimeError) as exc_info:
                await transcribe_audio(Path("test.mp3"))

            assert "not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_found(self):
        """Test transcribe_audio_file raises for missing file."""
        from aragora.speech.transcribe import transcribe_audio_file

        with pytest.raises(FileNotFoundError):
            await transcribe_audio_file("/nonexistent/path.mp3")

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_a_file(self):
        """Test transcribe_audio_file raises for directory."""
        from aragora.speech.transcribe import transcribe_audio_file

        with pytest.raises(ValueError) as exc_info:
            await transcribe_audio_file("/tmp")

        assert "not a file" in str(exc_info.value)


class TestSpeechHandlerImports:
    """Tests for speech handler imports and routes."""

    def test_handler_routes(self):
        """Test handler route definitions."""
        from aragora.server.handlers.features.speech import SpeechHandler

        # Check class-level ROUTES without instantiation
        assert "/api/speech/transcribe" in SpeechHandler.ROUTES
        assert "/api/speech/transcribe-url" in SpeechHandler.ROUTES
        assert "/api/speech/providers" in SpeechHandler.ROUTES

    def test_can_handle(self):
        """Test can_handle method."""
        from aragora.server.handlers.features.speech import SpeechHandler

        # Create mock server_context
        mock_context = MagicMock()
        handler = SpeechHandler(mock_context)

        assert handler.can_handle("/api/speech/transcribe") is True
        assert handler.can_handle("/api/speech/providers") is True
        assert handler.can_handle("/api/other") is False

    def test_get_providers(self):
        """Test GET /api/speech/providers handler."""
        from aragora.server.handlers.features.speech import SpeechHandler

        # Create mock server_context
        mock_context = MagicMock()
        handler = SpeechHandler(mock_context)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            result = handler.handle("/api/speech/providers", {}, None)

        assert result is not None

        # HandlerResult has status_code and body attributes
        import json

        if hasattr(result, "status_code"):
            status = result.status_code
            body = result.body
        elif hasattr(result, "body"):
            body = result.body
            status = getattr(result, "status", 200)
        else:
            # Tuple format fallback
            body, status, headers = result

        assert status == 200
        data = json.loads(body)
        assert "providers" in data
        assert "default" in data

    def test_supported_extensions(self):
        """Test supported file extensions."""
        from aragora.server.handlers.features.speech import SUPPORTED_EXTENSIONS

        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".wav" in SUPPORTED_EXTENSIONS
        assert ".webm" in SUPPORTED_EXTENSIONS
        assert ".m4a" in SUPPORTED_EXTENSIONS
        assert ".ogg" in SUPPORTED_EXTENSIONS
        assert ".flac" in SUPPORTED_EXTENSIONS

    def test_max_file_size(self):
        """Test file size limits."""
        from aragora.server.handlers.features.speech import MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES

        assert MAX_FILE_SIZE_MB == 25
        assert MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024


class TestSpeechModuleExports:
    """Tests for speech module exports."""

    def test_main_exports(self):
        """Test main module exports."""
        from aragora.speech import (
            transcribe_audio,
            transcribe_audio_file,
            TranscriptionResult,
            TranscriptionSegment,
            STTProvider,
            STTProviderConfig,
            OpenAIWhisperProvider,
        )

        # All exports should be importable
        assert transcribe_audio is not None
        assert transcribe_audio_file is not None
        assert TranscriptionResult is not None
        assert TranscriptionSegment is not None
        assert STTProvider is not None
        assert STTProviderConfig is not None
        assert OpenAIWhisperProvider is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
