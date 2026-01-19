"""
Tests for aragora.transcription.whisper_backend - Whisper transcription backends.

Tests cover:
- Backend initialization
- Audio format validation
- Transcription result structure
- Error handling (invalid file, API failure)
- Backend selection logic
- File size limits
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if openai is available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Check if faster-whisper is available
try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

requires_openai = pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="openai not installed"
)

requires_faster_whisper = pytest.mark.skipif(
    not FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)

from aragora.transcription.whisper_backend import (
    AUDIO_FORMATS,
    VIDEO_FORMATS,
    ALL_MEDIA_FORMATS,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionBackend,
    OpenAIWhisperBackend,
    FasterWhisperBackend,
    WhisperCppBackend,
    get_transcription_backend,
)


def is_supported_format(filename: str) -> bool:
    """Check if a file format is supported."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALL_MEDIA_FORMATS


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.audio.transcriptions.create = MagicMock(
        return_value=MagicMock(
            text="Hello, this is a test transcription.",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello, this is"},
                {"start": 2.0, "end": 4.0, "text": "a test transcription."},
            ],
            language="en",
            duration=4.0,
        )
    )
    return client


@pytest.fixture
def sample_audio_file():
    """Create a temporary sample audio file."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        # Write minimal MP3 header (not a valid audio file, but sufficient for testing)
        f.write(b"\xff\xfb\x90\x00" + b"\x00" * 100)
        return Path(f.name)


@pytest.fixture
def sample_video_file():
    """Create a temporary sample video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Write minimal data
        f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
        return Path(f.name)


# ===========================================================================
# Format Validation Tests
# ===========================================================================


class TestFormatValidation:
    """Tests for audio/video format validation."""

    def test_audio_formats_defined(self):
        """Verify audio formats are defined."""
        assert ".mp3" in AUDIO_FORMATS
        assert ".wav" in AUDIO_FORMATS
        assert ".m4a" in AUDIO_FORMATS
        assert ".webm" in AUDIO_FORMATS
        assert ".ogg" in AUDIO_FORMATS
        assert ".flac" in AUDIO_FORMATS

    def test_video_formats_defined(self):
        """Verify video formats are defined."""
        assert ".mp4" in VIDEO_FORMATS
        assert ".mov" in VIDEO_FORMATS
        assert ".webm" in VIDEO_FORMATS
        assert ".mkv" in VIDEO_FORMATS

    def test_is_supported_format_audio(self):
        """Test audio format detection."""
        assert is_supported_format("test.mp3") is True
        assert is_supported_format("test.wav") is True
        assert is_supported_format("test.MP3") is True  # Case insensitive
        assert is_supported_format("/path/to/audio.flac") is True

    def test_is_supported_format_video(self):
        """Test video format detection."""
        assert is_supported_format("video.mp4") is True
        assert is_supported_format("video.mov") is True
        assert is_supported_format("VIDEO.MKV") is True  # Case insensitive

    def test_is_supported_format_unsupported(self):
        """Test unsupported format detection."""
        assert is_supported_format("document.pdf") is False
        assert is_supported_format("image.png") is False
        assert is_supported_format("text.txt") is False
        assert is_supported_format("no_extension") is False


# ===========================================================================
# TranscriptionResult Tests
# ===========================================================================


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_create_result(self):
        """Test creating a transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello"),
                TranscriptionSegment(id=1, start=1.0, end=2.0, text="world"),
            ],
            language="en",
            duration=2.0,
            backend="test",
        )
        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.duration == 2.0

    def test_result_properties(self):
        """Test result has expected properties."""
        result = TranscriptionResult(
            text="Test",
            segments=[TranscriptionSegment(id=0, start=0.0, end=1.0, text="Test")],
            language="en",
            duration=1.0,
            backend="openai",
        )
        assert result.text == "Test"
        assert result.language == "en"
        assert result.duration == 1.0
        assert len(result.segments) == 1
        assert result.backend == "openai"

    def test_segment_properties(self):
        """Test segment has expected properties."""
        segment = TranscriptionSegment(
            id=0,
            start=1.5,
            end=3.0,
            text="Test segment",
        )
        assert segment.start == 1.5
        assert segment.end == 3.0
        assert segment.text == "Test segment"
        assert segment.duration == 1.5  # end - start


# ===========================================================================
# OpenAI Whisper Backend Tests
# ===========================================================================


@requires_openai
class TestOpenAIWhisperBackend:
    """Tests for OpenAI Whisper API backend."""

    def test_initialization(self):
        """Test backend initialization."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend()
            assert backend.name == "openai"
            assert backend.model == "whisper-1"

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend(model="whisper-1")
            assert backend.model == "whisper-1"

    def test_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIWhisperBackend()

    @pytest.mark.asyncio
    async def test_transcribe_success(self, sample_audio_file, mock_openai_client):
        """Test successful transcription."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend()
            backend._client = mock_openai_client

            result = await backend.transcribe(sample_audio_file)

            assert result.text == "Hello, this is a test transcription."
            assert result.language == "en"
            mock_openai_client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, sample_audio_file, mock_openai_client):
        """Test transcription with language hint."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend()
            backend._client = mock_openai_client

            await backend.transcribe(sample_audio_file, language="fr")

            call_kwargs = mock_openai_client.audio.transcriptions.create.call_args
            assert call_kwargs.kwargs.get("language") == "fr"

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self):
        """Test transcription with non-existent file."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend()

            with pytest.raises(FileNotFoundError):
                await backend.transcribe(Path("/nonexistent/file.mp3"))

    @pytest.mark.asyncio
    async def test_transcribe_api_error(self, sample_audio_file):
        """Test handling of API errors."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend()
            backend._client = MagicMock()
            backend._client.audio.transcriptions.create.side_effect = Exception(
                "API Error"
            )

            with pytest.raises(Exception, match="API Error"):
                await backend.transcribe(sample_audio_file)


# ===========================================================================
# Faster Whisper Backend Tests
# ===========================================================================


@requires_faster_whisper
class TestFasterWhisperBackend:
    """Tests for faster-whisper local backend."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = FasterWhisperBackend(model_size="base")
        assert backend.name == "faster-whisper"

    def test_initialization_different_sizes(self):
        """Test initialization with different model sizes."""
        for size in ["tiny", "base", "small", "medium", "large"]:
            backend = FasterWhisperBackend(model_size=size)
            assert backend is not None


@pytest.mark.skipif(FASTER_WHISPER_AVAILABLE, reason="faster-whisper is installed")
class TestFasterWhisperBackendUnavailable:
    """Tests for faster-whisper when not installed."""

    def test_is_not_available(self):
        """Test is_available returns False when faster-whisper not installed."""
        backend = FasterWhisperBackend()
        assert backend.is_available() is False


# ===========================================================================
# Whisper.cpp Backend Tests
# ===========================================================================


class TestWhisperCppBackend:
    """Tests for whisper.cpp backend."""

    def test_initialization_with_binary(self):
        """Test backend initialization when binary is available."""
        import shutil
        whisper_path = shutil.which("whisper")
        if whisper_path:
            backend = WhisperCppBackend()
            assert backend.name == "whisper-cpp"
        else:
            pytest.skip("whisper.cpp binary not in PATH")

    def test_is_not_available(self):
        """Test is_available when whisper.cpp not in PATH."""
        import shutil
        import os
        # Check for whisper-cpp binary (the actual name used)
        if shutil.which("whisper-cpp") is not None or os.getenv("WHISPER_CPP_PATH"):
            pytest.skip("whisper-cpp is installed")
        backend = WhisperCppBackend()
        assert backend.is_available() is False


# ===========================================================================
# Backend Selection Tests
# ===========================================================================


class TestBackendSelection:
    """Tests for automatic backend selection."""

    @requires_openai
    def test_get_openai_backend(self):
        """Test selecting OpenAI backend."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_transcription_backend("openai")
            assert isinstance(backend, OpenAIWhisperBackend)

    @requires_faster_whisper
    def test_get_faster_whisper_backend(self):
        """Test selecting faster-whisper backend."""
        backend = get_transcription_backend("faster-whisper")
        assert isinstance(backend, FasterWhisperBackend)

    def test_get_whisper_cpp_backend(self):
        """Test selecting whisper.cpp backend."""
        import shutil
        if shutil.which("whisper") is not None:
            backend = get_transcription_backend("whisper-cpp")
            assert isinstance(backend, WhisperCppBackend)
        else:
            with pytest.raises(RuntimeError):
                get_transcription_backend("whisper-cpp")

    @requires_openai
    def test_get_auto_backend_openai(self):
        """Test auto-selection prefers OpenAI when available."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_transcription_backend("auto")
            assert isinstance(backend, OpenAIWhisperBackend)

    def test_get_invalid_backend(self):
        """Test error on invalid backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_transcription_backend("invalid-backend")


# ===========================================================================
# File Size Limit Tests
# ===========================================================================


@requires_openai
class TestFileSizeLimits:
    """Tests for file size validation."""

    @pytest.mark.asyncio
    async def test_file_too_large(self, tmp_path):
        """Test rejection of files exceeding size limit."""
        # Create a file larger than the limit
        large_file = tmp_path / "large.mp3"
        large_file.write_bytes(b"\x00" * (26 * 1024 * 1024))  # 26MB

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = OpenAIWhisperBackend()

            with pytest.raises(ValueError, match="size"):
                await backend.transcribe(large_file)


# ===========================================================================
# Integration Pattern Tests
# ===========================================================================


@requires_openai
class TestBackendInterface:
    """Tests verifying backend interface consistency."""

    def test_all_backends_have_name(self):
        """All backends should have a name property."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            openai = OpenAIWhisperBackend()
            assert hasattr(openai, "name")
            assert isinstance(openai.name, str)

    def test_all_backends_have_transcribe(self):
        """All backends should have async transcribe method."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            openai = OpenAIWhisperBackend()
            assert hasattr(openai, "transcribe")
            assert callable(openai.transcribe)
