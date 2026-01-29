"""Tests for Whisper connector."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import io

from aragora.connectors.whisper import (
    WhisperConnector,
    TranscriptionResult,
    TranscriptionSegment,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    ALL_SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    MAX_FILE_SIZE_BYTES,
    is_supported_audio,
    is_supported_video,
    is_supported_media,
    get_supported_formats,
)
from aragora.connectors.base import Evidence
from aragora.connectors.exceptions import (
    ConnectorConfigError,
    ConnectorRateLimitError,
)
from aragora.reasoning.provenance import SourceType


# Sample Whisper API responses
SAMPLE_VERBOSE_JSON_RESPONSE = {
    "text": "Hello, this is a test transcription of audio content.",
    "language": "en",
    "duration": 5.5,
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 2.5,
            "text": "Hello, this is a test",
            "avg_logprob": -0.25,
        },
        {
            "id": 1,
            "start": 2.5,
            "end": 5.5,
            "text": " transcription of audio content.",
            "avg_logprob": -0.30,
        },
    ],
}

SAMPLE_SIMPLE_JSON_RESPONSE = {
    "text": "Simple transcription without segments.",
}


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_segment_creation(self):
        """Test creating a transcription segment."""
        segment = TranscriptionSegment(
            start=0.0,
            end=2.5,
            text="Hello world",
            confidence=0.95,
        )

        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.text == "Hello world"
        assert segment.confidence == 0.95

    def test_segment_default_confidence(self):
        """Test segment has default confidence of 0."""
        segment = TranscriptionSegment(
            start=0.0,
            end=1.0,
            text="Test",
        )

        assert segment.confidence == 0.0

    def test_segment_to_dict(self):
        """Test segment serialization."""
        segment = TranscriptionSegment(
            start=1.5,
            end=3.0,
            text="Test text",
            confidence=0.85,
        )

        data = segment.to_dict()

        assert data["start"] == 1.5
        assert data["end"] == 3.0
        assert data["text"] == "Test text"
        assert data["confidence"] == 0.85

    def test_segment_from_dict(self):
        """Test segment deserialization."""
        data = {
            "start": 2.0,
            "end": 4.0,
            "text": "From dict",
            "confidence": 0.9,
        }

        segment = TranscriptionSegment.from_dict(data)

        assert segment.start == 2.0
        assert segment.end == 4.0
        assert segment.text == "From dict"
        assert segment.confidence == 0.9

    def test_segment_from_dict_default_confidence(self):
        """Test segment deserialization with missing confidence."""
        data = {
            "start": 0.0,
            "end": 1.0,
            "text": "No confidence",
        }

        segment = TranscriptionSegment.from_dict(data)

        assert segment.confidence == 0.0


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_result_creation(self):
        """Test creating a transcription result."""
        result = TranscriptionResult(
            id="trans_abc123",
            text="Hello world",
            language="en",
            duration_seconds=5.0,
            source_filename="audio.mp3",
        )

        assert result.id == "trans_abc123"
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 5.0
        assert result.source_filename == "audio.mp3"

    def test_result_word_count_auto(self):
        """Test automatic word count calculation."""
        result = TranscriptionResult(
            id="test",
            text="This has five words here",
        )

        assert result.word_count == 5

    def test_result_word_count_explicit(self):
        """Test explicit word count overrides auto calculation."""
        result = TranscriptionResult(
            id="test",
            text="This has five words here",
            word_count=10,  # Explicit value
        )

        assert result.word_count == 10

    def test_result_with_segments(self):
        """Test result with segments."""
        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="First", confidence=0.9),
            TranscriptionSegment(start=2.0, end=4.0, text="Second", confidence=0.85),
        ]
        result = TranscriptionResult(
            id="test",
            text="First Second",
            segments=segments,
        )

        assert len(result.segments) == 2
        assert result.segments[0].text == "First"
        assert result.segments[1].text == "Second"

    def test_result_to_evidence(self):
        """Test converting result to Evidence."""
        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="Hello", confidence=0.9),
        ]
        result = TranscriptionResult(
            id="trans_123",
            text="Hello world",
            segments=segments,
            language="en",
            duration_seconds=2.0,
            source_filename="meeting.mp3",
            confidence=0.85,
        )

        evidence = result.to_evidence()

        assert isinstance(evidence, Evidence)
        assert evidence.id == "trans_123"
        assert evidence.source_type == SourceType.AUDIO_TRANSCRIPT
        assert evidence.source_id == "meeting.mp3"
        assert evidence.content == "Hello world"
        assert evidence.title == "Transcript: meeting.mp3"
        assert evidence.confidence == 0.85
        assert "segments" in evidence.metadata
        assert evidence.metadata["language"] == "en"
        assert evidence.metadata["duration_seconds"] == 2.0

    def test_result_to_evidence_default_confidence(self):
        """Test evidence uses default confidence when result has none."""
        result = TranscriptionResult(
            id="test",
            text="Test text",
            source_filename="file.mp3",
            confidence=0.0,  # Zero confidence
        )

        evidence = result.to_evidence()

        # Should use default 0.85 for Whisper transcriptions
        assert evidence.confidence == 0.85

    def test_result_to_dict(self):
        """Test result serialization."""
        result = TranscriptionResult(
            id="trans_456",
            text="Serialized text",
            language="es",
            duration_seconds=10.0,
            source_filename="spanish.mp3",
            confidence=0.9,
        )

        data = result.to_dict()

        assert data["id"] == "trans_456"
        assert data["text"] == "Serialized text"
        assert data["language"] == "es"
        assert data["duration_seconds"] == 10.0
        assert data["source_filename"] == "spanish.mp3"
        assert data["confidence"] == 0.9

    def test_result_from_dict(self):
        """Test result deserialization."""
        data = {
            "id": "trans_789",
            "text": "Deserialized text",
            "segments": [{"start": 0.0, "end": 1.0, "text": "Segment", "confidence": 0.8}],
            "language": "fr",
            "duration_seconds": 3.0,
            "source_filename": "french.mp3",
            "word_count": 2,
            "confidence": 0.88,
        }

        result = TranscriptionResult.from_dict(data)

        assert result.id == "trans_789"
        assert result.text == "Deserialized text"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Segment"
        assert result.language == "fr"
        assert result.duration_seconds == 3.0
        assert result.source_filename == "french.mp3"
        assert result.confidence == 0.88


class TestWhisperConnector:
    """Tests for WhisperConnector."""

    @pytest.fixture
    def connector(self):
        """Create a Whisper connector for testing."""
        return WhisperConnector(
            api_key="test-api-key",
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        with patch.dict("os.environ", {}, clear=True):
            return WhisperConnector(api_key="")

    def test_connector_properties(self, connector):
        """Test connector property methods."""
        assert connector.source_type == SourceType.AUDIO_TRANSCRIPT
        assert connector.name == "Whisper"
        assert connector.model == "whisper-1"

    def test_connector_is_available_with_httpx(self, connector):
        """Test is_available when httpx is installed."""
        with patch("aragora.connectors.whisper.HTTPX_AVAILABLE", True):
            assert connector.is_available

    def test_connector_is_available_without_httpx(self, connector):
        """Test is_available when httpx is not installed."""
        with patch("aragora.connectors.whisper.HTTPX_AVAILABLE", False):
            assert not connector.is_available

    def test_connector_is_available_without_api_key(self):
        """Test is_available when API key is missing."""
        connector = WhisperConnector(api_key="")
        assert not connector.is_available

    def test_connector_custom_settings(self):
        """Test connector with custom settings."""
        connector = WhisperConnector(
            api_key="key",
            model="whisper-2",
            language="en",
            response_format="text",
            timeout=60,
            default_confidence=0.9,
        )

        assert connector.model == "whisper-2"
        assert connector.language == "en"
        assert connector.response_format == "text"
        assert connector.timeout == 60
        assert connector.default_confidence == 0.9

    def test_get_mime_type_mp3(self, connector):
        """Test MIME type detection for mp3."""
        assert connector._get_mime_type("audio.mp3") == "audio/mpeg"

    def test_get_mime_type_wav(self, connector):
        """Test MIME type detection for wav."""
        assert connector._get_mime_type("audio.wav") == "audio/wav"

    def test_get_mime_type_mp4(self, connector):
        """Test MIME type detection for mp4."""
        assert connector._get_mime_type("video.mp4") == "video/mp4"

    def test_get_mime_type_unknown(self, connector):
        """Test MIME type detection for unknown extension."""
        assert connector._get_mime_type("file.xyz") == "application/octet-stream"

    def test_get_mime_type_no_extension(self, connector):
        """Test MIME type detection for file without extension."""
        assert connector._get_mime_type("filename") == "application/octet-stream"

    def test_validate_file_success(self, connector):
        """Test file validation for valid file."""
        content = b"x" * 1000  # Small file
        connector._validate_file(content, "audio.mp3")  # Should not raise

    def test_validate_file_too_large(self, connector):
        """Test file validation rejects oversized files."""
        content = b"x" * (MAX_FILE_SIZE_BYTES + 1)

        with pytest.raises(ConnectorConfigError) as exc_info:
            connector._validate_file(content, "large.mp3")

        assert "too large" in str(exc_info.value).lower()

    def test_validate_file_unsupported_extension(self, connector):
        """Test file validation rejects unsupported extensions."""
        content = b"x" * 100

        with pytest.raises(ConnectorConfigError) as exc_info:
            connector._validate_file(content, "file.txt")

        assert "unsupported" in str(exc_info.value).lower()

    def test_parse_response_verbose_json(self, connector):
        """Test parsing verbose JSON response."""
        result = connector._parse_response(SAMPLE_VERBOSE_JSON_RESPONSE, "test.mp3")

        assert result.text == "Hello, this is a test transcription of audio content."
        assert result.language == "en"
        assert result.duration_seconds == 5.5
        assert result.source_filename == "test.mp3"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello, this is a test"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5

    def test_parse_response_simple_json(self, connector):
        """Test parsing simple JSON response."""
        result = connector._parse_response(SAMPLE_SIMPLE_JSON_RESPONSE, "simple.mp3")

        assert result.text == "Simple transcription without segments."
        assert result.source_filename == "simple.mp3"
        assert len(result.segments) == 0

    def test_parse_response_plain_text(self, connector):
        """Test parsing plain text response."""
        result = connector._parse_response("Plain text response", "plain.mp3")

        assert result.text == "Plain text response"
        assert result.source_filename == "plain.mp3"

    @pytest.mark.asyncio
    async def test_transcribe_not_available_no_httpx(self, connector):
        """Test transcribe fails when httpx not available."""
        with patch("aragora.connectors.whisper.HTTPX_AVAILABLE", False):
            with pytest.raises(ConnectorConfigError) as exc_info:
                await connector.transcribe(b"audio", "test.mp3")

            assert "httpx not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_not_available_no_api_key(self):
        """Test transcribe fails when API key not configured."""
        connector = WhisperConnector(api_key="")

        with pytest.raises(ConnectorConfigError) as exc_info:
            await connector.transcribe(b"audio", "test.mp3")

        assert "OPENAI_API_KEY" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_mocked(self, connector):
        """Test transcribe with mocked HTTP response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_VERBOSE_JSON_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            # Disable rate limiting for test
            connector._last_request_time = 0

            result = await connector.transcribe(b"audio content", "test.mp3")

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription of audio content."
            assert result.language == "en"
            assert len(result.segments) == 2

    @pytest.mark.asyncio
    async def test_transcribe_with_prompt(self, connector):
        """Test transcribe with context prompt."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SIMPLE_JSON_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            connector._last_request_time = 0

            result = await connector.transcribe(
                b"audio content",
                "meeting.mp3",
                prompt="This is a business meeting about quarterly results.",
            )

            # Verify prompt was included in request
            call_args = mock_client_instance.post.call_args
            data = call_args[1]["data"]
            assert data["prompt"] == "This is a business meeting about quarterly results."

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self):
        """Test transcribe with specified language."""
        connector = WhisperConnector(api_key="test-key", language="es")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"text": "Hola mundo", "language": "es"}
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            connector._last_request_time = 0

            result = await connector.transcribe(b"audio", "spanish.mp3")

            call_args = mock_client_instance.post.call_args
            data = call_args[1]["data"]
            assert data["language"] == "es"

    @pytest.mark.asyncio
    async def test_transcribe_rate_limit_error(self):
        """Test transcribe handles rate limit errors."""
        # Create connector with no retries to test rate limit behavior directly
        connector = WhisperConnector(api_key="test-key", max_retries=0)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "30"}

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            connector._last_request_time = 0

            # The connector raises ConnectorRateLimitError on 429
            with pytest.raises(ConnectorRateLimitError) as exc_info:
                await connector.transcribe(b"audio", "test.mp3")

            assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_transcribe_validates_file_size(self, connector):
        """Test transcribe validates file size before request."""
        oversized_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)

        with pytest.raises(ConnectorConfigError) as exc_info:
            await connector.transcribe(oversized_content, "huge.mp3")

        assert "too large" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_transcribe_validates_file_type(self, connector):
        """Test transcribe validates file type before request."""
        with pytest.raises(ConnectorConfigError) as exc_info:
            await connector.transcribe(b"content", "document.pdf")

        assert "unsupported" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_not_implemented(self, connector):
        """Test search raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            await connector.search("query")

        assert "does not support search" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_returns_cached(self, connector):
        """Test fetch returns cached evidence."""
        cached_evidence = Evidence(
            id="trans_cached",
            source_type=SourceType.AUDIO_TRANSCRIPT,
            source_id="cached.mp3",
            content="Cached transcription",
            title="Transcript: cached.mp3",
        )
        connector._cache_put("trans_cached", cached_evidence)

        result = await connector.fetch("trans_cached")

        assert result is not None
        assert result.id == "trans_cached"
        assert result.content == "Cached transcription"

    @pytest.mark.asyncio
    async def test_fetch_returns_none_for_uncached(self, connector):
        """Test fetch returns None for uncached evidence."""
        result = await connector.fetch("nonexistent_id")

        assert result is None

    def test_cache_result(self, connector):
        """Test caching a transcription result."""
        result = TranscriptionResult(
            id="trans_to_cache",
            text="Text to cache",
            source_filename="file.mp3",
        )

        evidence = connector.cache_result(result)

        assert isinstance(evidence, Evidence)
        assert evidence.id == "trans_to_cache"
        assert evidence.content == "Text to cache"

        # Verify it's actually cached
        cached = connector._cache_get("trans_to_cache")
        assert cached is not None
        assert cached.id == "trans_to_cache"

    @pytest.mark.asyncio
    async def test_transcribe_stream_not_configured(self):
        """Test stream transcription fails when not configured."""
        connector = WhisperConnector(api_key="")

        async def mock_chunks():
            yield b"chunk1"

        with pytest.raises(ConnectorConfigError):
            async for _ in connector.transcribe_stream(mock_chunks()):
                pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_mocked(self, connector):
        """Test stream transcription with mocked responses."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "text": "Streamed segment",
                "duration": 3.0,
            }
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            connector._last_request_time = 0

            # Create enough audio data to trigger transcription
            # Assuming 16kHz, 16-bit mono: 32000 bytes/second, need 3 seconds = 96000 bytes
            async def audio_chunks():
                for _ in range(10):
                    yield b"x" * 10000

            segments = []
            async for segment in connector.transcribe_stream(
                audio_chunks(), chunk_duration_ms=3000
            ):
                segments.append(segment)

            # Should have at least one segment
            assert len(segments) >= 1
            assert all(isinstance(s, TranscriptionSegment) for s in segments)


class TestSupportedFormats:
    """Tests for format support functions."""

    def test_supported_audio_extensions(self):
        """Test audio extensions are defined."""
        assert ".mp3" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".m4a" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".webm" in SUPPORTED_AUDIO_EXTENSIONS

    def test_supported_video_extensions(self):
        """Test video extensions are defined."""
        assert ".mp4" in SUPPORTED_VIDEO_EXTENSIONS
        assert ".mov" in SUPPORTED_VIDEO_EXTENSIONS
        assert ".avi" in SUPPORTED_VIDEO_EXTENSIONS
        assert ".mkv" in SUPPORTED_VIDEO_EXTENSIONS

    def test_all_supported_extensions(self):
        """Test combined extensions."""
        assert ALL_SUPPORTED_EXTENSIONS == SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

    def test_max_file_size(self):
        """Test max file size constant."""
        assert MAX_FILE_SIZE_MB == 25
        assert MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024

    def test_is_supported_audio_true(self):
        """Test is_supported_audio returns True for audio files."""
        assert is_supported_audio("song.mp3")
        assert is_supported_audio("voice.wav")
        assert is_supported_audio("podcast.m4a")
        assert is_supported_audio("FILE.MP3")  # Case insensitive

    def test_is_supported_audio_false(self):
        """Test is_supported_audio returns False for non-audio files."""
        assert not is_supported_audio("video.mp4")
        assert not is_supported_audio("document.pdf")
        assert not is_supported_audio("image.jpg")
        assert not is_supported_audio("noextension")

    def test_is_supported_video_true(self):
        """Test is_supported_video returns True for video files."""
        assert is_supported_video("movie.mp4")
        assert is_supported_video("clip.mov")
        assert is_supported_video("video.avi")
        assert is_supported_video("RECORDING.MKV")  # Case insensitive

    def test_is_supported_video_false(self):
        """Test is_supported_video returns False for non-video files."""
        assert not is_supported_video("audio.mp3")
        assert not is_supported_video("document.pdf")
        assert not is_supported_video("image.jpg")

    def test_is_supported_media_audio(self):
        """Test is_supported_media returns True for audio."""
        assert is_supported_media("song.mp3")
        assert is_supported_media("voice.wav")

    def test_is_supported_media_video(self):
        """Test is_supported_media returns True for video."""
        assert is_supported_media("movie.mp4")
        assert is_supported_media("clip.mov")

    def test_is_supported_media_false(self):
        """Test is_supported_media returns False for unsupported."""
        assert not is_supported_media("document.pdf")
        assert not is_supported_media("image.jpg")

    def test_get_supported_formats(self):
        """Test get_supported_formats returns expected structure."""
        formats = get_supported_formats()

        assert "audio" in formats
        assert "video" in formats
        assert "max_size_mb" in formats
        assert "model" in formats

        assert ".mp3" in formats["audio"]
        assert ".mp4" in formats["video"]
        assert formats["max_size_mb"] == 25
        assert formats["model"] == "whisper-1"


class TestWhisperConnectorRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def connector(self):
        """Create a connector for rate limit testing."""
        return WhisperConnector(api_key="test-key")

    @pytest.mark.asyncio
    async def test_rate_limit_delay(self, connector):
        """Test rate limiting enforces delay between requests."""
        import time

        # Set last request to now
        connector._last_request_time = time.time()

        start = time.time()
        await connector._rate_limit()
        elapsed = time.time() - start

        # Should have waited approximately RATE_LIMIT_DELAY
        assert elapsed >= connector.RATE_LIMIT_DELAY * 0.9  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_no_delay_first_request(self, connector):
        """Test no delay on first request."""
        import time

        # Clear last request time
        connector._last_request_time = 0

        start = time.time()
        await connector._rate_limit()
        elapsed = time.time() - start

        # Should be nearly instant
        assert elapsed < 0.1


class TestWhisperConnectorAPIKey:
    """Tests for API key handling."""

    def test_api_key_from_constructor(self):
        """Test API key can be provided in constructor."""
        connector = WhisperConnector(api_key="explicit-key")
        assert connector.api_key == "explicit-key"

    def test_api_key_from_environment(self):
        """Test API key is read from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            connector = WhisperConnector()
            assert connector.api_key == "env-key"

    def test_api_key_constructor_overrides_environment(self):
        """Test constructor API key takes precedence over environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            connector = WhisperConnector(api_key="constructor-key")
            assert connector.api_key == "constructor-key"

    def test_no_api_key_logs_warning(self):
        """Test warning is logged when no API key configured."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("aragora.connectors.whisper.logger") as mock_logger:
                WhisperConnector(api_key="")
                mock_logger.warning.assert_called()


class TestWhisperConnectorVideoSupport:
    """Tests for video file handling."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        return WhisperConnector(api_key="test-key")

    def test_validate_video_files(self, connector):
        """Test video file validation passes."""
        content = b"video content"
        connector._validate_file(content, "meeting.mp4")  # Should not raise
        connector._validate_file(content, "recording.mov")  # Should not raise
        connector._validate_file(content, "video.avi")  # Should not raise

    def test_video_mime_types(self, connector):
        """Test video MIME types are correct."""
        assert connector._get_mime_type("video.mp4") == "video/mp4"
        assert connector._get_mime_type("clip.mov") == "video/quicktime"
        assert connector._get_mime_type("movie.avi") == "video/x-msvideo"
        assert connector._get_mime_type("video.mkv") == "video/x-matroska"

    @pytest.mark.asyncio
    async def test_transcribe_video_file(self, connector):
        """Test transcribing a video file."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "text": "Video transcription",
                "duration": 60.0,
            }
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            connector._last_request_time = 0

            result = await connector.transcribe(b"video content", "meeting.mp4")

            assert result.text == "Video transcription"
            assert result.source_filename == "meeting.mp4"

            # Verify correct MIME type was used
            call_args = mock_client_instance.post.call_args
            files = call_args[1]["files"]
            assert files["file"][2] == "video/mp4"
