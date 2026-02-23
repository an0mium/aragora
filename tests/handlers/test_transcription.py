"""Tests for transcription handler.

Covers:
- Circuit breaker state machine (CLOSED/OPEN/HALF_OPEN transitions)
- File security validation (null bytes, double extensions, magic bytes)
- Route matching (can_handle)
- GET /config endpoint
- GET /status/:id endpoint
- POST audio/video/youtube transcription endpoints
- Rate limiting
- Error handling and response codes
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.transcription import (
    AUDIO_FORMATS,
    MAX_AUDIO_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    VIDEO_FORMATS,
    TranscriptionCircuitBreaker,
    TranscriptionHandler,
    _validate_file_content,
    _validate_filename_security,
    reset_transcription_circuit_breaker,
)


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestTranscriptionCircuitBreaker:
    """Test circuit breaker state machine."""

    def setup_method(self):
        self.cb = TranscriptionCircuitBreaker(
            failure_threshold=3,
            cooldown_seconds=1.0,
            half_open_max_calls=2,
        )

    def test_initial_state_closed(self):
        assert self.cb.state == TranscriptionCircuitBreaker.CLOSED

    def test_can_proceed_when_closed(self):
        assert self.cb.can_proceed() is True

    def test_stays_closed_below_threshold(self):
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state == TranscriptionCircuitBreaker.CLOSED
        assert self.cb.can_proceed() is True

    def test_opens_at_threshold(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == TranscriptionCircuitBreaker.OPEN
        assert self.cb.can_proceed() is False

    def test_transitions_to_half_open_after_cooldown(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == TranscriptionCircuitBreaker.OPEN

        # Fast-forward past cooldown
        self.cb._last_failure_time = time.time() - 2.0
        assert self.cb.state == TranscriptionCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb._last_failure_time = time.time() - 2.0

        assert self.cb.can_proceed() is True  # 1st test call
        assert self.cb.can_proceed() is True  # 2nd test call
        assert self.cb.can_proceed() is False  # exceeded max

    def test_half_open_to_closed_on_success(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb._last_failure_time = time.time() - 2.0

        self.cb.can_proceed()
        self.cb.record_success()
        self.cb.can_proceed()
        self.cb.record_success()

        assert self.cb.state == TranscriptionCircuitBreaker.CLOSED

    def test_half_open_to_open_on_failure(self):
        for _ in range(3):
            self.cb.record_failure()
        self.cb._last_failure_time = time.time() - 2.0

        self.cb.can_proceed()
        self.cb.record_failure()

        assert self.cb.state == TranscriptionCircuitBreaker.OPEN

    def test_success_resets_failure_count_when_closed(self):
        self.cb.record_failure()
        self.cb.record_failure()
        self.cb.record_success()
        # Should be back to 0 failures, so 3 more needed
        self.cb.record_failure()
        self.cb.record_failure()
        assert self.cb.state == TranscriptionCircuitBreaker.CLOSED

    def test_get_status_returns_dict(self):
        status = self.cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status
        assert status["state"] == "closed"
        assert status["failure_count"] == 0

    def test_reset_returns_to_closed(self):
        for _ in range(3):
            self.cb.record_failure()
        assert self.cb.state == TranscriptionCircuitBreaker.OPEN

        self.cb.reset()
        assert self.cb.state == TranscriptionCircuitBreaker.CLOSED
        assert self.cb.can_proceed() is True


# ============================================================================
# Filename Security Validation
# ============================================================================


class TestFilenameSecurityValidation:
    """Test filename security checks."""

    def test_valid_filename(self):
        valid, err = _validate_filename_security("audio.mp3")
        assert valid is True
        assert err is None

    def test_null_byte_rejected(self):
        valid, err = _validate_filename_security("audio\x00.mp3")
        assert valid is False
        assert "null bytes" in err

    def test_double_extension_exe_rejected(self):
        valid, err = _validate_filename_security("audio.mp3.exe")
        assert valid is False
        assert "double extensions" in err

    def test_double_extension_bat_rejected(self):
        valid, err = _validate_filename_security("video.mp4.bat")
        assert valid is False
        assert "double extensions" in err

    def test_double_extension_py_rejected(self):
        valid, err = _validate_filename_security("recording.wav.py")
        assert valid is False
        assert "double extensions" in err

    def test_double_extension_case_insensitive(self):
        valid, err = _validate_filename_security("audio.MP3.EXE")
        assert valid is False
        assert "double extensions" in err

    def test_normal_dotted_name_allowed(self):
        valid, err = _validate_filename_security("my.recording.mp3")
        # This should pass since the pattern targets specific dangerous combos
        assert valid is True

    def test_empty_filename(self):
        valid, err = _validate_filename_security("")
        assert valid is True  # No null bytes or double extensions


# ============================================================================
# File Content (Magic Byte) Validation
# ============================================================================


class TestFileContentValidation:
    """Test magic byte content validation."""

    def test_empty_file_rejected(self):
        valid, err = _validate_file_content(b"", ".mp3")
        assert valid is False
        assert "Empty file" in err

    def test_valid_mp3_id3(self):
        valid, err = _validate_file_content(b"ID3" + b"\x00" * 100, ".mp3")
        assert valid is True

    def test_valid_mp3_frame_sync(self):
        valid, err = _validate_file_content(b"\xff\xfb" + b"\x00" * 100, ".mp3")
        assert valid is True

    def test_valid_wav(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".wav")
        assert valid is True

    def test_wav_riff_without_wave_rejected(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"JUNK" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".wav")
        assert valid is False

    def test_valid_ogg(self):
        valid, err = _validate_file_content(b"OggS" + b"\x00" * 100, ".ogg")
        assert valid is True

    def test_valid_flac(self):
        valid, err = _validate_file_content(b"fLaC" + b"\x00" * 100, ".flac")
        assert valid is True

    def test_valid_m4a(self):
        data = b"\x00\x00\x00\x00" + b"ftyp" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".m4a")
        assert valid is True

    def test_valid_webm(self):
        valid, err = _validate_file_content(b"\x1a\x45\xdf\xa3" + b"\x00" * 100, ".webm")
        assert valid is True

    def test_valid_mp4_video(self):
        data = b"\x00\x00\x00\x00" + b"ftyp" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".mp4", is_video=True)
        assert valid is True

    def test_valid_avi(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 100
        valid, err = _validate_file_content(data, ".avi", is_video=True)
        assert valid is True

    def test_avi_riff_without_avi_rejected(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".avi", is_video=True)
        assert valid is False

    def test_unknown_extension_rejected(self):
        valid, err = _validate_file_content(b"\x00" * 100, ".xyz")
        assert valid is False
        assert "Unknown file extension" in err

    def test_wrong_magic_bytes_rejected(self):
        valid, err = _validate_file_content(b"\x00" * 100, ".mp3")
        assert valid is False
        assert "does not match" in err

    def test_file_too_short_for_signature(self):
        valid, err = _validate_file_content(b"\xff", ".mp3")
        assert valid is False
        assert "does not match" in err


# ============================================================================
# Route Matching (can_handle)
# ============================================================================


class TestTranscriptionCanHandle:
    """Test route matching."""

    def setup_method(self):
        self.handler = TranscriptionHandler()

    def test_audio_route(self):
        assert self.handler.can_handle("/api/v1/transcription/audio") is True

    def test_video_route(self):
        assert self.handler.can_handle("/api/v1/transcription/video") is True

    def test_youtube_route(self):
        assert self.handler.can_handle("/api/v1/transcription/youtube") is True

    def test_youtube_info_route(self):
        assert self.handler.can_handle("/api/v1/transcription/youtube/info") is True

    def test_config_route(self):
        assert self.handler.can_handle("/api/v1/transcription/config") is True

    def test_status_route(self):
        assert self.handler.can_handle("/api/v1/transcription/status/abc123") is True

    def test_alias_audio_route(self):
        assert self.handler.can_handle("/api/v1/transcribe/audio") is True

    def test_alias_video_route(self):
        assert self.handler.can_handle("/api/v1/transcribe/video") is True

    def test_unrelated_route(self):
        assert self.handler.can_handle("/api/v1/debates") is False

    def test_partial_match_rejected(self):
        assert self.handler.can_handle("/api/v1/transcription") is False


# ============================================================================
# GET /config Endpoint
# ============================================================================


class TestGetConfig:
    """Test GET /api/v1/transcription/config."""

    def setup_method(self):
        self.handler = TranscriptionHandler()
        reset_transcription_circuit_breaker()

    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(False, "No backend"),
    )
    def test_config_when_unavailable(self, mock_check):
        result = self.handler._get_config()
        assert result.status_code == 200
        body = _body(result)
        assert body["available"] is False
        assert body["error"] == "No backend"
        assert "audio_formats" in body
        assert "video_formats" in body

    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    @patch("aragora.server.handlers.transcription.get_available_backends", create=True)
    @patch(
        "aragora.server.handlers.transcription.WHISPER_MODELS",
        {"base": {}, "large": {}},
        create=True,
    )
    def test_config_when_available(self, mock_backends, mock_check):
        # Need to patch the lazy imports inside _get_config
        with patch.dict(
            "sys.modules",
            {
                "aragora.transcription": MagicMock(
                    get_available_backends=lambda: ["whisper", "openai"]
                ),
                "aragora.transcription.whisper_backend": MagicMock(
                    WHISPER_MODELS={"base": {}, "large": {}}
                ),
            },
        ):
            result = self.handler._get_config()
            assert result.status_code == 200
            body = _body(result)
            assert body["available"] is True


# ============================================================================
# GET /status/:id Endpoint
# ============================================================================


class TestGetStatus:
    """Test GET /api/v1/transcription/status/:id."""

    def setup_method(self):
        self.handler = TranscriptionHandler()

    @patch(
        "aragora.server.handlers.transcription._get_job",
        return_value=None,
    )
    def test_status_not_found(self, mock_get):
        result = self.handler._get_status("nonexistent")
        assert result.status_code == 404

    @patch(
        "aragora.server.handlers.transcription._get_job",
        return_value={
            "status": "completed",
            "progress": 100,
            "result": {"text": "hello world"},
        },
    )
    def test_status_found(self, mock_get):
        result = self.handler._get_status("job123")
        assert result.status_code == 200
        body = _body(result)
        assert body["job_id"] == "job123"
        assert body["status"] == "completed"
        assert body["progress"] == 100

    @patch(
        "aragora.server.handlers.transcription._get_job",
        return_value={
            "status": "processing",
            "progress": 50,
        },
    )
    def test_status_processing(self, mock_get):
        result = self.handler._get_status("job456")
        assert result.status_code == 200
        body = _body(result)
        assert body["status"] == "processing"
        assert body["progress"] == 50


# ============================================================================
# POST Audio Transcription
# ============================================================================


@dataclass
class MockSegment:
    start: float = 0.0
    end: float = 1.0
    text: str = "hello"


@dataclass
class MockTranscriptionResult:
    text: str = "hello world"
    language: str = "en"
    duration: float = 5.0
    segments: list = field(default_factory=lambda: [MockSegment()])
    backend: str = "whisper"
    processing_time: float = 1.0

    def to_dict(self):
        return {"text": self.text, "language": self.language}


def _make_mock_handler(content_type="multipart/form-data; boundary=abc", content_length="100"):
    """Create a mock HTTP handler with proper headers."""
    mock = MagicMock()
    _headers = {
        "Content-Type": content_type,
        "Content-Length": content_length,
    }
    mock.headers = MagicMock()
    mock.headers.get = lambda k, d="": _headers.get(k, d)
    mock.headers.__getitem__ = lambda self_unused, k: _headers[k]
    mock.headers.__contains__ = lambda self_unused, k: k in _headers
    mock.client_address = ("127.0.0.1", 12345)
    return mock


class TestAudioTranscription:
    """Test POST /api/v1/transcription/audio."""

    def setup_method(self):
        self.handler = TranscriptionHandler()
        reset_transcription_circuit_breaker()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_request(self):
        # Open the circuit breaker
        cb = TranscriptionCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker",
            cb,
        ):
            result = await self.handler._handle_audio_transcription(_make_mock_handler())
            assert result.status_code == 503
            assert "temporarily unavailable" in _body(result)["error"]

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_not_multipart_rejected(self, mock_check):
        mock = _make_mock_handler(content_type="application/json")
        result = await self.handler._handle_audio_transcription(mock)
        assert result.status_code == 400
        assert "multipart" in _body(result)["error"]

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_content_length_too_large(self, mock_check):
        huge_size = str(MAX_AUDIO_SIZE_MB * 1024 * 1024 + 1)
        mock = _make_mock_handler(content_length=huge_size)
        result = await self.handler._handle_audio_transcription(mock)
        assert result.status_code == 413

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(False, "No backend"),
    )
    async def test_backend_unavailable(self, mock_check):
        result = await self.handler._handle_audio_transcription(_make_mock_handler())
        assert result.status_code == 503


# ============================================================================
# POST Video Transcription
# ============================================================================


class TestVideoTranscription:
    """Test POST /api/v1/transcription/video."""

    def setup_method(self):
        self.handler = TranscriptionHandler()
        reset_transcription_circuit_breaker()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_video(self):
        cb = TranscriptionCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker",
            cb,
        ):
            result = await self.handler._handle_video_transcription(_make_mock_handler())
            assert result.status_code == 503

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_not_multipart_rejected(self, mock_check):
        mock = _make_mock_handler(content_type="application/json")
        result = await self.handler._handle_video_transcription(mock)
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_video_content_length_too_large(self, mock_check):
        huge_size = str(MAX_VIDEO_SIZE_MB * 1024 * 1024 + 1)
        mock = _make_mock_handler(content_length=huge_size)
        result = await self.handler._handle_video_transcription(mock)
        assert result.status_code == 413


# ============================================================================
# POST YouTube Transcription
# ============================================================================


class TestYouTubeTranscription:
    """Test POST /api/v1/transcription/youtube."""

    def setup_method(self):
        self.handler = TranscriptionHandler()
        reset_transcription_circuit_breaker()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_youtube(self):
        cb = TranscriptionCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker",
            cb,
        ):
            result = await self.handler._handle_youtube_transcription(MagicMock())
            assert result.status_code == 503

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(False, "No backend"),
    )
    async def test_youtube_backend_unavailable(self, mock_check):
        result = await self.handler._handle_youtube_transcription(MagicMock())
        assert result.status_code == 503

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_youtube_missing_url(self, mock_check):
        self.handler.read_json_body_validated = MagicMock(return_value=({}, None))
        result = await self.handler._handle_youtube_transcription(MagicMock())
        assert result.status_code == 400
        assert "url" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_youtube_invalid_url(self, mock_check):
        self.handler.read_json_body_validated = MagicMock(
            return_value=({"url": "https://notyoutube.com/video"}, None)
        )
        mock_fetcher_cls = MagicMock()
        mock_fetcher_cls.is_youtube_url.return_value = False
        with patch.dict(
            "sys.modules",
            {
                "aragora.transcription.youtube": MagicMock(YouTubeFetcher=mock_fetcher_cls),
            },
        ):
            result = await self.handler._handle_youtube_transcription(MagicMock())
            assert result.status_code == 400
            assert "YouTube" in _body(result)["error"]

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    @patch("aragora.server.handlers.transcription._save_job")
    async def test_youtube_success(self, mock_save, mock_check):
        self.handler.read_json_body_validated = MagicMock(
            return_value=({"url": "https://youtube.com/watch?v=abc"}, None)
        )
        mock_result = MockTranscriptionResult()
        mock_fetcher_cls = MagicMock()
        mock_fetcher_cls.is_youtube_url.return_value = True

        with patch.dict(
            "sys.modules",
            {
                "aragora.transcription.youtube": MagicMock(YouTubeFetcher=mock_fetcher_cls),
                "aragora.transcription": MagicMock(
                    transcribe_youtube=AsyncMock(return_value=mock_result)
                ),
            },
        ):
            result = await self.handler._handle_youtube_transcription(MagicMock())
            assert result.status_code == 200
            body = _body(result)
            assert body["status"] == "completed"
            assert body["text"] == "hello world"


# ============================================================================
# POST YouTube Info
# ============================================================================


class TestYouTubeInfo:
    """Test POST /api/v1/transcription/youtube/info."""

    def setup_method(self):
        self.handler = TranscriptionHandler()

    @pytest.mark.asyncio
    async def test_missing_url(self):
        self.handler.read_json_body_validated = MagicMock(return_value=({}, None))
        result = await self.handler._handle_youtube_info(MagicMock())
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_url(self):
        self.handler.read_json_body_validated = MagicMock(
            return_value=({"url": "https://example.com"}, None)
        )
        mock_fetcher_cls = MagicMock()
        mock_fetcher_cls.is_youtube_url.return_value = False
        with patch.dict(
            "sys.modules",
            {
                "aragora.transcription.youtube": MagicMock(YouTubeFetcher=mock_fetcher_cls),
            },
        ):
            result = await self.handler._handle_youtube_info(MagicMock())
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_body_parse_error(self):
        from aragora.server.handlers.base import error_response

        err_result = error_response("bad json", 400)
        self.handler.read_json_body_validated = MagicMock(return_value=(None, err_result))
        result = await self.handler._handle_youtube_info(MagicMock())
        assert result.status_code == 400


# ============================================================================
# Handle (GET dispatcher)
# ============================================================================


class TestHandleGet:
    """Test handle() GET dispatcher."""

    def setup_method(self):
        self.handler = TranscriptionHandler()
        reset_transcription_circuit_breaker()

    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(False, "No backend"),
    )
    def test_config_route_dispatches(self, mock_check):
        result = self.handler.handle("/api/v1/transcription/config", {})
        # Should return a result (not None)
        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.transcription._get_job", return_value=None)
    def test_status_route_dispatches(self, mock_get):
        result = self.handler.handle("/api/v1/transcription/status/job123", {})
        assert result is not None
        assert result.status_code == 404  # job not found

    @patch(
        "aragora.server.handlers.transcription._get_job",
        return_value={"status": "completed", "progress": 100, "result": {}},
    )
    def test_status_extracts_job_id(self, mock_get):
        result = self.handler.handle("/api/v1/transcription/status/my-job-id", {})
        mock_get.assert_called_with("my-job-id")

    def test_unknown_get_route_returns_none(self):
        result = self.handler.handle("/api/v1/transcription/unknown", {})
        assert result is None


# ============================================================================
# Handle POST (dispatcher)
# ============================================================================


class TestHandlePost:
    """Test handle_post() dispatcher."""

    def setup_method(self):
        self.handler = TranscriptionHandler()
        reset_transcription_circuit_breaker()

    @pytest.mark.asyncio
    async def test_unknown_post_route_returns_none(self):
        mock = _make_mock_handler()
        with patch(
            "aragora.server.handlers.transcription._audio_limiter",
            MagicMock(is_allowed=MagicMock(return_value=True)),
        ):
            result = await self.handler.handle_post("/api/v1/transcription/unknown", {}, mock)
            assert result is None

    @pytest.mark.asyncio
    async def test_audio_rate_limit_429(self):
        mock = _make_mock_handler()
        limiter = MagicMock(is_allowed=MagicMock(return_value=False))
        with patch("aragora.server.handlers.transcription._audio_limiter", limiter):
            result = await self.handler.handle_post("/api/v1/transcription/audio", {}, mock)
            assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_youtube_rate_limit_429(self):
        mock = _make_mock_handler()
        limiter = MagicMock(is_allowed=MagicMock(return_value=False))
        with patch("aragora.server.handlers.transcription._youtube_limiter", limiter):
            result = await self.handler.handle_post("/api/v1/transcription/youtube", {}, mock)
            assert result.status_code == 429

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    )
    async def test_alias_route_audio(self, mock_check):
        """Test that /api/v1/transcribe/audio is routed to audio handler."""
        mock = _make_mock_handler(content_type="application/json")
        limiter = MagicMock(is_allowed=MagicMock(return_value=True))
        with patch("aragora.server.handlers.transcription._audio_limiter", limiter):
            result = await self.handler.handle_post("/api/v1/transcribe/audio", {}, mock)
            # Should reach the audio handler (which rejects non-multipart with 400)
            assert result is not None
            assert result.status_code == 400


# ============================================================================
# Constants Validation
# ============================================================================


class TestConstants:
    """Verify handler constants are reasonable."""

    def test_audio_formats(self):
        expected = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac"}
        assert AUDIO_FORMATS == expected

    def test_video_formats(self):
        expected = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
        assert VIDEO_FORMATS == expected

    def test_max_audio_size(self):
        assert MAX_AUDIO_SIZE_MB == 100

    def test_max_video_size(self):
        assert MAX_VIDEO_SIZE_MB == 500
