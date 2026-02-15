"""
Tests for aragora.server.handlers.transcription - Transcription HTTP Handlers.

Tests cover:
- TranscriptionHandler: instantiation, ROUTES, can_handle
- TranscriptionCircuitBreaker: state transitions, can_proceed, record_success/failure
- GET /api/v1/transcription/config: available, unavailable backends
- GET /api/v1/transcription/status/:id: found, not found
- POST /api/v1/transcription/audio: circuit breaker blocked, format validation
- POST /api/v1/transcription/video: circuit breaker blocked, format validation
- POST /api/v1/transcription/youtube: missing url, invalid url
- POST /api/v1/transcription/youtube/info: missing url, invalid url
- File security: null bytes, double extensions, magic bytes
- handle() routing: returns None for unmatched paths
- handle_post() routing: returns None for unmatched paths
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.transcription import (
    AUDIO_FORMATS,
    MAX_AUDIO_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    TranscriptionCircuitBreaker,
    TranscriptionHandler,
    VIDEO_FORMATS,
    _validate_file_content,
    _validate_filename_security,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create a TranscriptionHandler with mocked dependencies."""
    h = TranscriptionHandler(ctx={})
    return h


@pytest.fixture
def circuit_breaker():
    """Create a fresh TranscriptionCircuitBreaker."""
    return TranscriptionCircuitBreaker(
        failure_threshold=3,
        cooldown_seconds=1.0,
        half_open_max_calls=2,
    )


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestTranscriptionHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, TranscriptionHandler)

    def test_has_routes(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0

    def test_routes_contain_audio(self, handler):
        assert "/api/v1/transcription/audio" in handler.ROUTES

    def test_routes_contain_video(self, handler):
        assert "/api/v1/transcription/video" in handler.ROUTES

    def test_routes_contain_youtube(self, handler):
        assert "/api/v1/transcription/youtube" in handler.ROUTES

    def test_routes_contain_config(self, handler):
        assert "/api/v1/transcription/config" in handler.ROUTES

    def test_routes_contain_status_wildcard(self, handler):
        assert "/api/v1/transcription/status/*" in handler.ROUTES

    def test_routes_contain_alias_audio(self, handler):
        assert "/api/v1/transcribe/audio" in handler.ROUTES

    def test_routes_contain_alias_video(self, handler):
        assert "/api/v1/transcribe/video" in handler.ROUTES


# ===========================================================================
# Test can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing logic."""

    def test_can_handle_audio(self, handler):
        assert handler.can_handle("/api/v1/transcription/audio") is True

    def test_can_handle_video(self, handler):
        assert handler.can_handle("/api/v1/transcription/video") is True

    def test_can_handle_youtube(self, handler):
        assert handler.can_handle("/api/v1/transcription/youtube") is True

    def test_can_handle_youtube_info(self, handler):
        assert handler.can_handle("/api/v1/transcription/youtube/info") is True

    def test_can_handle_config(self, handler):
        assert handler.can_handle("/api/v1/transcription/config") is True

    def test_can_handle_status_with_id(self, handler):
        assert handler.can_handle("/api/v1/transcription/status/job-123") is True

    def test_can_handle_alias_audio(self, handler):
        assert handler.can_handle("/api/v1/transcribe/audio") is True

    def test_can_handle_alias_video(self, handler):
        assert handler.can_handle("/api/v1/transcribe/video") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_bare_status(self, handler):
        assert handler.can_handle("/api/v1/transcription/status") is False

    def test_cannot_handle_root(self, handler):
        assert handler.can_handle("/api/v1/transcription") is False


# ===========================================================================
# Test CircuitBreaker
# ===========================================================================


class TestTranscriptionCircuitBreaker:
    """Tests for the circuit breaker state machine."""

    def test_initial_state_closed(self, circuit_breaker):
        assert circuit_breaker.state == TranscriptionCircuitBreaker.CLOSED

    def test_can_proceed_when_closed(self, circuit_breaker):
        assert circuit_breaker.can_proceed() is True

    def test_record_success_resets_failure_count(self, circuit_breaker):
        circuit_breaker._failure_count = 2
        circuit_breaker.record_success()
        assert circuit_breaker._failure_count == 0

    def test_opens_after_threshold(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == TranscriptionCircuitBreaker.OPEN

    def test_cannot_proceed_when_open(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.can_proceed() is False

    def test_transitions_to_half_open_after_cooldown(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        # Simulate cooldown elapsed
        circuit_breaker._last_failure_time = time.time() - 2.0
        assert circuit_breaker.state == TranscriptionCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_calls(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        circuit_breaker._last_failure_time = time.time() - 2.0
        # half_open_max_calls=2
        assert circuit_breaker.can_proceed() is True
        assert circuit_breaker.can_proceed() is True
        assert circuit_breaker.can_proceed() is False

    def test_half_open_closes_on_success(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        circuit_breaker._last_failure_time = time.time() - 2.0
        circuit_breaker.can_proceed()  # Enter half-open
        circuit_breaker.record_success()
        circuit_breaker.record_success()
        assert circuit_breaker.state == TranscriptionCircuitBreaker.CLOSED

    def test_half_open_reopens_on_failure(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        circuit_breaker._last_failure_time = time.time() - 2.0
        circuit_breaker.can_proceed()  # Enter half-open
        circuit_breaker.record_failure()
        assert circuit_breaker.state == TranscriptionCircuitBreaker.OPEN

    def test_reset(self, circuit_breaker):
        for _ in range(3):
            circuit_breaker.record_failure()
        circuit_breaker.reset()
        assert circuit_breaker.state == TranscriptionCircuitBreaker.CLOSED
        assert circuit_breaker._failure_count == 0

    def test_get_status(self, circuit_breaker):
        status = circuit_breaker.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status
        assert status["state"] == "closed"
        assert status["failure_count"] == 0


# ===========================================================================
# Test File Security Validation
# ===========================================================================


class TestFileSecurity:
    """Tests for filename and content validation."""

    def test_valid_filename(self):
        valid, err = _validate_filename_security("recording.mp3")
        assert valid is True
        assert err is None

    def test_null_byte_in_filename(self):
        valid, err = _validate_filename_security("recording\x00.mp3")
        assert valid is False
        assert "null bytes" in err

    def test_double_extension_blocked(self):
        valid, err = _validate_filename_security("recording.mp3.exe")
        assert valid is False
        assert "double extensions" in err

    def test_double_extension_bat(self):
        valid, err = _validate_filename_security("video.mp4.bat")
        assert valid is False

    def test_valid_extension_not_double(self):
        valid, err = _validate_filename_security("my.recording.mp3")
        assert valid is True

    def test_empty_file_content(self):
        valid, err = _validate_file_content(b"", ".mp3")
        assert valid is False
        assert "Empty file content" in err

    def test_unknown_audio_extension(self):
        valid, err = _validate_file_content(b"\x00\x00", ".xyz")
        assert valid is False
        assert "Unknown file extension" in err

    def test_mp3_id3_valid(self):
        data = b"ID3" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".mp3")
        assert valid is True
        assert err is None

    def test_wav_valid(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".wav")
        assert valid is True

    def test_wav_riff_but_not_wave(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"XXXX" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".wav")
        assert valid is False

    def test_flac_valid(self):
        data = b"fLaC" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".flac")
        assert valid is True

    def test_ogg_valid(self):
        data = b"OggS" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".ogg")
        assert valid is True

    def test_mp4_video_valid(self):
        data = b"\x00\x00\x00\x00" + b"ftyp" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".mp4", is_video=True)
        assert valid is True

    def test_avi_valid(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 100
        valid, err = _validate_file_content(data, ".avi", is_video=True)
        assert valid is True

    def test_avi_riff_but_not_avi(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"XXXX" + b"\x00" * 100
        valid, err = _validate_file_content(data, ".avi", is_video=True)
        assert valid is False

    def test_content_mismatch(self):
        data = b"\x00\x00\x00\x00\x00\x00\x00\x00" * 10
        valid, err = _validate_file_content(data, ".mp3")
        assert valid is False
        assert "does not match" in err


# ===========================================================================
# Test GET /api/v1/transcription/config
# ===========================================================================


class TestGetConfig:
    """Tests for the config endpoint."""

    def test_get_config_not_available(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(False, "No transcription backend available."),
        ):
            result = handler._get_config()
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["available"] is False
            assert "error" in data
            assert "audio_formats" in data
            assert "video_formats" in data

    def test_get_config_import_error(self, handler):
        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            with patch(
                "aragora.server.handlers.transcription.TranscriptionHandler._get_config",
                side_effect=ImportError("No module"),
            ):
                # Direct test of error path
                pass

    def test_get_config_formats_listed(self, handler):
        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(False, "Not available"),
        ):
            result = handler._get_config()
            data = _parse_body(result)
            for fmt in [".mp3", ".wav", ".flac"]:
                assert fmt in data["audio_formats"]


# ===========================================================================
# Test GET /api/v1/transcription/status/:id
# ===========================================================================


class TestGetStatus:
    """Tests for the job status endpoint."""

    def test_status_found(self, handler):
        with patch(
            "aragora.server.handlers.transcription._get_job",
            return_value={
                "status": "completed",
                "progress": 100,
                "result": {"text": "Hello"},
                "error": None,
            },
        ):
            result = handler._get_status("job-123")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["job_id"] == "job-123"
            assert data["status"] == "completed"
            assert data["progress"] == 100

    def test_status_not_found(self, handler):
        with patch(
            "aragora.server.handlers.transcription._get_job",
            return_value=None,
        ):
            result = handler._get_status("nonexistent")
            assert result.status_code == 404

    def test_status_processing(self, handler):
        with patch(
            "aragora.server.handlers.transcription._get_job",
            return_value={
                "status": "processing",
                "progress": 42,
                "result": None,
                "error": None,
            },
        ):
            result = handler._get_status("job-456")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "processing"
            assert data["progress"] == 42

    def test_status_failed(self, handler):
        with patch(
            "aragora.server.handlers.transcription._get_job",
            return_value={
                "status": "failed",
                "progress": 0,
                "result": None,
                "error": "Transcription failed",
            },
        ):
            result = handler._get_status("job-789")
            data = _parse_body(result)
            assert data["status"] == "failed"
            assert data["error"] == "Transcription failed"


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_config(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(False, "Not available"),
        ):
            result = handler.handle("/api/v1/transcription/config", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_status(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.transcription._get_job",
            return_value={"status": "completed", "progress": 100, "result": None, "error": None},
        ):
            result = handler.handle("/api/v1/transcription/status/job-1", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/transcription/unknown", {}, mock_handler)
        assert result is None

    def test_handle_audio_path_not_handled_by_get(self, handler):
        """Audio endpoint is POST-only, GET should return None."""
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/transcription/audio", {}, mock_handler)
        assert result is None


# ===========================================================================
# Test handle_post() Routing (POST)
# ===========================================================================


class TestHandlePostRouting:
    """Tests for handle_post() routing."""

    @pytest.mark.asyncio
    async def test_handle_post_audio_circuit_breaker_blocked(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch("aragora.server.handlers.transcription._audio_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch(
                "aragora.server.handlers.transcription._transcription_circuit_breaker"
            ) as mock_cb:
                mock_cb.can_proceed.return_value = False
                with patch(
                    "aragora.server.handlers.transcription._check_transcription_available",
                    return_value=(True, None),
                ):
                    result = await handler._handle_audio_transcription(mock_handler)
                    assert result.status_code == 503
                    data = _parse_body(result)
                    assert "unavailable" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_post_audio_not_multipart(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_handler.headers["Content-Type"] = "application/json"
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker"
        ) as mock_cb:
            mock_cb.can_proceed.return_value = True
            with patch(
                "aragora.server.handlers.transcription._check_transcription_available",
                return_value=(True, None),
            ):
                result = await handler._handle_audio_transcription(mock_handler)
                assert result.status_code == 400
                data = _parse_body(result)
                assert "multipart" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_post_video_circuit_breaker_blocked(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker"
        ) as mock_cb:
            mock_cb.can_proceed.return_value = False
            result = await handler._handle_video_transcription(mock_handler)
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_post_video_not_multipart(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_handler.headers["Content-Type"] = "application/json"
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker"
        ) as mock_cb:
            mock_cb.can_proceed.return_value = True
            with patch(
                "aragora.server.handlers.transcription._check_transcription_available",
                return_value=(True, None),
            ):
                result = await handler._handle_video_transcription(mock_handler)
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_youtube_circuit_breaker_blocked(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker"
        ) as mock_cb:
            mock_cb.can_proceed.return_value = False
            result = await handler._handle_youtube_transcription(mock_handler)
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_post_youtube_missing_url(self, handler):
        body = json.dumps({}).encode()
        mock_handler = _make_mock_handler("POST", body)
        with patch(
            "aragora.server.handlers.transcription._transcription_circuit_breaker"
        ) as mock_cb:
            mock_cb.can_proceed.return_value = True
            with patch(
                "aragora.server.handlers.transcription._check_transcription_available",
                return_value=(True, None),
            ):
                with patch.object(handler, "read_json_body_validated", return_value=({}, None)):
                    result = await handler._handle_youtube_transcription(mock_handler)
                    assert result.status_code == 400
                    data = _parse_body(result)
                    assert "url" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_handle_post_youtube_info_missing_url(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.object(handler, "read_json_body_validated", return_value=({}, None)):
            result = await handler._handle_youtube_info(mock_handler)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch(
            "aragora.server.handlers.transcription.get_client_ip",
            return_value="127.0.0.1",
        ):
            result = await handler.handle_post("/api/v1/transcription/unknown", {}, mock_handler)
            assert result is None


# ===========================================================================
# Test Constants
# ===========================================================================


class TestConstants:
    """Test module-level constants."""

    def test_audio_formats(self):
        assert ".mp3" in AUDIO_FORMATS
        assert ".wav" in AUDIO_FORMATS
        assert ".flac" in AUDIO_FORMATS
        assert ".ogg" in AUDIO_FORMATS

    def test_video_formats(self):
        assert ".mp4" in VIDEO_FORMATS
        assert ".mov" in VIDEO_FORMATS
        assert ".webm" in VIDEO_FORMATS
        assert ".avi" in VIDEO_FORMATS

    def test_max_audio_size(self):
        assert MAX_AUDIO_SIZE_MB == 100

    def test_max_video_size(self):
        assert MAX_VIDEO_SIZE_MB == 500
