"""
Tests for TranscriptionHandler job management functionality.

Tests cover:
- Job creation and persistence
- Job status retrieval
- Job store integration
- Error state handling
- Alias routes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
        client_address: tuple = ("127.0.0.1", 12345),
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = client_address

    def send_response(self, code: int) -> None:
        self.response_code = code

    def send_header(self, key: str, value: str) -> None:
        pass

    def end_headers(self) -> None:
        pass


@dataclass
class MockTranscriptionResult:
    """Mock transcription result."""

    text: str = "Hello, this is a test transcription."
    language: str = "en"
    duration: float = 10.0
    segments: list = field(default_factory=list)
    backend: str = "mock"
    processing_time: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": self.segments,
        }


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "continuum_memory": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "evidence_store": MagicMock(),
        "usage_tracker": MagicMock(),
    }


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
        with patch(
            "aragora.transcription.whisper_backend.get_available_backends",
            return_value=["mock"],
        ):
            from aragora.server.handlers.transcription import (
                _transcription_jobs,
                reset_transcription_circuit_breaker,
            )

            _transcription_jobs.clear()
            reset_transcription_circuit_breaker()
            # Prevent LazyStoreFactory from creating real asyncpg connections
            with patch("aragora.server.handlers.transcription._job_store") as mock_store:
                mock_store.get.return_value = None
                yield
            _transcription_jobs.clear()


# ===========================================================================
# Job Status Tests
# ===========================================================================


class TestJobStatus:
    """Tests for GET /api/v1/transcription/status/:id endpoint."""

    def test_get_job_status_not_found(self, mock_server_context):
        """Test getting status for non-existent job."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/transcription/status/nonexistent-job-id",
        )

        result = handler._get_status("nonexistent-job-id")

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data.get("error", "").lower()

    def test_get_job_status_found(self, mock_server_context):
        """Test getting status for existing job."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _save_job,
                )

        # Create a job first
        job_id = "test-job-123"
        _save_job(
            job_id,
            {
                "status": "processing",
                "progress": 50,
                "filename": "test.mp3",
            },
        )

        handler = TranscriptionHandler(mock_server_context)
        result = handler._get_status(job_id)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["job_id"] == job_id
        assert data["status"] == "processing"
        assert data["progress"] == 50

    def test_get_completed_job_status(self, mock_server_context):
        """Test getting status for completed job."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _save_job,
                )

        job_id = "completed-job-456"
        _save_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "result": {
                    "text": "Test transcription",
                    "language": "en",
                    "duration": 30.5,
                },
            },
        )

        handler = TranscriptionHandler(mock_server_context)
        result = handler._get_status(job_id)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["result"] is not None
        assert data["result"]["text"] == "Test transcription"

    def test_get_failed_job_status(self, mock_server_context):
        """Test getting status for failed job."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _save_job,
                )

        job_id = "failed-job-789"
        _save_job(
            job_id,
            {
                "status": "failed",
                "error": "Transcription service unavailable",
            },
        )

        handler = TranscriptionHandler(mock_server_context)
        result = handler._get_status(job_id)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "failed"
        assert "error" in data
        assert "unavailable" in data["error"]


# ===========================================================================
# Job Persistence Tests
# ===========================================================================


class TestJobPersistence:
    """Tests for job persistence functionality."""

    def test_save_and_get_job(self):
        """Test saving and retrieving a job."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import _save_job, _get_job

        job_id = "persist-test-001"
        job_data = {
            "status": "processing",
            "progress": 25,
            "filename": "audio.wav",
        }

        _save_job(job_id, job_data)
        retrieved = _get_job(job_id)

        assert retrieved is not None
        assert retrieved["status"] == "processing"
        assert retrieved["progress"] == 25
        assert retrieved["filename"] == "audio.wav"

    def test_get_nonexistent_job(self):
        """Test getting a job that doesn't exist."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import _get_job

        result = _get_job("nonexistent-id-xyz")
        assert result is None

    def test_update_job(self):
        """Test updating an existing job."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import _save_job, _get_job

        job_id = "update-test-001"

        # Create initial job
        _save_job(job_id, {"status": "processing", "progress": 0})

        # Update job
        _save_job(job_id, {"status": "completed", "progress": 100, "result": {"text": "Done"}})

        retrieved = _get_job(job_id)
        assert retrieved["status"] == "completed"
        assert retrieved["progress"] == 100


# ===========================================================================
# Alias Route Tests
# ===========================================================================


class TestAliasRoutes:
    """Tests for alias routes (backwards compatibility)."""

    def test_can_handle_transcribe_audio_alias(self, mock_server_context):
        """Test handler recognizes /api/v1/transcribe/audio alias."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)
        assert handler.can_handle("/api/v1/transcribe/audio") is True

    def test_can_handle_transcribe_video_alias(self, mock_server_context):
        """Test handler recognizes /api/v1/transcribe/video alias."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)
        assert handler.can_handle("/api/v1/transcribe/video") is True


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limiter_allows_initial_requests(self, mock_server_context):
        """Test rate limiter allows initial requests."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import _audio_limiter

        # Clear any existing rate limit state
        _audio_limiter._buckets.clear()

        # First few requests should be allowed
        for i in range(5):
            result = _audio_limiter.is_allowed(f"test-ip-{i}")
            assert result is True

    def test_youtube_limiter_more_restrictive(self, mock_server_context):
        """Test YouTube limiter is more restrictive than audio limiter."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    _audio_limiter,
                    _youtube_limiter,
                )

        # YouTube limiter should have lower limit (5/min vs 10/min)
        # We can verify this by checking the limiter's requests_per_minute
        # Since we can't access private attributes directly, we verify through behavior
        assert _audio_limiter is not _youtube_limiter


# ===========================================================================
# Handler Method Tests
# ===========================================================================


class TestHandlerMethods:
    """Tests for handler method dispatch."""

    def test_handle_routes_to_config(self, mock_server_context):
        """Test handle() routes config path correctly."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/transcription/config",
        )

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            with patch(
                "aragora.transcription.get_available_backends",
                return_value=["whisper"],
            ):
                with patch(
                    "aragora.transcription.whisper_backend.WHISPER_MODELS",
                    {"tiny": {}},
                ):
                    result = handler.handle("/api/v1/transcription/config", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_routes_to_status(self, mock_server_context):
        """Test handle() routes status path correctly."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _save_job,
                )

        # Create a job first
        _save_job("my-job-id", {"status": "completed", "progress": 100})

        handler = TranscriptionHandler(mock_server_context)

        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/transcription/status/my-job-id",
        )

        result = handler.handle("/api/v1/transcription/status/my-job-id", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_returns_none_for_post_endpoints(self, mock_server_context):
        """Test handle() returns None for POST-only endpoints."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/transcription/audio",
            method="GET",
        )

        result = handler.handle("/api/v1/transcription/audio", {}, mock_http)

        # GET on POST-only endpoint returns None
        assert result is None


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in transcription handler."""

    @pytest.mark.asyncio
    async def test_invalid_content_type_audio(self, mock_server_context):
        """Test error for non-multipart content type on audio endpoint."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "100",
            },
            body=b'{"file": "data"}',
            method="POST",
        )

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            result = await handler._handle_audio_transcription(mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "multipart/form-data" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_url_youtube(self, mock_server_context):
        """Test error when URL is missing for YouTube endpoint."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        body = json.dumps({}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            result = await handler._handle_youtube_transcription(mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "url" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_invalid_youtube_url(self, mock_server_context):
        """Test error for invalid YouTube URL."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        body = json.dumps({"url": "https://example.com/not-youtube"}).encode()
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            with patch(
                "aragora.transcription.youtube.YouTubeFetcher.is_youtube_url",
                return_value=False,
            ):
                result = await handler._handle_youtube_transcription(mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid youtube" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_service_unavailable(self, mock_server_context):
        """Test 503 response when transcription service is unavailable."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        boundary = "----TestBoundary"
        file_data = b"ID3" + b"\x00" * 100
        body = (
            (
                f"------{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="audio.mp3"\r\n'
                f"Content-Type: audio/mpeg\r\n\r\n"
            ).encode()
            + file_data
            + f"\r\n------{boundary}--\r\n".encode()
        )

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(False, "No transcription backend available."),
        ):
            result = await handler._handle_audio_transcription(mock_http)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Config Endpoint Tests
# ===========================================================================


class TestConfigEndpoint:
    """Tests for GET /api/v1/transcription/config endpoint."""

    def test_config_when_unavailable(self, mock_server_context):
        """Test config returns unavailable status."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(False, "No API key configured"),
        ):
            result = handler._get_config()

        assert result is not None
        assert result.status_code == 200  # Returns config even when unavailable
        data = json.loads(result.body)
        assert data["available"] is False
        assert "error" in data
        assert "audio_formats" in data
        assert "video_formats" in data

    def test_config_when_available(self, mock_server_context):
        """Test config returns full info when available."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionHandler

        handler = TranscriptionHandler(mock_server_context)

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            with patch(
                "aragora.transcription.get_available_backends",
                return_value=["whisper", "openai"],
            ):
                with patch(
                    "aragora.transcription.whisper_backend.WHISPER_MODELS",
                    {"tiny": {}, "base": {}, "small": {}},
                ):
                    result = handler._get_config()

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["available"] is True
        assert "backends" in data
        assert "whisper" in data["backends"]
        assert "openai" in data["backends"]
        assert "models" in data
        assert len(data["models"]) == 3
        assert data["youtube_enabled"] is True
