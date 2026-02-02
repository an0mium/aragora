"""
Tests for TranscriptionHandler circuit breaker functionality.

Tests cover:
- Circuit breaker state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold triggering circuit open
- Cooldown period before half-open
- Recovery after successful calls
- Integration with transcription endpoints
"""

from __future__ import annotations

import json
import time
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
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

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


# ===========================================================================
# Circuit Breaker Unit Tests
# ===========================================================================


class TestTranscriptionCircuitBreaker:
    """Tests for TranscriptionCircuitBreaker class."""

    def test_circuit_breaker_creation(self):
        """Test circuit breaker initializes in closed state."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_proceed() is True

    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(failure_threshold=3, cooldown_seconds=60)

        # Record failures below threshold
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "closed"

        # Third failure should open circuit
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_proceed() is False

    def test_circuit_breaker_cooldown(self):
        """Test circuit transitions to half-open after cooldown."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Wait for cooldown
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.state == "half_open"
        assert cb.can_proceed() is True

    def test_circuit_breaker_recovery(self):
        """Test circuit closes after successful recovery."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=0.1,
            half_open_max_calls=2,
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Wait for cooldown
        time.sleep(0.15)
        assert cb.state == "half_open"

        # Record successful calls
        cb.record_success()
        cb.record_success()

        # Should be closed now
        assert cb.state == "closed"
        assert cb.can_proceed() is True

    def test_circuit_breaker_reopens_on_half_open_failure(self):
        """Test circuit reopens if failure occurs in half-open state."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for half-open
        time.sleep(0.15)
        assert cb.state == "half_open"

        # Failure in half-open reopens
        cb.record_failure()
        assert cb.state == "open"

    def test_circuit_breaker_status(self):
        """Test circuit breaker status report."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(failure_threshold=5, cooldown_seconds=30)

        status = cb.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(failure_threshold=2)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        # Reset
        cb.reset()
        assert cb.state == "closed"
        assert cb.can_proceed() is True

    def test_success_resets_failure_count(self):
        """Test successful call resets failure count in closed state."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(failure_threshold=3)

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        status = cb.get_status()
        assert status["failure_count"] == 2

        # Success resets count
        cb.record_success()
        status = cb.get_status()
        assert status["failure_count"] == 0

    def test_half_open_limits_calls(self):
        """Test half-open state limits the number of test calls."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import TranscriptionCircuitBreaker

        cb = TranscriptionCircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=0.1,
            half_open_max_calls=2,
        )

        # Open circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for half-open
        time.sleep(0.15)

        # First two calls allowed
        assert cb.can_proceed() is True
        assert cb.can_proceed() is True

        # Third call blocked (until recovery)
        assert cb.can_proceed() is False


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker functions."""

    def test_get_circuit_breaker_status(self):
        """Test getting global circuit breaker status."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    get_transcription_circuit_breaker_status,
                    reset_transcription_circuit_breaker,
                )

        # Reset to known state
        reset_transcription_circuit_breaker()

        status = get_transcription_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status
        assert status["state"] == "closed"

    def test_reset_circuit_breaker(self):
        """Test resetting global circuit breaker."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    get_transcription_circuit_breaker_status,
                    reset_transcription_circuit_breaker,
                    _transcription_circuit_breaker,
                )

        # Open circuit
        for _ in range(5):
            _transcription_circuit_breaker.record_failure()

        status = get_transcription_circuit_breaker_status()
        assert status["state"] == "open"

        # Reset
        reset_transcription_circuit_breaker()

        status = get_transcription_circuit_breaker_status()
        assert status["state"] == "closed"


# ===========================================================================
# Integration Tests with Handler
# ===========================================================================


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with handler."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        """Reset circuit breaker before each test."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    reset_transcription_circuit_breaker,
                )

                reset_transcription_circuit_breaker()
                yield
                reset_transcription_circuit_breaker()

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_audio_requests(self, mock_server_context):
        """Test audio transcription blocked when circuit is open."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _transcription_circuit_breaker,
                )

        # Open circuit
        for _ in range(5):
            _transcription_circuit_breaker.record_failure()

        handler = TranscriptionHandler(mock_server_context)

        # Create multipart request
        boundary = "----TestBoundary"
        file_data = b"ID3" + b"\x00" * 100  # Valid MP3 header
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
            return_value=(True, None),
        ):
            result = await handler._handle_audio_transcription(mock_http)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "temporarily unavailable" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_video_requests(self, mock_server_context):
        """Test video transcription blocked when circuit is open."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _transcription_circuit_breaker,
                )

        # Open circuit
        for _ in range(5):
            _transcription_circuit_breaker.record_failure()

        handler = TranscriptionHandler(mock_server_context)

        # Create multipart request with MP4
        boundary = "----TestBoundary"
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100  # Valid MP4 header
        body = (
            (
                f"------{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="video.mp4"\r\n'
                f"Content-Type: video/mp4\r\n\r\n"
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
            return_value=(True, None),
        ):
            result = await handler._handle_video_transcription(mock_http)

        assert result is not None
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_youtube_requests(self, mock_server_context):
        """Test YouTube transcription blocked when circuit is open."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _transcription_circuit_breaker,
                )

        # Open circuit
        for _ in range(5):
            _transcription_circuit_breaker.record_failure()

        handler = TranscriptionHandler(mock_server_context)

        body = json.dumps({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}).encode()
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
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_successful_transcription_records_success(self, mock_server_context):
        """Test successful transcription records success with circuit breaker."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    TranscriptionHandler,
                    _transcription_circuit_breaker,
                    reset_transcription_circuit_breaker,
                )

        reset_transcription_circuit_breaker()

        # Record some failures first (below threshold)
        _transcription_circuit_breaker.record_failure()
        _transcription_circuit_breaker.record_failure()
        status = _transcription_circuit_breaker.get_status()
        assert status["failure_count"] == 2

        handler = TranscriptionHandler(mock_server_context)

        # Create valid MP3 upload
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

        mock_result = MockTranscriptionResult()

        with patch(
            "aragora.server.handlers.transcription._check_transcription_available",
            return_value=(True, None),
        ):
            with patch(
                "aragora.transcription.transcribe_audio",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                result = await handler._handle_audio_transcription(mock_http)

        assert result is not None
        # After successful transcription, failure count should be reset
        status = _transcription_circuit_breaker.get_status()
        assert status["failure_count"] == 0


# ===========================================================================
# Config Endpoint Tests (with circuit breaker status)
# ===========================================================================


class TestConfigWithCircuitBreaker:
    """Tests for config endpoint reporting circuit breaker status."""

    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        """Reset circuit breaker before each test."""
        with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                from aragora.server.handlers.transcription import (
                    reset_transcription_circuit_breaker,
                )

                reset_transcription_circuit_breaker()
                yield

    def test_config_available_with_circuit_closed(self, mock_server_context):
        """Test config returns available when circuit is closed."""
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
                return_value=["whisper", "openai"],
            ):
                with patch(
                    "aragora.transcription.whisper_backend.WHISPER_MODELS",
                    {"tiny": {}, "base": {}},
                ):
                    result = handler._get_config()

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["available"] is True
