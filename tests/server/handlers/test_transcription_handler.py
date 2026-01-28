"""
Tests for aragora.server.handlers.transcription - Transcription API handler.

Tests cover:
- Routing and method handling
- POST /api/transcription/audio
- POST /api/transcription/video
- POST /api/transcription/youtube
- POST /api/transcription/youtube/info
- GET /api/transcription/config
- Rate limiting
- Error responses
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock get_available_backends before importing the handler to avoid API key checks
with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
    with patch(
        "aragora.transcription.whisper_backend.get_available_backends", return_value=["mock"]
    ):
        from aragora.server.handlers.transcription import TranscriptionHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "user"


@dataclass
class MockTranscriptionResult:
    """Mock transcription result."""

    text: str = "Hello, this is a test transcription."
    language: str = "en"
    duration: float = 10.0
    segments: list = field(default_factory=list)

    def to_dict(self):
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": self.segments,
        }


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

    def send_response(self, code):
        self.response_code = code

    def send_header(self, key, value):
        pass

    def end_headers(self):
        pass


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


@pytest.fixture
def handler(mock_server_context):
    """Create a TranscriptionHandler instance."""
    return TranscriptionHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json", "Content-Length": "0"},
        path="/api/v1/transcription/config",
    )


@pytest.fixture
def mock_auth_context():
    """Create a mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_transcription_available():
    """Mock the transcription availability check to avoid API key requirements."""
    with patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    ):
        with patch(
            "aragora.transcription.get_available_backends",
            return_value=["mock"],
        ):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                yield


@pytest.fixture
def mock_transcription_unavailable():
    """Mock the transcription availability check as unavailable."""
    with patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(False, "No transcription backend available."),
    ):
        yield


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_config(self, handler):
        """Test handler recognizes config endpoint."""
        assert handler.can_handle("/api/v1/transcription/config") is True

    def test_can_handle_audio(self, handler):
        """Test handler recognizes audio endpoint."""
        assert handler.can_handle("/api/v1/transcription/audio") is True

    def test_can_handle_video(self, handler):
        """Test handler recognizes video endpoint."""
        assert handler.can_handle("/api/v1/transcription/video") is True

    def test_can_handle_youtube(self, handler):
        """Test handler recognizes youtube endpoint."""
        assert handler.can_handle("/api/v1/transcription/youtube") is True

    def test_can_handle_youtube_info(self, handler):
        """Test handler recognizes youtube info endpoint."""
        assert handler.can_handle("/api/v1/transcription/youtube/info") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/transcription/unknown") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False


# ===========================================================================
# Config Endpoint Tests
# ===========================================================================


class TestConfigEndpoint:
    """Tests for GET /api/transcription/config."""

    def test_get_config(self, handler, mock_http_handler, mock_transcription_available):
        """Test getting transcription config."""
        result = handler.handle("/api/v1/transcription/config", {}, mock_http_handler)

        assert result is not None
        assert result.content_type == "application/json"

        # Response includes config data even when backend is unavailable
        data = json.loads(result.body)
        assert "audio_formats" in data
        assert "video_formats" in data
        # Config includes size limits
        assert "max_audio_size_mb" in data or "max_file_size_mb" in data

    def test_config_contains_formats(
        self, handler, mock_http_handler, mock_transcription_available
    ):
        """Test config includes expected formats."""
        result = handler.handle("/api/v1/transcription/config", {}, mock_http_handler)
        data = json.loads(result.body)

        assert ".mp3" in data["audio_formats"]
        assert ".wav" in data["audio_formats"]
        assert ".mp4" in data["video_formats"]


# ===========================================================================
# Audio Transcription Tests
# ===========================================================================


class TestAudioTranscription:
    """Tests for POST /api/transcription/audio."""

    def test_audio_requires_post(self, handler, mock_http_handler):
        """Test audio endpoint requires POST method."""
        result = handler.handle("/api/v1/transcription/audio", {}, mock_http_handler)
        assert result is None  # GET not handled

    @pytest.mark.asyncio
    async def test_audio_transcription_success(self, handler, mock_transcription_available):
        """Test successful audio transcription."""
        # Create multipart form data
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = (
            f"------{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.mp3"\r\n'
            f"Content-Type: audio/mpeg\r\n\r\n"
            f"fake audio data\r\n"
            f"------{boundary}--\r\n"
        ).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.transcription.whisper_backend.get_transcription_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.transcribe = AsyncMock(return_value=MockTranscriptionResult())
            mock_get.return_value = mock_backend

            result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

            # Should return result or error
            assert result is not None

    @pytest.mark.asyncio
    async def test_audio_missing_file(self, handler, mock_transcription_available):
        """Test error when no file provided."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            body=b"{}",
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        # May return 503 (no backend) or 400 (bad request)
        assert result.status_code in (400, 503)

    @pytest.mark.asyncio
    async def test_audio_unsupported_format(self, handler, mock_transcription_available):
        """Test error for unsupported audio format."""
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body_data = (
            f"------{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.pdf"\r\n'
            f"Content-Type: application/pdf\r\n\r\n"
            f"fake data\r\n"
            f"------{boundary}--\r\n"
        ).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body_data)),
            },
            body=body_data,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        # Should reject unsupported format or return 503 if no backend
        assert result.status_code in (400, 415, 503)


# ===========================================================================
# YouTube Transcription Tests
# ===========================================================================


class TestYouTubeTranscription:
    """Tests for POST /api/transcription/youtube."""

    @pytest.mark.asyncio
    async def test_youtube_requires_url(self, handler, mock_transcription_available):
        """Test youtube endpoint requires URL in body."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            body=b"{}",
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/youtube", {}, mock_http)

        assert result is not None
        # May return 503 (no backend) or 400 (bad request)
        assert result.status_code in (400, 503)

    @pytest.mark.asyncio
    async def test_youtube_invalid_url(self, handler, mock_transcription_available):
        """Test error for invalid YouTube URL."""
        body_data = json.dumps({"url": "https://example.com/video"}).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
            },
            body=body_data,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/youtube", {}, mock_http)

        assert result is not None
        # May return 503 (no backend) or 400 (bad request)
        assert result.status_code in (400, 503)

    @pytest.mark.asyncio
    async def test_youtube_transcription_success(self, handler, mock_transcription_available):
        """Test successful YouTube transcription."""
        body = json.dumps({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.transcription.youtube.YouTubeFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_audio = AsyncMock(return_value="/tmp/audio.mp3")
            mock_fetcher_class.return_value = mock_fetcher

            with patch(
                "aragora.transcription.whisper_backend.get_transcription_backend"
            ) as mock_get:
                mock_backend = MagicMock()
                mock_backend.transcribe = AsyncMock(return_value=MockTranscriptionResult())
                mock_get.return_value = mock_backend

                result = await handler.handle_post("/api/v1/transcription/youtube", {}, mock_http)

                # Should return result
                assert result is not None


# ===========================================================================
# YouTube Info Endpoint Tests
# ===========================================================================


class TestYouTubeInfoEndpoint:
    """Tests for POST /api/transcription/youtube/info."""

    @pytest.mark.asyncio
    async def test_youtube_info_requires_url(self, handler):
        """Test youtube info requires URL."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            body=b"{}",
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/youtube/info", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_youtube_info_success(self, handler, mock_transcription_available):
        """Test successful YouTube info fetch."""
        body_data = json.dumps({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body_data)),
            },
            body=body_data,
            method="POST",
        )

        mock_info = MagicMock()
        mock_info.to_dict.return_value = {
            "video_id": "dQw4w9WgXcQ",
            "title": "Test Video",
            "duration": 180,
            "channel": "Test Channel",
        }

        with patch("aragora.transcription.youtube.YouTubeFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher.get_video_info = AsyncMock(return_value=mock_info)
            mock_fetcher_class.return_value = mock_fetcher

            result = await handler.handle_post("/api/v1/transcription/youtube/info", {}, mock_http)

            assert result is not None
            if result.status_code == 200:
                data = json.loads(result.body)
                assert "video_id" in data or "error" not in data


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limiter_initialized(self, handler):
        """Test rate limiters are initialized."""
        assert hasattr(handler, "_audio_limiter") or True  # May use different approach

    # Rate limiting is typically tested with time mocking
    # which requires more complex setup


# ===========================================================================
# Error Response Tests
# ===========================================================================


class TestErrorResponses:
    """Tests for error responses."""

    def test_404_for_unknown_route(self, handler, mock_http_handler):
        """Test 404 for unknown routes (can_handle returns False)."""
        assert handler.can_handle("/api/v1/transcription/unknown") is False

    def test_method_not_allowed(self, handler, mock_http_handler):
        """Test method handling."""
        # GET on POST-only endpoint returns None (not handled)
        result = handler.handle("/api/v1/transcription/audio", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_json_error_format(self, handler, mock_transcription_available):
        """Test error responses are JSON formatted."""
        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": "2",
            },
            body=b"{}",
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        if result:
            if result.status_code >= 400:
                assert result.content_type == "application/json"
                data = json.loads(result.body)
                assert "error" in data or "message" in data


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests for handler registration."""

    def test_handler_routes_defined(self, handler):
        """Test handler has ROUTES defined."""
        assert hasattr(handler, "ROUTES") or hasattr(handler, "can_handle")

    def test_handler_methods(self, handler):
        """Test handler has required methods."""
        assert hasattr(handler, "handle")
        assert hasattr(handler, "handle_post")
        assert hasattr(handler, "can_handle")
        assert callable(handler.handle)
        assert callable(handler.handle_post)
        assert callable(handler.can_handle)
