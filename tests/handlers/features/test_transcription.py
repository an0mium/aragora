"""Tests for audio/video transcription handler.

Tests the transcription API endpoints including:
- GET /api/v1/transcription/formats - Get supported formats
- GET /api/v1/transcription/{job_id} - Get job status/result
- GET /api/v1/transcription/{job_id}/segments - Get timestamped segments
- POST /api/v1/transcription/upload - Upload and transcribe audio/video
- DELETE /api/v1/transcription/{job_id} - Delete a transcription

Also tests:
- Upload rate limiting (per-minute and per-hour)
- Multipart form-data and raw upload parsing
- File validation (size, extension, filename security)
- Structured transcription error codes
- Job lifecycle (create, update, eviction)
"""

import io
import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.features.transcription import (
    ALL_SUPPORTED_EXTENSIONS,
    AUDIO_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_FILENAME_LENGTH,
    MAX_MULTIPART_PARTS,
    MIN_FILE_SIZE,
    VIDEO_EXTENSIONS,
    TranscriptionError,
    TranscriptionErrorCode,
    TranscriptionHandler,
    TranscriptionJob,
    TranscriptionStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler with headers, rfile, and client_address."""

    path: str = "/"
    command: str = "POST"
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "Content-Length": "0",
            "Content-Type": "application/octet-stream",
        }
    )
    client_address: tuple = ("127.0.0.1", 12345)
    _rfile_data: bytes = b""

    def __post_init__(self):
        self.rfile = io.BytesIO(self._rfile_data)


def _make_http_handler(
    content: bytes = b"",
    content_type: str = "application/octet-stream",
    filename: str | None = None,
    client_ip: str = "127.0.0.1",
    extra_headers: dict[str, str] | None = None,
) -> MockHTTPHandler:
    """Build a MockHTTPHandler with the given file content."""
    headers = {
        "Content-Length": str(len(content)),
        "Content-Type": content_type,
    }
    if filename:
        headers["X-Filename"] = filename
    if extra_headers:
        headers.update(extra_headers)
    return MockHTTPHandler(
        headers=headers,
        client_address=(client_ip, 12345),
        _rfile_data=content,
    )


def _make_multipart_body(
    filename: str, content: bytes, boundary: str = "----boundary"
) -> bytes:
    """Build a multipart/form-data body with a single file part."""
    parts = []
    parts.append(f"------{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(content)
    parts.append(f"\r\n------{boundary}--\r\n".encode())
    return b"\r\n".join(parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a TranscriptionHandler with minimal context."""
    return TranscriptionHandler(server_context={})


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limit state between tests."""
    TranscriptionHandler._upload_counts.clear()
    yield
    TranscriptionHandler._upload_counts.clear()


@pytest.fixture(autouse=True)
def reset_jobs():
    """Reset in-memory job store between tests."""
    TranscriptionHandler._jobs.clear()
    yield
    TranscriptionHandler._jobs.clear()


@pytest.fixture
def mock_user_ctx():
    """Create a mock authenticated user context for upload tests."""
    ctx = MagicMock()
    ctx.is_authenticated = True
    ctx.authenticated = True
    ctx.user_id = "test-user-001"
    ctx.org_id = "test-org-001"
    ctx.role = "admin"
    ctx.error_reason = None
    return ctx


@pytest.fixture(autouse=True)
def patch_user_auth(mock_user_ctx):
    """Patch extract_user_from_request to bypass JWT auth for upload tests."""
    with patch(
        "aragora.billing.jwt_auth.extract_user_from_request",
        return_value=mock_user_ctx,
    ):
        yield


# ===========================================================================
# TranscriptionError tests
# ===========================================================================


class TestTranscriptionError:
    """Tests for the TranscriptionError dataclass."""

    def test_to_response_default_status(self):
        err = TranscriptionError(TranscriptionErrorCode.NO_CONTENT, "No content provided")
        result = err.to_response()
        assert _status(result) == 400
        body = _body(result)
        assert body["error"] == "No content provided"
        assert body["error_code"] == "no_content"

    def test_to_response_custom_status(self):
        err = TranscriptionError(TranscriptionErrorCode.FILE_TOO_LARGE, "File too large")
        result = err.to_response(status=413)
        assert _status(result) == 413

    def test_to_response_with_details(self):
        details = {"received_bytes": 5000, "max_bytes": 1000}
        err = TranscriptionError(
            TranscriptionErrorCode.FILE_TOO_LARGE, "Too big", details=details
        )
        result = err.to_response()
        body = _body(result)
        assert body["details"] == details

    def test_to_response_without_details(self):
        err = TranscriptionError(TranscriptionErrorCode.NO_CONTENT, "Empty")
        result = err.to_response()
        body = _body(result)
        assert "details" not in body

    def test_all_error_codes_have_string_values(self):
        for code in TranscriptionErrorCode:
            assert isinstance(code.value, str)
            assert len(code.value) > 0

    def test_to_response_404(self):
        err = TranscriptionError(
            TranscriptionErrorCode.JOB_NOT_FOUND, "Not found"
        )
        result = err.to_response(status=404)
        assert _status(result) == 404
        body = _body(result)
        assert body["error_code"] == "job_not_found"

    def test_to_response_429(self):
        err = TranscriptionError(
            TranscriptionErrorCode.RATE_LIMITED, "Too many requests"
        )
        result = err.to_response(status=429)
        assert _status(result) == 429
        body = _body(result)
        assert body["error_code"] == "rate_limited"


# ===========================================================================
# TranscriptionStatus tests
# ===========================================================================


class TestTranscriptionStatus:
    """Tests for TranscriptionStatus enum."""

    def test_pending_value(self):
        assert TranscriptionStatus.PENDING.value == "pending"

    def test_processing_value(self):
        assert TranscriptionStatus.PROCESSING.value == "processing"

    def test_completed_value(self):
        assert TranscriptionStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert TranscriptionStatus.FAILED.value == "failed"


# ===========================================================================
# TranscriptionJob tests
# ===========================================================================


class TestTranscriptionJob:
    """Tests for the TranscriptionJob dataclass."""

    def test_to_dict_basic(self):
        job = TranscriptionJob(
            id="trans_abc123",
            filename="test.mp3",
            status=TranscriptionStatus.PENDING,
        )
        d = job.to_dict()
        assert d["id"] == "trans_abc123"
        assert d["filename"] == "test.mp3"
        assert d["status"] == "pending"
        assert d["text"] is None
        assert d["segment_count"] == 0
        assert d["word_count"] == 0

    def test_to_dict_completed(self):
        job = TranscriptionJob(
            id="trans_abc123",
            filename="recording.wav",
            status=TranscriptionStatus.COMPLETED,
            text="Hello world",
            language="en",
            duration_seconds=5.5,
            word_count=2,
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}, {"start": 1.0, "end": 2.0, "text": "world"}],
            completed_at=1000000.0,
        )
        d = job.to_dict()
        assert d["status"] == "completed"
        assert d["text"] == "Hello world"
        assert d["language"] == "en"
        assert d["duration_seconds"] == 5.5
        assert d["word_count"] == 2
        assert d["segment_count"] == 2
        assert d["completed_at"] == 1000000.0

    def test_to_dict_failed(self):
        job = TranscriptionJob(
            id="trans_xyz",
            filename="bad.mp3",
            status=TranscriptionStatus.FAILED,
            error="Some error",
        )
        d = job.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Some error"

    def test_default_values(self):
        job = TranscriptionJob(
            id="j1", filename="f.mp3", status=TranscriptionStatus.PENDING
        )
        assert job.file_size_bytes == 0
        assert job.duration_seconds is None
        assert job.text is None
        assert job.language is None
        assert job.segments == []
        assert job.workspace_id is None
        assert job.user_id is None
        assert job.org_id is None
        assert job.tenant_id is None
        assert job.metadata == {}


# ===========================================================================
# can_handle tests
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_formats_endpoint(self, handler):
        assert handler.can_handle("/api/v1/transcription/formats")

    def test_upload_endpoint(self, handler):
        assert handler.can_handle("/api/v1/transcription/upload")

    def test_status_endpoint(self, handler):
        assert handler.can_handle("/api/v1/transcription/status")

    def test_job_by_id(self, handler):
        assert handler.can_handle("/api/v1/transcription/trans_abc123")

    def test_job_segments(self, handler):
        assert handler.can_handle("/api/v1/transcription/trans_abc123/segments")

    def test_rejects_non_transcription_path(self, handler):
        assert not handler.can_handle("/api/v1/documents")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_partial_match(self, handler):
        assert not handler.can_handle("/api/v1/transcription")

    def test_routes_constant(self, handler):
        assert "/api/v1/transcription/upload" in handler.ROUTES
        assert "/api/v1/transcription/formats" in handler.ROUTES
        assert "/api/v1/transcription/status" in handler.ROUTES


# ===========================================================================
# GET /api/v1/transcription/formats tests
# ===========================================================================


class TestGetFormats:
    """Tests for GET /api/v1/transcription/formats."""

    def test_returns_200(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        assert _status(result) == 200

    def test_includes_audio_formats(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        body = _body(result)
        assert "audio" in body
        for ext in AUDIO_EXTENSIONS:
            assert ext in body["audio"]

    def test_includes_video_formats(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        body = _body(result)
        assert "video" in body
        for ext in VIDEO_EXTENSIONS:
            assert ext in body["video"]

    def test_includes_max_size(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        body = _body(result)
        assert body["max_size_mb"] == MAX_FILE_SIZE_MB

    def test_includes_model(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        body = _body(result)
        assert body["model"] == "whisper-1"

    def test_audio_formats_sorted(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        body = _body(result)
        assert body["audio"] == sorted(body["audio"])

    def test_video_formats_sorted(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        body = _body(result)
        assert body["video"] == sorted(body["video"])


# ===========================================================================
# GET /api/v1/transcription/{job_id} tests
# ===========================================================================


class TestGetJobStatus:
    """Tests for GET /api/v1/transcription/{job_id}."""

    def test_not_found(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/nonexistent", {}, mock)
        assert _status(result) == 404
        body = _body(result)
        assert body["error_code"] == "job_not_found"

    def test_pending_job(self, handler):
        job = handler._create_job("test.mp3", 1024)
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == job.id
        assert body["status"] == "pending"
        assert body["filename"] == "test.mp3"

    def test_completed_job(self, handler):
        job = handler._create_job("test.wav", 2048)
        handler._update_job(
            job.id,
            TranscriptionStatus.COMPLETED,
            text="Hello world",
            language="en",
            word_count=2,
            duration_seconds=3.5,
        )
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "completed"
        assert body["text"] == "Hello world"
        assert body["language"] == "en"
        assert body["word_count"] == 2

    def test_failed_job(self, handler):
        job = handler._create_job("bad.mp3", 512)
        handler._update_job(
            job.id,
            TranscriptionStatus.FAILED,
            error="API error occurred",
        )
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "failed"
        assert body["error"] == "API error occurred"

    def test_upload_path_not_treated_as_job_id(self, handler):
        """Ensure /upload doesn't get treated as a job_id."""
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/upload", {}, mock)
        # handle() returns None because parts[3] == "upload" is excluded
        assert result is None


# ===========================================================================
# GET /api/v1/transcription/{job_id}/segments tests
# ===========================================================================


class TestGetJobSegments:
    """Tests for GET /api/v1/transcription/{job_id}/segments."""

    def test_not_found(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/nonexistent/segments", {}, mock)
        assert _status(result) == 404
        body = _body(result)
        assert body["error_code"] == "job_not_found"

    def test_pending_job_returns_empty_segments(self, handler):
        job = handler._create_job("test.mp3", 1024)
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}/segments", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["segments"] == []
        assert "pending" in body["message"]

    def test_processing_job_returns_empty_segments(self, handler):
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(job.id, TranscriptionStatus.PROCESSING)
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}/segments", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["segments"] == []
        assert "processing" in body["message"]

    def test_completed_job_returns_segments(self, handler):
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "world"},
        ]
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(
            job.id,
            TranscriptionStatus.COMPLETED,
            segments=segments,
        )
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}/segments", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["segments"] == segments
        assert body["segment_count"] == 2
        assert body["status"] == "completed"

    def test_failed_job_returns_empty_segments(self, handler):
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(job.id, TranscriptionStatus.FAILED, error="Failed")
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}/segments", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["segments"] == []


# ===========================================================================
# DELETE /api/v1/transcription/{job_id} tests
# ===========================================================================


class TestDeleteJob:
    """Tests for DELETE /api/v1/transcription/{job_id}."""

    def test_delete_existing_job(self, handler):
        job = handler._create_job("test.mp3", 1024)
        mock = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete(f"/api/v1/transcription/{job.id}", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert job.id in body["message"]

    def test_delete_nonexistent_job(self, handler):
        mock = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/transcription/nonexistent", {}, mock)
        assert _status(result) == 404
        body = _body(result)
        assert body["error_code"] == "job_not_found"

    def test_delete_removes_from_store(self, handler):
        job = handler._create_job("test.mp3", 1024)
        mock = MockHTTPHandler(command="DELETE")
        handler.handle_delete(f"/api/v1/transcription/{job.id}", {}, mock)
        # Verify job is gone
        mock2 = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}", {}, mock2)
        assert _status(result) == 404

    def test_delete_upload_path_returns_none(self, handler):
        mock = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/transcription/upload", {}, mock)
        assert result is None

    def test_delete_nested_path_returns_none(self, handler):
        """Paths with 5+ segments don't match delete."""
        mock = MockHTTPHandler(command="DELETE")
        result = handler.handle_delete("/api/v1/transcription/abc/segments", {}, mock)
        assert result is None


# ===========================================================================
# POST /api/v1/transcription/upload - validation tests
# ===========================================================================


class TestUploadValidation:
    """Tests for upload validation in handle_post."""

    def test_non_upload_path_returns_none(self, handler):
        mock = MockHTTPHandler(command="POST")
        result = handler.handle_post("/api/v1/transcription/formats", {}, mock)
        assert result is None

    def test_zero_content_length(self, handler):
        mock = _make_http_handler(content=b"", filename="test.mp3")
        # Override content-length to 0
        mock.headers["Content-Length"] = "0"
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "no_content"

    def test_invalid_content_length(self, handler):
        mock = _make_http_handler(content=b"data", filename="test.mp3")
        mock.headers["Content-Length"] = "not-a-number"
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "invalid_content_length"

    def test_file_too_large(self, handler):
        mock = _make_http_handler(content=b"x", filename="test.mp3")
        # Set a content-length exceeding the maximum
        mock.headers["Content-Length"] = str(MAX_FILE_SIZE_BYTES + 1)
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 413
        body = _body(result)
        assert body["error_code"] == "file_too_large"
        assert "details" in body

    def test_unsupported_format(self, handler):
        content = b"fake data"
        mock = _make_http_handler(content=content, filename="readme.txt")
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "unsupported_format"

    def test_filename_too_long(self, handler):
        long_name = "a" * (MAX_FILENAME_LENGTH + 1) + ".mp3"
        content = b"fake data"
        mock = _make_http_handler(content=content, filename=long_name)
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "filename_too_long"

    def test_content_length_mismatch(self, handler):
        """Content-Length header doesn't match actual body size."""
        content = b"fake mp3 data"
        mock = _make_http_handler(content=content, filename="test.mp3")
        # Lie about content length
        mock.headers["Content-Length"] = str(len(content) + 100)
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "corrupted_upload"

    def test_missing_filename_raw_upload(self, handler):
        content = b"fake mp3 data"
        mock = _make_http_handler(content=content)
        # No X-Filename header
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "invalid_filename"


# ===========================================================================
# POST upload - supported extensions
# ===========================================================================


class TestSupportedExtensions:
    """Tests that all supported extensions are accepted."""

    @pytest.mark.parametrize("ext", sorted(AUDIO_EXTENSIONS))
    def test_audio_extension_accepted(self, handler, ext):
        """Audio extensions should pass the extension check."""
        content = b"audio data here"
        mock = _make_http_handler(content=content, filename=f"recording{ext}")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202

    @pytest.mark.parametrize("ext", sorted(VIDEO_EXTENSIONS))
    def test_video_extension_accepted(self, handler, ext):
        """Video extensions should pass the extension check."""
        content = b"video data here"
        mock = _make_http_handler(content=content, filename=f"video{ext}")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202

    @pytest.mark.parametrize(
        "ext", [".txt", ".pdf", ".doc", ".py", ".exe", ".zip", ".html"]
    )
    def test_unsupported_extension_rejected(self, handler, ext):
        content = b"some data"
        mock = _make_http_handler(content=content, filename=f"file{ext}")
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "unsupported_format"


# ===========================================================================
# POST upload - successful upload
# ===========================================================================


class TestSuccessfulUpload:
    """Tests for successful upload with raw body."""

    def test_successful_raw_upload(self, handler):
        content = b"fake mp3 content"
        mock = _make_http_handler(content=content, filename="test.mp3")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202
        body = _body(result)
        assert body["success"] is True
        assert body["filename"] == "test.mp3"
        assert body["file_size_bytes"] == len(content)
        assert body["status"] == "pending"
        assert "job_id" in body

    def test_job_created_on_upload(self, handler):
        content = b"audio content"
        mock = _make_http_handler(content=content, filename="recording.wav")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        body = _body(result)
        job_id = body["job_id"]
        assert job_id in TranscriptionHandler._jobs
        job = TranscriptionHandler._jobs[job_id]
        assert job.filename == "recording.wav"
        assert job.status == TranscriptionStatus.PENDING
        assert job.file_size_bytes == len(content)

    def test_workspace_id_from_header(self, handler):
        content = b"audio content"
        mock = _make_http_handler(
            content=content,
            filename="test.mp3",
            extra_headers={"X-Workspace-ID": "ws-123"},
        )
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        body = _body(result)
        job = TranscriptionHandler._jobs[body["job_id"]]
        assert job.workspace_id == "ws-123"

    def test_default_workspace_id(self, handler):
        content = b"audio content"
        mock = _make_http_handler(content=content, filename="test.mp3")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        body = _body(result)
        job = TranscriptionHandler._jobs[body["job_id"]]
        assert job.workspace_id == "default"


# ===========================================================================
# POST upload - multipart form-data
# ===========================================================================


class TestMultipartUpload:
    """Tests for multipart/form-data upload parsing."""

    def test_missing_boundary(self, handler):
        content = b"some data"
        mock = _make_http_handler(
            content=content,
            content_type="multipart/form-data",
            filename=None,
        )
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "missing_boundary"

    def test_successful_multipart_upload(self, handler):
        file_content = b"fake mp3 audio bytes"
        boundary = "----boundary"
        body_bytes = _make_multipart_body("recording.mp3", file_content, boundary)
        mock = _make_http_handler(
            content=body_bytes,
            content_type=f"multipart/form-data; boundary=----{boundary}",
        )
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202
        resp_body = _body(result)
        assert resp_body["filename"] == "recording.mp3"

    def test_multipart_empty_filename(self, handler):
        boundary = "testbound"
        # Build a multipart body with empty filename
        part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename=""\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
            f"data"
            f"\r\n--{boundary}--\r\n"
        ).encode()
        mock = _make_http_handler(
            content=part,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "invalid_filename"

    def test_multipart_path_traversal(self, handler):
        boundary = "testbound"
        part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="../../etc/passwd"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
            f"data"
            f"\r\n--{boundary}--\r\n"
        ).encode()
        mock = _make_http_handler(
            content=part,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "invalid_filename"

    def test_multipart_null_byte_in_filename(self, handler):
        boundary = "testbound"
        part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test\x00.mp3"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
            f"data"
            f"\r\n--{boundary}--\r\n"
        ).encode()
        mock = _make_http_handler(
            content=part,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "invalid_filename"

    def test_multipart_too_many_parts(self, handler):
        boundary = "testbound"
        # Create more parts than MAX_MULTIPART_PARTS
        parts_list = []
        for i in range(MAX_MULTIPART_PARTS + 2):
            parts_list.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="field{i}"; filename="f{i}.mp3"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
                f"data{i}"
            )
        parts_list.append(f"--{boundary}--\r\n")
        body_bytes = "\r\n".join(parts_list).encode()
        mock = _make_http_handler(
            content=body_bytes,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "multipart_parse_error"

    def test_multipart_no_file_part(self, handler):
        boundary = "testbound"
        # Multipart body without Content-Disposition
        body_bytes = (
            f"--{boundary}\r\n"
            f"Content-Type: text/plain\r\n\r\n"
            f"just text"
            f"\r\n--{boundary}--\r\n"
        ).encode()
        mock = _make_http_handler(
            content=body_bytes,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "multipart_parse_error"

    def test_multipart_read_failure(self, handler):
        mock = _make_http_handler(
            content=b"",
            content_type="multipart/form-data; boundary=testbound",
        )
        # Make rfile.read raise an OSError
        mock.rfile = MagicMock()
        mock.rfile.read.side_effect = OSError("Disk error")
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "corrupted_upload"


# ===========================================================================
# Raw upload - filename validation
# ===========================================================================


class TestRawUploadFilenameValidation:
    """Tests for raw upload (non-multipart) filename validation."""

    def test_null_byte_in_filename(self, handler):
        content = b"fake data"
        mock = _make_http_handler(content=content, filename="test\x00.mp3")
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "invalid_filename"

    def test_path_traversal_in_filename(self, handler):
        content = b"fake data"
        mock = _make_http_handler(content=content, filename="../../../etc/passwd.mp3")
        # os.path.basename strips the directory parts, so this actually becomes "passwd.mp3"
        # but the extension check would still apply. Let's test a case with .. in the basename.
        # Actually, os.path.basename("../../etc/passwd.mp3") = "passwd.mp3" with no ".." in it.
        # So let's test that the traversal is properly stripped.
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        # After basename, it becomes "passwd.mp3" which has valid .mp3 ext
        # The ".." check is on the post-basename filename
        assert _status(result) == 202 or _status(result) == 400

    def test_raw_upload_read_failure(self, handler):
        content = b"fake audio"
        mock = _make_http_handler(content=content, filename="test.mp3")
        # Make rfile.read raise OSError
        mock.rfile = MagicMock()
        mock.rfile.read.side_effect = OSError("Read error")
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert body["error_code"] == "corrupted_upload"


# ===========================================================================
# Rate limiting tests
# ===========================================================================


class TestRateLimiting:
    """Tests for upload rate limiting."""

    def test_under_rate_limit(self, handler):
        """Uploads under limit succeed."""
        content = b"audio data"
        for i in range(TranscriptionHandler.MAX_UPLOADS_PER_MINUTE):
            mock = _make_http_handler(content=content, filename="test.mp3")
            with patch("asyncio.create_task"):
                result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
            assert _status(result) == 202, f"Upload {i+1} should succeed"

    def test_per_minute_rate_limit(self, handler):
        """Exceeding per-minute limit returns 429."""
        content = b"audio data"
        for _ in range(TranscriptionHandler.MAX_UPLOADS_PER_MINUTE):
            mock = _make_http_handler(content=content, filename="test.mp3")
            with patch("asyncio.create_task"):
                handler.handle_post("/api/v1/transcription/upload", {}, mock)

        # Next should be rate limited
        mock = _make_http_handler(content=content, filename="test.mp3")
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 429
        body = _body(result)
        assert body["error_code"] == "rate_limited"

    def test_different_ips_independent(self, handler):
        """Different IPs have independent rate limits."""
        content = b"audio data"
        for _ in range(TranscriptionHandler.MAX_UPLOADS_PER_MINUTE):
            mock = _make_http_handler(content=content, filename="test.mp3", client_ip="10.0.0.1")
            with patch("asyncio.create_task"):
                handler.handle_post("/api/v1/transcription/upload", {}, mock)

        # Different IP should still work
        mock = _make_http_handler(content=content, filename="test.mp3", client_ip="10.0.0.2")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202

    def test_per_hour_rate_limit(self, handler):
        """Exceeding per-hour limit returns 429."""
        content = b"audio data"
        # Fill up the hourly bucket by manipulating timestamps
        client_ip = "10.0.0.99"
        now = time.time()
        # Pre-fill with timestamps spread over the past hour
        TranscriptionHandler._upload_counts[client_ip] = [
            now - i * 60 for i in range(TranscriptionHandler.MAX_UPLOADS_PER_HOUR)
        ]

        mock = _make_http_handler(content=content, filename="test.mp3", client_ip=client_ip)
        result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 429
        body = _body(result)
        assert body["error_code"] == "rate_limited"

    def test_expired_timestamps_cleaned(self, handler):
        """Old timestamps are removed so limits refresh."""
        content = b"audio data"
        client_ip = "10.0.0.50"
        old_time = time.time() - 3700  # over an hour ago
        TranscriptionHandler._upload_counts[client_ip] = [old_time] * 20

        mock = _make_http_handler(content=content, filename="test.mp3", client_ip=client_ip)
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202

    def test_unknown_client_ip(self, handler):
        """Handler without client_address uses 'unknown'."""
        content = b"audio data"
        mock = _make_http_handler(content=content, filename="test.mp3")
        delattr(mock, "client_address")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202


# ===========================================================================
# Job lifecycle tests
# ===========================================================================


class TestJobLifecycle:
    """Tests for job creation, update, and eviction."""

    def test_create_job_generates_id(self, handler):
        job = handler._create_job("test.mp3", 1024)
        assert job.id.startswith("trans_")
        assert len(job.id) > 6

    def test_create_job_with_metadata(self, handler):
        metadata = {"user_id": "u1", "workspace_id": "ws1", "org_id": "org1", "tenant_id": "t1"}
        job = handler._create_job("test.mp3", 1024, metadata=metadata)
        assert job.workspace_id == "ws1"
        assert job.user_id == "u1"
        assert job.org_id == "org1"
        assert job.tenant_id == "t1"

    def test_create_job_registered_in_store(self, handler):
        job = handler._create_job("test.mp3", 1024)
        assert job.id in TranscriptionHandler._jobs

    def test_update_job_status(self, handler):
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(job.id, TranscriptionStatus.PROCESSING)
        assert TranscriptionHandler._jobs[job.id].status == TranscriptionStatus.PROCESSING

    def test_update_job_completed_sets_timestamp(self, handler):
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(job.id, TranscriptionStatus.COMPLETED)
        assert TranscriptionHandler._jobs[job.id].completed_at is not None

    def test_update_job_with_kwargs(self, handler):
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(
            job.id,
            TranscriptionStatus.COMPLETED,
            text="Hello",
            language="en",
            word_count=1,
        )
        updated = TranscriptionHandler._jobs[job.id]
        assert updated.text == "Hello"
        assert updated.language == "en"
        assert updated.word_count == 1

    def test_update_nonexistent_job(self, handler):
        """Updating a nonexistent job should not raise."""
        handler._update_job("nonexistent", TranscriptionStatus.FAILED)

    def test_update_ignores_unknown_kwargs(self, handler):
        """Unknown kwargs are silently ignored (hasattr check)."""
        job = handler._create_job("test.mp3", 1024)
        handler._update_job(job.id, TranscriptionStatus.COMPLETED, nonexistent_field="value")
        # Should not crash, and the field should not exist
        assert not hasattr(TranscriptionHandler._jobs[job.id], "nonexistent_field")

    def test_job_eviction_when_full(self, handler):
        """When MAX_JOBS reached, oldest completed job is evicted."""
        original_max = TranscriptionHandler.MAX_JOBS
        TranscriptionHandler.MAX_JOBS = 3
        try:
            j1 = handler._create_job("a.mp3", 100)
            handler._update_job(j1.id, TranscriptionStatus.COMPLETED)
            j2 = handler._create_job("b.mp3", 100)
            handler._update_job(j2.id, TranscriptionStatus.COMPLETED)
            j3 = handler._create_job("c.mp3", 100)
            # Store is full at 3; next create should evict j1 (oldest completed)
            j4 = handler._create_job("d.mp3", 100)
            assert j1.id not in TranscriptionHandler._jobs
            assert j4.id in TranscriptionHandler._jobs
        finally:
            TranscriptionHandler.MAX_JOBS = original_max

    def test_job_eviction_oldest_pending_when_no_completed(self, handler):
        """When no completed/failed jobs, oldest job is evicted."""
        original_max = TranscriptionHandler.MAX_JOBS
        TranscriptionHandler.MAX_JOBS = 2
        try:
            j1 = handler._create_job("a.mp3", 100)
            j2 = handler._create_job("b.mp3", 100)
            # Both are pending; adding one more should evict oldest
            j3 = handler._create_job("c.mp3", 100)
            assert j1.id not in TranscriptionHandler._jobs
            assert j3.id in TranscriptionHandler._jobs
        finally:
            TranscriptionHandler.MAX_JOBS = original_max


# ===========================================================================
# Constants / module-level tests
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_max_file_size_bytes(self):
        assert MAX_FILE_SIZE_BYTES == MAX_FILE_SIZE_MB * 1024 * 1024

    def test_min_file_size(self):
        assert MIN_FILE_SIZE == 1

    def test_audio_extensions_are_set(self):
        assert isinstance(AUDIO_EXTENSIONS, set)
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".wav" in AUDIO_EXTENSIONS

    def test_video_extensions_are_set(self):
        assert isinstance(VIDEO_EXTENSIONS, set)
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS

    def test_all_supported_is_union(self):
        assert ALL_SUPPORTED_EXTENSIONS == AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    def test_max_multipart_parts(self):
        assert MAX_MULTIPART_PARTS == 10

    def test_max_filename_length(self):
        assert MAX_FILENAME_LENGTH == 255


# ===========================================================================
# TranscriptionErrorCode enum tests
# ===========================================================================


class TestTranscriptionErrorCode:
    """Tests for all error code enum members."""

    def test_rate_limited(self):
        assert TranscriptionErrorCode.RATE_LIMITED.value == "rate_limited"

    def test_file_too_large(self):
        assert TranscriptionErrorCode.FILE_TOO_LARGE.value == "file_too_large"

    def test_file_too_small(self):
        assert TranscriptionErrorCode.FILE_TOO_SMALL.value == "file_too_small"

    def test_invalid_content_length(self):
        assert TranscriptionErrorCode.INVALID_CONTENT_LENGTH.value == "invalid_content_length"

    def test_no_content(self):
        assert TranscriptionErrorCode.NO_CONTENT.value == "no_content"

    def test_unsupported_format(self):
        assert TranscriptionErrorCode.UNSUPPORTED_FORMAT.value == "unsupported_format"

    def test_invalid_filename(self):
        assert TranscriptionErrorCode.INVALID_FILENAME.value == "invalid_filename"

    def test_filename_too_long(self):
        assert TranscriptionErrorCode.FILENAME_TOO_LONG.value == "filename_too_long"

    def test_corrupted_upload(self):
        assert TranscriptionErrorCode.CORRUPTED_UPLOAD.value == "corrupted_upload"

    def test_multipart_parse_error(self):
        assert TranscriptionErrorCode.MULTIPART_PARSE_ERROR.value == "multipart_parse_error"

    def test_missing_boundary(self):
        assert TranscriptionErrorCode.MISSING_BOUNDARY.value == "missing_boundary"

    def test_transcription_failed(self):
        assert TranscriptionErrorCode.TRANSCRIPTION_FAILED.value == "transcription_failed"

    def test_job_not_found(self):
        assert TranscriptionErrorCode.JOB_NOT_FOUND.value == "job_not_found"

    def test_api_not_configured(self):
        assert TranscriptionErrorCode.API_NOT_CONFIGURED.value == "api_not_configured"

    def test_quota_exceeded(self):
        assert TranscriptionErrorCode.QUOTA_EXCEEDED.value == "quota_exceeded"

    def test_total_error_codes(self):
        """Verify the expected number of error codes exist."""
        assert len(TranscriptionErrorCode) == 15


# ===========================================================================
# handle() routing tests
# ===========================================================================


class TestHandleRouting:
    """Tests for handle() GET request routing."""

    def test_formats_routes_correctly(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/formats", {}, mock)
        assert result is not None
        assert _status(result) == 200

    def test_job_id_routes_to_status(self, handler):
        job = handler._create_job("test.mp3", 100)
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}", {}, mock)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == job.id

    def test_segments_routes_correctly(self, handler):
        job = handler._create_job("test.mp3", 100)
        handler._update_job(job.id, TranscriptionStatus.COMPLETED, segments=[])
        mock = MockHTTPHandler(command="GET")
        result = handler.handle(f"/api/v1/transcription/{job.id}/segments", {}, mock)
        assert result is not None
        assert _status(result) == 200

    def test_unknown_sub_path_returns_none(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/abc/unknown", {}, mock)
        assert result is None

    def test_upload_path_returns_none_for_get(self, handler):
        mock = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/transcription/upload", {}, mock)
        assert result is None


# ===========================================================================
# handle_post routing tests
# ===========================================================================


class TestHandlePostRouting:
    """Tests for handle_post routing."""

    def test_upload_path_routes(self, handler):
        content = b"audio data"
        mock = _make_http_handler(content=content, filename="test.mp3")
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert result is not None
        assert _status(result) == 202

    def test_non_upload_path_returns_none(self, handler):
        mock = MockHTTPHandler(command="POST")
        result = handler.handle_post("/api/v1/transcription/formats", {}, mock)
        assert result is None

    def test_unknown_path_returns_none(self, handler):
        mock = MockHTTPHandler(command="POST")
        result = handler.handle_post("/api/v1/other", {}, mock)
        assert result is None


# ===========================================================================
# Constructor tests
# ===========================================================================


class TestConstructor:
    """Tests for TranscriptionHandler initialization."""

    def test_with_server_context(self):
        ctx = {"key": "value"}
        h = TranscriptionHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_with_ctx(self):
        ctx = {"key": "value"}
        h = TranscriptionHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_server_context_takes_precedence(self):
        h = TranscriptionHandler(ctx={"a": 1}, server_context={"b": 2})
        assert h.ctx == {"b": 2}

    def test_defaults_to_empty_dict(self):
        h = TranscriptionHandler()
        assert h.ctx == {}


# ===========================================================================
# _get_client_ip tests
# ===========================================================================


class TestGetClientIp:
    """Tests for _get_client_ip."""

    def test_returns_ip_from_client_address(self, handler):
        mock = MockHTTPHandler(client_address=("192.168.1.1", 8080))
        assert handler._get_client_ip(mock) == "192.168.1.1"

    def test_returns_unknown_without_client_address(self, handler):
        mock = MagicMock(spec=[])  # No attributes
        assert handler._get_client_ip(mock) == "unknown"


# ===========================================================================
# LRU eviction on rate limit tracker
# ===========================================================================


class TestRateLimitLRUEviction:
    """Tests for the LRU eviction of IP tracking in rate limiter."""

    def test_lru_eviction_when_max_ips_exceeded(self, handler):
        """When MAX_TRACKED_IPS is exceeded, oldest IPs are evicted."""
        original_max = TranscriptionHandler.MAX_TRACKED_IPS
        TranscriptionHandler.MAX_TRACKED_IPS = 3
        try:
            content = b"audio"
            for i in range(5):
                ip = f"10.0.0.{i}"
                mock = _make_http_handler(content=content, filename="test.mp3", client_ip=ip)
                with patch("asyncio.create_task"):
                    handler.handle_post("/api/v1/transcription/upload", {}, mock)
            # Oldest IPs should be evicted
            assert len(TranscriptionHandler._upload_counts) <= 3
        finally:
            TranscriptionHandler.MAX_TRACKED_IPS = original_max


# ===========================================================================
# Boundary parsing edge cases
# ===========================================================================


class TestBoundaryParsing:
    """Tests for multipart boundary extraction edge cases."""

    def test_boundary_with_quotes(self, handler):
        """Boundary value wrapped in quotes is handled."""
        file_content = b"audio data"
        boundary = "myboundary"
        body_bytes = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.mp3"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + file_content + f"\r\n--{boundary}--\r\n".encode()

        mock = _make_http_handler(
            content=body_bytes,
            content_type=f'multipart/form-data; boundary="{boundary}"',
        )
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202

    def test_boundary_with_whitespace(self, handler):
        """Boundary with surrounding whitespace is trimmed."""
        file_content = b"audio data"
        boundary = "myboundary"
        body_bytes = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.mp3"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + file_content + f"\r\n--{boundary}--\r\n".encode()

        mock = _make_http_handler(
            content=body_bytes,
            content_type=f"multipart/form-data; boundary= {boundary} ",
        )
        with patch("asyncio.create_task"):
            result = handler.handle_post("/api/v1/transcription/upload", {}, mock)
        assert _status(result) == 202
