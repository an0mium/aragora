"""
Tests for Transcription Handler.

Tests cover enums, dataclasses, constants, and basic handler creation.
"""

import pytest
import time

from aragora.server.handlers.features.transcription import (
    TranscriptionHandler,
    TranscriptionErrorCode,
    TranscriptionStatus,
    TranscriptionError,
    TranscriptionJob,
    MAX_FILE_SIZE_MB,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ALL_SUPPORTED_EXTENSIONS,
    MAX_FILENAME_LENGTH,
)


class TestTranscriptionConstants:
    """Tests for transcription module constants."""

    def test_file_size_limits(self):
        """Test file size limit constants."""
        assert MAX_FILE_SIZE_MB == 25
        assert MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024
        assert MIN_FILE_SIZE == 1

    def test_audio_extensions(self):
        """Test supported audio extensions."""
        expected = {".mp3", ".m4a", ".wav", ".webm", ".mpga", ".mpeg"}
        assert AUDIO_EXTENSIONS == expected

    def test_video_extensions(self):
        """Test supported video extensions."""
        expected = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
        assert VIDEO_EXTENSIONS == expected

    def test_all_extensions_combined(self):
        """Test that all extensions is union of audio and video."""
        assert ALL_SUPPORTED_EXTENSIONS == AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

    def test_filename_length_limit(self):
        """Test filename length limit."""
        assert MAX_FILENAME_LENGTH == 255


class TestTranscriptionErrorCodeEnum:
    """Tests for TranscriptionErrorCode enum."""

    def test_all_error_codes_defined(self):
        """Test that expected error codes are defined."""
        expected = [
            "rate_limited",
            "file_too_large",
            "file_too_small",
            "unsupported_format",
            "transcription_failed",
            "job_not_found",
        ]
        for code in expected:
            assert TranscriptionErrorCode(code) is not None

    def test_error_code_values(self):
        """Test error code enum values."""
        assert TranscriptionErrorCode.RATE_LIMITED.value == "rate_limited"
        assert TranscriptionErrorCode.FILE_TOO_LARGE.value == "file_too_large"
        assert TranscriptionErrorCode.JOB_NOT_FOUND.value == "job_not_found"


class TestTranscriptionStatusEnum:
    """Tests for TranscriptionStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all statuses are defined."""
        expected = ["pending", "processing", "completed", "failed"]
        for status in expected:
            assert TranscriptionStatus(status) is not None

    def test_status_values(self):
        """Test status enum values."""
        assert TranscriptionStatus.PENDING.value == "pending"
        assert TranscriptionStatus.PROCESSING.value == "processing"
        assert TranscriptionStatus.COMPLETED.value == "completed"
        assert TranscriptionStatus.FAILED.value == "failed"


class TestTranscriptionError:
    """Tests for TranscriptionError dataclass."""

    def test_error_creation(self):
        """Test creating a transcription error."""
        error = TranscriptionError(
            code=TranscriptionErrorCode.FILE_TOO_LARGE,
            message="File exceeds maximum size",
            details={"received_bytes": 30000000, "max_bytes": MAX_FILE_SIZE_BYTES},
        )

        assert error.code == TranscriptionErrorCode.FILE_TOO_LARGE
        assert "maximum size" in error.message
        assert error.details["received_bytes"] == 30000000

    def test_error_to_response(self):
        """Test error response generation."""
        error = TranscriptionError(
            code=TranscriptionErrorCode.UNSUPPORTED_FORMAT,
            message="Unsupported file type: .xyz",
        )

        response = error.to_response(400)
        assert response is not None


class TestTranscriptionJob:
    """Tests for TranscriptionJob dataclass."""

    def test_job_creation(self):
        """Test creating a transcription job."""
        job = TranscriptionJob(
            id="trans_abc123",
            filename="audio.mp3",
            status=TranscriptionStatus.PENDING,
            file_size_bytes=1024000,
        )

        assert job.id == "trans_abc123"
        assert job.filename == "audio.mp3"
        assert job.status == TranscriptionStatus.PENDING
        assert job.file_size_bytes == 1024000

    def test_job_defaults(self):
        """Test job with default values."""
        job = TranscriptionJob(
            id="trans_def456",
            filename="video.mp4",
            status=TranscriptionStatus.PENDING,
        )

        assert job.completed_at is None
        assert job.transcription_id is None
        assert job.error is None
        assert job.text is None
        assert job.language is None
        assert job.word_count == 0
        assert job.segments == []

    def test_job_completed(self):
        """Test completed transcription job."""
        job = TranscriptionJob(
            id="trans_ghi789",
            filename="interview.m4a",
            status=TranscriptionStatus.COMPLETED,
            file_size_bytes=5000000,
            text="Hello, this is a transcription test.",
            language="en",
            word_count=7,
            duration_seconds=30.5,
        )

        assert job.status == TranscriptionStatus.COMPLETED
        assert job.text is not None
        assert job.word_count == 7
        assert job.duration_seconds == 30.5

    def test_job_to_dict(self):
        """Test job serialization."""
        job = TranscriptionJob(
            id="trans_test",
            filename="test.mp3",
            status=TranscriptionStatus.COMPLETED,
            file_size_bytes=2048,
            text="Test transcription",
            word_count=2,
        )

        data = job.to_dict()
        assert data["id"] == "trans_test"
        assert data["filename"] == "test.mp3"
        assert data["status"] == "completed"
        assert data["word_count"] == 2
        assert data["segment_count"] == 0


class TestTranscriptionHandler:
    """Tests for TranscriptionHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = TranscriptionHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(TranscriptionHandler, "ROUTES")
        routes = TranscriptionHandler.ROUTES
        assert "/api/v1/transcription/upload" in routes
        assert "/api/v1/transcription/formats" in routes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = TranscriptionHandler(server_context={})

        assert handler.can_handle("/api/v1/transcription/formats") is True
        assert handler.can_handle("/api/v1/transcription/upload") is True
        # Dynamic routes
        assert handler.can_handle("/api/v1/transcription/job123") is True
        assert handler.can_handle("/api/v1/transcription/job123/segments") is True
        # Invalid routes
        assert handler.can_handle("/api/v1/invalid/route") is False

    def test_handler_rate_limit_config(self):
        """Test rate limit configuration."""
        assert TranscriptionHandler.MAX_UPLOADS_PER_MINUTE == 3
        assert TranscriptionHandler.MAX_UPLOADS_PER_HOUR == 20
