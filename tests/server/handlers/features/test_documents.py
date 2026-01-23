"""
Tests for Documents Handler.

Tests cover enums, dataclasses, constants, and basic handler creation.
"""

import pytest

from aragora.server.handlers.features.documents import (
    DocumentHandler,
    UploadErrorCode,
    UploadError,
    MAX_MULTIPART_PARTS,
    MAX_FILENAME_LENGTH,
    MIN_FILE_SIZE,
)


class TestDocumentsConstants:
    """Tests for documents module constants."""

    def test_multipart_parts_limit(self):
        """Test multipart parts limit."""
        assert MAX_MULTIPART_PARTS == 10

    def test_filename_length_limit(self):
        """Test filename length limit."""
        assert MAX_FILENAME_LENGTH == 255

    def test_min_file_size(self):
        """Test minimum file size."""
        assert MIN_FILE_SIZE == 1


class TestUploadErrorCodeEnum:
    """Tests for UploadErrorCode enum."""

    def test_all_error_codes_defined(self):
        """Test that expected error codes are defined."""
        expected = [
            "rate_limited",
            "file_too_large",
            "file_too_small",
            "unsupported_format",
            "invalid_filename",
            "corrupted_upload",
            "storage_not_configured",
        ]
        for code in expected:
            assert UploadErrorCode(code) is not None

    def test_error_code_values(self):
        """Test error code enum values."""
        assert UploadErrorCode.RATE_LIMITED.value == "rate_limited"
        assert UploadErrorCode.FILE_TOO_LARGE.value == "file_too_large"
        assert UploadErrorCode.UNSUPPORTED_FORMAT.value == "unsupported_format"


class TestUploadError:
    """Tests for UploadError dataclass."""

    def test_error_creation(self):
        """Test creating an upload error."""
        error = UploadError(
            code=UploadErrorCode.FILE_TOO_LARGE,
            message="File exceeds maximum size",
            details={"received_bytes": 50000000, "max_bytes": 25000000},
        )

        assert error.code == UploadErrorCode.FILE_TOO_LARGE
        assert "maximum size" in error.message
        assert error.details["received_bytes"] == 50000000

    def test_error_without_details(self):
        """Test error without details."""
        error = UploadError(
            code=UploadErrorCode.UNSUPPORTED_FORMAT,
            message="Unsupported file type",
        )

        assert error.code == UploadErrorCode.UNSUPPORTED_FORMAT
        assert error.details is None

    def test_error_to_response(self):
        """Test error response generation."""
        error = UploadError(
            code=UploadErrorCode.INVALID_FILENAME,
            message="Invalid filename",
        )

        response = error.to_response(400)
        assert response is not None


class TestDocumentHandler:
    """Tests for DocumentHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = DocumentHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(DocumentHandler, "ROUTES")
        routes = DocumentHandler.ROUTES
        assert "/api/v1/documents" in routes
        assert "/api/v1/documents/formats" in routes
        assert "/api/v1/documents/upload" in routes

    def test_can_handle_method(self):
        """Test can_handle method for valid routes."""
        handler = DocumentHandler(server_context={})

        assert handler.can_handle("/api/v1/documents") is True
        assert handler.can_handle("/api/v1/documents/formats") is True
        assert handler.can_handle("/api/v1/documents/upload") is True
        assert handler.can_handle("/api/v1/documents/doc123") is True

        # Invalid routes
        assert handler.can_handle("/api/v1/invalid/route") is False

    def test_handler_rate_limit_config(self):
        """Test rate limit configuration."""
        assert DocumentHandler.MAX_UPLOADS_PER_MINUTE == 5
        assert DocumentHandler.MAX_UPLOADS_PER_HOUR == 30
        assert DocumentHandler.MAX_TRACKED_IPS == 10000
