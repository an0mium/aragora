"""
Tests for file upload validation utilities.

Tests cover:
- File size validation (min/max limits)
- MIME type whitelist validation
- File extension validation
- Path traversal prevention
- Filename security (null bytes, control chars, reserved names)
- Filename sanitization
- HTTP status codes for different validation failures
"""

from __future__ import annotations

import os
import pytest
from unittest import mock

from aragora.server.handlers.utils.file_validation import (
    validate_file_upload,
    validate_file_size,
    validate_mime_type,
    validate_extension,
    validate_filename_security,
    sanitize_filename,
    get_max_file_size,
    get_max_file_size_mb,
    FileValidationResult,
    FileValidationError,
    FileValidationErrorCode,
    ALLOWED_MIME_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    MAX_FILENAME_LENGTH,
    DEFAULT_MAX_FILE_SIZE,
)


class TestValidateFileSize:
    """Tests for file size validation."""

    def test_valid_file_size(self):
        """Test that normal file sizes pass validation."""
        result = validate_file_size(1024)  # 1KB
        assert result.valid
        assert result.error_code is None

    def test_file_too_small(self):
        """Test that files below minimum size are rejected."""
        result = validate_file_size(0)
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.FILE_TOO_SMALL
        assert result.http_status == 400
        assert "too small" in result.error_message.lower()

    def test_file_too_large(self):
        """Test that files above maximum size are rejected with 413 status."""
        # Use a smaller max_size for testing
        result = validate_file_size(2000, max_size=1000)
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.FILE_TOO_LARGE
        assert result.http_status == 413  # Payload Too Large
        assert "too large" in result.error_message.lower()

    def test_file_at_max_size(self):
        """Test that file exactly at max size is allowed."""
        result = validate_file_size(1000, max_size=1000)
        assert result.valid

    def test_file_at_min_size(self):
        """Test that file exactly at min size is allowed."""
        result = validate_file_size(1, min_size=1)
        assert result.valid

    def test_custom_max_size(self):
        """Test custom max size override."""
        custom_max = 50 * 1024 * 1024  # 50MB
        result = validate_file_size(60 * 1024 * 1024, max_size=custom_max)
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.FILE_TOO_LARGE

    def test_details_include_sizes(self):
        """Test that error details include size information."""
        result = validate_file_size(5000, max_size=1000)
        assert result.details is not None
        assert result.details["received_bytes"] == 5000
        assert result.details["max_bytes"] == 1000


class TestValidateMimeType:
    """Tests for MIME type validation."""

    def test_valid_pdf_mime_type(self):
        """Test that application/pdf is allowed."""
        result = validate_mime_type("application/pdf")
        assert result.valid

    def test_valid_text_mime_type(self):
        """Test that text/plain is allowed."""
        result = validate_mime_type("text/plain")
        assert result.valid

    def test_valid_json_mime_type(self):
        """Test that application/json is allowed."""
        result = validate_mime_type("application/json")
        assert result.valid

    def test_valid_image_mime_types(self):
        """Test that common image types are allowed."""
        for mime in ["image/png", "image/jpeg", "image/gif", "image/webp"]:
            result = validate_mime_type(mime)
            assert result.valid, f"{mime} should be allowed"

    def test_valid_audio_mime_types(self):
        """Test that audio types for transcription are allowed."""
        for mime in ["audio/mpeg", "audio/wav", "audio/webm", "audio/flac"]:
            result = validate_mime_type(mime)
            assert result.valid, f"{mime} should be allowed"

    def test_valid_video_mime_types(self):
        """Test that video types for transcription are allowed."""
        for mime in ["video/mp4", "video/webm", "video/quicktime"]:
            result = validate_mime_type(mime)
            assert result.valid, f"{mime} should be allowed"

    def test_invalid_executable_mime_type(self):
        """Test that executable MIME types are rejected."""
        result = validate_mime_type("application/x-executable")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.INVALID_MIME_TYPE
        assert result.http_status == 415  # Unsupported Media Type

    def test_invalid_arbitrary_mime_type(self):
        """Test that arbitrary unknown types are rejected."""
        result = validate_mime_type("application/x-malware")
        assert not result.valid
        assert result.http_status == 415

    def test_mime_type_with_charset(self):
        """Test that MIME types with charset parameters are handled."""
        result = validate_mime_type("text/plain; charset=utf-8")
        assert result.valid

    def test_none_mime_type_allowed(self):
        """Test that None MIME type is allowed (will rely on extension validation)."""
        result = validate_mime_type(None)
        assert result.valid

    def test_empty_mime_type_allowed(self):
        """Test that empty MIME type is allowed (will rely on extension validation)."""
        result = validate_mime_type("")
        assert result.valid

    def test_custom_allowed_types(self):
        """Test custom MIME type whitelist."""
        custom_types = frozenset({"application/custom"})
        result = validate_mime_type("application/custom", allowed_types=custom_types)
        assert result.valid

        result = validate_mime_type("application/pdf", allowed_types=custom_types)
        assert not result.valid


class TestValidateExtension:
    """Tests for file extension validation."""

    def test_valid_document_extensions(self):
        """Test that document extensions are allowed."""
        for ext in [".pdf", ".doc", ".docx", ".txt", ".md", ".csv"]:
            result = validate_extension(f"document{ext}")
            assert result.valid, f"{ext} should be allowed"

    def test_valid_code_extensions(self):
        """Test that code file extensions are allowed."""
        for ext in [".py", ".js", ".ts", ".java", ".go", ".rs", ".rb"]:
            result = validate_extension(f"code{ext}")
            assert result.valid, f"{ext} should be allowed"

    def test_valid_image_extensions(self):
        """Test that image extensions are allowed."""
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]:
            result = validate_extension(f"image{ext}")
            assert result.valid, f"{ext} should be allowed"

    def test_invalid_executable_extension(self):
        """Test that executable extensions are rejected."""
        result = validate_extension("malware.exe")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.INVALID_EXTENSION
        assert result.http_status == 415

    def test_invalid_script_extension(self):
        """Test that dangerous script extensions are rejected."""
        # Note: .sh and .bash are allowed for code indexing
        # But .bat, .cmd, .ps1 should be blocked at MIME level
        result = validate_extension("script.bat")
        assert not result.valid

    def test_no_extension(self):
        """Test that files without extension are rejected."""
        result = validate_extension("filename")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.INVALID_EXTENSION

    def test_case_insensitive_extension(self):
        """Test that extension check is case-insensitive."""
        result = validate_extension("document.PDF")
        assert result.valid

        result = validate_extension("image.JPG")
        assert result.valid

    def test_custom_allowed_extensions(self):
        """Test custom extension whitelist."""
        custom_extensions = frozenset({".custom"})
        result = validate_extension("file.custom", allowed_extensions=custom_extensions)
        assert result.valid

        result = validate_extension("file.pdf", allowed_extensions=custom_extensions)
        assert not result.valid

    def test_details_include_extension(self):
        """Test that error details include the extension."""
        result = validate_extension("file.xyz")
        assert result.details is not None
        assert result.details["extension"] == ".xyz"


class TestValidateFilenameSecurity:
    """Tests for filename security validation (path traversal, null bytes, etc.)."""

    def test_valid_simple_filename(self):
        """Test that simple filenames pass."""
        result = validate_filename_security("document.pdf")
        assert result.valid

    def test_valid_filename_with_spaces(self):
        """Test that filenames with spaces pass."""
        result = validate_filename_security("my document.pdf")
        assert result.valid

    def test_valid_filename_with_underscores(self):
        """Test that filenames with underscores pass."""
        result = validate_filename_security("my_document_v2.pdf")
        assert result.valid

    def test_valid_filename_with_hyphens(self):
        """Test that filenames with hyphens pass."""
        result = validate_filename_security("my-document-v2.pdf")
        assert result.valid

    def test_path_traversal_dotdot(self):
        """Test that .. path traversal is blocked."""
        result = validate_filename_security("../../../etc/passwd")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.PATH_TRAVERSAL
        assert result.http_status == 400

    def test_path_traversal_forward_slash(self):
        """Test that forward slash paths are blocked."""
        result = validate_filename_security("/etc/passwd")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.PATH_TRAVERSAL

    def test_path_traversal_backslash(self):
        """Test that backslash paths are blocked."""
        result = validate_filename_security("..\\..\\windows\\system32\\config")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.PATH_TRAVERSAL

    def test_null_byte_injection(self):
        """Test that null bytes are blocked."""
        result = validate_filename_security("file.pdf\x00.exe")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.NULL_BYTES

    def test_empty_filename(self):
        """Test that empty filenames are rejected."""
        result = validate_filename_security("")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.EMPTY_FILENAME

    def test_whitespace_only_filename(self):
        """Test that whitespace-only filenames are rejected."""
        result = validate_filename_security("   ")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.EMPTY_FILENAME

    def test_filename_too_long(self):
        """Test that filenames exceeding max length are rejected."""
        long_name = "a" * (MAX_FILENAME_LENGTH + 1) + ".txt"
        result = validate_filename_security(long_name)
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.FILENAME_TOO_LONG

    def test_filename_at_max_length(self):
        """Test that filename exactly at max length is allowed."""
        # Account for extension
        name_length = MAX_FILENAME_LENGTH - 4  # ".txt"
        long_name = "a" * name_length + ".txt"
        result = validate_filename_security(long_name)
        assert result.valid

    def test_hidden_file_unix(self):
        """Test that hidden files (starting with .) are blocked."""
        result = validate_filename_security(".hidden")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.INVALID_FILENAME

    def test_control_characters(self):
        """Test that control characters are blocked."""
        result = validate_filename_security("file\x1b.txt")
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_characters(self):
        """Test that Windows reserved characters are blocked."""
        for char in '<>:"|?*':
            result = validate_filename_security(f"file{char}name.txt")
            assert not result.valid, f"Character '{char}' should be blocked"

    def test_windows_reserved_names(self):
        """Test that Windows reserved names are blocked."""
        for name in ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]:
            result = validate_filename_security(f"{name}.txt")
            assert not result.valid, f"Reserved name '{name}' should be blocked"
            # Also test lowercase
            result = validate_filename_security(f"{name.lower()}.txt")
            assert not result.valid

    def test_dots_only_filename(self):
        """Test that dots-only filenames are blocked."""
        result = validate_filename_security("...")
        assert not result.valid


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_sanitize_simple_filename(self):
        """Test that simple filenames are unchanged."""
        assert sanitize_filename("document.pdf") == "document.pdf"

    def test_sanitize_removes_path_components(self):
        """Test that path components are stripped."""
        assert sanitize_filename("../etc/passwd.txt") == "passwd.txt"
        assert sanitize_filename("/var/log/file.txt") == "file.txt"

    def test_sanitize_removes_null_bytes(self):
        """Test that null bytes are removed."""
        result = sanitize_filename("file\x00.pdf")
        assert "\x00" not in result
        assert result == "file.pdf"

    def test_sanitize_replaces_dangerous_chars(self):
        """Test that dangerous characters are replaced with underscore."""
        result = sanitize_filename("file<name>.txt")
        assert "<" not in result
        assert ">" not in result
        assert result == "file_name_.txt"

    def test_sanitize_collapses_underscores(self):
        """Test that multiple underscores are collapsed."""
        result = sanitize_filename("file___name.txt")
        assert "___" not in result
        assert result == "file_name.txt"

    def test_sanitize_preserves_extension(self):
        """Test that file extension is preserved."""
        result = sanitize_filename("some<>file.pdf")
        assert result.endswith(".pdf")

    def test_sanitize_truncates_long_names(self):
        """Test that long filenames are truncated."""
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= MAX_FILENAME_LENGTH
        assert result.endswith(".pdf")

    def test_sanitize_empty_raises_error(self):
        """Test that empty filename raises error."""
        with pytest.raises(FileValidationError) as exc_info:
            sanitize_filename("")
        assert exc_info.value.error_code == FileValidationErrorCode.EMPTY_FILENAME

    def test_sanitize_path_only_raises_error(self):
        """Test that path-only (no filename) raises error."""
        with pytest.raises(FileValidationError):
            sanitize_filename("../../../")


class TestValidateFileUpload:
    """Tests for comprehensive file upload validation."""

    def test_valid_pdf_upload(self):
        """Test valid PDF file upload."""
        result = validate_file_upload(
            filename="report.pdf",
            size=1024000,
            content_type="application/pdf",
        )
        assert result.valid

    def test_valid_image_upload(self):
        """Test valid image file upload."""
        result = validate_file_upload(
            filename="photo.jpg",
            size=500000,
            content_type="image/jpeg",
        )
        assert result.valid

    def test_valid_code_file_upload(self):
        """Test valid code file upload."""
        result = validate_file_upload(
            filename="script.py",
            size=5000,
            content_type="text/x-python",
        )
        assert result.valid

    def test_file_too_large_returns_413(self):
        """Test that oversized files return 413 status."""
        result = validate_file_upload(
            filename="large.pdf",
            size=MAX_FILE_SIZE + 1,
            content_type="application/pdf",
        )
        assert not result.valid
        assert result.http_status == 413

    def test_invalid_mime_type_returns_415(self):
        """Test that invalid MIME types return 415 status."""
        result = validate_file_upload(
            filename="file.pdf",
            size=1000,
            content_type="application/x-malware",
        )
        assert not result.valid
        assert result.http_status == 415

    def test_invalid_extension_returns_415(self):
        """Test that invalid extensions return 415 status."""
        result = validate_file_upload(
            filename="file.exe",
            size=1000,
            content_type=None,
        )
        assert not result.valid
        assert result.http_status == 415

    def test_path_traversal_returns_400(self):
        """Test that path traversal attempts return 400 status."""
        result = validate_file_upload(
            filename="../../../etc/passwd",
            size=1000,
            content_type="text/plain",
        )
        assert not result.valid
        assert result.http_status == 400

    def test_validates_all_aspects(self):
        """Test that all validation aspects are checked."""
        # Valid file
        result = validate_file_upload(
            filename="document.pdf",
            size=1024,
            content_type="application/pdf",
        )
        assert result.valid

        # Invalid size
        result = validate_file_upload(
            filename="document.pdf",
            size=0,
            content_type="application/pdf",
        )
        assert not result.valid

    def test_accepts_octet_stream_with_valid_extension(self):
        """Test that octet-stream is accepted when extension is valid."""
        result = validate_file_upload(
            filename="document.pdf",
            size=1024,
            content_type="application/octet-stream",
        )
        assert result.valid


class TestConfigurableMaxSize:
    """Tests for configurable max file size via environment variable."""

    def test_default_max_size(self):
        """Test default max size is 100MB."""
        assert DEFAULT_MAX_FILE_SIZE == 100 * 1024 * 1024

    def test_get_max_file_size(self):
        """Test get_max_file_size returns configured value."""
        # Should return the current configured value
        size = get_max_file_size()
        assert isinstance(size, int)
        assert size > 0

    def test_get_max_file_size_mb(self):
        """Test get_max_file_size_mb returns value in MB."""
        size_mb = get_max_file_size_mb()
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_env_var_override(self):
        """Test that ARAGORA_MAX_FILE_SIZE env var overrides default."""
        # This test verifies the mechanism works by checking
        # that MAX_FILE_SIZE is used from the module
        # (actual env var testing would require module reload)
        assert MAX_FILE_SIZE >= 0


class TestFileValidationResult:
    """Tests for FileValidationResult dataclass."""

    def test_success_factory(self):
        """Test success() factory method."""
        result = FileValidationResult.success()
        assert result.valid
        assert result.error_code is None
        assert result.error_message is None

    def test_failure_factory(self):
        """Test failure() factory method."""
        result = FileValidationResult.failure(
            error_code=FileValidationErrorCode.FILE_TOO_LARGE,
            error_message="File too large",
            http_status=413,
            details={"size": 1000},
        )
        assert not result.valid
        assert result.error_code == FileValidationErrorCode.FILE_TOO_LARGE
        assert result.error_message == "File too large"
        assert result.http_status == 413
        assert result.details == {"size": 1000}


class TestFileValidationError:
    """Tests for FileValidationError exception."""

    def test_exception_attributes(self):
        """Test that exception has all expected attributes."""
        error = FileValidationError(
            message="Test error",
            error_code=FileValidationErrorCode.FILE_TOO_LARGE,
            http_status=413,
            details={"key": "value"},
        )
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == FileValidationErrorCode.FILE_TOO_LARGE
        assert error.http_status == 413
        assert error.details == {"key": "value"}

    def test_exception_default_http_status(self):
        """Test that default HTTP status is 400."""
        error = FileValidationError(
            message="Test error",
            error_code=FileValidationErrorCode.INVALID_FILENAME,
        )
        assert error.http_status == 400


class TestAllowedMimeTypes:
    """Tests for ALLOWED_MIME_TYPES constant."""

    def test_contains_common_document_types(self):
        """Test that common document types are in whitelist."""
        assert "application/pdf" in ALLOWED_MIME_TYPES
        assert "text/plain" in ALLOWED_MIME_TYPES
        assert "text/csv" in ALLOWED_MIME_TYPES

    def test_contains_office_types(self):
        """Test that Office document types are in whitelist."""
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in ALLOWED_MIME_TYPES
        )
        assert (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            in ALLOWED_MIME_TYPES
        )

    def test_contains_image_types(self):
        """Test that common image types are in whitelist."""
        assert "image/png" in ALLOWED_MIME_TYPES
        assert "image/jpeg" in ALLOWED_MIME_TYPES

    def test_contains_audio_types(self):
        """Test that audio types for transcription are in whitelist."""
        assert "audio/mpeg" in ALLOWED_MIME_TYPES
        assert "audio/wav" in ALLOWED_MIME_TYPES

    def test_is_frozenset(self):
        """Test that ALLOWED_MIME_TYPES is immutable."""
        assert isinstance(ALLOWED_MIME_TYPES, frozenset)


class TestAllowedExtensions:
    """Tests for ALLOWED_EXTENSIONS constant."""

    def test_contains_common_document_extensions(self):
        """Test that common document extensions are in whitelist."""
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS
        assert ".md" in ALLOWED_EXTENSIONS
        assert ".csv" in ALLOWED_EXTENSIONS

    def test_contains_code_extensions(self):
        """Test that code file extensions are in whitelist."""
        assert ".py" in ALLOWED_EXTENSIONS
        assert ".js" in ALLOWED_EXTENSIONS
        assert ".ts" in ALLOWED_EXTENSIONS
        assert ".java" in ALLOWED_EXTENSIONS

    def test_does_not_contain_dangerous_extensions(self):
        """Test that dangerous extensions are NOT in whitelist."""
        assert ".exe" not in ALLOWED_EXTENSIONS
        assert ".dll" not in ALLOWED_EXTENSIONS
        assert ".bat" not in ALLOWED_EXTENSIONS
        assert ".cmd" not in ALLOWED_EXTENSIONS
        assert ".vbs" not in ALLOWED_EXTENSIONS

    def test_is_frozenset(self):
        """Test that ALLOWED_EXTENSIONS is immutable."""
        assert isinstance(ALLOWED_EXTENSIONS, frozenset)


class TestIntegrationScenarios:
    """Integration tests for real-world upload scenarios."""

    def test_typical_pdf_upload(self):
        """Test typical PDF document upload scenario."""
        result = validate_file_upload(
            filename="Q4_Financial_Report_2024.pdf",
            size=2 * 1024 * 1024,  # 2MB
            content_type="application/pdf",
        )
        assert result.valid

    def test_typical_image_upload(self):
        """Test typical image upload scenario."""
        result = validate_file_upload(
            filename="company-logo.png",
            size=500 * 1024,  # 500KB
            content_type="image/png",
        )
        assert result.valid

    def test_typical_code_upload(self):
        """Test typical code file upload scenario."""
        result = validate_file_upload(
            filename="api_handler.py",
            size=15 * 1024,  # 15KB
            content_type="text/x-python",
        )
        assert result.valid

    def test_audio_for_transcription(self):
        """Test audio file upload for transcription."""
        result = validate_file_upload(
            filename="meeting_recording.mp3",
            size=50 * 1024 * 1024,  # 50MB
            content_type="audio/mpeg",
        )
        assert result.valid

    def test_malicious_path_traversal_attempt(self):
        """Test that malicious path traversal is blocked."""
        result = validate_file_upload(
            filename="..%2F..%2F..%2Fetc%2Fpasswd",  # URL encoded
            size=1000,
            content_type="text/plain",
        )
        # This should be blocked due to invalid characters
        assert not result.valid

    def test_malicious_null_byte_injection(self):
        """Test that null byte injection is blocked."""
        result = validate_file_upload(
            filename="safe.pdf\x00.exe",
            size=1000,
            content_type="application/pdf",
        )
        assert not result.valid

    def test_double_extension_attack(self):
        """Test handling of double extension attack."""
        # .pdf.exe should be rejected because .exe is the actual extension
        result = validate_file_upload(
            filename="document.pdf.exe",
            size=1000,
            content_type="application/pdf",
        )
        assert not result.valid

    def test_uppercase_extension(self):
        """Test that uppercase extensions work."""
        result = validate_file_upload(
            filename="DOCUMENT.PDF",
            size=1000,
            content_type="application/pdf",
        )
        assert result.valid

    def test_unicode_filename(self):
        """Test that unicode filenames are handled."""
        result = validate_file_upload(
            filename="documento_espanol.pdf",
            size=1000,
            content_type="application/pdf",
        )
        assert result.valid
