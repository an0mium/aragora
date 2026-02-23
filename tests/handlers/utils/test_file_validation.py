"""Comprehensive tests for aragora.server.handlers.utils.file_validation.

Tests cover:
- FileValidationErrorCode enum values
- FileValidationResult dataclass (success, failure, fields)
- FileValidationError exception
- validate_file_size (min/max bounds, custom limits)
- validate_mime_type (whitelist, normalization, edge cases)
- validate_extension (whitelist, case handling, no-extension)
- validate_filename_security (traversal, null bytes, control chars, reserved names, length)
- sanitize_filename (dangerous chars, collapse, truncation, extension recovery)
- validate_file_upload (end-to-end orchestration)
- get_max_file_size / get_max_file_size_mb helpers
- Constants: ALLOWED_MIME_TYPES, ALLOWED_EXTENSIONS, DANGEROUS_FILENAME_PATTERNS
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.utils.file_validation import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    DANGEROUS_FILENAME_PATTERNS,
    DEFAULT_MAX_FILE_SIZE,
    MAX_FILE_SIZE,
    MAX_FILENAME_LENGTH,
    MIN_FILE_SIZE,
    FileValidationError,
    FileValidationErrorCode,
    FileValidationResult,
    get_max_file_size,
    get_max_file_size_mb,
    sanitize_filename,
    validate_extension,
    validate_file_size,
    validate_file_upload,
    validate_filename_security,
    validate_mime_type,
)


# ---------------------------------------------------------------------------
# FileValidationErrorCode enum
# ---------------------------------------------------------------------------

class TestFileValidationErrorCode:
    """Tests for the FileValidationErrorCode enum."""

    def test_all_error_codes_exist(self):
        expected = {
            "FILE_TOO_LARGE",
            "FILE_TOO_SMALL",
            "INVALID_MIME_TYPE",
            "INVALID_EXTENSION",
            "PATH_TRAVERSAL",
            "INVALID_FILENAME",
            "FILENAME_TOO_LONG",
            "NULL_BYTES",
            "EMPTY_FILENAME",
        }
        actual = {e.name for e in FileValidationErrorCode}
        assert actual == expected

    def test_error_code_values_are_strings(self):
        for code in FileValidationErrorCode:
            assert isinstance(code.value, str)

    def test_specific_code_values(self):
        assert FileValidationErrorCode.FILE_TOO_LARGE.value == "file_too_large"
        assert FileValidationErrorCode.PATH_TRAVERSAL.value == "path_traversal"
        assert FileValidationErrorCode.NULL_BYTES.value == "null_bytes"


# ---------------------------------------------------------------------------
# FileValidationResult dataclass
# ---------------------------------------------------------------------------

class TestFileValidationResult:
    """Tests for the FileValidationResult dataclass."""

    def test_success_factory(self):
        r = FileValidationResult.success()
        assert r.valid is True
        assert r.error_code is None
        assert r.error_message is None
        assert r.http_status == 400  # default
        assert r.details is None

    def test_failure_factory_defaults(self):
        r = FileValidationResult.failure(
            error_code=FileValidationErrorCode.FILE_TOO_LARGE,
            error_message="Too big",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_LARGE
        assert r.error_message == "Too big"
        assert r.http_status == 400
        assert r.details is None

    def test_failure_factory_custom_status_and_details(self):
        details = {"received_bytes": 999}
        r = FileValidationResult.failure(
            error_code=FileValidationErrorCode.FILE_TOO_SMALL,
            error_message="Too small",
            http_status=422,
            details=details,
        )
        assert r.http_status == 422
        assert r.details == details

    def test_direct_construction(self):
        r = FileValidationResult(valid=True)
        assert r.valid is True


# ---------------------------------------------------------------------------
# FileValidationError exception
# ---------------------------------------------------------------------------

class TestFileValidationError:
    """Tests for the FileValidationError exception."""

    def test_basic_exception(self):
        err = FileValidationError(
            message="Bad file",
            error_code=FileValidationErrorCode.INVALID_FILENAME,
        )
        assert str(err) == "Bad file"
        assert err.message == "Bad file"
        assert err.error_code is FileValidationErrorCode.INVALID_FILENAME
        assert err.http_status == 400
        assert err.details == {}

    def test_custom_status_and_details(self):
        err = FileValidationError(
            message="Nope",
            error_code=FileValidationErrorCode.FILE_TOO_LARGE,
            http_status=413,
            details={"max": 100},
        )
        assert err.http_status == 413
        assert err.details == {"max": 100}

    def test_is_exception_subclass(self):
        err = FileValidationError(
            message="test",
            error_code=FileValidationErrorCode.EMPTY_FILENAME,
        )
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(FileValidationError) as exc_info:
            raise FileValidationError(
                message="bad",
                error_code=FileValidationErrorCode.NULL_BYTES,
            )
        assert exc_info.value.error_code is FileValidationErrorCode.NULL_BYTES


# ---------------------------------------------------------------------------
# validate_file_size
# ---------------------------------------------------------------------------

class TestValidateFileSize:
    """Tests for validate_file_size."""

    def test_valid_size(self):
        r = validate_file_size(1024)
        assert r.valid is True

    def test_minimum_valid_size(self):
        r = validate_file_size(MIN_FILE_SIZE)
        assert r.valid is True

    def test_below_minimum_size(self):
        r = validate_file_size(0)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_SMALL
        assert r.details["received_bytes"] == 0
        assert r.details["min_bytes"] == MIN_FILE_SIZE

    def test_negative_size(self):
        r = validate_file_size(-10)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_SMALL

    def test_exceeds_default_max(self):
        r = validate_file_size(MAX_FILE_SIZE + 1)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_LARGE
        assert r.http_status == 413

    def test_exact_max_size_is_valid(self):
        r = validate_file_size(MAX_FILE_SIZE)
        assert r.valid is True

    def test_custom_max_size(self):
        r = validate_file_size(500, max_size=1000)
        assert r.valid is True

    def test_custom_max_exceeded(self):
        r = validate_file_size(1001, max_size=1000)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_LARGE
        assert r.details["max_bytes"] == 1000

    def test_custom_min_size(self):
        r = validate_file_size(10, min_size=10)
        assert r.valid is True

    def test_custom_min_not_met(self):
        r = validate_file_size(5, min_size=10)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_SMALL
        assert r.details["min_bytes"] == 10

    def test_error_message_includes_mb(self):
        r = validate_file_size(200_000_000, max_size=100_000_000)
        assert r.valid is False
        assert "MB" in r.error_message

    def test_too_small_error_message_includes_bytes(self):
        r = validate_file_size(0)
        assert "bytes" in r.error_message


# ---------------------------------------------------------------------------
# validate_mime_type
# ---------------------------------------------------------------------------

class TestValidateMimeType:
    """Tests for validate_mime_type."""

    def test_valid_pdf(self):
        r = validate_mime_type("application/pdf")
        assert r.valid is True

    def test_valid_plain_text(self):
        r = validate_mime_type("text/plain")
        assert r.valid is True

    def test_valid_image_png(self):
        r = validate_mime_type("image/png")
        assert r.valid is True

    def test_invalid_mime_type(self):
        r = validate_mime_type("application/x-executable")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_MIME_TYPE
        assert r.http_status == 415

    def test_none_content_type_passes(self):
        """No content type falls through to extension validation."""
        r = validate_mime_type(None)
        assert r.valid is True

    def test_empty_string_passes(self):
        r = validate_mime_type("")
        assert r.valid is True

    def test_mime_with_charset_parameter(self):
        r = validate_mime_type("text/plain; charset=utf-8")
        assert r.valid is True

    def test_mime_with_boundary_parameter(self):
        r = validate_mime_type("application/json; boundary=something")
        assert r.valid is True

    def test_case_insensitive(self):
        r = validate_mime_type("APPLICATION/PDF")
        assert r.valid is True

    def test_mixed_case_with_params(self):
        r = validate_mime_type("Text/HTML; Charset=UTF-8")
        assert r.valid is True

    def test_custom_allowed_types(self):
        custom = frozenset({"custom/type"})
        r = validate_mime_type("custom/type", allowed_types=custom)
        assert r.valid is True

    def test_custom_allowed_rejects_default(self):
        custom = frozenset({"custom/type"})
        r = validate_mime_type("application/pdf", allowed_types=custom)
        assert r.valid is False

    def test_error_details_contain_content_type(self):
        r = validate_mime_type("application/x-shellscript")
        assert r.valid is False
        assert r.details["content_type"] == "application/x-shellscript"

    def test_octet_stream_allowed(self):
        r = validate_mime_type("application/octet-stream")
        assert r.valid is True

    def test_audio_types_allowed(self):
        for t in ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"]:
            r = validate_mime_type(t)
            assert r.valid is True, f"{t} should be allowed"

    def test_video_types_allowed(self):
        for t in ["video/mp4", "video/webm", "video/quicktime"]:
            r = validate_mime_type(t)
            assert r.valid is True, f"{t} should be allowed"


# ---------------------------------------------------------------------------
# validate_extension
# ---------------------------------------------------------------------------

class TestValidateExtension:
    """Tests for validate_extension."""

    def test_valid_pdf_extension(self):
        r = validate_extension("report.pdf")
        assert r.valid is True

    def test_valid_txt_extension(self):
        r = validate_extension("notes.txt")
        assert r.valid is True

    def test_valid_python_extension(self):
        r = validate_extension("script.py")
        assert r.valid is True

    def test_uppercase_extension_normalized(self):
        r = validate_extension("report.PDF")
        assert r.valid is True

    def test_mixed_case_extension(self):
        r = validate_extension("image.JpEg")
        assert r.valid is True

    def test_no_extension_fails(self):
        r = validate_extension("noextension")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_EXTENSION
        assert "must have an extension" in r.error_message

    def test_disallowed_extension(self):
        r = validate_extension("malware.exe")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_EXTENSION
        assert r.http_status == 415

    def test_double_extension_uses_last(self):
        r = validate_extension("archive.tar.gz")
        assert r.valid is True  # .gz is in ALLOWED_EXTENSIONS

    def test_custom_allowed_extensions(self):
        custom = frozenset({".xyz"})
        r = validate_extension("data.xyz", allowed_extensions=custom)
        assert r.valid is True

    def test_custom_allowed_rejects_default(self):
        custom = frozenset({".xyz"})
        r = validate_extension("report.pdf", allowed_extensions=custom)
        assert r.valid is False

    def test_error_details_contain_extension(self):
        r = validate_extension("bad.exe")
        assert r.details["extension"] == ".exe"
        assert r.details["filename"] == "bad.exe"

    def test_dot_only_filename(self):
        """A file named just '.' has no real extension."""
        r = validate_extension(".")
        # rsplit('.', 1)[-1] is '' so ext becomes '.'
        # '.' is not in allowed extensions
        assert r.valid is False

    def test_code_extensions_allowed(self):
        for ext in [".js", ".ts", ".java", ".go", ".rs", ".rb", ".swift"]:
            r = validate_extension(f"file{ext}")
            assert r.valid is True, f"{ext} should be allowed"

    def test_data_format_extensions_allowed(self):
        for ext in [".json", ".yaml", ".yml", ".toml", ".ini"]:
            r = validate_extension(f"config{ext}")
            assert r.valid is True, f"{ext} should be allowed"

    def test_archive_extensions_allowed(self):
        for ext in [".zip", ".tar", ".gz", ".tgz"]:
            r = validate_extension(f"archive{ext}")
            assert r.valid is True, f"{ext} should be allowed"


# ---------------------------------------------------------------------------
# validate_filename_security
# ---------------------------------------------------------------------------

class TestValidateFilenameSecurity:
    """Tests for validate_filename_security."""

    def test_valid_simple_filename(self):
        r = validate_filename_security("report.pdf")
        assert r.valid is True

    def test_valid_filename_with_spaces(self):
        r = validate_filename_security("my report.pdf")
        assert r.valid is True

    def test_valid_filename_with_underscores(self):
        r = validate_filename_security("my_report_v2.pdf")
        assert r.valid is True

    def test_valid_filename_with_hyphens(self):
        r = validate_filename_security("my-report-v2.pdf")
        assert r.valid is True

    def test_null_byte_injection(self):
        r = validate_filename_security("file.pdf\x00.exe")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.NULL_BYTES

    def test_empty_filename(self):
        r = validate_filename_security("")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.EMPTY_FILENAME

    def test_whitespace_only_filename(self):
        r = validate_filename_security("   ")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.EMPTY_FILENAME

    def test_path_traversal_dotdot(self):
        r = validate_filename_security("../../etc/passwd")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.PATH_TRAVERSAL

    def test_path_traversal_forward_slash(self):
        r = validate_filename_security("path/to/file.pdf")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.PATH_TRAVERSAL

    def test_path_traversal_backslash(self):
        r = validate_filename_security("path\\to\\file.pdf")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.PATH_TRAVERSAL

    def test_hidden_file_unix(self):
        r = validate_filename_security(".htaccess")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_hidden_file_dotenv(self):
        r = validate_filename_security(".env")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_con(self):
        r = validate_filename_security("CON")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_nul(self):
        r = validate_filename_security("NUL")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_com1(self):
        r = validate_filename_security("COM1")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_lpt3(self):
        r = validate_filename_security("LPT3")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_prn_with_extension(self):
        r = validate_filename_security("PRN.txt")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_aux(self):
        r = validate_filename_security("AUX")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_windows_reserved_characters(self):
        for ch in '<>:"|?*':
            r = validate_filename_security(f"file{ch}name.txt")
            assert r.valid is False, f"Character {ch!r} should be rejected"

    def test_control_characters(self):
        r = validate_filename_security("file\x01name.txt")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_filename_too_long(self):
        long_name = "a" * 256 + ".txt"
        r = validate_filename_security(long_name)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILENAME_TOO_LONG
        assert r.details["max_length"] == MAX_FILENAME_LENGTH

    def test_filename_at_max_length(self):
        name = "a" * (MAX_FILENAME_LENGTH - 4) + ".txt"
        assert len(name) == MAX_FILENAME_LENGTH
        r = validate_filename_security(name)
        assert r.valid is True

    def test_special_characters_rejected(self):
        # Parentheses, brackets, etc. via the safe_chars check
        r = validate_filename_security("file(1).pdf")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_semicolon_rejected(self):
        r = validate_filename_security("file;drop.pdf")
        assert r.valid is False

    def test_unicode_letters_allowed(self):
        """Unicode word characters should be allowed by \\w with re.UNICODE."""
        r = validate_filename_security("rapport_2024.pdf")
        assert r.valid is True

    def test_numbers_in_filename(self):
        r = validate_filename_security("report_2024_v3.pdf")
        assert r.valid is True


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    """Tests for sanitize_filename."""

    def test_clean_filename_unchanged(self):
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_strips_path_components(self):
        result = sanitize_filename("/path/to/report.pdf")
        assert result == "report.pdf"

    def test_removes_null_bytes(self):
        result = sanitize_filename("file\x00.pdf")
        assert "\x00" not in result
        assert result == "file.pdf"

    def test_replaces_dangerous_characters(self):
        result = sanitize_filename("file<>name.pdf")
        # < and > get replaced by _
        assert "<" not in result
        assert ">" not in result
        assert result.endswith(".pdf")

    def test_collapses_multiple_underscores(self):
        result = sanitize_filename("file___name.pdf")
        assert "___" not in result

    def test_collapses_multiple_spaces(self):
        result = sanitize_filename("file   name.pdf")
        # Spaces become underscores, then collapsed
        assert "   " not in result

    def test_strips_leading_trailing_underscores_periods(self):
        result = sanitize_filename("___file.pdf___")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_empty_filename_raises(self):
        with pytest.raises(FileValidationError) as exc_info:
            sanitize_filename("")
        assert exc_info.value.error_code is FileValidationErrorCode.EMPTY_FILENAME

    def test_all_invalid_chars_raises(self):
        with pytest.raises(FileValidationError) as exc_info:
            sanitize_filename("...")
        assert exc_info.value.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_path_only_raises(self):
        with pytest.raises(FileValidationError) as exc_info:
            sanitize_filename("/")
        assert exc_info.value.error_code is FileValidationErrorCode.INVALID_FILENAME

    def test_preserves_extension(self):
        result = sanitize_filename("my file!.pdf")
        assert result.endswith(".pdf")

    def test_extension_recovery(self):
        """When sanitization removes the dot, extension should be recovered."""
        # Construct a filename where the name part gets sanitized away but ext is valid
        result = sanitize_filename("!!!.pdf")
        # After replacing non-word chars and stripping, we get just the extension
        # The function tries to recover the extension
        assert ".pdf" in result or result.endswith("pdf")

    def test_truncates_long_filename_with_extension(self):
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= MAX_FILENAME_LENGTH
        assert result.endswith(".pdf")

    def test_truncates_long_filename_without_extension(self):
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= MAX_FILENAME_LENGTH

    def test_windows_path_stripped(self):
        result = sanitize_filename("C:\\Users\\test\\file.pdf")
        # os.path.basename on posix gives "C:\\Users\\test\\file.pdf" literally
        # but the backslashes get replaced by underscores
        # The key point is no path traversal remains
        assert "/" not in result
        assert "\\" not in result

    def test_unicode_filename(self):
        result = sanitize_filename("rapport_2024.pdf")
        assert "rapport" in result

    def test_mixed_dangerous_and_safe_chars(self):
        result = sanitize_filename("my<>file|name.txt")
        assert result.endswith(".txt")
        # Dangerous chars replaced
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result


# ---------------------------------------------------------------------------
# validate_file_upload (end-to-end)
# ---------------------------------------------------------------------------

class TestValidateFileUpload:
    """Tests for validate_file_upload orchestrating all validators."""

    def test_valid_upload(self):
        r = validate_file_upload(
            filename="report.pdf",
            size=1024,
            content_type="application/pdf",
        )
        assert r.valid is True

    def test_invalid_filename_checked_first(self):
        """Filename security is validated before size/mime/extension."""
        r = validate_file_upload(
            filename="../../etc/passwd",
            size=1024,
            content_type="application/pdf",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.PATH_TRAVERSAL

    def test_file_too_large(self):
        r = validate_file_upload(
            filename="report.pdf",
            size=MAX_FILE_SIZE + 1,
            content_type="application/pdf",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_LARGE

    def test_file_too_small(self):
        r = validate_file_upload(
            filename="report.pdf",
            size=0,
            content_type="application/pdf",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_SMALL

    def test_invalid_mime_type(self):
        r = validate_file_upload(
            filename="script.pdf",
            size=100,
            content_type="application/x-executable",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_MIME_TYPE

    def test_invalid_extension(self):
        r = validate_file_upload(
            filename="malware.exe",
            size=100,
            content_type="application/octet-stream",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.INVALID_EXTENSION

    def test_no_content_type_still_validates_extension(self):
        r = validate_file_upload(
            filename="report.pdf",
            size=100,
            content_type=None,
        )
        assert r.valid is True

    def test_custom_max_size(self):
        r = validate_file_upload(
            filename="small.txt",
            size=500,
            content_type="text/plain",
            max_size=100,
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILE_TOO_LARGE

    def test_custom_mime_types(self):
        custom_mime = frozenset({"custom/special"})
        r = validate_file_upload(
            filename="data.txt",
            size=100,
            content_type="custom/special",
            allowed_mime_types=custom_mime,
        )
        assert r.valid is True

    def test_custom_extensions(self):
        custom_ext = frozenset({".special"})
        r = validate_file_upload(
            filename="data.special",
            size=100,
            content_type=None,
            allowed_extensions=custom_ext,
        )
        assert r.valid is True

    def test_null_byte_in_filename(self):
        r = validate_file_upload(
            filename="file\x00.pdf",
            size=100,
            content_type="application/pdf",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.NULL_BYTES

    def test_empty_filename(self):
        r = validate_file_upload(
            filename="",
            size=100,
            content_type="application/pdf",
        )
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.EMPTY_FILENAME


# ---------------------------------------------------------------------------
# get_max_file_size / get_max_file_size_mb
# ---------------------------------------------------------------------------

class TestGetMaxFileSize:
    """Tests for size helper functions."""

    def test_get_max_file_size_returns_int(self):
        result = get_max_file_size()
        assert isinstance(result, int)
        assert result == MAX_FILE_SIZE

    def test_get_max_file_size_mb_returns_float(self):
        result = get_max_file_size_mb()
        assert isinstance(result, float)
        assert result == MAX_FILE_SIZE / (1024 * 1024)

    def test_default_max_size_is_100mb(self):
        assert DEFAULT_MAX_FILE_SIZE == 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_allowed_mime_types_is_frozenset(self):
        assert isinstance(ALLOWED_MIME_TYPES, frozenset)

    def test_allowed_extensions_is_frozenset(self):
        assert isinstance(ALLOWED_EXTENSIONS, frozenset)

    def test_dangerous_patterns_is_tuple(self):
        assert isinstance(DANGEROUS_FILENAME_PATTERNS, tuple)

    def test_mime_types_include_common_types(self):
        expected = {
            "application/pdf",
            "text/plain",
            "image/png",
            "image/jpeg",
            "application/json",
            "text/csv",
        }
        assert expected.issubset(ALLOWED_MIME_TYPES)

    def test_extensions_include_common_types(self):
        expected = {".pdf", ".txt", ".png", ".jpg", ".json", ".csv", ".py"}
        assert expected.issubset(ALLOWED_EXTENSIONS)

    def test_all_extensions_start_with_dot(self):
        for ext in ALLOWED_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext!r} should start with a dot"

    def test_min_file_size_is_positive(self):
        assert MIN_FILE_SIZE >= 1

    def test_max_filename_length(self):
        assert MAX_FILENAME_LENGTH == 255

    def test_dangerous_patterns_match_dotdot(self):
        matched = any(p.search("..") for p in DANGEROUS_FILENAME_PATTERNS)
        assert matched is True

    def test_dangerous_patterns_match_hidden_files(self):
        matched = any(p.search(".hidden") for p in DANGEROUS_FILENAME_PATTERNS)
        assert matched is True

    def test_dangerous_patterns_match_windows_reserved(self):
        for name in ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]:
            matched = any(p.search(name) for p in DANGEROUS_FILENAME_PATTERNS)
            assert matched is True, f"{name} should match a dangerous pattern"


# ---------------------------------------------------------------------------
# Edge cases and security scenarios
# ---------------------------------------------------------------------------

class TestSecurityEdgeCases:
    """Edge-case and adversarial input tests."""

    def test_dotdot_in_middle_of_name(self):
        """'..' anywhere in filename triggers PATH_TRAVERSAL check."""
        r = validate_filename_security("file..name.pdf")
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.PATH_TRAVERSAL

    def test_encoded_path_traversal(self):
        """URL-encoded path traversal should still fail on the dots."""
        r = validate_filename_security("..%2f..%2fetc%2fpasswd")
        # The .. at the start triggers the DANGEROUS_FILENAME_PATTERNS
        assert r.valid is False

    def test_very_long_extension(self):
        name = "file." + "a" * 300
        r = validate_filename_security(name)
        assert r.valid is False
        assert r.error_code is FileValidationErrorCode.FILENAME_TOO_LONG

    def test_only_dots_filename(self):
        r = validate_filename_security("...")
        assert r.valid is False

    def test_tab_character_in_filename(self):
        r = validate_filename_security("file\tname.txt")
        assert r.valid is False

    def test_newline_in_filename(self):
        r = validate_filename_security("file\nname.txt")
        assert r.valid is False

    def test_zero_width_space(self):
        """Zero-width space (U+200B) is not in \\w so should fail safe chars check."""
        r = validate_filename_security("file\u200bname.txt")
        assert r.valid is False

    def test_upload_validates_all_layers(self):
        """A valid-looking file passes all validation layers."""
        r = validate_file_upload(
            filename="quarterly_report_2024.pdf",
            size=50_000,
            content_type="application/pdf",
        )
        assert r.valid is True

    def test_upload_with_mime_charset(self):
        r = validate_file_upload(
            filename="data.csv",
            size=100,
            content_type="text/csv; charset=utf-8",
        )
        assert r.valid is True

    def test_extension_case_insensitive_in_upload(self):
        r = validate_file_upload(
            filename="image.PNG",
            size=100,
            content_type="image/png",
        )
        assert r.valid is True

    def test_sanitize_preserves_safe_unicode(self):
        result = sanitize_filename("notes_2024.txt")
        assert "notes" in result
        assert result.endswith(".txt")

    def test_sanitize_handles_all_underscores_stripped(self):
        """Edge: if stripping leading/trailing gives empty, raise."""
        with pytest.raises(FileValidationError):
            sanitize_filename("___")

    def test_sanitize_handles_all_dots_stripped(self):
        with pytest.raises(FileValidationError):
            sanitize_filename(".....")
