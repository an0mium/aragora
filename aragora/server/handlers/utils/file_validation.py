"""
File upload validation utilities.

Provides comprehensive validation for file uploads including:
- File size limits (configurable via environment)
- MIME type whitelist validation
- Path traversal prevention
- File extension validation
- Filename sanitization

Usage:
    from aragora.server.handlers.utils.file_validation import (
        validate_file_upload,
        FileValidationError,
        FileValidationResult,
    )

    result = validate_file_upload(
        filename="report.pdf",
        size=1024000,
        content_type="application/pdf",
    )
    if not result.valid:
        return error_response(result.error_message, result.http_status)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

# Maximum file size in bytes (default 100MB, configurable via env)
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILE_SIZE = int(os.environ.get("ARAGORA_MAX_FILE_SIZE", str(DEFAULT_MAX_FILE_SIZE)))

# Maximum filename length
MAX_FILENAME_LENGTH = 255

# Minimum file size (prevent empty file attacks)
MIN_FILE_SIZE = 1


class FileValidationErrorCode(Enum):
    """Specific error codes for file validation failures."""

    FILE_TOO_LARGE = "file_too_large"
    FILE_TOO_SMALL = "file_too_small"
    INVALID_MIME_TYPE = "invalid_mime_type"
    INVALID_EXTENSION = "invalid_extension"
    PATH_TRAVERSAL = "path_traversal"
    INVALID_FILENAME = "invalid_filename"
    FILENAME_TOO_LONG = "filename_too_long"
    NULL_BYTES = "null_bytes"
    EMPTY_FILENAME = "empty_filename"


# MIME types allowed for upload (whitelist approach)
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {
        # Documents
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
        "application/rtf",
        "application/vnd.oasis.opendocument.text",  # .odt
        "application/vnd.oasis.opendocument.spreadsheet",  # .ods
        "application/vnd.oasis.opendocument.presentation",  # .odp
        # Text
        "text/plain",
        "text/csv",
        "text/markdown",
        "text/html",
        "text/xml",
        "text/x-python",
        "text/x-java",
        "text/x-c",
        "text/x-c++",
        "text/x-javascript",
        "text/x-typescript",
        # Data formats
        "application/json",
        "application/xml",
        "application/x-yaml",
        "text/yaml",
        # Images (for OCR, documentation)
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "image/tiff",
        "image/bmp",
        # Audio (for transcription)
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/flac",
        "audio/aac",
        "audio/m4a",
        "audio/x-m4a",
        # Video (for transcription)
        "video/mp4",
        "video/webm",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        # Archives (for batch processing)
        "application/zip",
        "application/x-tar",
        "application/gzip",
        "application/x-gzip",
        # Generic binary (allow with extension check)
        "application/octet-stream",
    }
)

# Allowed file extensions (whitelist approach)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Documents
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".rtf",
        ".odt",
        ".ods",
        ".odp",
        ".epub",
        # Text
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".html",
        ".htm",
        ".xml",
        # Code (for indexing)
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".jl",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".graphql",
        ".proto",
        ".ex",
        ".exs",
        # Data formats
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".conf",
        ".cfg",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".tiff",
        ".tif",
        ".bmp",
        ".heic",
        # Audio
        ".mp3",
        ".m4a",
        ".wav",
        ".webm",
        ".ogg",
        ".flac",
        ".aac",
        ".wma",
        # Video
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".wmv",
        ".m4v",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
    }
)

# Dangerous patterns in filenames
DANGEROUS_FILENAME_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\.\."),  # Path traversal
    re.compile(r"^\."),  # Hidden files (Unix)
    re.compile(r"[<>:\"|?*]"),  # Windows reserved characters
    re.compile(r"[\x00-\x1f]"),  # Control characters
    re.compile(
        r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\.|$)", re.IGNORECASE
    ),  # Windows reserved names
)


@dataclass
class FileValidationResult:
    """Result of file validation."""

    valid: bool
    error_code: FileValidationErrorCode | None = None
    error_message: str | None = None
    http_status: int = 400
    details: dict | None = None

    @classmethod
    def success(cls) -> FileValidationResult:
        """Create a successful validation result."""
        return cls(valid=True)

    @classmethod
    def failure(
        cls,
        error_code: FileValidationErrorCode,
        error_message: str,
        http_status: int = 400,
        details: dict | None = None,
    ) -> FileValidationResult:
        """Create a failed validation result."""
        return cls(
            valid=False,
            error_code=error_code,
            error_message=error_message,
            http_status=http_status,
            details=details,
        )


class FileValidationError(Exception):
    """Exception raised when file validation fails."""

    def __init__(
        self,
        message: str,
        error_code: FileValidationErrorCode,
        http_status: int = 400,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.details = details or {}


def validate_file_size(
    size: int,
    max_size: int | None = None,
    min_size: int = MIN_FILE_SIZE,
) -> FileValidationResult:
    """
    Validate file size is within acceptable limits.

    Args:
        size: File size in bytes
        max_size: Maximum allowed size (defaults to MAX_FILE_SIZE env var)
        min_size: Minimum allowed size (defaults to MIN_FILE_SIZE)

    Returns:
        FileValidationResult indicating success or failure
    """
    if max_size is None:
        max_size = MAX_FILE_SIZE

    if size < min_size:
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.FILE_TOO_SMALL,
            error_message=f"File too small. Minimum size: {min_size} bytes",
            http_status=400,
            details={"received_bytes": size, "min_bytes": min_size},
        )

    if size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.FILE_TOO_LARGE,
            error_message=f"File too large. Maximum size: {max_size_mb:.1f}MB",
            http_status=413,  # Payload Too Large
            details={"received_bytes": size, "max_bytes": max_size},
        )

    return FileValidationResult.success()


def validate_mime_type(
    content_type: str | None,
    allowed_types: frozenset[str] | None = None,
) -> FileValidationResult:
    """
    Validate MIME type is in the allowed whitelist.

    Args:
        content_type: MIME type from Content-Type header
        allowed_types: Set of allowed MIME types (defaults to ALLOWED_MIME_TYPES)

    Returns:
        FileValidationResult indicating success or failure
    """
    if allowed_types is None:
        allowed_types = ALLOWED_MIME_TYPES

    if not content_type:
        # If no content type provided, we'll rely on extension validation
        return FileValidationResult.success()

    # Normalize: strip parameters (e.g., "text/plain; charset=utf-8" -> "text/plain")
    mime_base = content_type.split(";")[0].strip().lower()

    if mime_base not in allowed_types:
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.INVALID_MIME_TYPE,
            error_message=f"Unsupported file type: {mime_base}",
            http_status=415,  # Unsupported Media Type
            details={"content_type": mime_base},
        )

    return FileValidationResult.success()


def validate_extension(
    filename: str,
    allowed_extensions: frozenset[str] | None = None,
) -> FileValidationResult:
    """
    Validate file extension is in the allowed whitelist.

    Args:
        filename: Original filename
        allowed_extensions: Set of allowed extensions (defaults to ALLOWED_EXTENSIONS)

    Returns:
        FileValidationResult indicating success or failure
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_EXTENSIONS

    # Extract extension
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    if not ext:
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.INVALID_EXTENSION,
            error_message="File must have an extension",
            http_status=400,
            details={"filename": filename},
        )

    if ext not in allowed_extensions:
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.INVALID_EXTENSION,
            error_message=f"Unsupported file extension: {ext}",
            http_status=415,  # Unsupported Media Type
            details={"extension": ext, "filename": filename},
        )

    return FileValidationResult.success()


def validate_filename_security(filename: str) -> FileValidationResult:
    """
    Validate filename for path traversal and other security issues.

    Checks for:
    - Path traversal sequences (.., /, \\)
    - Null bytes
    - Empty or whitespace-only filenames
    - Dangerous patterns (control chars, reserved names)
    - Overly long filenames

    Args:
        filename: The filename to validate

    Returns:
        FileValidationResult indicating success or failure
    """
    # Check for null bytes (potential injection attack)
    if "\x00" in filename:
        logger.warning("[SECURITY] Null byte injection attempt in filename: %r", filename[:50])
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.NULL_BYTES,
            error_message="Filename contains invalid characters (null bytes)",
            http_status=400,
        )

    # Check for empty filename
    if not filename or not filename.strip():
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.EMPTY_FILENAME,
            error_message="Filename cannot be empty",
            http_status=400,
        )

    # Sanitize: get just the basename (prevent path injection)
    basename = os.path.basename(filename)

    # Also check POSIX path interpretation
    posix_basename = PurePosixPath(filename).name

    # If basename is empty after extraction, the filename was malicious
    if not basename or not posix_basename:
        logger.warning("[SECURITY] Path traversal attempt: %r", filename[:100])
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.PATH_TRAVERSAL,
            error_message="Invalid filename: path components not allowed",
            http_status=400,
        )

    # Check for path traversal sequences in the original filename
    if ".." in filename or "/" in filename or "\\" in filename:
        logger.warning("[SECURITY] Path traversal attempt: %r", filename[:100])
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.PATH_TRAVERSAL,
            error_message="Filename contains path traversal sequences",
            http_status=400,
        )

    # Check filename length
    if len(basename) > MAX_FILENAME_LENGTH:
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.FILENAME_TOO_LONG,
            error_message=f"Filename too long. Maximum: {MAX_FILENAME_LENGTH} characters",
            http_status=400,
            details={"filename_length": len(basename), "max_length": MAX_FILENAME_LENGTH},
        )

    # Check for dangerous patterns
    for pattern in DANGEROUS_FILENAME_PATTERNS:
        if pattern.search(basename):
            logger.warning("[SECURITY] Dangerous filename pattern: %r", basename[:50])
            return FileValidationResult.failure(
                error_code=FileValidationErrorCode.INVALID_FILENAME,
                error_message="Filename contains invalid characters or patterns",
                http_status=400,
                details={"filename": basename[:50]},
            )

    # Check that filename consists only of safe characters
    # Allow alphanumeric, spaces, hyphens, underscores, periods
    safe_chars_pattern = re.compile(r"^[\w\s\-_.]+$", re.UNICODE)
    if not safe_chars_pattern.match(basename):
        logger.warning("[SECURITY] Unsafe characters in filename: %r", basename[:50])
        return FileValidationResult.failure(
            error_code=FileValidationErrorCode.INVALID_FILENAME,
            error_message="Filename contains special characters that are not allowed",
            http_status=400,
            details={"filename": basename[:50]},
        )

    return FileValidationResult.success()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing dangerous characters and path components.

    Args:
        filename: The original filename

    Returns:
        Sanitized filename safe for storage

    Raises:
        FileValidationError: If filename cannot be sanitized to a valid form
    """
    if not filename:
        raise FileValidationError(
            message="Filename cannot be empty",
            error_code=FileValidationErrorCode.EMPTY_FILENAME,
        )

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Get basename only (strip path components)
    basename = os.path.basename(filename)

    if not basename:
        raise FileValidationError(
            message="Invalid filename: no valid filename after sanitization",
            error_code=FileValidationErrorCode.INVALID_FILENAME,
        )

    # Remove/replace dangerous characters
    # Keep alphanumeric, spaces, hyphens, underscores, periods
    sanitized = re.sub(r"[^\w\s\-_.]", "_", basename)

    # Collapse multiple underscores/spaces
    sanitized = re.sub(r"[_\s]+", "_", sanitized)

    # Remove leading/trailing underscores and periods
    sanitized = sanitized.strip("_.")

    # Ensure we still have a valid filename
    if not sanitized:
        raise FileValidationError(
            message="Invalid filename: no valid characters after sanitization",
            error_code=FileValidationErrorCode.INVALID_FILENAME,
        )

    # Ensure extension is preserved
    if "." not in sanitized and "." in basename:
        # Try to recover extension from original
        ext = basename.rsplit(".", 1)[-1]
        if ext and re.match(r"^[a-zA-Z0-9]+$", ext):
            sanitized = f"{sanitized}.{ext}"

    # Enforce max length
    if len(sanitized) > MAX_FILENAME_LENGTH:
        # Truncate while preserving extension
        if "." in sanitized:
            name, ext = sanitized.rsplit(".", 1)
            max_name_len = MAX_FILENAME_LENGTH - len(ext) - 1
            if max_name_len > 0:
                sanitized = f"{name[:max_name_len]}.{ext}"
            else:
                sanitized = sanitized[:MAX_FILENAME_LENGTH]
        else:
            sanitized = sanitized[:MAX_FILENAME_LENGTH]

    return sanitized


def validate_file_upload(
    filename: str,
    size: int,
    content_type: str | None = None,
    max_size: int | None = None,
    allowed_mime_types: frozenset[str] | None = None,
    allowed_extensions: frozenset[str] | None = None,
) -> FileValidationResult:
    """
    Comprehensive file upload validation.

    Validates:
    1. File size (within min/max limits)
    2. MIME type (against whitelist)
    3. File extension (against whitelist)
    4. Filename security (path traversal, null bytes, etc.)

    Args:
        filename: Original filename from upload
        size: File size in bytes
        content_type: MIME type from Content-Type header
        max_size: Optional maximum file size override
        allowed_mime_types: Optional MIME type whitelist override
        allowed_extensions: Optional extension whitelist override

    Returns:
        FileValidationResult with validation status and error details if failed

    Example:
        result = validate_file_upload(
            filename="report.pdf",
            size=1024000,
            content_type="application/pdf",
        )
        if not result.valid:
            return error_response(result.error_message, result.http_status)
    """
    # 1. Validate filename security first (path traversal, null bytes, etc.)
    result = validate_filename_security(filename)
    if not result.valid:
        return result

    # 2. Validate file size
    result = validate_file_size(size, max_size=max_size)
    if not result.valid:
        return result

    # 3. Validate MIME type
    result = validate_mime_type(content_type, allowed_types=allowed_mime_types)
    if not result.valid:
        return result

    # 4. Validate file extension
    result = validate_extension(filename, allowed_extensions=allowed_extensions)
    if not result.valid:
        return result

    return FileValidationResult.success()


def get_max_file_size() -> int:
    """Get the configured maximum file size in bytes."""
    return MAX_FILE_SIZE


def get_max_file_size_mb() -> float:
    """Get the configured maximum file size in megabytes."""
    return MAX_FILE_SIZE / (1024 * 1024)


__all__ = [
    # Main validation function
    "validate_file_upload",
    # Individual validators
    "validate_file_size",
    "validate_mime_type",
    "validate_extension",
    "validate_filename_security",
    # Utilities
    "sanitize_filename",
    "get_max_file_size",
    "get_max_file_size_mb",
    # Types and constants
    "FileValidationResult",
    "FileValidationError",
    "FileValidationErrorCode",
    "ALLOWED_MIME_TYPES",
    "ALLOWED_EXTENSIONS",
    "MAX_FILE_SIZE",
    "MAX_FILENAME_LENGTH",
]
