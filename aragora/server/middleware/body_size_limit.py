"""
Request Body Size Limit Middleware.

Provides protection against memory exhaustion DoS attacks by enforcing
body size limits on incoming requests:
- Check Content-Length header for fast rejection
- Track actual bytes read for chunked transfers
- Configurable per-endpoint overrides for file uploads
- Logging of violations for monitoring

Usage:
    from aragora.server.middleware.body_size_limit import (
        BodySizeLimitMiddleware,
        check_body_size,
        with_body_size_limit,
    )

    # Decorator style
    @with_body_size_limit(max_bytes=10 * 1024 * 1024)  # 10MB
    def upload_handler(self, handler):
        ...

    # Middleware instance style
    middleware = BodySizeLimitMiddleware()
    result = middleware.check_content_length(headers)
    if not result.allowed:
        return error_response(413, result.message)

Configuration via environment:
    ARAGORA_MAX_REQUEST_SIZE=10485760  # Default max body size (10MB)
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, TypeVar

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Default maximum request body size: 10MB
DEFAULT_MAX_REQUEST_SIZE = 10 * 1024 * 1024

# HTTP status code for Payload Too Large
HTTP_PAYLOAD_TOO_LARGE = 413

# Endpoints with larger size limits (file uploads)
DEFAULT_LARGE_ENDPOINTS = {
    "/api/documents/upload": 100 * 1024 * 1024,  # 100MB
    "/api/files/upload": 100 * 1024 * 1024,  # 100MB
    "/api/backup/upload": 500 * 1024 * 1024,  # 500MB
    "/api/import": 50 * 1024 * 1024,  # 50MB
}

# Endpoints with smaller size limits (JSON API)
DEFAULT_SMALL_ENDPOINTS = {
    "/api/auth/": 1 * 1024 * 1024,  # 1MB
    "/api/login": 1 * 1024 * 1024,  # 1MB
    "/api/register": 1 * 1024 * 1024,  # 1MB
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BodySizeLimitConfig:
    """Configuration for request body size limits."""

    # Default maximum body size in bytes
    max_request_size: int = field(
        default_factory=lambda: int(
            os.environ.get("ARAGORA_MAX_REQUEST_SIZE", str(DEFAULT_MAX_REQUEST_SIZE))
        )
    )

    # Per-endpoint size overrides (endpoint pattern -> max bytes)
    endpoint_limits: dict[str, int] = field(default_factory=dict)

    # Whether to log violations
    log_violations: bool = True

    # Whether to include size details in error messages
    include_size_in_error: bool = True

    def __post_init__(self) -> None:
        """Initialize with default endpoint limits if none provided."""
        if not self.endpoint_limits:
            self.endpoint_limits = {
                **DEFAULT_LARGE_ENDPOINTS,
                **DEFAULT_SMALL_ENDPOINTS,
            }

    def get_limit_for_endpoint(self, path: str) -> int:
        """Get the body size limit for a specific endpoint.

        Args:
            path: Request path

        Returns:
            Maximum allowed body size in bytes
        """
        # Check explicit overrides first (longest match wins)
        matched_pattern = ""
        matched_limit = self.max_request_size

        for pattern, limit in self.endpoint_limits.items():
            if pattern in path and len(pattern) > len(matched_pattern):
                matched_pattern = pattern
                matched_limit = limit

        return matched_limit


# Global config instance with thread-safe initialization
_body_size_config: BodySizeLimitConfig | None = None
_body_size_config_lock = threading.Lock()


def get_body_size_config() -> BodySizeLimitConfig:
    """Get or create the global body size configuration."""
    global _body_size_config
    if _body_size_config is None:
        with _body_size_config_lock:
            if _body_size_config is None:
                _body_size_config = BodySizeLimitConfig()
    return _body_size_config


def configure_body_size_limit(
    max_request_size: int | None = None,
    endpoint_limits: dict[str, int] | None = None,
    log_violations: bool | None = None,
    include_size_in_error: bool | None = None,
) -> BodySizeLimitConfig:
    """Configure body size limit settings.

    Args:
        max_request_size: Default maximum body size in bytes
        endpoint_limits: Per-endpoint size overrides
        log_violations: Whether to log violations
        include_size_in_error: Whether to include size details in error

    Returns:
        Updated configuration
    """
    global _body_size_config
    config = get_body_size_config()

    if max_request_size is not None:
        config.max_request_size = max_request_size
    if endpoint_limits is not None:
        config.endpoint_limits.update(endpoint_limits)
    if log_violations is not None:
        config.log_violations = log_violations
    if include_size_in_error is not None:
        config.include_size_in_error = include_size_in_error

    _body_size_config = config
    return config


def reset_body_size_config() -> None:
    """Reset the global body size configuration (for testing)."""
    global _body_size_config
    with _body_size_config_lock:
        _body_size_config = None


# =============================================================================
# Check Result
# =============================================================================


@dataclass
class BodySizeCheckResult:
    """Result of a body size check."""

    allowed: bool
    message: str = ""
    status_code: int = HTTP_PAYLOAD_TOO_LARGE
    content_length: int | None = None
    max_allowed: int | None = None

    @classmethod
    def ok(cls) -> "BodySizeCheckResult":
        """Return a successful check result."""
        return cls(allowed=True)

    @classmethod
    def too_large(
        cls,
        content_length: int,
        max_allowed: int,
        include_size: bool = True,
    ) -> "BodySizeCheckResult":
        """Return a failure result for payload too large."""
        if include_size:
            max_mb = max_allowed / (1024 * 1024)
            request_mb = content_length / (1024 * 1024)
            message = (
                f"Request body too large: {request_mb:.2f}MB exceeds maximum allowed {max_mb:.2f}MB"
            )
        else:
            message = "Request body too large"

        return cls(
            allowed=False,
            message=message,
            status_code=HTTP_PAYLOAD_TOO_LARGE,
            content_length=content_length,
            max_allowed=max_allowed,
        )

    @classmethod
    def invalid_content_length(cls, value: str) -> "BodySizeCheckResult":
        """Return a failure result for invalid Content-Length header."""
        return cls(
            allowed=False,
            message=f"Invalid Content-Length header: {value}",
            status_code=400,
        )

    @classmethod
    def negative_content_length(cls) -> "BodySizeCheckResult":
        """Return a failure result for negative Content-Length."""
        return cls(
            allowed=False,
            message="Content-Length cannot be negative",
            status_code=400,
        )


# =============================================================================
# Middleware Class
# =============================================================================


class BodySizeLimitMiddleware:
    """Middleware for enforcing request body size limits.

    Usage:
        middleware = BodySizeLimitMiddleware()

        # Check Content-Length header (fast path)
        result = middleware.check_content_length(headers, path="/api/upload")
        if not result.allowed:
            return error_response(result.status_code, result.message)

        # Wrap body reader for chunked transfers
        limited_body = middleware.wrap_body_reader(body_stream, path="/api/upload")
        try:
            data = limited_body.read()
        except BodySizeLimitExceeded as e:
            return error_response(413, str(e))
    """

    def __init__(self, config: BodySizeLimitConfig | None = None):
        """Initialize middleware with optional configuration.

        Args:
            config: Optional configuration. Uses global config if not provided.
        """
        self.config = config or get_body_size_config()
        # Track violation counts for metrics
        self._violation_count = 0
        self._violation_lock = threading.Lock()

    def check_content_length(
        self,
        headers: dict[str, str],
        path: str = "",
        max_size_override: int | None = None,
    ) -> BodySizeCheckResult:
        """Check Content-Length header against size limit.

        This provides fast rejection before reading the body.

        Args:
            headers: Request headers dict (case-insensitive lookup)
            path: Request path for endpoint-specific limits
            max_size_override: Override the configured limit

        Returns:
            BodySizeCheckResult indicating if request is allowed
        """
        # Get max size for this request
        max_size = max_size_override or self.config.get_limit_for_endpoint(path)

        # Look up Content-Length header (case-insensitive)
        content_length_str = headers.get("Content-Length") or headers.get("content-length")

        if not content_length_str:
            # No Content-Length - could be chunked encoding
            # Will need to track during body read
            return BodySizeCheckResult.ok()

        # Parse Content-Length
        try:
            content_length = int(content_length_str)
        except ValueError:
            return BodySizeCheckResult.invalid_content_length(content_length_str)

        # Check for negative values
        if content_length < 0:
            return BodySizeCheckResult.negative_content_length()

        # Check against limit
        if content_length > max_size:
            self._record_violation(path, content_length, max_size)
            return BodySizeCheckResult.too_large(
                content_length=content_length,
                max_allowed=max_size,
                include_size=self.config.include_size_in_error,
            )

        return BodySizeCheckResult.ok()

    def wrap_body_reader(
        self,
        body: BinaryIO,
        path: str = "",
        max_size_override: int | None = None,
    ) -> "LimitedBodyReader":
        """Wrap a body reader to enforce size limits during chunked reads.

        Args:
            body: Body stream to wrap
            path: Request path for endpoint-specific limits
            max_size_override: Override the configured limit

        Returns:
            LimitedBodyReader that raises on size limit exceeded
        """
        max_size = max_size_override or self.config.get_limit_for_endpoint(path)
        return LimitedBodyReader(
            body,
            max_size=max_size,
            on_exceeded=lambda bytes_read: self._record_violation(path, bytes_read, max_size),
        )

    def _record_violation(self, path: str, content_length: int, max_size: int) -> None:
        """Record a size limit violation.

        Args:
            path: Request path
            content_length: Attempted content length
            max_size: Maximum allowed size
        """
        with self._violation_lock:
            self._violation_count += 1

        if self.config.log_violations:
            logger.warning(
                "Request body size limit exceeded",
                extra={
                    "path": path,
                    "content_length": content_length,
                    "max_size": max_size,
                    "content_length_mb": content_length / (1024 * 1024),
                    "max_size_mb": max_size / (1024 * 1024),
                    "violation_count": self._violation_count,
                },
            )

    def get_violation_count(self) -> int:
        """Get the total number of size limit violations recorded."""
        with self._violation_lock:
            return self._violation_count

    def reset_violation_count(self) -> None:
        """Reset the violation counter (for testing)."""
        with self._violation_lock:
            self._violation_count = 0


# =============================================================================
# Limited Body Reader
# =============================================================================


class BodySizeLimitExceeded(Exception):
    """Exception raised when body size limit is exceeded during chunked read."""

    def __init__(self, bytes_read: int, max_size: int):
        self.bytes_read = bytes_read
        self.max_size = max_size
        max_mb = max_size / (1024 * 1024)
        super().__init__(
            f"Request body size limit exceeded: read {bytes_read} bytes, "
            f"maximum allowed is {max_mb:.2f}MB"
        )


class LimitedBodyReader:
    """Wrapper around a body stream that enforces size limits.

    Used for chunked transfer encoding where Content-Length is not known upfront.
    """

    def __init__(
        self,
        body: BinaryIO,
        max_size: int,
        on_exceeded: Callable[[int], None] | None = None,
    ):
        """Initialize limited reader.

        Args:
            body: Underlying body stream
            max_size: Maximum bytes to allow
            on_exceeded: Optional callback when limit is exceeded
        """
        self._body = body
        self._max_size = max_size
        self._bytes_read = 0
        self._on_exceeded = on_exceeded

    def read(self, size: int = -1) -> bytes:
        """Read from the body stream, enforcing size limit.

        Args:
            size: Number of bytes to read (-1 for all)

        Returns:
            Bytes read from stream

        Raises:
            BodySizeLimitExceeded: If size limit would be exceeded
        """
        if size == -1:
            # Read all - need to read in chunks to enforce limit
            chunks = []
            while True:
                chunk = self._read_chunk(8192)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)

        return self._read_chunk(size)

    def _read_chunk(self, size: int) -> bytes:
        """Read a chunk, checking size limit.

        Args:
            size: Maximum bytes to read

        Returns:
            Bytes read

        Raises:
            BodySizeLimitExceeded: If limit exceeded
        """
        # Check if we would exceed limit
        remaining = self._max_size - self._bytes_read
        if remaining <= 0:
            if self._on_exceeded:
                self._on_exceeded(self._bytes_read)
            raise BodySizeLimitExceeded(self._bytes_read, self._max_size)

        # Read up to remaining allowed bytes
        chunk = self._body.read(min(size, remaining + 1))

        if chunk:
            self._bytes_read += len(chunk)

            # Check if we exceeded after read
            if self._bytes_read > self._max_size:
                if self._on_exceeded:
                    self._on_exceeded(self._bytes_read)
                raise BodySizeLimitExceeded(self._bytes_read, self._max_size)

        return chunk

    def readline(self, limit: int = -1) -> bytes:
        """Read a line from the body stream, enforcing size limit.

        Args:
            limit: Maximum bytes to read for line (-1 for unlimited line length)

        Returns:
            Line bytes read from stream

        Raises:
            BodySizeLimitExceeded: If size limit would be exceeded
        """
        # Check current state
        if self._bytes_read >= self._max_size:
            if self._on_exceeded:
                self._on_exceeded(self._bytes_read)
            raise BodySizeLimitExceeded(self._bytes_read, self._max_size)

        remaining = self._max_size - self._bytes_read
        effective_limit = remaining + 1 if limit < 0 else min(limit, remaining + 1)

        line = self._body.readline(effective_limit)

        if line:
            self._bytes_read += len(line)
            if self._bytes_read > self._max_size:
                if self._on_exceeded:
                    self._on_exceeded(self._bytes_read)
                raise BodySizeLimitExceeded(self._bytes_read, self._max_size)

        return line

    @property
    def bytes_read(self) -> int:
        """Get total bytes read so far."""
        return self._bytes_read


# =============================================================================
# Decorator
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def with_body_size_limit(
    max_bytes: int | None = None,
    check_content_length: bool = True,
) -> Callable[[F], F]:
    """Decorator to enforce body size limits on a handler.

    Args:
        max_bytes: Maximum body size in bytes (uses config default if None)
        check_content_length: Whether to check Content-Length header

    Returns:
        Decorator function

    Usage:
        @with_body_size_limit(max_bytes=10 * 1024 * 1024)  # 10MB
        def upload_handler(self, handler):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, handler: Any, *args: Any, **kwargs: Any) -> Any:
            # Get headers from handler
            headers = {key: val for key, val in handler.headers.items()}
            path = getattr(handler, "path", "")

            # Check Content-Length if requested
            if check_content_length:
                middleware = BodySizeLimitMiddleware()
                result = middleware.check_content_length(
                    headers, path=path, max_size_override=max_bytes
                )

                if not result.allowed:
                    # Send error response
                    handler.send_response(result.status_code)
                    handler.send_header("Content-Type", "application/json")
                    handler.end_headers()
                    import json

                    error_body = json.dumps(
                        {"error": result.message, "status_code": result.status_code}
                    )
                    handler.wfile.write(error_body.encode("utf-8"))
                    return None

            return func(self, handler, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================


def check_body_size(
    headers: dict[str, str],
    path: str = "",
    max_size: int | None = None,
) -> BodySizeCheckResult:
    """Check if request body size is within limits.

    Convenience function using the global middleware instance.

    Args:
        headers: Request headers dict
        path: Request path for endpoint-specific limits
        max_size: Override the configured limit

    Returns:
        BodySizeCheckResult indicating if request is allowed
    """
    middleware = BodySizeLimitMiddleware()
    return middleware.check_content_length(headers, path=path, max_size_override=max_size)


def get_body_size_stats() -> dict[str, Any]:
    """Get statistics about body size limit enforcement.

    Returns:
        Dict with violation count and configuration info
    """
    config = get_body_size_config()
    middleware = BodySizeLimitMiddleware(config)

    return {
        "max_request_size": config.max_request_size,
        "max_request_size_mb": config.max_request_size / (1024 * 1024),
        "endpoint_limits_count": len(config.endpoint_limits),
        "log_violations": config.log_violations,
        "violation_count": middleware.get_violation_count(),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "BodySizeLimitConfig",
    "get_body_size_config",
    "configure_body_size_limit",
    "reset_body_size_config",
    # Constants
    "DEFAULT_MAX_REQUEST_SIZE",
    "HTTP_PAYLOAD_TOO_LARGE",
    "DEFAULT_LARGE_ENDPOINTS",
    "DEFAULT_SMALL_ENDPOINTS",
    # Check result
    "BodySizeCheckResult",
    # Middleware
    "BodySizeLimitMiddleware",
    # Body reader
    "BodySizeLimitExceeded",
    "LimitedBodyReader",
    # Decorator
    "with_body_size_limit",
    # Convenience functions
    "check_body_size",
    "get_body_size_stats",
]
