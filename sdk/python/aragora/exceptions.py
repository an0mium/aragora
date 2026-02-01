"""
Aragora SDK Exceptions

Custom exception classes for handling API errors.
"""

from __future__ import annotations

from typing import Any


class AragoraError(Exception):
    """Base exception for all Aragora SDK errors.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code, if applicable.
        error_code: Machine-readable error code from the API (e.g. ``"RATE_LIMITED"``).
        trace_id: Unique request trace ID for debugging and support.
        response_body: Raw parsed response body, if available.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        trace_id: str | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.trace_id = trace_id
        self.response_body = response_body

    def __str__(self) -> str:
        parts = ["AragoraError"]
        if self.status_code:
            parts[0] += f" ({self.status_code})"
        if self.error_code:
            parts[0] += f" [{self.error_code}]"
        parts.append(self.message)
        if self.trace_id:
            parts.append(f"(trace: {self.trace_id})")
        return ": ".join(parts[:2]) + (f" {parts[2]}" if len(parts) > 2 else "")


class AuthenticationError(AragoraError):
    """Raised when authentication fails (401 errors)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message, status_code=401, error_code=error_code, trace_id=trace_id, **kwargs
        )


class AuthorizationError(AragoraError):
    """Raised when authorization fails (403 errors)."""

    def __init__(
        self,
        message: str = "Access denied",
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message, status_code=403, error_code=error_code, trace_id=trace_id, **kwargs
        )


class NotFoundError(AragoraError):
    """Raised when a resource is not found (404 errors)."""

    def __init__(
        self,
        message: str = "Resource not found",
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message, status_code=404, error_code=error_code, trace_id=trace_id, **kwargs
        )


class RateLimitError(AragoraError):
    """Raised when rate limits are exceeded (429 errors)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message, status_code=429, error_code=error_code, trace_id=trace_id, **kwargs
        )
        self.retry_after = retry_after

    def __str__(self) -> str:
        base = super().__str__().replace("AragoraError", "RateLimitError")
        if self.retry_after:
            return f"{base} (retry after {self.retry_after}s)"
        return base


class ValidationError(AragoraError):
    """Raised when request validation fails (400 errors)."""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: list[dict[str, Any]] | None = None,
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message, status_code=400, error_code=error_code, trace_id=trace_id, **kwargs
        )
        self.errors = errors or []


class ServerError(AragoraError):
    """Raised for server errors (5xx errors)."""

    def __init__(
        self,
        message: str = "Server error",
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code=error_code, trace_id=trace_id, **kwargs)


class TimeoutError(AragoraError):
    """Raised when a request times out."""

    def __init__(
        self,
        message: str = "Request timed out",
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code=error_code, trace_id=trace_id, **kwargs)


class ConnectionError(AragoraError):
    """Raised when a connection cannot be established."""

    def __init__(
        self,
        message: str = "Connection failed",
        error_code: str | None = None,
        trace_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, error_code=error_code, trace_id=trace_id, **kwargs)
