"""
Aragora SDK Exceptions

Custom exception classes for handling API errors.
"""

from typing import Any


class AragoraError(Exception):
    """Base exception for all Aragora SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        if self.status_code:
            return f"AragoraError ({self.status_code}): {self.message}"
        return f"AragoraError: {self.message}"


class AuthenticationError(AragoraError):
    """Raised when authentication fails (401 errors)."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(AragoraError):
    """Raised when authorization fails (403 errors)."""

    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class NotFoundError(AragoraError):
    """Raised when a resource is not found (404 errors)."""

    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class RateLimitError(AragoraError):
    """Raised when rate limits are exceeded (429 errors)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after

    def __str__(self) -> str:
        if self.retry_after:
            return f"RateLimitError: {self.message} (retry after {self.retry_after}s)"
        return f"RateLimitError: {self.message}"


class ValidationError(AragoraError):
    """Raised when request validation fails (400 errors)."""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: list[dict] | None = None,
        **kwargs,
    ):
        super().__init__(message, status_code=400, **kwargs)
        self.errors = errors or []


class ServerError(AragoraError):
    """Raised for server errors (5xx errors)."""

    def __init__(self, message: str = "Server error", **kwargs):
        super().__init__(message, **kwargs)


class TimeoutError(AragoraError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, **kwargs)


class ConnectionError(AragoraError):
    """Raised when a connection cannot be established."""

    def __init__(self, message: str = "Connection failed", **kwargs):
        super().__init__(message, **kwargs)
