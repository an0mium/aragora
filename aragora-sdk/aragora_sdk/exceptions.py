"""
Aragora SDK Exceptions.

Custom exception classes for the Aragora SDK.
"""

from __future__ import annotations


class AragoraError(Exception):
    """Base exception for all Aragora SDK errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(AragoraError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationError(AragoraError):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403)


class NotFoundError(AragoraError):
    """Raised when a resource is not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class RateLimitError(AragoraError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ValidationError(AragoraError):
    """Raised when request validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, status_code=400)
        self.field = field


class ServerError(AragoraError):
    """Raised when the server returns an error."""

    def __init__(self, message: str = "Server error"):
        super().__init__(message, status_code=500)


class TimeoutError(AragoraError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message)


class ConnectionError(AragoraError):
    """Raised when connection to the server fails."""

    def __init__(self, message: str = "Connection failed"):
        super().__init__(message)
