"""
Client SDK error classes.

Provides a hierarchy of typed exceptions for API errors.
"""

from __future__ import annotations

from aragora.exceptions import AragoraError


class AragoraAPIError(AragoraError):
    """Base exception for API errors in the SDK client.

    Inherits from AragoraError to unify the error hierarchy, allowing
    all Aragora exceptions to be caught with `except AragoraError`.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code (e.g., "NOT_FOUND", "RATE_LIMITED")
        status_code: HTTP status code
        suggestion: Optional suggestion for resolution
    """

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN",
        status_code: int = 500,
        suggestion: str | None = None,
    ):
        self.code = code
        self.status_code = status_code
        self.suggestion = suggestion
        self._base_message = message
        full_message = message
        if suggestion:
            full_message = f"{message}. Suggestion: {suggestion}"
        self._full_message = full_message
        # Initialize AragoraError with message
        super().__init__(full_message, {"code": code, "status_code": status_code})

    def __str__(self) -> str:
        """Return simple message format for SDK backward compatibility."""
        return self._full_message


class RateLimitError(AragoraAPIError):
    """Raised when rate limit is exceeded (HTTP 429).

    The server may return Retry-After header indicating when to retry.
    """

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        super().__init__(
            message,
            code="RATE_LIMITED",
            status_code=429,
            suggestion="Wait before retrying. Consider using RetryConfig with exponential backoff.",
        )
        self.retry_after = retry_after


class AuthenticationError(AragoraAPIError):
    """Raised when authentication fails (HTTP 401).

    Usually indicates missing or invalid API token.
    """

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message,
            code="UNAUTHORIZED",
            status_code=401,
            suggestion="Check that ARAGORA_API_TOKEN is set correctly.",
        )


class NotFoundError(AragoraAPIError):
    """Raised when a resource is not found (HTTP 404)."""

    def __init__(self, message: str = "Resource not found", resource_type: str = "resource"):
        super().__init__(
            message,
            code="NOT_FOUND",
            status_code=404,
            suggestion=f"Verify the {resource_type} ID is correct.",
        )
        self.resource_type = resource_type


class QuotaExceededError(AragoraAPIError):
    """Raised when usage quota is exceeded (HTTP 402)."""

    def __init__(self, message: str = "Quota exceeded"):
        super().__init__(
            message,
            code="QUOTA_EXCEEDED",
            status_code=402,
            suggestion="Upgrade your plan or wait for quota reset.",
        )


class ValidationError(AragoraAPIError):
    """Raised when request validation fails (HTTP 400)."""

    def __init__(self, message: str = "Validation error", field: str | None = None):
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            status_code=400,
            suggestion=f"Check the '{field}' parameter." if field else "Check request parameters.",
        )
        self.field = field


__all__ = [
    "AragoraAPIError",
    "RateLimitError",
    "AuthenticationError",
    "NotFoundError",
    "QuotaExceededError",
    "ValidationError",
]
