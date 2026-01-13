"""
Standardized error handling for Aragora server.

This module provides:
1. Base exception hierarchy for API errors
2. Error response formatting
3. Error logging utilities
4. HTTP status code mapping

Usage:
    from aragora.server.errors import (
        AragoraAPIError,
        NotFoundError,
        ValidationError,
        AuthenticationError,
        RateLimitError,
        format_error_response,
    )

    # Raise specific errors
    raise NotFoundError("Debate not found", debate_id="abc123")

    # Format error for API response
    response = format_error_response(error, include_trace=False)
"""

from __future__ import annotations

import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


# =============================================================================
# Error Codes
# =============================================================================


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # Client errors (4xx)
    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNPROCESSABLE_ENTITY = "UNPROCESSABLE_ENTITY"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"

    # Domain-specific errors
    DEBATE_ERROR = "DEBATE_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"


# =============================================================================
# Base Exception Classes
# =============================================================================


@dataclass
class ErrorContext:
    """Context information for error tracking."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    path: Optional[str] = None
    method: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AragoraAPIError(Exception):
    """Base exception for all Aragora API errors.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        status_code: HTTP status code
        details: Additional error details
        suggestion: Suggested fix for the user
        context: Error tracking context
    """

    default_message = "An error occurred"
    default_code = ErrorCode.INTERNAL_ERROR
    default_status_code = 500

    def __init__(
        self,
        message: Optional[str] = None,
        code: Optional[ErrorCode] = None,
        status_code: Optional[int] = None,
        details: Optional[str] = None,
        suggestion: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        **extra: Any,
    ):
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.status_code = status_code or self.default_status_code
        self.details = details
        self.suggestion = suggestion
        self.context = context or ErrorContext(extra=extra)

        # Merge extra into context
        if extra:
            self.context.extra.update(extra)

        super().__init__(self.message)

    def to_dict(self, include_trace: bool = False) -> Dict[str, Any]:
        """Convert error to dictionary for API response."""
        result: Dict[str, Any] = {
            "error": self.code.value if isinstance(self.code, ErrorCode) else self.code,
            "message": self.message,
            "error_id": self.context.error_id,
        }

        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.context.extra:
            result["context"] = self.context.extra

        if include_trace:
            result["trace"] = traceback.format_exc()

        return result

    def log(self, level: int = logging.ERROR) -> None:
        """Log the error with context."""
        logger.log(
            level,
            f"[{self.context.error_id}] {self.code.value}: {self.message}",
            extra={
                "error_id": self.context.error_id,
                "error_code": self.code.value,
                "status_code": self.status_code,
                "context": self.context.extra,
            },
        )


# =============================================================================
# Client Errors (4xx)
# =============================================================================


class BadRequestError(AragoraAPIError):
    """Invalid request format or parameters."""

    default_message = "Invalid request"
    default_code = ErrorCode.BAD_REQUEST
    default_status_code = 400


class ValidationError(AragoraAPIError):
    """Request validation failed."""

    default_message = "Validation failed"
    default_code = ErrorCode.VALIDATION_ERROR
    default_status_code = 400

    def __init__(
        self,
        message: Optional[str] = None,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if field:
            self.context.extra["field"] = field
        if value is not None:
            self.context.extra["value"] = str(value)[:100]


class AuthenticationError(AragoraAPIError):
    """Authentication required or failed."""

    default_message = "Authentication required"
    default_code = ErrorCode.UNAUTHORIZED
    default_status_code = 401


class ForbiddenError(AragoraAPIError):
    """Access denied to resource."""

    default_message = "Access denied"
    default_code = ErrorCode.FORBIDDEN
    default_status_code = 403


class NotFoundError(AragoraAPIError):
    """Requested resource not found."""

    default_message = "Resource not found"
    default_code = ErrorCode.NOT_FOUND
    default_status_code = 404

    def __init__(
        self,
        message: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if resource_type:
            self.context.extra["resource_type"] = resource_type
        if resource_id:
            self.context.extra["resource_id"] = resource_id


class MethodNotAllowedError(AragoraAPIError):
    """HTTP method not allowed for this endpoint."""

    default_message = "Method not allowed"
    default_code = ErrorCode.METHOD_NOT_ALLOWED
    default_status_code = 405


class ConflictError(AragoraAPIError):
    """Resource conflict (e.g., duplicate)."""

    default_message = "Resource conflict"
    default_code = ErrorCode.CONFLICT
    default_status_code = 409


class RateLimitError(AragoraAPIError):
    """Rate limit exceeded."""

    default_message = "Rate limit exceeded"
    default_code = ErrorCode.RATE_LIMITED
    default_status_code = 429

    def __init__(
        self,
        message: Optional[str] = None,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if retry_after:
            self.context.extra["retry_after_seconds"] = retry_after
        if limit:
            self.context.extra["limit"] = limit


class PayloadTooLargeError(AragoraAPIError):
    """Request payload too large."""

    default_message = "Payload too large"
    default_code = ErrorCode.PAYLOAD_TOO_LARGE
    default_status_code = 413


# =============================================================================
# Server Errors (5xx)
# =============================================================================


class InternalError(AragoraAPIError):
    """Internal server error."""

    default_message = "Internal server error"
    default_code = ErrorCode.INTERNAL_ERROR
    default_status_code = 500


class ServiceUnavailableError(AragoraAPIError):
    """Service temporarily unavailable."""

    default_message = "Service temporarily unavailable"
    default_code = ErrorCode.SERVICE_UNAVAILABLE
    default_status_code = 503


class GatewayTimeoutError(AragoraAPIError):
    """Upstream service timeout."""

    default_message = "Gateway timeout"
    default_code = ErrorCode.GATEWAY_TIMEOUT
    default_status_code = 504


class DatabaseError(AragoraAPIError):
    """Database operation failed."""

    default_message = "Database error"
    default_code = ErrorCode.DATABASE_ERROR
    default_status_code = 500


class ExternalServiceError(AragoraAPIError):
    """External service (API, connector) failed."""

    default_message = "External service error"
    default_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    default_status_code = 502

    def __init__(
        self,
        message: Optional[str] = None,
        service: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if service:
            self.context.extra["service"] = service


# =============================================================================
# Domain-Specific Errors
# =============================================================================


class DebateError(AragoraAPIError):
    """Debate-related error."""

    default_message = "Debate error"
    default_code = ErrorCode.DEBATE_ERROR
    default_status_code = 500

    def __init__(
        self,
        message: Optional[str] = None,
        debate_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if debate_id:
            self.context.extra["debate_id"] = debate_id


class VerificationError(AragoraAPIError):
    """Formal verification error."""

    default_message = "Verification error"
    default_code = ErrorCode.VERIFICATION_ERROR
    default_status_code = 500


class MemoryError(AragoraAPIError):
    """Memory system error."""

    default_message = "Memory system error"
    default_code = ErrorCode.MEMORY_ERROR
    default_status_code = 500


# =============================================================================
# Error Formatting
# =============================================================================


def format_error_response(
    error: Exception,
    include_trace: bool = False,
    path: Optional[str] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """Format any exception as a standardized API error response.

    Args:
        error: The exception to format
        include_trace: Whether to include stack trace (dev mode only)
        path: Request path for context
        method: HTTP method for context

    Returns:
        Dictionary suitable for JSON response
    """
    if isinstance(error, AragoraAPIError):
        if path:
            error.context.path = path
        if method:
            error.context.method = method
        return error.to_dict(include_trace=include_trace)

    # Convert unknown exceptions to InternalError
    internal = InternalError(
        message=str(error) if str(error) else "An unexpected error occurred",
        details=type(error).__name__,
    )
    if path:
        internal.context.path = path
    if method:
        internal.context.method = method

    return internal.to_dict(include_trace=include_trace)


def get_status_code(error: Exception) -> int:
    """Get HTTP status code for an exception."""
    if isinstance(error, AragoraAPIError):
        return error.status_code
    return 500


# =============================================================================
# Error Mapping
# =============================================================================

# Map common exception types to API errors
EXCEPTION_MAP: Dict[Type[Exception], Type[AragoraAPIError]] = {
    ValueError: ValidationError,
    KeyError: NotFoundError,
    PermissionError: ForbiddenError,
    TimeoutError: GatewayTimeoutError,
    ConnectionError: ExternalServiceError,
}


def wrap_exception(
    error: Exception,
    default_class: Type[AragoraAPIError] = InternalError,
) -> AragoraAPIError:
    """Wrap a standard exception in an AragoraAPIError.

    Args:
        error: The exception to wrap
        default_class: Default error class if no mapping found

    Returns:
        An AragoraAPIError instance
    """
    if isinstance(error, AragoraAPIError):
        return error

    error_class = EXCEPTION_MAP.get(type(error), default_class)
    return error_class(message=str(error), details=type(error).__name__)


# =============================================================================
# Logging Helpers
# =============================================================================


def log_error(
    error: Exception,
    level: int = logging.ERROR,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Log an error with context and return the error ID.

    Args:
        error: The exception to log
        level: Logging level
        context: Additional context to include

    Returns:
        Error ID for tracking
    """
    if isinstance(error, AragoraAPIError):
        if context:
            error.context.extra.update(context)
        error.log(level)
        return error.context.error_id

    # Create error ID for non-API errors
    error_id = str(uuid.uuid4())[:8]
    logger.log(
        level,
        f"[{error_id}] {type(error).__name__}: {error}",
        extra={"error_id": error_id, "context": context or {}},
        exc_info=True,
    )
    return error_id


__all__ = [
    # Error codes
    "ErrorCode",
    # Context
    "ErrorContext",
    # Base class
    "AragoraAPIError",
    # Client errors
    "BadRequestError",
    "ValidationError",
    "AuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "MethodNotAllowedError",
    "ConflictError",
    "RateLimitError",
    "PayloadTooLargeError",
    # Server errors
    "InternalError",
    "ServiceUnavailableError",
    "GatewayTimeoutError",
    "DatabaseError",
    "ExternalServiceError",
    # Domain errors
    "DebateError",
    "VerificationError",
    "MemoryError",
    # Utilities
    "format_error_response",
    "get_status_code",
    "wrap_exception",
    "log_error",
    "EXCEPTION_MAP",
]
