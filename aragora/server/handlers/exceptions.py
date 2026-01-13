"""
Handler-specific exception utilities.

Provides structured exception handling for HTTP handlers with:
- HTTP status code mapping
- Consistent error responses
- Logging integration
- Exception classification for metrics

Usage:
    from aragora.server.handlers.exceptions import (
        handle_handler_error,
        HandlerValidationError,
        HandlerNotFoundError,
        HandlerAuthorizationError,
    )

    try:
        # handler code
    except HandlerValidationError as e:
        return error_response(str(e), e.status_code)
    except Exception as e:
        return handle_handler_error(e, "operation_name", logger)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any, Optional, Tuple

# Import base exceptions from main module
from aragora.exceptions import (
    AragoraError,
    ValidationError,
    InputValidationError,
    DatabaseError,
    RecordNotFoundError,
    AuthError,
    AuthenticationError,
    AuthorizationError,
    RateLimitExceededError,
)

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult


# =============================================================================
# Handler-Specific Exceptions
# =============================================================================


class HandlerError(AragoraError):
    """Base exception for handler errors with HTTP status code."""

    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None, details: dict | None = None):
        super().__init__(message, details)
        if status_code is not None:
            self.status_code = status_code


class HandlerValidationError(HandlerError):
    """Request validation failed (400 Bad Request)."""

    status_code = 400

    def __init__(self, message: str, field: str | None = None, details: dict | None = None):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, details=details)
        self.field = field


class HandlerNotFoundError(HandlerError):
    """Resource not found (404 Not Found)."""

    status_code = 404

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class HandlerAuthorizationError(HandlerError):
    """User lacks permission (403 Forbidden)."""

    status_code = 403

    def __init__(self, action: str, resource: str | None = None):
        msg = f"Not authorized to {action}"
        if resource:
            msg += f" on {resource}"
        super().__init__(msg, details={"action": action, "resource": resource})
        self.action = action
        self.resource = resource


class HandlerConflictError(HandlerError):
    """Resource conflict (409 Conflict)."""

    status_code = 409

    def __init__(self, message: str, resource_type: str | None = None):
        super().__init__(message, details={"resource_type": resource_type})


class HandlerRateLimitError(HandlerError):
    """Rate limit exceeded (429 Too Many Requests)."""

    status_code = 429

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, details=details)
        self.retry_after = retry_after


class HandlerExternalServiceError(HandlerError):
    """External service failed (502 Bad Gateway or 503 Service Unavailable)."""

    status_code = 502

    def __init__(self, service: str, message: str, unavailable: bool = False):
        if unavailable:
            self.status_code = 503
        super().__init__(
            f"{service} service error: {message}",
            details={"service": service, "unavailable": unavailable},
        )
        self.service = service


class HandlerDatabaseError(HandlerError):
    """Database operation failed (500 Internal Server Error)."""

    status_code = 500

    def __init__(self, operation: str, message: str | None = None):
        msg = f"Database error during {operation}"
        if message:
            msg += f": {message}"
        super().__init__(msg, details={"operation": operation})
        self.operation = operation


# =============================================================================
# Exception Classification
# =============================================================================


# Map exception types to (status_code, log_level, should_include_message)
EXCEPTION_MAP: dict[type, Tuple[int, str, bool]] = {
    # Validation errors - client's fault, include message
    HandlerValidationError: (400, "info", True),
    InputValidationError: (400, "info", True),
    ValidationError: (400, "info", True),
    ValueError: (400, "info", True),
    KeyError: (400, "info", False),  # Don't expose internal key names
    TypeError: (400, "warning", False),
    # Not found - client's fault, include message
    HandlerNotFoundError: (404, "info", True),
    RecordNotFoundError: (404, "info", True),
    # Authorization - client's fault, include message
    HandlerAuthorizationError: (403, "info", True),
    AuthorizationError: (403, "info", True),
    AuthenticationError: (401, "info", True),
    AuthError: (401, "info", True),
    # Rate limiting - client's fault, include message
    HandlerRateLimitError: (429, "info", True),
    RateLimitExceededError: (429, "info", True),
    # Conflict - depends on situation
    HandlerConflictError: (409, "info", True),
    # External service - not our fault, generic message
    HandlerExternalServiceError: (502, "error", True),
    # Database - our fault, generic message
    HandlerDatabaseError: (500, "error", False),
    DatabaseError: (500, "error", False),
    sqlite3.Error: (500, "error", False),
    sqlite3.IntegrityError: (409, "warning", False),  # Often a conflict
    # Timeouts
    TimeoutError: (504, "warning", False),
}

# Generic error messages for exceptions that shouldn't expose details
GENERIC_ERROR_MESSAGES: dict[int, str] = {
    400: "Invalid request",
    401: "Authentication required",
    403: "Access denied",
    404: "Resource not found",
    409: "Resource conflict",
    429: "Too many requests",
    500: "Internal server error",
    502: "Service temporarily unavailable",
    503: "Service unavailable",
    504: "Request timeout",
}


def classify_exception(exc: Exception) -> Tuple[int, str, str]:
    """
    Classify an exception for HTTP response.

    Returns:
        Tuple of (status_code, log_level, error_message)
    """
    exc_type = type(exc)

    # Check for exact match first
    if exc_type in EXCEPTION_MAP:
        status_code, log_level, include_msg = EXCEPTION_MAP[exc_type]
        if include_msg:
            return status_code, log_level, str(exc)
        return status_code, log_level, GENERIC_ERROR_MESSAGES.get(status_code, "Error")

    # Check for subclass match
    for mapped_type, (status_code, log_level, include_msg) in EXCEPTION_MAP.items():
        if isinstance(exc, mapped_type):
            if include_msg:
                return status_code, log_level, str(exc)
            return status_code, log_level, GENERIC_ERROR_MESSAGES.get(status_code, "Error")

    # Check for HandlerError subclass (has status_code attribute)
    if isinstance(exc, HandlerError):
        return exc.status_code, "error", str(exc)

    # Check for AragoraError (our base exception)
    if isinstance(exc, AragoraError):
        return 500, "error", str(exc)

    # Unknown exception - log as error, return generic message
    return 500, "error", "Internal server error"


def handle_handler_error(
    exc: Exception,
    operation: str,
    logger: logging.Logger,
    include_traceback: bool = False,
) -> Tuple[int, str]:
    """
    Handle an exception in a handler, logging appropriately.

    Args:
        exc: The exception that occurred
        operation: Name of the operation (for logging)
        logger: Logger instance to use
        include_traceback: Whether to log full traceback

    Returns:
        Tuple of (status_code, error_message) for response

    Example:
        try:
            do_something()
        except Exception as e:
            status, message = handle_handler_error(e, "create debate", logger)
            return error_response(message, status)
    """
    status_code, log_level, message = classify_exception(exc)

    # Build log message
    log_msg = f"{operation} failed: {type(exc).__name__}: {exc}"

    # Log at appropriate level
    if include_traceback or log_level == "error":
        getattr(logger, log_level)(log_msg, exc_info=True)
    else:
        getattr(logger, log_level)(log_msg)

    return status_code, message


def is_client_error(exc: Exception) -> bool:
    """Check if exception represents a client error (4xx)."""
    status_code, _, _ = classify_exception(exc)
    return 400 <= status_code < 500


def is_server_error(exc: Exception) -> bool:
    """Check if exception represents a server error (5xx)."""
    status_code, _, _ = classify_exception(exc)
    return status_code >= 500


def is_retryable_error(exc: Exception) -> bool:
    """Check if exception represents a retryable error."""
    status_code, _, _ = classify_exception(exc)
    # 429, 502, 503, 504 are typically retryable
    return status_code in (429, 502, 503, 504)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Handler exceptions
    "HandlerError",
    "HandlerValidationError",
    "HandlerNotFoundError",
    "HandlerAuthorizationError",
    "HandlerConflictError",
    "HandlerRateLimitError",
    "HandlerExternalServiceError",
    "HandlerDatabaseError",
    # Classification utilities
    "classify_exception",
    "handle_handler_error",
    "is_client_error",
    "is_server_error",
    "is_retryable_error",
    # Re-exports from main exceptions module
    "ValidationError",
    "InputValidationError",
    "DatabaseError",
    "RecordNotFoundError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitExceededError",
]
