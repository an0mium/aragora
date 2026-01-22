"""
Standardized Error Handler Middleware.

Ensures all API errors follow a consistent response format:
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message",
        "details": {...},  // Optional additional context
        "request_id": "req_xxx",
        "timestamp": "2026-01-21T...",
        "path": "/api/endpoint"
    }
}

This middleware catches exceptions and converts them to standardized responses,
improving client error handling and debugging.

Usage:
    from aragora.server.middleware.error_handler import (
        ErrorHandlerMiddleware,
        APIError,
        raise_api_error,
    )

    # Raise API errors with standard codes
    raise APIError(
        code=ErrorCode.VALIDATION_ERROR,
        message="Invalid email format",
        details={"field": "email", "value": "not-an-email"}
    )

    # Or use the helper function
    raise_api_error(
        ErrorCode.NOT_FOUND,
        "Debate not found",
        details={"debate_id": debate_id}
    )
"""

from __future__ import annotations

import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Type

from ..error_codes import ErrorCode, get_status_for_code

logger = logging.getLogger(__name__)


@dataclass
class APIError(Exception):
    """
    Standardized API error exception.

    Raise this exception in handlers for automatic conversion to
    a standardized error response.

    Attributes:
        code: Error code from ErrorCode class
        message: Human-readable error message
        status: HTTP status code (auto-derived from code if not specified)
        details: Optional dict with additional error context
        headers: Optional headers to include in response
    """

    code: str
    message: str
    status: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.status is None:
            self.status = get_status_for_code(self.code)
        super().__init__(self.message)

    def to_dict(
        self, request_id: Optional[str] = None, path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert to standardized error response dict."""
        error_dict: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

        if self.details:
            error_dict["details"] = self.details

        if request_id:
            error_dict["request_id"] = request_id

        if path:
            error_dict["path"] = path

        return {"error": error_dict}


def raise_api_error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status: Optional[int] = None,
) -> None:
    """
    Raise a standardized API error.

    Convenience function for raising APIError exceptions.

    Args:
        code: Error code from ErrorCode class
        message: Human-readable error message
        details: Optional dict with additional error context
        status: HTTP status code (auto-derived from code if not specified)

    Raises:
        APIError: Always raises this exception
    """
    raise APIError(code=code, message=message, details=details, status=status)


# Exception to error code mapping for common Python exceptions
EXCEPTION_ERROR_MAP: Dict[Type[Exception], str] = {
    ValueError: ErrorCode.VALIDATION_ERROR,
    KeyError: ErrorCode.MISSING_FIELD,
    TypeError: ErrorCode.INVALID_FIELD,
    PermissionError: ErrorCode.PERMISSION_DENIED,
    FileNotFoundError: ErrorCode.NOT_FOUND,
    TimeoutError: ErrorCode.EXTERNAL_SERVICE_ERROR,
    ConnectionError: ErrorCode.EXTERNAL_SERVICE_ERROR,
}


@dataclass
class ErrorResponse:
    """Represents a standardized error response."""

    status: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)


class ErrorHandlerMiddleware:
    """
    Middleware for standardizing error responses.

    Catches exceptions and converts them to standardized JSON error responses.
    Also adds request IDs for tracking and debugging.
    """

    def __init__(
        self,
        app: Callable,
        include_traceback: bool = False,
        log_errors: bool = True,
        exclude_paths: Optional[list[str]] = None,
    ):
        """
        Initialize error handler middleware.

        Args:
            app: The WSGI/ASGI application to wrap
            include_traceback: Include stack trace in 5xx errors (dev only)
            log_errors: Log errors to logger
            exclude_paths: Paths to exclude from error handling (e.g., /healthz)
        """
        self.app = app
        self.include_traceback = include_traceback
        self.log_errors = log_errors
        self.exclude_paths = exclude_paths or ["/healthz", "/readyz"]

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"req_{uuid.uuid4().hex[:16]}"

    def handle_exception(
        self,
        exc: Exception,
        request_id: str,
        path: str,
    ) -> ErrorResponse:
        """
        Convert an exception to a standardized error response.

        Args:
            exc: The exception that was raised
            request_id: Request ID for tracking
            path: Request path

        Returns:
            ErrorResponse with status, body, and headers
        """
        # Handle APIError (already standardized)
        if isinstance(exc, APIError):
            body = exc.to_dict(request_id=request_id, path=path)
            headers = exc.headers or {}

            if self.log_errors:
                logger.warning(
                    f"API error: {exc.code} - {exc.message}",
                    extra={
                        "request_id": request_id,
                        "error_code": exc.code,
                        "path": path,
                    },
                )

            return ErrorResponse(
                status=exc.status or 400,
                body=body,
                headers=headers,
            )

        # Map known exceptions to error codes
        error_code = EXCEPTION_ERROR_MAP.get(type(exc), ErrorCode.INTERNAL_ERROR)
        status = get_status_for_code(error_code)

        # Build error message
        error_message = str(exc) if str(exc) else f"{type(exc).__name__} occurred"

        # For 5xx errors, sanitize message for production
        if status >= 500:
            if self.log_errors:
                logger.exception(
                    f"Internal error: {exc}",
                    extra={
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "path": path,
                    },
                )

            # Don't expose internal details in production
            if not self.include_traceback:
                error_message = "An internal error occurred. Please try again later."

        error_dict: Dict[str, Any] = {
            "code": error_code,
            "message": error_message,
            "request_id": request_id,
            "path": path,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

        # Include traceback in development mode for 5xx errors
        if self.include_traceback and status >= 500:
            error_dict["traceback"] = traceback.format_exc()

        return ErrorResponse(
            status=status,
            body={"error": error_dict},
            headers={},
        )


def create_error_response(
    code: str,
    message: str,
    status: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a standardized error response dict.

    Utility function for handlers that want to return error responses
    without raising exceptions.

    Args:
        code: Error code from ErrorCode class
        message: Human-readable error message
        status: HTTP status code (auto-derived from code if not specified)
        details: Optional dict with additional error context
        request_id: Request ID for tracking
        path: Request path

    Returns:
        Dict containing standardized error response body
    """
    error_dict: Dict[str, Any] = {
        "code": code,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }

    if details:
        error_dict["details"] = details

    if request_id:
        error_dict["request_id"] = request_id

    if path:
        error_dict["path"] = path

    return {"error": error_dict}


# Pre-built common error responses
def validation_error(
    message: str,
    field: Optional[str] = None,
    value: Any = None,
) -> APIError:
    """Create a validation error."""
    details = {}
    if field:
        details["field"] = field
    if value is not None:
        details["value"] = str(value)[:100]  # Truncate for safety

    return APIError(
        code=ErrorCode.VALIDATION_ERROR,
        message=message,
        details=details if details else None,
    )


def not_found_error(
    resource: str,
    resource_id: Optional[str] = None,
) -> APIError:
    """Create a not found error."""
    message = f"{resource} not found"
    if resource_id:
        message = f"{resource} '{resource_id}' not found"

    return APIError(
        code=ErrorCode.NOT_FOUND,
        message=message,
        details={"resource": resource, "id": resource_id} if resource_id else None,
    )


def permission_error(
    action: str,
    resource: Optional[str] = None,
) -> APIError:
    """Create a permission denied error."""
    message = f"Permission denied for action: {action}"
    if resource:
        message = f"Permission denied: cannot {action} {resource}"

    return APIError(
        code=ErrorCode.PERMISSION_DENIED,
        message=message,
        details={"action": action, "resource": resource} if resource else None,
    )


def rate_limit_error(
    retry_after: Optional[int] = None,
) -> APIError:
    """Create a rate limit error."""
    headers = {}
    if retry_after:
        headers["Retry-After"] = str(retry_after)

    return APIError(
        code=ErrorCode.RATE_LIMITED,
        message="Too many requests. Please slow down.",
        details={"retry_after_seconds": retry_after} if retry_after else None,
        headers=headers,
    )


__all__ = [
    "APIError",
    "ErrorHandlerMiddleware",
    "ErrorResponse",
    "create_error_response",
    "raise_api_error",
    "validation_error",
    "not_found_error",
    "permission_error",
    "rate_limit_error",
]
