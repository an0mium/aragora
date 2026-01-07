"""Centralized error handling and structured error responses.

Provides consistent error message handling across the server:
- sanitize_error_text: Redacts sensitive data from error strings
- safe_error_message: Maps exceptions to user-friendly messages
- APIError: Structured error class with codes and metadata
- ErrorCode: Standard error code constants
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import shared sanitization utilities
from aragora.utils.error_sanitizer import (
    sanitize_error_text,
    SENSITIVE_PATTERNS as _SENSITIVE_PATTERNS,  # For backwards compatibility
)


# =============================================================================
# Error Codes
# =============================================================================

class ErrorCode(str, Enum):
    """Standard error codes for structured API responses."""

    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    INVALID_FORMAT = "INVALID_FORMAT"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"

    # Domain-specific errors
    DEBATE_ERROR = "DEBATE_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"


# Map HTTP status codes to error codes
_STATUS_TO_CODE = {
    400: ErrorCode.INVALID_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL_ERROR,
    502: ErrorCode.EXTERNAL_SERVICE_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
    504: ErrorCode.TIMEOUT,
}


# =============================================================================
# APIError Class
# =============================================================================

@dataclass
class APIError(Exception):
    """
    Structured API error with code, message, and metadata.

    Provides consistent error format across all API endpoints:
    {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid agent name",
            "status": 400,
            "trace_id": "abc123",
            "details": {"field": "agent"},
            "suggestion": "Use GET /api/agents for valid names"
        }
    }
    """

    code: ErrorCode
    message: str
    status: int = 400
    trace_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None

    def __post_init__(self):
        # Initialize exception base
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        result = {
            "code": self.code.value if isinstance(self.code, ErrorCode) else self.code,
            "message": self.message,
            "status": self.status,
        }
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result

    def to_response(self) -> dict:
        """Convert to full error response envelope."""
        return {"error": self.to_dict()}

    @classmethod
    def from_status(
        cls,
        status: int,
        message: str,
        trace_id: Optional[str] = None,
        details: Optional[dict] = None,
        suggestion: Optional[str] = None,
    ) -> "APIError":
        """Create APIError from HTTP status code."""
        code = _STATUS_TO_CODE.get(status, ErrorCode.INTERNAL_ERROR)
        return cls(
            code=code,
            message=message,
            status=status,
            trace_id=trace_id,
            details=details or {},
            suggestion=suggestion,
        )

    @classmethod
    def validation_error(
        cls,
        message: str,
        field: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> "APIError":
        """Create a validation error."""
        details = {"field": field} if field else {}
        return cls(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status=400,
            trace_id=trace_id,
            details=details,
        )

    @classmethod
    def not_found(
        cls,
        resource: str,
        resource_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> "APIError":
        """Create a not found error."""
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} '{resource_id}' not found"
        return cls(
            code=ErrorCode.NOT_FOUND,
            message=message,
            status=404,
            trace_id=trace_id,
            details={"resource": resource, "id": resource_id} if resource_id else {},
        )

    @classmethod
    def rate_limited(
        cls,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> "APIError":
        """Create a rate limit error."""
        details = {"retry_after": retry_after} if retry_after else {}
        suggestion = f"Retry after {retry_after} seconds" if retry_after else "Please try again later"
        return cls(
            code=ErrorCode.RATE_LIMITED,
            message=message,
            status=429,
            trace_id=trace_id,
            details=details,
            suggestion=suggestion,
        )

    @classmethod
    def internal_error(
        cls,
        message: str = "An internal error occurred",
        trace_id: Optional[str] = None,
    ) -> "APIError":
        """Create an internal server error."""
        return cls(
            code=ErrorCode.INTERNAL_ERROR,
            message=message,
            status=500,
            trace_id=trace_id,
            suggestion="If this persists, contact support with the trace ID",
        )


# =============================================================================
# Helper Functions
# =============================================================================

def safe_error_message(e: Exception, context: str = "") -> str:
    """Return a sanitized error message for client responses.

    Logs the full error server-side while returning a generic message to clients.
    This prevents information disclosure of internal details like file paths,
    stack traces, or sensitive configuration.

    Args:
        e: The exception that occurred
        context: Optional context string for logging (e.g., "debate creation")

    Returns:
        User-friendly error message safe to return to clients
    """
    # Log full details server-side for debugging
    logger.error(f"Error in {context}: {type(e).__name__}: {e}", exc_info=True)

    # Map common exceptions to user-friendly messages
    error_type = type(e).__name__
    if error_type in ("FileNotFoundError", "OSError"):
        return "Resource not found"
    elif error_type in ("json.JSONDecodeError", "ValueError"):
        return "Invalid data format"
    elif error_type in ("PermissionError",):
        return "Access denied"
    elif error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "Operation timed out"
    else:
        return "An error occurred"


# =============================================================================
# Error Formatter
# =============================================================================

class ErrorFormatter:
    """Unified error formatter for consistent API responses.

    Provides a single interface for formatting both client and server errors
    with automatic classification, sanitization, and trace ID generation.

    Example:
        formatter = ErrorFormatter()

        # Format client error (bad input)
        response = formatter.format_client_error(
            "Invalid agent name",
            field="agent",
            trace_id="abc123"
        )

        # Format server error (internal failure)
        response = formatter.format_server_error(
            exception,
            context="debate creation",
            trace_id="abc123"
        )

        # Auto-classify an exception
        api_error = formatter.classify_exception(exception, "debate creation")
    """

    # Map exception types to (status_code, error_code, default_message)
    EXCEPTION_MAP: dict[str, tuple[int, ErrorCode, str]] = {
        "FileNotFoundError": (404, ErrorCode.NOT_FOUND, "Resource not found"),
        "OSError": (500, ErrorCode.INTERNAL_ERROR, "System error"),
        "JSONDecodeError": (400, ErrorCode.INVALID_FORMAT, "Invalid JSON format"),
        "ValueError": (400, ErrorCode.VALIDATION_ERROR, "Invalid value"),
        "KeyError": (400, ErrorCode.MISSING_PARAMETER, "Missing required field"),
        "PermissionError": (403, ErrorCode.FORBIDDEN, "Access denied"),
        "TimeoutError": (504, ErrorCode.TIMEOUT, "Operation timed out"),
        "ConnectionError": (502, ErrorCode.EXTERNAL_SERVICE_ERROR, "Connection failed"),
        "sqlite3.OperationalError": (503, ErrorCode.DATABASE_ERROR, "Database error"),
    }

    @classmethod
    def format_client_error(
        cls,
        message: str,
        status: int = 400,
        code: Optional[ErrorCode] = None,
        field: Optional[str] = None,
        trace_id: Optional[str] = None,
        suggestion: Optional[str] = None,
    ) -> dict:
        """Format a client error (4xx) response.

        Use for validation failures, bad requests, and missing resources.

        Args:
            message: Human-readable error message
            status: HTTP status code (default 400)
            code: Error code (auto-detected from status if not provided)
            field: Field that caused the error (for validation errors)
            trace_id: Request trace ID for correlation
            suggestion: Helpful suggestion for the user

        Returns:
            Formatted error response dict
        """
        error_code = code or _STATUS_TO_CODE.get(status, ErrorCode.INVALID_REQUEST)
        details = {"field": field} if field else {}

        return APIError(
            code=error_code,
            message=message,
            status=status,
            trace_id=trace_id,
            details=details,
            suggestion=suggestion,
        ).to_response()

    @classmethod
    def format_server_error(
        cls,
        exception: Exception,
        context: str = "",
        trace_id: Optional[str] = None,
        log_full: bool = True,
    ) -> dict:
        """Format a server error (5xx) response.

        Logs full details server-side, returns sanitized message to client.

        Args:
            exception: The exception that occurred
            context: Context for logging (e.g., "debate creation")
            trace_id: Request trace ID for correlation
            log_full: Whether to log full exception with traceback

        Returns:
            Formatted error response dict with sanitized message
        """
        # Log full details server-side
        if log_full:
            logger.exception(
                f"[{trace_id or 'no-trace'}] Error in {context}: "
                f"{type(exception).__name__}: {exception}"
            )

        # Get classification
        status, code, message = cls._classify_exception_type(exception)

        return APIError(
            code=code,
            message=message,
            status=status,
            trace_id=trace_id,
            suggestion="If this persists, contact support with the trace ID" if trace_id else None,
        ).to_response()

    @classmethod
    def classify_exception(
        cls,
        exception: Exception,
        context: str = "",
        trace_id: Optional[str] = None,
    ) -> APIError:
        """Classify an exception into an APIError.

        Args:
            exception: The exception to classify
            context: Context for the error
            trace_id: Request trace ID

        Returns:
            APIError instance with appropriate classification
        """
        status, code, message = cls._classify_exception_type(exception)

        # Log for debugging
        logger.debug(
            f"Classified {type(exception).__name__} as {code.value} ({status})"
        )

        return APIError(
            code=code,
            message=message,
            status=status,
            trace_id=trace_id,
            details={"context": context} if context else {},
        )

    @classmethod
    def _classify_exception_type(
        cls, exception: Exception
    ) -> tuple[int, ErrorCode, str]:
        """Classify exception type to (status, code, message)."""
        exception_type = type(exception).__name__

        # Check direct mapping
        if exception_type in cls.EXCEPTION_MAP:
            return cls.EXCEPTION_MAP[exception_type]

        # Check fully qualified name for nested types
        full_type = f"{type(exception).__module__}.{exception_type}"
        if full_type in cls.EXCEPTION_MAP:
            return cls.EXCEPTION_MAP[full_type]

        # Check base classes
        for base_type, mapping in cls.EXCEPTION_MAP.items():
            if base_type in str(type(exception).__mro__):
                return mapping

        # Default to internal error
        return 500, ErrorCode.INTERNAL_ERROR, "An error occurred"

    @classmethod
    def categorize_status(cls, status: int) -> str:
        """Categorize HTTP status code.

        Returns:
            Category string: "success", "redirect", "client_error", "server_error"
        """
        if 200 <= status < 300:
            return "success"
        elif 300 <= status < 400:
            return "redirect"
        elif 400 <= status < 500:
            return "client_error"
        else:
            return "server_error"
