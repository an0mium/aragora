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
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

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

    # Agent-specific errors
    INVALID_API_KEY = "INVALID_API_KEY"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_RATE_LIMITED = "AGENT_RATE_LIMITED"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    CONFIG_ERROR = "CONFIG_ERROR"


# =============================================================================
# Error Suggestions Registry
# =============================================================================

ERROR_SUGGESTIONS: dict[ErrorCode, dict[str, str]] = {
    ErrorCode.INVALID_API_KEY: {
        "message": "Invalid or missing API key",
        "suggestion": "Check your API key configuration",
        "cli_help": "Run `aragora doctor` to verify API keys",
        "docs": "See docs/ENVIRONMENT.md for configuration",
    },
    ErrorCode.AGENT_TIMEOUT: {
        "message": "Agent timed out",
        "suggestion": "Increase timeout or use fewer agents",
        "cli_help": "Set ARAGORA_DEBATE_TIMEOUT=1200 in .env",
        "docs": "See docs/TROUBLESHOOTING.md",
    },
    ErrorCode.AGENT_RATE_LIMITED: {
        "message": "Agent rate limited by provider",
        "suggestion": "Wait and retry, or use fallback provider",
        "cli_help": "Set OPENROUTER_API_KEY for fallback access",
        "docs": "See docs/ENVIRONMENT.md for fallback configuration",
    },
    ErrorCode.AGENT_NOT_FOUND: {
        "message": "Agent not found",
        "suggestion": "Check agent name spelling",
        "cli_help": "Run `aragora status` to list available agents",
        "docs": "See docs/CUSTOM_AGENTS.md for agent configuration",
    },
    ErrorCode.CONFIG_ERROR: {
        "message": "Configuration error",
        "suggestion": "Check your configuration file",
        "cli_help": "Run `aragora config show` to view current settings",
        "docs": "See docs/ENVIRONMENT.md for configuration options",
    },
    ErrorCode.UNAUTHORIZED: {
        "message": "Authentication required",
        "suggestion": "Provide valid authentication token",
        "cli_help": "Set ARAGORA_API_TOKEN in .env",
        "docs": "See docs/API_REFERENCE.md for authentication",
    },
    ErrorCode.RATE_LIMITED: {
        "message": "Rate limit exceeded",
        "suggestion": "Wait before retrying",
        "cli_help": "Rate limits reset after 60 seconds",
        "docs": "See docs/API_REFERENCE.md for rate limit details",
    },
    ErrorCode.NOT_FOUND: {
        "message": "Resource not found",
        "suggestion": "Verify the resource ID exists",
        "cli_help": "Use `aragora status` to check available resources",
        "docs": "Check the API endpoint path",
    },
    ErrorCode.DATABASE_ERROR: {
        "message": "Database error",
        "suggestion": "Check database connectivity",
        "cli_help": "Run `aragora doctor` to check database status",
        "docs": "See docs/TROUBLESHOOTING.md for database issues",
    },
    ErrorCode.TIMEOUT: {
        "message": "Operation timed out",
        "suggestion": "Increase timeout or reduce complexity",
        "cli_help": "Set ARAGORA_DEBATE_TIMEOUT to a higher value",
        "docs": "See docs/ENVIRONMENT.md for timeout configuration",
    },
    ErrorCode.SERVICE_UNAVAILABLE: {
        "message": "Service temporarily unavailable",
        "suggestion": "Wait and retry",
        "cli_help": "Check server status with `aragora doctor`",
        "docs": "See docs/TROUBLESHOOTING.md",
    },
}


def get_error_suggestion(code: ErrorCode) -> dict[str, str]:
    """Get suggestion details for an error code.

    Returns:
        Dict with 'message', 'suggestion', 'cli_help', and 'docs' keys
    """
    return ERROR_SUGGESTIONS.get(
        code,
        {
            "message": "An error occurred",
            "suggestion": "Check the error details",
            "cli_help": "Run `aragora doctor` for diagnostics",
            "docs": "See docs/TROUBLESHOOTING.md",
        },
    )


def format_cli_error(code: ErrorCode, details: str = "") -> str:
    """Format an error message for CLI output with actionable suggestions.

    Args:
        code: Error code
        details: Additional error details

    Returns:
        Formatted error string for terminal output
    """
    info = get_error_suggestion(code)
    lines = [
        f"[{code.value}] {info['message']}",
    ]
    if details:
        lines.append(f"  Details: {details}")
    lines.append(f"  -> {info['suggestion']}")
    lines.append(f"  -> Try: {info['cli_help']}")
    return "\n".join(lines)


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
        # Auto-populate suggestion from registry if not provided
        if not self.suggestion and self.code in ERROR_SUGGESTIONS:
            self.suggestion = ERROR_SUGGESTIONS[self.code].get("suggestion")

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
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
        suggestion = (
            f"Retry after {retry_after} seconds" if retry_after else "Please try again later"
        )
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
        logger.debug(f"Classified {type(exception).__name__} as {code.value} ({status})")

        return APIError(
            code=code,
            message=message,
            status=status,
            trace_id=trace_id,
            details={"context": context} if context else {},
        )

    @classmethod
    def _classify_exception_type(cls, exception: Exception) -> tuple[int, ErrorCode, str]:
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


# =============================================================================
# Error Context Utilities
# =============================================================================


class ErrorContext:
    """Context manager that adds context to exceptions.

    Wraps operations with additional context for better debugging.
    On exception, logs the context and optionally re-raises with enriched info.

    Example:
        with ErrorContext("loading debate", debate_id=debate_id):
            debate = storage.get_debate(debate_id)

        # Or as decorator:
        @with_error_context("agent creation")
        def create_agent(spec):
            ...
    """

    def __init__(
        self,
        operation: str,
        reraise: bool = True,
        log_level: str = "error",
        **context_data,
    ):
        """Initialize error context.

        Args:
            operation: Description of the operation (e.g., "loading debate")
            reraise: Whether to re-raise the exception (default True)
            log_level: Log level for error messages ("error", "warning", "debug")
            **context_data: Additional context (e.g., debate_id="abc123")
        """
        self.operation = operation
        self.reraise = reraise
        self.log_level = log_level
        self.context_data = context_data
        self.exception: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.exception = exc_val

            # Format context for logging
            context_str = ", ".join(f"{k}={v}" for k, v in self.context_data.items())
            full_context = f"{self.operation}"
            if context_str:
                full_context += f" ({context_str})"

            # Log with appropriate level
            log_func = getattr(logger, self.log_level, logger.error)
            log_func(
                f"Error during {full_context}: {type(exc_val).__name__}: {exc_val}",
                exc_info=self.log_level == "error",
            )

            if not self.reraise:
                return True  # Suppress exception

        return False  # Propagate exception


def with_error_context(operation: str, **default_context: Any) -> Callable[[F], F]:
    """Decorator version of ErrorContext.

    Example:
        @with_error_context("debate creation", phase="initialization")
        def create_debate(config):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Merge default context with any runtime context
            context = {**default_context}
            with ErrorContext(operation, **context):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def log_and_suppress(operation: str, default_value: Any = None, **context: Any) -> "ErrorContext":
    """Context manager that logs errors but returns a default value.

    Useful for optional operations that shouldn't fail the request.

    Example:
        with log_and_suppress("fetching metadata", default_value={}) as ctx:
            metadata = fetch_metadata(id)
        # If fetch fails, metadata = {}
    """

    class SuppressingContext(ErrorContext):
        def __init__(self) -> None:
            super().__init__(operation, reraise=False, log_level="warning", **context)
            self.result = default_value

        def __enter__(self) -> "SuppressingContext":
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: Any,
        ) -> bool:
            if exc_val is not None:
                super().__exit__(exc_type, exc_val, exc_tb)
                return True
            return False

    return SuppressingContext()
