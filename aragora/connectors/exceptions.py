"""
Connector Exception Hierarchy.

Provides standardized exceptions for all connector operations,
enabling consistent error handling, retry logic, and logging.

Hierarchy:
    ConnectorError (base)
    ├── ConnectorAuthError - Authentication/authorization failures
    ├── ConnectorRateLimitError - Rate limit exceeded
    ├── ConnectorTimeoutError - Request timeout
    ├── ConnectorNetworkError - Network connectivity issues
    ├── ConnectorAPIError - API returned error response
    ├── ConnectorValidationError - Invalid input/parameters
    ├── ConnectorNotFoundError - Resource not found
    └── ConnectorQuotaError - Quota exhausted

Each exception includes:
- connector_name: Identifies which connector raised the error
- retry_after: Optional seconds to wait before retry
- is_retryable: Whether the operation can be retried
"""

from typing import Optional


class ConnectorError(Exception):
    """Base exception for all connector errors.

    All connector implementations should raise subclasses of this
    exception for consistent error handling.

    Attributes:
        connector_name: Name of the connector that raised the error
        retry_after: Seconds to wait before retry (if applicable)
        is_retryable: Whether the operation can be retried
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        retry_after: Optional[float] = None,
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.connector_name = connector_name
        self.retry_after = retry_after
        self.is_retryable = is_retryable

    def __str__(self) -> str:
        base = super().__str__()
        if self.connector_name != "unknown":
            return f"[{self.connector_name}] {base}"
        return base


class ConnectorAuthError(ConnectorError):
    """Authentication or authorization failure.

    Raised when:
    - API keys are invalid or missing
    - OAuth tokens are expired
    - Permission denied for requested operation

    Generally NOT retryable without credential refresh.
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            retry_after=retry_after,
            is_retryable=False,  # Auth errors need credential fix
        )


class ConnectorRateLimitError(ConnectorError):
    """Rate limit exceeded.

    Raised when:
    - API returns 429 status code
    - Too many requests in time window
    - Provider throttles requests

    IS retryable after waiting for retry_after period.
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            retry_after=retry_after or 60.0,  # Default to 60 seconds
            is_retryable=True,
        )


class ConnectorTimeoutError(ConnectorError):
    """Request timeout.

    Raised when:
    - Connection timeout
    - Read timeout
    - Overall request timeout

    IS retryable (transient network issue).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            retry_after=5.0,  # Short delay before retry
            is_retryable=True,
        )
        self.timeout_seconds = timeout_seconds


class ConnectorNetworkError(ConnectorError):
    """Network connectivity issues.

    Raised when:
    - DNS resolution failure
    - Connection refused
    - SSL/TLS errors
    - Network unreachable

    IS retryable (transient network issue).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            retry_after=5.0,
            is_retryable=True,
        )


class ConnectorAPIError(ConnectorError):
    """API returned an error response.

    Raised when:
    - API returns 4xx/5xx status
    - API returns error in response body
    - Unexpected API response format

    Retryability depends on status code:
    - 5xx: Generally retryable
    - 4xx: Generally not retryable (except 429)
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        # 5xx errors are retryable, 4xx are not
        is_retryable = status_code is not None and 500 <= status_code < 600
        super().__init__(
            message,
            connector_name=connector_name,
            retry_after=retry_after,
            is_retryable=is_retryable,
        )
        self.status_code = status_code


class ConnectorValidationError(ConnectorError):
    """Invalid input or parameters.

    Raised when:
    - Required parameters missing
    - Parameter format invalid
    - Value out of allowed range

    NOT retryable (need to fix input).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        field: Optional[str] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            is_retryable=False,
        )
        self.field = field


class ConnectorNotFoundError(ConnectorError):
    """Requested resource not found.

    Raised when:
    - File/document doesn't exist
    - API resource 404
    - Search returns no results

    NOT retryable (resource doesn't exist).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            is_retryable=False,
        )
        self.resource_id = resource_id


class ConnectorQuotaError(ConnectorError):
    """Quota or usage limit exhausted.

    Raised when:
    - Daily/monthly quota exceeded
    - Storage limit reached
    - API usage cap hit

    NOT retryable immediately (need quota reset or upgrade).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        quota_reset: Optional[float] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            retry_after=quota_reset,
            is_retryable=False,  # Can't retry until quota resets
        )
        self.quota_reset = quota_reset


class ConnectorParseError(ConnectorError):
    """Failed to parse response content.

    Raised when:
    - JSON decode error
    - HTML parsing failure
    - Unexpected response format

    NOT retryable (response format issue).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        content_type: Optional[str] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            is_retryable=False,
        )
        self.content_type = content_type


class ConnectorConfigError(ConnectorError):
    """Configuration or setup error.

    Raised when:
    - Missing required configuration
    - Invalid configuration values
    - Connector not properly initialized

    NOT retryable (need to fix configuration).
    """

    def __init__(
        self,
        message: str,
        connector_name: str = "unknown",
        config_key: Optional[str] = None,
    ):
        super().__init__(
            message,
            connector_name=connector_name,
            is_retryable=False,
        )
        self.config_key = config_key


# =============================================================================
# Exception Utilities
# =============================================================================


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if the error suggests a retry might succeed
    """
    if isinstance(error, ConnectorError):
        return error.is_retryable

    # Check common retryable error types
    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Timeout errors
    if "timeout" in error_type.lower() or "timeout" in error_msg:
        return True

    # Connection errors
    if "connection" in error_type.lower() or "connection" in error_msg:
        return True

    # Rate limit indicators
    if "429" in error_msg or "rate" in error_msg:
        return True

    return False


def get_retry_delay(error: Exception, default: float = 5.0) -> float:
    """Get recommended retry delay for an error.

    Args:
        error: Exception to get delay for
        default: Default delay if not specified

    Returns:
        Recommended delay in seconds
    """
    if isinstance(error, ConnectorError) and error.retry_after:
        return error.retry_after

    # Rate limit errors should wait longer
    if isinstance(error, ConnectorRateLimitError):
        return 60.0

    return default


def classify_exception(
    error: Exception,
    connector_name: str = "unknown",
) -> ConnectorError:
    """
    Convert a generic exception to an appropriate ConnectorError subclass.

    This enables consistent error handling by converting standard Python
    exceptions (TimeoutError, ConnectionError, etc.) to the connector
    exception hierarchy.

    Args:
        error: The original exception
        connector_name: Name of the connector for error context

    Returns:
        A ConnectorError subclass appropriate for the error type

    Example:
        try:
            response = await client.get(url)
        except Exception as e:
            raise classify_exception(e, "web_connector") from e
    """
    import asyncio
    import json
    import ssl

    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Already a ConnectorError - just update connector name if needed
    if isinstance(error, ConnectorError):
        if error.connector_name == "unknown":
            error.connector_name = connector_name
        return error

    # Timeout errors
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return ConnectorTimeoutError(
            f"Request timed out: {error}",
            connector_name=connector_name,
        )
    if "timeout" in error_type.lower() or "timeout" in error_msg:
        return ConnectorTimeoutError(
            str(error),
            connector_name=connector_name,
        )

    # Connection/network errors
    if isinstance(error, (ConnectionError, OSError)):
        if "refused" in error_msg or "reset" in error_msg:
            return ConnectorNetworkError(
                f"Connection failed: {error}",
                connector_name=connector_name,
            )
        return ConnectorNetworkError(
            str(error),
            connector_name=connector_name,
        )

    # SSL errors
    if isinstance(error, ssl.SSLError):
        return ConnectorNetworkError(
            f"SSL error: {error}",
            connector_name=connector_name,
        )

    # JSON parsing errors
    if isinstance(error, json.JSONDecodeError):
        return ConnectorParseError(
            f"JSON decode error: {error}",
            connector_name=connector_name,
            content_type="application/json",
        )

    # Rate limit detection from error messages
    if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
        return ConnectorRateLimitError(
            str(error),
            connector_name=connector_name,
        )

    # Auth detection from error messages
    if any(
        x in error_msg for x in ["401", "403", "unauthorized", "forbidden", "invalid.*key", "auth"]
    ):
        return ConnectorAuthError(
            str(error),
            connector_name=connector_name,
        )

    # Not found detection
    if "404" in error_msg or "not found" in error_msg:
        return ConnectorNotFoundError(
            str(error),
            connector_name=connector_name,
        )

    # Server errors
    if any(x in error_msg for x in ["500", "502", "503", "504", "server error"]):
        return ConnectorAPIError(
            str(error),
            connector_name=connector_name,
            status_code=500,
        )

    # Validation errors
    if isinstance(error, (ValueError, TypeError)):
        return ConnectorValidationError(
            str(error),
            connector_name=connector_name,
        )

    # Default: wrap as generic ConnectorAPIError
    return ConnectorAPIError(
        f"Unexpected error: {error}",
        connector_name=connector_name,
        status_code=None,
    )


class connector_error_handler:
    """
    Context manager that converts exceptions to ConnectorError types.

    Usage:
        async with connector_error_handler("github"):
            response = await client.get(url)
            return response.json()

    Any exception raised within the block will be converted to an
    appropriate ConnectorError subclass with the connector name set.
    """

    def __init__(self, connector_name: str):
        self.connector_name = connector_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            raise classify_exception(exc_val, self.connector_name) from exc_val
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            raise classify_exception(exc_val, self.connector_name) from exc_val
        return False


__all__ = [
    # Base
    "ConnectorError",
    # Specific errors
    "ConnectorAuthError",
    "ConnectorRateLimitError",
    "ConnectorTimeoutError",
    "ConnectorNetworkError",
    "ConnectorAPIError",
    "ConnectorValidationError",
    "ConnectorNotFoundError",
    "ConnectorQuotaError",
    "ConnectorParseError",
    "ConnectorConfigError",
    # Utilities
    "is_retryable_error",
    "get_retry_delay",
    "classify_exception",
    "connector_error_handler",
]
