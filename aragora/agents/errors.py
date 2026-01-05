"""
Standardized error handling for agent operations.

Provides:
- Custom exception hierarchy for agent errors
- Async error handling decorators with retry logic
- Structured error logging with sanitization
"""

import asyncio
import functools
import logging
import re
from typing import Any, Callable, Optional, Type, TypeVar

import aiohttp

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar("T")


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class AgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        cause: Exception = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.agent_name = agent_name
        self.cause = cause
        self.recoverable = recoverable

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.agent_name:
            parts.insert(0, f"[{self.agent_name}]")
        if self.cause:
            parts.append(f"(caused by: {type(self.cause).__name__}: {self.cause})")
        return " ".join(parts)


class AgentConnectionError(AgentError):
    """Network connection or HTTP errors."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        status_code: int = None,
        cause: Exception = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.status_code = status_code


class AgentTimeoutError(AgentError):
    """Timeout during agent operation."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        timeout_seconds: float = None,
        cause: Exception = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.timeout_seconds = timeout_seconds


class AgentRateLimitError(AgentError):
    """Rate limit or quota exceeded."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        retry_after: float = None,
        cause: Exception = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.retry_after = retry_after


class AgentAPIError(AgentError):
    """API-specific error (invalid request, auth failure, etc.)."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        status_code: int = None,
        error_type: str = None,
        cause: Exception = None,
    ):
        # 4xx errors are generally not recoverable (bad request, auth)
        recoverable = status_code is None or status_code >= 500
        super().__init__(message, agent_name, cause, recoverable=recoverable)
        self.status_code = status_code
        self.error_type = error_type


class AgentResponseError(AgentError):
    """Error parsing or validating agent response."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        response_data: Any = None,
        cause: Exception = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=False)
        self.response_data = response_data


class AgentStreamError(AgentError):
    """Error during streaming response."""

    def __init__(
        self,
        message: str,
        agent_name: str = None,
        partial_content: str = None,
        cause: Exception = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.partial_content = partial_content


# =============================================================================
# Sensitive Data Sanitization
# =============================================================================

_SENSITIVE_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "<REDACTED_KEY>"),
    (r"AIza[a-zA-Z0-9_-]{35}", "<REDACTED_KEY>"),
    (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+["\']?', "api_key=<REDACTED>"),
    (r'["\']?authorization["\']?\s*[:=]\s*["\']?Bearer\s+[\w.-]+["\']?', "auth=<REDACTED>"),
    (r'["\']?token["\']?\s*[:=]\s*["\']?[\w.-]+["\']?', "token=<REDACTED>"),
    (r'["\']?secret["\']?\s*[:=]\s*["\']?[\w-]+["\']?', "secret=<REDACTED>"),
    (r"x-api-key:\s*[\w-]+", "x-api-key: <REDACTED>"),
    (r"x-goog-api-key:\s*[\w-]+", "x-goog-api-key: <REDACTED>"),
]


def sanitize_error(text: str, max_length: int = 500) -> str:
    """Sanitize error message to remove potential secrets."""
    sanitized = str(text)
    for pattern, replacement in _SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"
    return sanitized


# =============================================================================
# Error Handling Decorators
# =============================================================================


def handle_agent_errors(
    agent_name_attr: str = "name",
    max_retries: int = 0,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (AgentConnectionError, AgentTimeoutError),
):
    """
    Decorator for async agent methods that standardizes error handling.

    Wraps aiohttp and other common exceptions in AgentError types,
    logs errors appropriately, and optionally retries transient failures.

    Args:
        agent_name_attr: Attribute name on self containing agent name
        max_retries: Maximum retry attempts for recoverable errors (0 = no retry)
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Multiplier for delay between retries
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of AgentError subclasses to retry

    Usage:
        @handle_agent_errors(max_retries=3)
        async def generate(self, prompt: str) -> str:
            async with aiohttp.ClientSession() as session:
                ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> T:
            agent_name = getattr(self, agent_name_attr, "unknown")
            attempt = 0
            delay = retry_delay
            last_error: Optional[AgentError] = None

            while True:
                attempt += 1
                try:
                    return await func(self, *args, **kwargs)

                # Timeout errors
                except asyncio.TimeoutError as e:
                    timeout = getattr(self, "timeout", None)
                    last_error = AgentTimeoutError(
                        f"Operation timed out after {timeout}s",
                        agent_name=agent_name,
                        timeout_seconds=timeout,
                        cause=e,
                    )
                    logger.warning(
                        f"[{agent_name}] Timeout (attempt {attempt}): {last_error}"
                    )

                # aiohttp connection errors
                except aiohttp.ClientConnectorError as e:
                    last_error = AgentConnectionError(
                        f"Connection failed: {sanitize_error(str(e))}",
                        agent_name=agent_name,
                        cause=e,
                    )
                    logger.warning(
                        f"[{agent_name}] Connection error (attempt {attempt}): {last_error}"
                    )

                except aiohttp.ServerDisconnectedError as e:
                    last_error = AgentConnectionError(
                        f"Server disconnected: {sanitize_error(str(e))}",
                        agent_name=agent_name,
                        cause=e,
                    )
                    logger.warning(
                        f"[{agent_name}] Server disconnected (attempt {attempt}): {last_error}"
                    )

                except aiohttp.ClientPayloadError as e:
                    last_error = AgentStreamError(
                        f"Payload error during streaming: {sanitize_error(str(e))}",
                        agent_name=agent_name,
                        cause=e,
                    )
                    logger.warning(
                        f"[{agent_name}] Stream error (attempt {attempt}): {last_error}"
                    )

                except aiohttp.ClientResponseError as e:
                    if e.status == 429:
                        retry_after = None
                        if e.headers and "Retry-After" in e.headers:
                            try:
                                retry_after = float(e.headers["Retry-After"])
                            except (ValueError, TypeError):
                                pass
                        last_error = AgentRateLimitError(
                            f"Rate limit exceeded (HTTP 429)",
                            agent_name=agent_name,
                            retry_after=retry_after,
                            cause=e,
                        )
                        logger.warning(
                            f"[{agent_name}] Rate limited (attempt {attempt}): {last_error}"
                        )
                    elif e.status >= 500:
                        last_error = AgentConnectionError(
                            f"Server error (HTTP {e.status})",
                            agent_name=agent_name,
                            status_code=e.status,
                            cause=e,
                        )
                        logger.warning(
                            f"[{agent_name}] Server error (attempt {attempt}): {last_error}"
                        )
                    else:
                        last_error = AgentAPIError(
                            f"API error (HTTP {e.status}): {sanitize_error(str(e))}",
                            agent_name=agent_name,
                            status_code=e.status,
                            cause=e,
                        )
                        logger.error(
                            f"[{agent_name}] API error (attempt {attempt}): {last_error}"
                        )

                # Already wrapped AgentErrors - re-raise or retry
                except AgentError as e:
                    e.agent_name = e.agent_name or agent_name
                    last_error = e
                    if e.recoverable:
                        logger.warning(
                            f"[{agent_name}] Agent error (attempt {attempt}): {e}"
                        )
                    else:
                        logger.error(
                            f"[{agent_name}] Non-recoverable error: {e}"
                        )
                        raise

                # JSON decode errors
                except ValueError as e:
                    if "json" in str(e).lower() or "decode" in str(e).lower():
                        last_error = AgentResponseError(
                            f"Invalid JSON response: {sanitize_error(str(e))}",
                            agent_name=agent_name,
                            cause=e,
                        )
                        logger.error(
                            f"[{agent_name}] Response parse error: {last_error}"
                        )
                        raise last_error from e
                    raise

                # Catch-all for unexpected errors
                except Exception as e:
                    last_error = AgentError(
                        f"Unexpected error: {sanitize_error(str(e))}",
                        agent_name=agent_name,
                        cause=e,
                        recoverable=False,
                    )
                    logger.error(
                        f"[{agent_name}] Unexpected error (attempt {attempt}): {last_error}",
                        exc_info=True,
                    )
                    raise last_error from e

                # Check if we should retry
                if (
                    max_retries > 0
                    and attempt <= max_retries
                    and last_error
                    and isinstance(last_error, retryable_exceptions)
                    and last_error.recoverable
                ):
                    # Use Retry-After if available for rate limits
                    if isinstance(last_error, AgentRateLimitError) and last_error.retry_after:
                        wait_time = min(last_error.retry_after, max_delay)
                    else:
                        wait_time = min(delay, max_delay)

                    logger.info(
                        f"[{agent_name}] Retrying in {wait_time:.1f}s "
                        f"(attempt {attempt}/{max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                    delay = min(delay * retry_backoff, max_delay)
                    continue

                # No more retries - raise the last error
                raise last_error

        return wrapper

    return decorator


def handle_stream_errors(agent_name_attr: str = "name"):
    """
    Decorator specifically for streaming methods.

    Wraps errors that occur during async iteration and attempts to
    preserve any partial content received.

    Usage:
        @handle_stream_errors()
        async def generate_stream(self, prompt: str):
            async for chunk in ...:
                yield chunk
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            agent_name = getattr(self, agent_name_attr, "unknown")
            partial_content = []

            try:
                async for chunk in func(self, *args, **kwargs):
                    if isinstance(chunk, str):
                        partial_content.append(chunk)
                    yield chunk

            except asyncio.TimeoutError as e:
                timeout = getattr(self, "timeout", None)
                raise AgentTimeoutError(
                    f"Stream timed out after {timeout}s",
                    agent_name=agent_name,
                    timeout_seconds=timeout,
                    partial_content="".join(partial_content) if partial_content else None,
                    cause=e,
                ) from e

            except (aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError) as e:
                raise AgentStreamError(
                    f"Stream interrupted: {sanitize_error(str(e))}",
                    agent_name=agent_name,
                    partial_content="".join(partial_content) if partial_content else None,
                    cause=e,
                ) from e

            except AgentError:
                raise

            except Exception as e:
                raise AgentStreamError(
                    f"Unexpected stream error: {sanitize_error(str(e))}",
                    agent_name=agent_name,
                    partial_content="".join(partial_content) if partial_content else None,
                    cause=e,
                ) from e

        return wrapper

    return decorator
