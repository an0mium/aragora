"""
Standardized error handling for agent operations.

Provides:
- Custom exception hierarchy for agent errors
- Centralized error classification for fallback decisions
- Async error handling decorators with retry logic
- Structured error logging with sanitization
"""

import asyncio
import functools
import logging
import random
import re
import subprocess
from typing import Any, Callable, Optional, Type, TypeVar

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Error Pattern Constants
# =============================================================================

# Patterns that indicate rate limiting, quota errors, or service issues
RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    # Rate limiting
    "rate limit", "rate_limit", "ratelimit",
    "429", "too many requests",
    "throttl",  # throttled, throttling
    # Quota/usage limit errors
    "quota exceeded", "quota_exceeded",
    "resource exhausted", "resource_exhausted",
    "insufficient_quota", "limit exceeded",
    "usage_limit", "usage limit",
    "limit has been reached",
    # Billing errors
    "billing", "credit balance", "payment required",
    "purchase credits", "402",
)

NETWORK_ERROR_PATTERNS: tuple[str, ...] = (
    # Capacity/availability errors
    "503", "service unavailable",
    "502", "bad gateway",
    "overloaded", "capacity",
    "temporarily unavailable", "try again later",
    "server busy", "high demand",
    # Connection errors
    "connection refused", "connection reset",
    "timed out", "timeout",
    "network error", "socket error",
    "could not resolve host", "name or service not known",
    "econnrefused", "econnreset", "etimedout",
    "no route to host", "network is unreachable",
)

CLI_ERROR_PATTERNS: tuple[str, ...] = (
    # API-specific errors
    "model overloaded", "model is currently overloaded",
    "engine is currently overloaded",
    "model_not_found", "model not found",
    "invalid_api_key", "invalid api key", "unauthorized",
    "authentication failed", "auth error",
    # CLI-specific errors
    "argument list too long",  # E2BIG - prompt too large for CLI
    "command not found", "no such file or directory",
    "permission denied", "access denied",
    "broken pipe",  # EPIPE - connection closed unexpectedly
)

# Combined patterns for fallback decisions (all error types that should trigger fallback)
ALL_FALLBACK_PATTERNS: tuple[str, ...] = (
    RATE_LIMIT_PATTERNS + NETWORK_ERROR_PATTERNS + CLI_ERROR_PATTERNS
)

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
        agent_name: str | None = None,
        cause: Exception | None = None,
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
        agent_name: str | None = None,
        status_code: int | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.status_code = status_code


class AgentTimeoutError(AgentError):
    """Timeout during agent operation."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        timeout_seconds: float | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.timeout_seconds = timeout_seconds


class AgentRateLimitError(AgentError):
    """Rate limit or quota exceeded."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        retry_after: float | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.retry_after = retry_after


class AgentAPIError(AgentError):
    """API-specific error (invalid request, auth failure, etc.)."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        cause: Exception | None = None,
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
        agent_name: str | None = None,
        response_data: Any = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=False)
        self.response_data = response_data


class AgentStreamError(AgentError):
    """Error during streaming response."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        partial_content: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.partial_content = partial_content


class AgentCircuitOpenError(AgentError):
    """Circuit breaker is open, blocking requests to agent.

    This error is raised when too many consecutive failures have occurred
    and the circuit breaker has opened to protect the system from cascading
    failures. The request should be retried after the cooldown period.
    """

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        cooldown_seconds: float | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, cause, recoverable=True)
        self.cooldown_seconds = cooldown_seconds


# =============================================================================
# CLI Agent Errors
# =============================================================================


class CLIAgentError(AgentError):
    """Base class for CLI agent errors."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        returncode: int | None = None,
        stderr: str | None = None,
        cause: Exception | None = None,
        recoverable: bool = True,
    ):
        super().__init__(message, agent_name, cause, recoverable)
        self.returncode = returncode
        self.stderr = stderr


class CLIParseError(CLIAgentError):
    """Error parsing CLI agent output (invalid JSON, etc.)."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        returncode: int | None = None,
        stderr: str | None = None,
        raw_output: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, returncode, stderr, cause, recoverable=False)
        self.raw_output = raw_output


class CLITimeoutError(CLIAgentError):
    """CLI agent subprocess timed out."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        timeout_seconds: float | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, returncode=-9, cause=cause, recoverable=True)
        self.timeout_seconds = timeout_seconds


class CLISubprocessError(CLIAgentError):
    """CLI subprocess failed with non-zero exit code."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        returncode: int | None = None,
        stderr: str | None = None,
        cause: Exception | None = None,
    ):
        # Non-zero exit codes are generally recoverable (transient failures)
        super().__init__(message, agent_name, returncode, stderr, cause, recoverable=True)


class CLINotFoundError(CLIAgentError):
    """CLI tool not found or not installed."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        cli_name: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, agent_name, returncode=127, cause=cause, recoverable=False)
        self.cli_name = cli_name


# =============================================================================
# Error Classification Utilities
# =============================================================================


class ErrorClassifier:
    """Centralized error classification for fallback and retry decisions.

    Provides consistent error classification across CLI and API agents.
    Use this class to determine if an error should trigger fallback,
    retry, or other recovery mechanisms.

    Example:
        classifier = ErrorClassifier()

        # Check if exception should trigger fallback
        if classifier.should_fallback(error):
            return await fallback_agent.generate(prompt)

        # Check specific error types
        if classifier.is_rate_limit("Error: 429 Too Many Requests"):
            await asyncio.sleep(retry_after)
    """

    # OS error numbers that indicate connection/network issues
    NETWORK_ERRNO: frozenset[int] = frozenset({
        7,    # E2BIG - Argument list too long (prompt too large for CLI)
        32,   # EPIPE - Broken pipe (connection closed)
        104,  # ECONNRESET - Connection reset by peer
        110,  # ETIMEDOUT - Connection timed out
        111,  # ECONNREFUSED - Connection refused
        113,  # EHOSTUNREACH - No route to host
    })

    @classmethod
    def is_rate_limit(cls, error_message: str) -> bool:
        """Check if error message indicates rate limiting or quota exceeded.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates rate limiting
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in RATE_LIMIT_PATTERNS)

    @classmethod
    def is_network_error(cls, error_message: str) -> bool:
        """Check if error message indicates network/connection issues.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates network issues
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in NETWORK_ERROR_PATTERNS)

    @classmethod
    def is_cli_error(cls, error_message: str) -> bool:
        """Check if error message indicates CLI-specific issues.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates CLI issues
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in CLI_ERROR_PATTERNS)

    @classmethod
    def should_fallback(cls, error: Exception) -> bool:
        """Determine if an exception should trigger fallback to alternative agent.

        Checks exception type and message for patterns that indicate the
        primary agent is unavailable and fallback should be attempted.

        Args:
            error: The exception to classify

        Returns:
            True if fallback should be attempted
        """
        error_str = str(error).lower()

        # Check for pattern matches in error message
        if any(pattern in error_str for pattern in ALL_FALLBACK_PATTERNS):
            return True

        # Timeout errors should trigger fallback
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return True

        # Connection errors should trigger fallback
        if isinstance(error, (ConnectionError, ConnectionRefusedError,
                              ConnectionResetError, BrokenPipeError)):
            return True

        # OS-level errors (file not found for CLI, etc.)
        if isinstance(error, OSError) and error.errno in cls.NETWORK_ERRNO:
            return True

        # CLI command failures
        if isinstance(error, RuntimeError):
            if "cli command failed" in error_str or "cli" in error_str:
                return True
            if any(kw in error_str for kw in ["api error", "http error", "status"]):
                return True

        # Subprocess errors
        if isinstance(error, subprocess.SubprocessError):
            return True

        return False

    @classmethod
    def get_error_category(cls, error: Exception) -> str:
        """Get the category of an error for logging/metrics.

        Args:
            error: The exception to categorize

        Returns:
            Category string: "rate_limit", "network", "cli", "timeout", or "unknown"
        """
        error_str = str(error).lower()

        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return "timeout"

        if cls.is_rate_limit(error_str):
            return "rate_limit"

        if cls.is_network_error(error_str) or isinstance(
            error, (ConnectionError, ConnectionRefusedError,
                    ConnectionResetError, BrokenPipeError)
        ):
            return "network"

        if cls.is_cli_error(error_str) or isinstance(error, subprocess.SubprocessError):
            return "cli"

        return "unknown"


def classify_cli_error(
    returncode: int,
    stderr: str,
    stdout: str,
    agent_name: str | None = None,
    timeout_seconds: float | None = None,
) -> CLIAgentError:
    """
    Classify a CLI agent error based on return code and output.

    This function analyzes subprocess results to determine the appropriate
    error type for proper handling and retry decisions.

    Args:
        returncode: Subprocess exit code
        stderr: Standard error output
        stdout: Standard output
        agent_name: Name of the agent for error context
        timeout_seconds: Timeout value if applicable

    Returns:
        Appropriate CLIAgentError subclass instance
    """
    stderr_lower = stderr.lower() if stderr else ""
    stdout_lower = stdout.lower() if stdout else ""

    # Rate limit detection using centralized patterns
    if ErrorClassifier.is_rate_limit(stderr_lower):
        return CLIAgentError(
            f"Rate limit exceeded",
            agent_name=agent_name,
            returncode=returncode,
            stderr=stderr[:500] if stderr else None,
            recoverable=True,
        )

    # Timeout detection (SIGKILL = -9)
    if returncode == -9 or "timeout" in stderr_lower or "timed out" in stderr_lower:
        return CLITimeoutError(
            f"CLI command timed out after {timeout_seconds}s" if timeout_seconds else "CLI command timed out",
            agent_name=agent_name,
            timeout_seconds=timeout_seconds,
        )

    # Command not found
    if returncode == 127 or "command not found" in stderr_lower or "not found" in stderr_lower:
        return CLINotFoundError(
            f"CLI tool not found",
            agent_name=agent_name,
        )

    # Permission denied
    if returncode == 126 or "permission denied" in stderr_lower:
        return CLISubprocessError(
            f"Permission denied executing CLI",
            agent_name=agent_name,
            returncode=returncode,
            stderr=stderr[:500] if stderr else None,
        )

    # JSON parse error detection
    if stdout and not stdout.strip():
        return CLIParseError(
            f"Empty response from CLI",
            agent_name=agent_name,
            returncode=returncode,
            stderr=stderr[:500] if stderr else None,
            raw_output=stdout[:200] if stdout else None,
        )

    # Check for JSON error responses
    if stdout and stdout.strip().startswith("{"):
        try:
            import json
            data = json.loads(stdout)
            if "error" in data:
                return CLIAgentError(
                    f"CLI returned error: {data.get('error', 'Unknown error')[:200]}",
                    agent_name=agent_name,
                    returncode=returncode,
                    stderr=stderr[:500] if stderr else None,
                    recoverable=True,
                )
        except json.JSONDecodeError:
            return CLIParseError(
                f"Invalid JSON response from CLI",
                agent_name=agent_name,
                returncode=returncode,
                stderr=stderr[:500] if stderr else None,
                raw_output=stdout[:200] if stdout else None,
            )

    # Generic subprocess error
    return CLISubprocessError(
        f"CLI exited with code {returncode}: {stderr[:200] if stderr else 'no error output'}",
        agent_name=agent_name,
        returncode=returncode,
        stderr=stderr[:500] if stderr else None,
    )


# =============================================================================
# Sensitive Data Sanitization
# =============================================================================

# Import shared sanitization utilities
from aragora.utils.error_sanitizer import (
    sanitize_error,
    SENSITIVE_PATTERNS as _SENSITIVE_PATTERNS,  # For backwards compatibility
)


# =============================================================================
# Retry Delay Calculation
# =============================================================================


def _calculate_retry_delay_with_jitter(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter_factor: float = 0.3,
) -> float:
    """
    Calculate retry delay with exponential backoff and random jitter.

    Jitter prevents thundering herd when multiple clients recover simultaneously
    after a provider outage.

    Args:
        attempt: Current retry attempt (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Fraction of delay to randomize (default: 0.3 = ±30%)

    Returns:
        Delay in seconds with jitter applied
    """
    # Calculate base exponential delay
    delay = min(base_delay * (2 ** attempt), max_delay)

    # Apply random jitter: delay ± (jitter_factor * delay)
    jitter = delay * jitter_factor * random.uniform(-1, 1)

    # Ensure minimum delay of 0.1s
    return max(0.1, delay + jitter)


# =============================================================================
# Error Handling Decorators
# =============================================================================


def handle_agent_errors(
    agent_name_attr: str = "name",
    max_retries: int = 0,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (AgentConnectionError, AgentTimeoutError, AgentRateLimitError),
    circuit_breaker_attr: str = "_circuit_breaker",
):
    """
    Decorator for async agent methods that standardizes error handling.

    Wraps aiohttp and other common exceptions in AgentError types,
    logs errors appropriately, and optionally retries transient failures.
    Integrates with CircuitBreaker for graceful failure handling.

    Args:
        agent_name_attr: Attribute name on self containing agent name
        max_retries: Maximum retry attempts for recoverable errors (0 = no retry)
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Multiplier for delay between retries
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of AgentError subclasses to retry
        circuit_breaker_attr: Attribute name on self for CircuitBreaker instance.
            If the attribute exists and circuit is open, raises AgentCircuitOpenError.
            Records success/failure to circuit breaker after each attempt.

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
            circuit_breaker = getattr(self, circuit_breaker_attr, None)

            # Check circuit breaker before attempting call
            if circuit_breaker is not None and not circuit_breaker.can_proceed():
                raise AgentCircuitOpenError(
                    f"Circuit breaker is open for agent",
                    agent_name=agent_name,
                    cooldown_seconds=circuit_breaker.cooldown_seconds,
                )

            attempt = 0
            delay = retry_delay
            last_error: Optional[AgentError] = None

            while True:
                attempt += 1
                try:
                    result = await func(self, *args, **kwargs)
                    # Record success to circuit breaker
                    if circuit_breaker is not None:
                        circuit_breaker.record_success()
                    return result

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
                    # Record failure to circuit breaker before raising
                    if circuit_breaker is not None:
                        circuit_breaker.record_failure()
                    raise last_error from e

                # Record failure to circuit breaker
                if circuit_breaker is not None and last_error is not None:
                    circuit_breaker.record_failure()

                # Check if we should retry
                if (
                    max_retries > 0
                    and attempt <= max_retries
                    and last_error
                    and isinstance(last_error, retryable_exceptions)
                    and last_error.recoverable
                ):
                    # Use Retry-After header if available for rate limits
                    if isinstance(last_error, AgentRateLimitError) and last_error.retry_after:
                        # Add small jitter to Retry-After to avoid thundering herd
                        base_wait = min(last_error.retry_after, max_delay)
                        jitter = base_wait * 0.1 * random.uniform(0, 1)
                        wait_time = base_wait + jitter
                    else:
                        # Calculate delay with jitter for exponential backoff
                        wait_time = _calculate_retry_delay_with_jitter(
                            attempt - 1,  # 0-indexed for calculation
                            retry_delay,
                            max_delay,
                        )

                    logger.info(
                        f"[{agent_name}] Retrying in {wait_time:.1f}s "
                        f"(attempt {attempt}/{max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
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
