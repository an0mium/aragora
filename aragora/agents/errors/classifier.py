"""
Error classification for fallback and retry decisions.

Provides centralized error pattern matching and classification to determine:
- Whether to trigger fallback to alternative agents
- Error categorization for metrics/logging
- CLI error classification from subprocess results
"""

import asyncio
import subprocess
from dataclasses import dataclass

from .exceptions import (
    AgentError,
    CLIAgentError,
    CLINotFoundError,
    CLIParseError,
    CLISubprocessError,
    CLITimeoutError,
)


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


# =============================================================================
# Error Context and Action Dataclasses
# =============================================================================


@dataclass
class ErrorContext:
    """Context for error handling decisions."""

    agent_name: str
    attempt: int
    max_retries: int
    retry_delay: float
    max_delay: float
    timeout: float | None = None


@dataclass
class ErrorAction:
    """Result of error classification for retry/handling decisions."""

    error: "AgentError"
    should_retry: bool
    delay_seconds: float = 0.0
    log_level: str = "warning"


# =============================================================================
# Error Classifier
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


# =============================================================================
# CLI Error Classification
# =============================================================================


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


__all__ = [
    # Constants
    "RATE_LIMIT_PATTERNS",
    "NETWORK_ERROR_PATTERNS",
    "CLI_ERROR_PATTERNS",
    "ALL_FALLBACK_PATTERNS",
    # Dataclasses
    "ErrorContext",
    "ErrorAction",
    # Classifier
    "ErrorClassifier",
    "classify_cli_error",
]
