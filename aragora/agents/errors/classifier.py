"""
Error classification for fallback and retry decisions.

Provides centralized error pattern matching and classification to determine:
- Whether to trigger fallback to alternative agents
- Error categorization for metrics/logging
- CLI error classification from subprocess results
- Recommended recovery actions

Error Categories:
    - timeout: Request/response timeouts
    - rate_limit: Rate limiting, quota exceeded
    - network: Connection issues, service unavailable
    - cli: CLI subprocess failures
    - auth: Authentication/authorization failures
    - validation: Input validation errors (context too long, etc.)
    - model: Model-specific errors (not found, unavailable)
    - content_policy: Content moderation violations
    - unknown: Unclassified errors
"""

import asyncio
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .exceptions import (
    AgentError,
    CLIAgentError,
    CLINotFoundError,
    CLIParseError,
    CLISubprocessError,
    CLITimeoutError,
)

# =============================================================================
# Error Category Enum
# =============================================================================


class ErrorCategory(Enum):
    """Categorization of errors for metrics and handling decisions."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    CLI = "cli"
    AUTH = "auth"
    VALIDATION = "validation"
    MODEL = "model"
    CONTENT_POLICY = "content_policy"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity level of errors for logging and alerting."""

    CRITICAL = "critical"  # System failure, requires immediate attention
    ERROR = "error"  # Operation failed, may need investigation
    WARNING = "warning"  # Transient failure, likely recoverable
    INFO = "info"  # Expected failure (e.g., rate limit, retry will succeed)


class RecoveryAction(Enum):
    """Recommended recovery action for an error."""

    RETRY = "retry"  # Retry with exponential backoff
    RETRY_IMMEDIATE = "retry_immediate"  # Retry immediately (transient)
    FALLBACK = "fallback"  # Switch to alternative agent
    WAIT = "wait"  # Wait for specified duration (rate limit)
    ABORT = "abort"  # Do not retry, operation cannot succeed
    ESCALATE = "escalate"  # Requires human intervention


# =============================================================================
# Error Pattern Constants
# =============================================================================

# Patterns that indicate rate limiting, quota errors, or service issues
RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    # Rate limiting
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "too many requests",
    "throttl",  # throttled, throttling
    # Quota/usage limit errors
    "quota exceeded",
    "quota_exceeded",
    "resource exhausted",
    "resource_exhausted",
    "insufficient_quota",
    "limit exceeded",
    "usage_limit",
    "usage limit",
    "limit has been reached",
    # Billing errors
    "billing",
    "credit balance",
    "payment required",
    "purchase credits",
    "402",
)

NETWORK_ERROR_PATTERNS: tuple[str, ...] = (
    # Capacity/availability errors
    "503",
    "service unavailable",
    "502",
    "bad gateway",
    "500",
    "internal server error",
    "overloaded",
    "capacity",
    "temporarily unavailable",
    "try again later",
    "server busy",
    "high demand",
    # Connection errors
    "connection refused",
    "connection reset",
    "timed out",
    "timeout",
    "network error",
    "socket error",
    "could not resolve host",
    "name or service not known",
    "econnrefused",
    "econnreset",
    "etimedout",
    "no route to host",
    "network is unreachable",
    # SSL/TLS errors
    "ssl error",
    "ssl_error",
    "certificate verify failed",
    "ssl handshake",
    "ssl: certificate",
    "certificate expired",
    "cert verify",
    # Proxy errors
    "proxy error",
    "proxy authentication",
    "tunnel connection failed",
    "407",
    # DNS errors
    "dns resolution failed",
    "getaddrinfo failed",
    "nodename nor servname provided",
)

CLI_ERROR_PATTERNS: tuple[str, ...] = (
    # CLI-specific errors
    "argument list too long",  # E2BIG - prompt too large for CLI
    "command not found",
    "no such file or directory",
    "permission denied",
    "access denied",
    "broken pipe",  # EPIPE - connection closed unexpectedly
)

AUTH_ERROR_PATTERNS: tuple[str, ...] = (
    # Authentication/authorization errors
    "invalid_api_key",
    "invalid api key",
    "unauthorized",
    "401",
    "authentication failed",
    "auth error",
    "forbidden",
    "403",
    "access denied",
    "permission denied",
    "invalid credentials",
    "bad credentials",
    "token expired",
    "session expired",
    "api key not found",
    "missing api key",
)

VALIDATION_ERROR_PATTERNS: tuple[str, ...] = (
    # Input validation errors
    "context length",
    "context_length",
    "context window",
    "context_window",
    "too long",
    "max_tokens",
    "maximum context",
    "token limit",
    "input too large",
    "prompt too long",
    "invalid input",
    "invalid_input",
    "bad request",
    "400",
    "validation error",
    "validation_error",
    "malformed",
    "invalid format",
    "missing required",
    "required field",
    "out of range",
    "invalid value",
    # Provider-specific validation
    "string_above_max_length",
    "reduce your prompt",
    "reduce the length",
    "exceeds the model",
    "exceeds maximum",
)

MODEL_ERROR_PATTERNS: tuple[str, ...] = (
    # Model-specific errors
    "model overloaded",
    "model is currently overloaded",
    "engine is currently overloaded",
    "model_not_found",
    "model not found",
    "model unavailable",
    "model_unavailable",
    "unsupported model",
    "invalid model",
    "model deprecated",
    "model_deprecated",
    # Provider-specific model errors
    "does not exist",
    "does not support",
    "model access",
    "model_access_denied",
    "decommissioned",
    "no longer available",
    "not available in your region",
)

CONTENT_POLICY_PATTERNS: tuple[str, ...] = (
    # Content moderation/policy violations
    "content policy",
    "content_policy",
    "content filter",
    "content_filter",
    "safety filter",
    "safety_filter",
    "moderation",
    "flagged",
    "harmful content",
    "inappropriate",
    "policy violation",
    "terms of service",
    "refused to generate",
    "cannot generate",
    # Provider-specific content blocks
    "i cannot",
    "i'm unable to",
    "output blocked",
    "response blocked",
    "content violation",
    "safety system",
    "ethical guidelines",
    "usage policies",
)

# Combined patterns for fallback decisions (all error types that should trigger fallback)
ALL_FALLBACK_PATTERNS: tuple[str, ...] = (
    RATE_LIMIT_PATTERNS
    + NETWORK_ERROR_PATTERNS
    + CLI_ERROR_PATTERNS
    + AUTH_ERROR_PATTERNS
    + MODEL_ERROR_PATTERNS
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
class ClassifiedError:
    """Full error classification result with category, severity, and recommended action.

    This is the primary result type for comprehensive error classification,
    providing all information needed for error handling decisions.

    Attributes:
        category: The error category (timeout, rate_limit, network, etc.)
        severity: The error severity (critical, error, warning, info)
        action: Recommended recovery action (retry, fallback, abort, etc.)
        should_fallback: Whether to trigger fallback to alternative agent
        message: Human-readable error description
        retry_after: Suggested wait time in seconds (for rate limits)
        details: Additional error-specific details
    """

    category: ErrorCategory
    severity: ErrorSeverity
    action: RecoveryAction
    should_fallback: bool
    message: str = ""
    retry_after: Optional[float] = None
    details: dict = field(default_factory=dict)

    @property
    def is_recoverable(self) -> bool:
        """Whether the error is potentially recoverable."""
        return self.action not in (RecoveryAction.ABORT, RecoveryAction.ESCALATE)

    @property
    def category_str(self) -> str:
        """Category as string for backward compatibility."""
        return self.category.value


@dataclass
class ErrorAction:
    """Result of error classification for retry/handling decisions.

    DEPRECATED: Use ClassifiedError for new code. This class is maintained
    for backward compatibility.
    """

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
    NETWORK_ERRNO: frozenset[int] = frozenset(
        {
            7,  # E2BIG - Argument list too long (prompt too large for CLI)
            32,  # EPIPE - Broken pipe (connection closed)
            104,  # ECONNRESET - Connection reset by peer
            110,  # ETIMEDOUT - Connection timed out
            111,  # ECONNREFUSED - Connection refused
            113,  # EHOSTUNREACH - No route to host
        }
    )

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
    def is_auth_error(cls, error_message: str) -> bool:
        """Check if error message indicates authentication/authorization issues.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates auth issues
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in AUTH_ERROR_PATTERNS)

    @classmethod
    def is_validation_error(cls, error_message: str) -> bool:
        """Check if error message indicates input validation issues.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates validation issues
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in VALIDATION_ERROR_PATTERNS)

    @classmethod
    def is_model_error(cls, error_message: str) -> bool:
        """Check if error message indicates model-specific issues.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates model issues
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in MODEL_ERROR_PATTERNS)

    @classmethod
    def is_content_policy_error(cls, error_message: str) -> bool:
        """Check if error message indicates content policy violations.

        Args:
            error_message: Error message string to check

        Returns:
            True if error indicates content policy violation
        """
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in CONTENT_POLICY_PATTERNS)

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
        if isinstance(
            error, (ConnectionError, ConnectionRefusedError, ConnectionResetError, BrokenPipeError)
        ):
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
            Category string: "timeout", "rate_limit", "network", "cli",
            "auth", "validation", "model", "content_policy", or "unknown"
        """
        error_str = str(error).lower()

        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return "timeout"

        if cls.is_rate_limit(error_str):
            return "rate_limit"

        if cls.is_network_error(error_str) or isinstance(
            error, (ConnectionError, ConnectionRefusedError, ConnectionResetError, BrokenPipeError)
        ):
            return "network"

        if cls.is_auth_error(error_str):
            return "auth"

        if cls.is_content_policy_error(error_str):
            return "content_policy"

        if cls.is_model_error(error_str):
            return "model"

        if cls.is_validation_error(error_str):
            return "validation"

        if cls.is_cli_error(error_str) or isinstance(error, subprocess.SubprocessError):
            return "cli"

        return "unknown"

    @classmethod
    def classify_error(cls, error: Exception) -> tuple[bool, str]:
        """Classify an error in a single pass, returning both fallback decision and category.

        More efficient than calling should_fallback() and get_error_category() separately
        when both pieces of information are needed.

        Args:
            error: The exception to classify

        Returns:
            Tuple of (should_fallback, category) where:
            - should_fallback: True if fallback should be attempted
            - category: "timeout", "rate_limit", "network", "cli",
              "auth", "validation", "model", "content_policy", or "unknown"
        """
        error_str = str(error).lower()

        # Check timeout first (common case)
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return True, "timeout"

        # Check rate limit patterns
        if cls.is_rate_limit(error_str):
            return True, "rate_limit"

        # Check network errors
        if cls.is_network_error(error_str) or isinstance(
            error, (ConnectionError, ConnectionRefusedError, ConnectionResetError, BrokenPipeError)
        ):
            return True, "network"

        # Check OS-level network errors
        if isinstance(error, OSError) and error.errno in cls.NETWORK_ERRNO:
            return True, "network"

        # Check auth errors (should fallback to try different credentials)
        if cls.is_auth_error(error_str):
            return True, "auth"

        # Check model errors (should fallback to different model)
        if cls.is_model_error(error_str):
            return True, "model"

        # Check CLI errors
        if cls.is_cli_error(error_str) or isinstance(error, subprocess.SubprocessError):
            return True, "cli"

        # Content policy errors - don't fallback (same content will fail elsewhere)
        if cls.is_content_policy_error(error_str):
            return False, "content_policy"

        # Validation errors - don't fallback (input is invalid)
        if cls.is_validation_error(error_str):
            return False, "validation"

        # Check RuntimeError patterns
        if isinstance(error, RuntimeError):
            if "cli command failed" in error_str or "cli" in error_str:
                return True, "cli"
            if any(kw in error_str for kw in ["api error", "http error", "status"]):
                return True, "unknown"

        # Check remaining fallback patterns (not categorized above)
        if any(pattern in error_str for pattern in ALL_FALLBACK_PATTERNS):
            return True, "unknown"

        return False, "unknown"

    @classmethod
    def classify_full(cls, error: Exception) -> ClassifiedError:
        """Comprehensive error classification with category, severity, and action.

        Provides full classification including recommended recovery action and
        severity level, suitable for advanced error handling pipelines.

        Args:
            error: The exception to classify

        Returns:
            ClassifiedError with category, severity, action, and metadata
        """
        error_str = str(error).lower()
        error_message = str(error)

        # Classification rules: (category, severity, action, should_fallback)
        # Order matters - more specific patterns should come first

        # Timeout
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.WARNING,
                action=RecoveryAction.RETRY,
                should_fallback=True,
                message=error_message,
            )

        # Rate limit
        if cls.is_rate_limit(error_str):
            # Try to extract retry-after hint
            retry_after = None
            for pattern in ["retry after ", "retry-after: ", "wait "]:
                if pattern in error_str:
                    import re

                    match = re.search(rf"{pattern}(\d+)", error_str)
                    if match:
                        retry_after = float(match.group(1))
                        break

            return ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.INFO,
                action=RecoveryAction.WAIT,
                should_fallback=True,
                message=error_message,
                retry_after=retry_after or 60.0,  # Default 60s if not specified
            )

        # Network errors
        if cls.is_network_error(error_str) or isinstance(
            error, (ConnectionError, ConnectionRefusedError, ConnectionResetError, BrokenPipeError)
        ):
            return ClassifiedError(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.WARNING,
                action=RecoveryAction.RETRY,
                should_fallback=True,
                message=error_message,
            )

        # OS-level network errors
        if isinstance(error, OSError) and error.errno in cls.NETWORK_ERRNO:
            return ClassifiedError(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.WARNING,
                action=RecoveryAction.RETRY,
                should_fallback=True,
                message=error_message,
                details={"errno": error.errno},
            )

        # Auth errors - critical, needs human intervention
        if cls.is_auth_error(error_str):
            return ClassifiedError(
                category=ErrorCategory.AUTH,
                severity=ErrorSeverity.CRITICAL,
                action=RecoveryAction.ESCALATE,
                should_fallback=True,  # Try different agent with valid creds
                message=error_message,
            )

        # Content policy - can't retry, content is problematic
        if cls.is_content_policy_error(error_str):
            return ClassifiedError(
                category=ErrorCategory.CONTENT_POLICY,
                severity=ErrorSeverity.ERROR,
                action=RecoveryAction.ABORT,
                should_fallback=False,  # Same content will fail elsewhere
                message=error_message,
            )

        # Model errors - fallback to different model
        if cls.is_model_error(error_str):
            return ClassifiedError(
                category=ErrorCategory.MODEL,
                severity=ErrorSeverity.WARNING,
                action=RecoveryAction.FALLBACK,
                should_fallback=True,
                message=error_message,
            )

        # Validation errors - input is bad, can't retry
        if cls.is_validation_error(error_str):
            return ClassifiedError(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                action=RecoveryAction.ABORT,
                should_fallback=False,
                message=error_message,
            )

        # CLI errors
        if cls.is_cli_error(error_str) or isinstance(error, subprocess.SubprocessError):
            return ClassifiedError(
                category=ErrorCategory.CLI,
                severity=ErrorSeverity.WARNING,
                action=RecoveryAction.FALLBACK,
                should_fallback=True,
                message=error_message,
            )

        # RuntimeError patterns
        if isinstance(error, RuntimeError):
            if "cli command failed" in error_str or "cli" in error_str:
                return ClassifiedError(
                    category=ErrorCategory.CLI,
                    severity=ErrorSeverity.WARNING,
                    action=RecoveryAction.FALLBACK,
                    should_fallback=True,
                    message=error_message,
                )

        # Unknown - default to retry with fallback
        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=any(pattern in error_str for pattern in ALL_FALLBACK_PATTERNS),
            message=error_message,
        )


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
            "Rate limit exceeded",
            agent_name=agent_name,
            returncode=returncode,
            stderr=stderr[:500] if stderr else None,
            recoverable=True,
        )

    # Timeout detection (SIGKILL = -9)
    if returncode == -9 or "timeout" in stderr_lower or "timed out" in stderr_lower:
        return CLITimeoutError(
            (
                f"CLI command timed out after {timeout_seconds}s"
                if timeout_seconds
                else "CLI command timed out"
            ),
            agent_name=agent_name,
            timeout_seconds=timeout_seconds,
        )

    # Command not found
    if returncode == 127 or "command not found" in stderr_lower or "not found" in stderr_lower:
        return CLINotFoundError(
            "CLI tool not found",
            agent_name=agent_name,
        )

    # Permission denied
    if returncode == 126 or "permission denied" in stderr_lower:
        return CLISubprocessError(
            "Permission denied executing CLI",
            agent_name=agent_name,
            returncode=returncode,
            stderr=stderr[:500] if stderr else None,
        )

    # JSON parse error detection
    if stdout and not stdout.strip():
        return CLIParseError(
            "Empty response from CLI",
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
                "Invalid JSON response from CLI",
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
    # Enums
    "ErrorCategory",
    "ErrorSeverity",
    "RecoveryAction",
    # Constants
    "RATE_LIMIT_PATTERNS",
    "NETWORK_ERROR_PATTERNS",
    "AUTH_ERROR_PATTERNS",
    "VALIDATION_ERROR_PATTERNS",
    "MODEL_ERROR_PATTERNS",
    "CONTENT_POLICY_PATTERNS",
    "CLI_ERROR_PATTERNS",
    "ALL_FALLBACK_PATTERNS",
    # Dataclasses
    "ErrorContext",
    "ClassifiedError",
    "ErrorAction",
    # Classifier
    "ErrorClassifier",
    "classify_cli_error",
]
