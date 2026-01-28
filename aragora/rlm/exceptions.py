"""
RLM-specific exceptions for robust error handling.

These exceptions provide fine-grained error reporting for RLM operations,
enabling proper retry logic, circuit breaker integration, and debugging.
"""

from __future__ import annotations


class RLMError(Exception):
    """Base exception for all RLM-related errors.

    All RLM exceptions inherit from this class, making it easy to catch
    any RLM error with a single except clause.

    Attributes:
        message: Human-readable error description
        operation: The RLM operation that failed (e.g., "query", "summarize")
        content_id: Optional content ID associated with the error
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        content_id: str | None = None,
    ):
        self.message = message
        self.operation = operation
        self.content_id = content_id
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.operation:
            parts.insert(0, f"[{self.operation}]")
        if self.content_id:
            parts.append(f"(content_id={self.content_id})")
        return " ".join(parts)


class RLMTimeoutError(RLMError):
    """RLM operation timed out.

    Raised when an RLM operation (query, summarization, etc.) exceeds
    the configured timeout. This is a transient error that may succeed
    on retry.

    Attributes:
        timeout_seconds: The timeout that was exceeded
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: str | None = None,
        content_id: str | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, operation, content_id)

    def __str__(self) -> str:
        return f"{super().__str__()} (timeout={self.timeout_seconds}s)"


class RLMContextOverflowError(RLMError):
    """Context window exceeded.

    Raised when the content being processed exceeds the maximum
    context window size. This is NOT a transient error - the content
    must be reduced or summarized before retrying.

    Attributes:
        content_size: Size of the content in characters
        max_size: Maximum allowed size
    """

    def __init__(
        self,
        message: str,
        content_size: int,
        max_size: int,
        operation: str | None = None,
        content_id: str | None = None,
    ):
        self.content_size = content_size
        self.max_size = max_size
        super().__init__(message, operation, content_id)

    def __str__(self) -> str:
        return f"{super().__str__()} (size={self.content_size}, max={self.max_size})"


class RLMProviderError(RLMError):
    """Provider-specific error.

    Raised when the underlying LLM provider returns an error.
    May be transient (rate limits, server errors) or permanent
    (invalid API key, quota exceeded).

    Attributes:
        provider: Name of the provider (e.g., "openai", "anthropic")
        status_code: HTTP status code if available
        is_transient: Whether the error is likely transient
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        is_transient: bool = True,
        operation: str | None = None,
        content_id: str | None = None,
    ):
        self.provider = provider
        self.status_code = status_code
        self.is_transient = is_transient
        super().__init__(message, operation, content_id)

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        return " ".join(parts)


class RLMCircuitOpenError(RLMError):
    """Circuit breaker is open.

    Raised when the circuit breaker is open due to repeated failures.
    The caller should either fail fast or wait for the circuit to
    enter half-open state.

    Attributes:
        cooldown_remaining: Seconds until circuit may close
    """

    def __init__(
        self,
        message: str,
        cooldown_remaining: float | None = None,
        operation: str | None = None,
        content_id: str | None = None,
    ):
        self.cooldown_remaining = cooldown_remaining
        super().__init__(message, operation, content_id)

    def __str__(self) -> str:
        if self.cooldown_remaining is not None:
            return f"{super().__str__()} (retry in {self.cooldown_remaining:.1f}s)"
        return super().__str__()


class RLMContentNotFoundError(RLMError):
    """Requested content not found in registry.

    Raised when trying to access content that hasn't been registered
    or has been unregistered.
    """

    pass


class RLMREPLError(RLMError):
    """Error in REPL environment execution.

    Raised when code execution in the REPL environment fails.

    Attributes:
        code: The code that failed to execute
        line_number: Line number where the error occurred
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        line_number: int | None = None,
        operation: str | None = None,
        content_id: str | None = None,
    ):
        self.code = code
        self.line_number = line_number
        super().__init__(message, operation, content_id)


__all__ = [
    "RLMError",
    "RLMTimeoutError",
    "RLMContextOverflowError",
    "RLMProviderError",
    "RLMCircuitOpenError",
    "RLMContentNotFoundError",
    "RLMREPLError",
]
