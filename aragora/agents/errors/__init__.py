"""
Standardized error handling for agent operations.

This package provides:
- Custom exception hierarchy for agent errors (inheriting from AragoraError)
- Centralized error classification for fallback decisions
- Async error handling decorators with retry logic
- Structured error logging with sanitization

Modules:
- exceptions: Custom exception classes for agent errors
- classifier: Error pattern matching and classification
- decorators: Retry decorators and error handlers

All exports are available at the package level for backward compatibility.
"""

# Import everything for backward compatibility
from .exceptions import (
    AgentError,
    AgentConnectionError,
    AgentTimeoutError,
    AgentRateLimitError,
    AgentAPIError,
    AgentResponseError,
    AgentStreamError,
    AgentCircuitOpenError,
    CLIAgentError,
    CLIParseError,
    CLITimeoutError,
    CLISubprocessError,
    CLINotFoundError,
)

from .classifier import (
    RATE_LIMIT_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    CLI_ERROR_PATTERNS,
    ALL_FALLBACK_PATTERNS,
    ErrorContext,
    ErrorAction,
    ErrorClassifier,
    classify_cli_error,
)

from .decorators import (
    calculate_retry_delay_with_jitter,
    _calculate_retry_delay_with_jitter,
    _handle_timeout_error,
    _handle_connection_error,
    _handle_payload_error,
    _handle_response_error,
    _handle_agent_error,
    _handle_json_error,
    _handle_unexpected_error,
    handle_agent_errors,
    with_error_handling,
    handle_stream_errors,
)

from .handlers import (
    handle_agent_operation,
    AgentErrorHandler,
    make_fallback_message,
)

# Re-export sanitization from utils for backward compatibility
from aragora.utils.error_sanitizer import (
    sanitize_error,
    SENSITIVE_PATTERNS as _SENSITIVE_PATTERNS,
)

# Type variable for generic return types (backward compat)
from typing import TypeVar
T = TypeVar("T")


__all__ = [
    # Type variable
    "T",
    # Exceptions
    "AgentError",
    "AgentConnectionError",
    "AgentTimeoutError",
    "AgentRateLimitError",
    "AgentAPIError",
    "AgentResponseError",
    "AgentStreamError",
    "AgentCircuitOpenError",
    "CLIAgentError",
    "CLIParseError",
    "CLITimeoutError",
    "CLISubprocessError",
    "CLINotFoundError",
    # Patterns
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
    # Retry calculation
    "calculate_retry_delay_with_jitter",
    "_calculate_retry_delay_with_jitter",
    # Handler functions
    "_handle_timeout_error",
    "_handle_connection_error",
    "_handle_payload_error",
    "_handle_response_error",
    "_handle_agent_error",
    "_handle_json_error",
    "_handle_unexpected_error",
    # Decorators
    "handle_agent_errors",
    "with_error_handling",
    "handle_stream_errors",
    # Autonomic handlers
    "handle_agent_operation",
    "AgentErrorHandler",
    "make_fallback_message",
    # Sanitization
    "sanitize_error",
    "_SENSITIVE_PATTERNS",
]
