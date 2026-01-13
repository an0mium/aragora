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
# Type variable for generic return types (re-export from types module)
from aragora.agents.types import T
from aragora.utils.error_sanitizer import (
    SENSITIVE_PATTERNS as _SENSITIVE_PATTERNS,
)

# Re-export sanitization from utils for backward compatibility
from aragora.utils.error_sanitizer import (
    sanitize_error,
)

from .classifier import (
    ALL_FALLBACK_PATTERNS,
    AUTH_ERROR_PATTERNS,
    CLI_ERROR_PATTERNS,
    CONTENT_POLICY_PATTERNS,
    MODEL_ERROR_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    # Patterns
    RATE_LIMIT_PATTERNS,
    VALIDATION_ERROR_PATTERNS,
    ClassifiedError,
    ErrorAction,
    # Enums
    ErrorCategory,
    # Classifier
    ErrorClassifier,
    # Dataclasses
    ErrorContext,
    ErrorSeverity,
    RecoveryAction,
    classify_cli_error,
)
from .decorators import (
    _calculate_retry_delay_with_jitter,
    _handle_agent_error,
    _handle_connection_error,
    _handle_json_error,
    _handle_payload_error,
    _handle_response_error,
    _handle_timeout_error,
    _handle_unexpected_error,
    calculate_retry_delay_with_jitter,
    handle_agent_errors,
    handle_stream_errors,
    with_error_handling,
)
from .exceptions import (
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    AgentError,
    AgentRateLimitError,
    AgentResponseError,
    AgentStreamError,
    AgentTimeoutError,
    CLIAgentError,
    CLINotFoundError,
    CLIParseError,
    CLISubprocessError,
    CLITimeoutError,
)
from .handlers import (
    AgentErrorHandler,
    _build_error_action,
    handle_agent_operation,
    make_fallback_message,
)

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
    # Enums
    "ErrorCategory",
    "ErrorSeverity",
    "RecoveryAction",
    # Patterns
    "RATE_LIMIT_PATTERNS",
    "NETWORK_ERROR_PATTERNS",
    "CLI_ERROR_PATTERNS",
    "AUTH_ERROR_PATTERNS",
    "VALIDATION_ERROR_PATTERNS",
    "MODEL_ERROR_PATTERNS",
    "CONTENT_POLICY_PATTERNS",
    "ALL_FALLBACK_PATTERNS",
    # Dataclasses
    "ErrorContext",
    "ErrorAction",
    "ClassifiedError",
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
    "_build_error_action",
    # Sanitization
    "sanitize_error",
    "_SENSITIVE_PATTERNS",
]
