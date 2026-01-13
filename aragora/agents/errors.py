"""
Standardized error handling for agent operations.

This module re-exports from aragora.agents.errors package for backward compatibility.

For new code, prefer importing directly from the package:
    from aragora.agents.errors import AgentError, handle_agent_errors
    from aragora.agents.errors.exceptions import AgentTimeoutError
    from aragora.agents.errors.classifier import ErrorClassifier
    from aragora.agents.errors.decorators import with_error_handling
"""

# Explicit re-exports from the errors package for backward compatibility
# (Avoid wildcard imports for better IDE analysis and explicit dependencies)
from aragora.agents.errors import (  # noqa: F401
    _SENSITIVE_PATTERNS,
    ALL_FALLBACK_PATTERNS,
    CLI_ERROR_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    # Patterns
    RATE_LIMIT_PATTERNS,
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    # Exceptions
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
    ErrorAction,
    # Classifier
    ErrorClassifier,
    # Dataclasses
    ErrorContext,
    # Type variable
    T,
    _calculate_retry_delay_with_jitter,
    _handle_agent_error,
    _handle_connection_error,
    _handle_json_error,
    _handle_payload_error,
    _handle_response_error,
    # Handler functions
    _handle_timeout_error,
    _handle_unexpected_error,
    # Retry calculation
    calculate_retry_delay_with_jitter,
    classify_cli_error,
    # Decorators
    handle_agent_errors,
    handle_stream_errors,
    # Sanitization
    sanitize_error,
    with_error_handling,
)
