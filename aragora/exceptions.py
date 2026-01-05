"""
Custom exception types for Aragora.

This module defines a hierarchy of exceptions used throughout the codebase.
Using specific exception types enables:
- More precise error handling with targeted except blocks
- Better error messages and debugging
- Cleaner separation of error domains
"""

from typing import Optional


class AragoraError(Exception):
    """Base exception for all Aragora errors.

    All custom exceptions in Aragora should inherit from this class
    to enable catching all Aragora-specific errors with a single handler.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# ============================================================================
# Debate Errors
# ============================================================================

class DebateError(AragoraError):
    """Base exception for debate-related errors."""
    pass


class DebateNotFoundError(DebateError):
    """Raised when a requested debate cannot be found."""

    def __init__(self, debate_id: str):
        super().__init__(f"Debate not found: {debate_id}", {"debate_id": debate_id})
        self.debate_id = debate_id


class DebateConfigurationError(DebateError):
    """Raised when debate configuration is invalid."""
    pass


class ConsensusError(DebateError):
    """Raised when consensus cannot be reached or is invalid."""
    pass


class RoundLimitExceededError(DebateError):
    """Raised when maximum rounds are exceeded."""

    def __init__(self, max_rounds: int, current_round: int):
        super().__init__(
            f"Round limit exceeded: {current_round}/{max_rounds}",
            {"max_rounds": max_rounds, "current_round": current_round}
        )
        self.max_rounds = max_rounds
        self.current_round = current_round


class EarlyStopError(DebateError):
    """Raised when debate is stopped early (not necessarily an error)."""

    def __init__(self, reason: str, round_stopped: int):
        super().__init__(f"Debate stopped early: {reason}", {"round": round_stopped})
        self.reason = reason
        self.round_stopped = round_stopped


# ============================================================================
# Agent Errors
# ============================================================================

class AgentError(AragoraError):
    """Base exception for agent-related errors."""
    pass


class AgentNotFoundError(AgentError):
    """Raised when a requested agent cannot be found."""

    def __init__(self, agent_name: str):
        super().__init__(f"Agent not found: {agent_name}", {"agent_name": agent_name})
        self.agent_name = agent_name


class AgentConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""
    pass


class AgentResponseError(AgentError):
    """Raised when an agent fails to generate a response."""

    def __init__(self, agent_name: str, reason: str):
        super().__init__(
            f"Agent '{agent_name}' failed to respond: {reason}",
            {"agent_name": agent_name, "reason": reason}
        )
        self.agent_name = agent_name
        self.reason = reason


class AgentTimeoutError(AgentError):
    """Raised when an agent takes too long to respond."""

    def __init__(self, agent_name: str, timeout_seconds: float):
        super().__init__(
            f"Agent '{agent_name}' timed out after {timeout_seconds}s",
            {"agent_name": agent_name, "timeout": timeout_seconds}
        )
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds


class APIKeyError(AgentError):
    """Raised when an API key is missing or invalid."""

    def __init__(self, provider: str):
        super().__init__(
            f"Missing or invalid API key for {provider}",
            {"provider": provider}
        )
        self.provider = provider


class RateLimitError(AgentError):
    """Raised when an API rate limit is hit."""

    def __init__(self, provider: str, retry_after: Optional[float] = None):
        super().__init__(
            f"Rate limit exceeded for {provider}",
            {"provider": provider, "retry_after": retry_after}
        )
        self.provider = provider
        self.retry_after = retry_after


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(AragoraError):
    """Base exception for validation errors."""
    pass


class InputValidationError(ValidationError):
    """Raised when user input fails validation."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Invalid input for '{field}': {reason}",
            {"field": field, "reason": reason}
        )
        self.field = field
        self.reason = reason


class SchemaValidationError(ValidationError):
    """Raised when data fails schema validation."""

    def __init__(self, schema_name: str, errors: list[str]):
        super().__init__(
            f"Schema validation failed for {schema_name}",
            {"schema": schema_name, "errors": errors}
        )
        self.schema_name = schema_name
        self.errors = errors


# ============================================================================
# Storage Errors
# ============================================================================

class StorageError(AragoraError):
    """Base exception for storage-related errors."""
    pass


class DatabaseError(StorageError):
    """Raised when a database operation fails."""
    pass


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails."""

    def __init__(self, db_path: str, reason: str):
        super().__init__(
            f"Failed to connect to database at {db_path}: {reason}",
            {"db_path": db_path, "reason": reason}
        )
        self.db_path = db_path
        self.reason = reason


class RecordNotFoundError(StorageError):
    """Raised when a requested record cannot be found."""

    def __init__(self, table: str, record_id: str):
        super().__init__(
            f"Record not found in {table}: {record_id}",
            {"table": table, "record_id": record_id}
        )
        self.table = table
        self.record_id = record_id


# ============================================================================
# Memory Errors
# ============================================================================

class MemoryError(AragoraError):
    """Base exception for memory system errors."""
    pass


class MemoryRetrievalError(MemoryError):
    """Raised when memory retrieval fails."""
    pass


class MemoryStorageError(MemoryError):
    """Raised when memory storage fails."""
    pass


class EmbeddingError(MemoryError):
    """Raised when embedding generation fails."""

    def __init__(self, text_preview: str, reason: str):
        # Truncate text preview for error message
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        super().__init__(
            f"Failed to generate embedding: {reason}",
            {"text_preview": preview, "reason": reason}
        )
        self.reason = reason


# ============================================================================
# Mode Errors
# ============================================================================

class ModeError(AragoraError):
    """Base exception for debate mode errors."""
    pass


class ModeNotFoundError(ModeError):
    """Raised when a requested mode cannot be found."""

    def __init__(self, mode_name: str):
        super().__init__(f"Mode not found: {mode_name}", {"mode_name": mode_name})
        self.mode_name = mode_name


class ModeConfigurationError(ModeError):
    """Raised when mode configuration is invalid."""
    pass


# ============================================================================
# Plugin Errors
# ============================================================================

class PluginError(AragoraError):
    """Base exception for plugin-related errors."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found."""

    def __init__(self, plugin_name: str):
        super().__init__(f"Plugin not found: {plugin_name}", {"plugin_name": plugin_name})
        self.plugin_name = plugin_name


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""

    def __init__(self, plugin_name: str, reason: str):
        super().__init__(
            f"Plugin '{plugin_name}' execution failed: {reason}",
            {"plugin_name": plugin_name, "reason": reason}
        )
        self.plugin_name = plugin_name
        self.reason = reason


# ============================================================================
# Authentication Errors
# ============================================================================

class AuthError(AragoraError):
    """Base exception for authentication errors."""
    pass


class AuthenticationError(AuthError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(AuthError):
    """Raised when authorization fails."""
    pass


class TokenExpiredError(AuthError):
    """Raised when an authentication token has expired."""
    pass


class RateLimitExceededError(AuthError):
    """Raised when rate limit is exceeded."""

    def __init__(self, limit: int, window_seconds: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            {"limit": limit, "window_seconds": window_seconds}
        )
        self.limit = limit
        self.window_seconds = window_seconds


# ============================================================================
# Nomic Errors
# ============================================================================

class NomicError(AragoraError):
    """Base exception for Nomic self-improvement loop errors."""
    pass


class NomicCycleError(NomicError):
    """Raised when a Nomic cycle fails."""

    def __init__(self, cycle: int, phase: str, reason: str):
        super().__init__(
            f"Nomic cycle {cycle} failed in {phase}: {reason}",
            {"cycle": cycle, "phase": phase, "reason": reason}
        )
        self.cycle = cycle
        self.phase = phase
        self.reason = reason


class NomicStateError(NomicError):
    """Raised when Nomic state is invalid or corrupted."""
    pass


# ============================================================================
# Verification Errors
# ============================================================================

class VerificationError(AragoraError):
    """Base exception for formal verification errors."""
    pass


class Z3NotAvailableError(VerificationError):
    """Raised when Z3 solver is not available."""

    def __init__(self):
        super().__init__("Z3 solver not available. Install with: pip install z3-solver")


class VerificationTimeoutError(VerificationError):
    """Raised when verification times out."""

    def __init__(self, timeout_ms: int):
        super().__init__(
            f"Verification timed out after {timeout_ms}ms",
            {"timeout_ms": timeout_ms}
        )
        self.timeout_ms = timeout_ms
