"""
Custom exception types for Aragora.

This module defines a hierarchy of exceptions used throughout the codebase.
Using specific exception types enables:
- More precise error handling with targeted except blocks
- Better error messages and debugging
- Cleaner separation of error domains
"""

from __future__ import annotations


class AragoraError(Exception):
    """Base exception for all Aragora errors.

    All custom exceptions in Aragora should inherit from this class
    to enable catching all Aragora-specific errors with a single handler.
    """

    def __init__(self, message: str, details: dict | None = None):
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
# Agent Errors (re-exported from aragora.agents.errors for unified imports)
# ============================================================================
# The canonical agent error hierarchy is in aragora.agents.errors, which provides
# richer functionality (recoverable flag, cause chaining, circuit breaker support).
# Re-exports are provided here for convenience and unified imports.

# Import will be done at module level after all base classes are defined
# to avoid circular imports. See end of file for re-exports.

# Legacy aliases for backwards compatibility (these were never used but kept
# for any future code that might expect them from exceptions.py)


class AgentNotFoundError(AragoraError):
    """Raised when a requested agent cannot be found.

    Note: This is a simple lookup error, distinct from AgentError hierarchy
    which handles runtime agent failures.
    """

    def __init__(self, agent_name: str):
        super().__init__(f"Agent not found: {agent_name}", {"agent_name": agent_name})
        self.agent_name = agent_name


class AgentConfigurationError(AragoraError):
    """Raised when agent configuration is invalid.

    Note: Configuration errors are distinct from runtime AgentErrors.
    """
    pass


class APIKeyError(AragoraError):
    """Raised when an API key is missing or invalid."""

    def __init__(self, provider: str):
        super().__init__(
            f"Missing or invalid API key for {provider}",
            {"provider": provider}
        )
        self.provider = provider


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


# ============================================================================
# Agent Errors - Usage Guide
# ============================================================================
# For runtime agent failures (timeouts, rate limits, connection issues), import
# from aragora.agents.errors which provides the full exception hierarchy with:
#   - recoverable flag for retry decisions
#   - cause chaining for debugging
#   - circuit breaker integration
#
# Example:
#   from aragora.agents.errors import (
#       AgentError, AgentTimeoutError, AgentRateLimitError,
#       CLIAgentError, ErrorClassifier,
#   )
#
# The exceptions in this file (AgentNotFoundError, AgentConfigurationError,
# APIKeyError) are for configuration-time errors, not runtime failures.
