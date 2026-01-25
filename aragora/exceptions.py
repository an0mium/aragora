"""
Custom exception types for Aragora.

This module defines a hierarchy of exceptions used throughout the codebase.
Using specific exception types enables:
- More precise error handling with targeted except blocks
- Better error messages and debugging
- Cleaner separation of error domains
"""

from __future__ import annotations

from typing import Any


class AragoraError(Exception):
    """Base exception for all Aragora errors.

    All custom exceptions in Aragora should inherit from this class
    to enable catching all Aragora-specific errors with a single handler.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
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


class ConsensusTimeoutError(DebateError):
    """Raised when consensus detection times out."""

    def __init__(self, timeout_seconds: float, rounds_completed: int):
        super().__init__(
            f"Consensus timed out after {timeout_seconds}s ({rounds_completed} rounds)",
            {"timeout_seconds": timeout_seconds, "rounds_completed": rounds_completed},
        )
        self.timeout_seconds = timeout_seconds
        self.rounds_completed = rounds_completed


class VoteValidationError(DebateError):
    """Raised when vote validation fails."""

    def __init__(self, agent_name: str, reason: str, vote_data: dict[str, Any] | None = None):
        super().__init__(
            f"Invalid vote from {agent_name}: {reason}",
            {"agent_name": agent_name, "reason": reason},
        )
        self.agent_name = agent_name
        self.reason = reason
        self.vote_data = vote_data


class PhaseExecutionError(DebateError):
    """Raised when a debate phase fails to execute."""

    def __init__(self, phase_name: str, reason: str, recoverable: bool = False):
        super().__init__(
            f"Phase '{phase_name}' failed: {reason}",
            {"phase_name": phase_name, "reason": reason, "recoverable": recoverable},
        )
        self.phase_name = phase_name
        self.reason = reason
        self.recoverable = recoverable


class RoundLimitExceededError(DebateError):
    """Raised when maximum rounds are exceeded."""

    def __init__(self, max_rounds: int, current_round: int):
        super().__init__(
            f"Round limit exceeded: {current_round}/{max_rounds}",
            {"max_rounds": max_rounds, "current_round": current_round},
        )
        self.max_rounds = max_rounds
        self.current_round = current_round


class EarlyStopError(DebateError):
    """Raised when debate is stopped early (not necessarily an error)."""

    def __init__(self, reason: str, round_stopped: int):
        super().__init__(f"Debate stopped early: {reason}", {"round": round_stopped})
        self.reason = reason
        self.round_stopped = round_stopped


class DebateStartError(DebateError):
    """Raised when a debate fails to start."""

    def __init__(self, debate_id: str, reason: str):
        super().__init__(
            f"Failed to start debate {debate_id}: {reason}",
            {"debate_id": debate_id, "reason": reason},
        )
        self.debate_id = debate_id
        self.reason = reason


class DebateBatchError(DebateError):
    """Raised when a batch debate operation fails."""

    def __init__(self, operation: str, reason: str, failed_ids: list[str] | None = None):
        super().__init__(
            f"Batch operation '{operation}' failed: {reason}",
            {"operation": operation, "reason": reason, "failed_ids": failed_ids or []},
        )
        self.operation = operation
        self.reason = reason
        self.failed_ids = failed_ids or []


class DebateExecutionError(DebateError):
    """Raised when debate execution fails mid-process."""

    def __init__(self, debate_id: str, phase: str, reason: str):
        super().__init__(
            f"Debate {debate_id} failed during {phase}: {reason}",
            {"debate_id": debate_id, "phase": phase, "reason": reason},
        )
        self.debate_id = debate_id
        self.phase = phase
        self.reason = reason


class VoteProcessingError(DebateError):
    """Raised when vote processing fails."""

    def __init__(self, debate_id: str, reason: str, agent_name: str | None = None):
        msg = f"Vote processing failed for debate {debate_id}: {reason}"
        details = {"debate_id": debate_id, "reason": reason}
        if agent_name:
            details["agent_name"] = agent_name
        super().__init__(msg, details)
        self.debate_id = debate_id
        self.reason = reason
        self.agent_name = agent_name


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


class ConfigurationError(AragoraError):
    """Raised when a component's configuration is missing or invalid.

    Used for missing callbacks, invalid settings, or incomplete setup.
    """

    def __init__(self, component: str, reason: str):
        super().__init__(
            f"Configuration error in {component}: {reason}",
            {"component": component, "reason": reason},
        )
        self.component = component
        self.reason = reason


class APIKeyError(AragoraError):
    """Raised when an API key is missing or invalid."""

    def __init__(self, provider: str):
        super().__init__(f"Missing or invalid API key for {provider}", {"provider": provider})
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
            f"Invalid input for '{field}': {reason}", {"field": field, "reason": reason}
        )
        self.field = field
        self.reason = reason


class SchemaValidationError(ValidationError):
    """Raised when data fails schema validation."""

    def __init__(self, schema_name: str, errors: list[str]):
        super().__init__(
            f"Schema validation failed for {schema_name}", {"schema": schema_name, "errors": errors}
        )
        self.schema_name = schema_name
        self.errors = errors


class JSONParseError(ValidationError):
    """Raised when JSON parsing fails."""

    def __init__(self, source: str, reason: str, raw_text: str | None = None):
        # Truncate raw text for error message
        preview = ""
        if raw_text:
            preview = raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
        super().__init__(
            f"Failed to parse JSON from {source}: {reason}",
            {"source": source, "reason": reason, "preview": preview},
        )
        self.source = source
        self.reason = reason
        self.raw_text = raw_text


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
            {"db_path": db_path, "reason": reason},
        )
        self.db_path = db_path
        self.reason = reason


class RecordNotFoundError(StorageError):
    """Raised when a requested record cannot be found."""

    def __init__(self, table: str, record_id: str):
        super().__init__(
            f"Record not found in {table}: {record_id}", {"table": table, "record_id": record_id}
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


class TierTransitionError(MemoryError):
    """Raised when memory tier transition fails."""

    def __init__(self, from_tier: str, to_tier: str, reason: str):
        super().__init__(
            f"Failed to transition from {from_tier} to {to_tier}: {reason}",
            {"from_tier": from_tier, "to_tier": to_tier, "reason": reason},
        )
        self.from_tier = from_tier
        self.to_tier = to_tier
        self.reason = reason


class EmbeddingError(MemoryError):
    """Raised when embedding generation fails."""

    def __init__(self, text_preview: str, reason: str):
        # Truncate text preview for error message
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        super().__init__(
            f"Failed to generate embedding: {reason}", {"text_preview": preview, "reason": reason}
        )
        self.reason = reason


class MemoryOperationError(MemoryError):
    """Raised when memory tier operations fail."""

    def __init__(self, tier: str, operation: str, reason: str):
        super().__init__(
            f"Memory {tier} {operation} failed: {reason}",
            {"tier": tier, "operation": operation, "reason": reason},
        )
        self.tier = tier
        self.operation = operation
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
            {"plugin_name": plugin_name, "reason": reason},
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
            {"limit": limit, "window_seconds": window_seconds},
        )
        self.limit = limit
        self.window_seconds = window_seconds


class OAuthStateError(AuthError):
    """Raised when OAuth state validation fails."""

    def __init__(self, reason: str):
        super().__init__(f"OAuth state validation failed: {reason}", {"reason": reason})
        self.reason = reason


# ============================================================================
# Billing/Budget Errors
# ============================================================================


class BillingError(AragoraError):
    """Base exception for billing-related errors."""

    pass


class BudgetExceededError(BillingError):
    """Raised when organization budget is exceeded and hard-stop is enforced."""

    def __init__(self, message: str, org_id: str = "", remaining_usd: float = 0.0):
        super().__init__(message, {"org_id": org_id, "remaining_usd": remaining_usd})
        self.org_id = org_id
        self.remaining_usd = remaining_usd


class InsufficientCreditsError(BillingError):
    """Raised when organization has insufficient credits for an operation."""

    def __init__(self, required: float, available: float, org_id: str = ""):
        super().__init__(
            f"Insufficient credits: required {required:.2f}, available {available:.2f}",
            {"required": required, "available": available, "org_id": org_id},
        )
        self.required = required
        self.available = available
        self.org_id = org_id


# ============================================================================
# Infrastructure Errors
# ============================================================================


class InfrastructureError(AragoraError):
    """Base exception for infrastructure-related errors."""

    pass


class RedisUnavailableError(InfrastructureError):
    """Raised when Redis is not available."""

    def __init__(self, operation: str | None = None):
        msg = "Redis not available"
        if operation:
            msg = f"Redis not available for {operation}"
        super().__init__(msg, {"operation": operation})
        self.operation = operation


class ExternalServiceError(InfrastructureError):
    """Raised when an external service call fails."""

    def __init__(self, service: str, reason: str, status_code: int | None = None):
        super().__init__(
            f"External service '{service}' failed: {reason}",
            {"service": service, "reason": reason, "status_code": status_code},
        )
        self.service = service
        self.reason = reason
        self.status_code = status_code


class CircuitBreakerError(InfrastructureError):
    """Raised when circuit breaker operations fail."""

    def __init__(self, service: str, state: str, reason: str):
        super().__init__(
            f"Circuit breaker for {service} ({state}): {reason}",
            {"service": service, "state": state, "reason": reason},
        )
        self.service = service
        self.state = state
        self.reason = reason


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
            {"cycle": cycle, "phase": phase, "reason": reason},
        )
        self.cycle = cycle
        self.phase = phase
        self.reason = reason


class NomicStateError(NomicError):
    """Raised when Nomic state is invalid or corrupted.

    Examples:
        - FileNotFoundError during state load
        - json.JSONDecodeError for corrupted state
        - PermissionError for state file access
    """

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message, {"path": path} if path else None)
        self.path = path


class NomicInitError(NomicError):
    """Raised when a Nomic component fails to initialize.

    The loop can continue with degraded functionality when these occur.

    Components that may raise this:
    - CircuitBreaker restore
    - Constitution loader
    - OutcomeTracker
    - WebhookDispatcher
    - InsightStore
    - FlipDetector
    """

    def __init__(self, component: str, reason: str, recoverable: bool = True):
        super().__init__(
            f"Failed to initialize {component}: {reason}",
            {"component": component, "reason": reason, "recoverable": recoverable},
        )
        self.component = component
        self.reason = reason
        self.recoverable = recoverable


class NomicMemoryError(NomicError):
    """Raised when Nomic memory operations fail.

    Operations include:
    - ContinuumMemory read/write
    - ConsensusMemory operations
    - InsightStore queries
    - PositionLedger operations
    """

    def __init__(self, operation: str, reason: str, tier: str | None = None):
        details = {"operation": operation, "reason": reason}
        if tier:
            details["tier"] = tier
        super().__init__(f"Memory {operation} failed: {reason}", details)
        self.operation = operation
        self.reason = reason
        self.tier = tier


class NomicAgentError(NomicError):
    """Raised when Nomic agent operations fail.

    Operations include:
    - Agent health checks
    - Agent probing
    - Team selection
    - Persona injection
    """

    def __init__(self, agent_name: str, operation: str, reason: str):
        super().__init__(
            f"Agent '{agent_name}' {operation} failed: {reason}",
            {"agent_name": agent_name, "operation": operation, "reason": reason},
        )
        self.agent_name = agent_name
        self.operation = operation
        self.reason = reason


class NomicPhaseError(NomicError):
    """Raised when a Nomic loop phase fails.

    Phases: context, debate, design, implement, verify, commit
    """

    def __init__(self, phase: str, reason: str, stage: str | None = None, recoverable: bool = True):
        details = {"phase": phase, "reason": reason, "recoverable": recoverable}
        if stage:
            details["stage"] = stage
        super().__init__(f"Phase '{phase}' failed: {reason}", details)
        self.phase = phase
        self.reason = reason
        self.stage = stage
        self.recoverable = recoverable


class NomicIntegrationError(NomicError):
    """Raised when external integrations fail.

    Integrations include:
    - Webhook dispatch
    - Supabase persistence
    - Stream event emission
    - Prometheus metrics
    """

    def __init__(self, integration: str, reason: str):
        super().__init__(
            f"Integration '{integration}' failed: {reason}",
            {"integration": integration, "reason": reason},
        )
        self.integration = integration
        self.reason = reason


class NomicAnalyticsError(NomicError):
    """Raised when analytics/tracking operations fail.

    Non-critical - analytics can be skipped without breaking the loop.

    Trackers include:
    - ELO rating updates
    - Calibration recording
    - Consensus storage
    - Risk tracking
    """

    def __init__(self, tracker: str, reason: str):
        super().__init__(
            f"Analytics tracker '{tracker}' failed: {reason}",
            {"tracker": tracker, "reason": reason},
        )
        self.tracker = tracker
        self.reason = reason


class NomicVerificationError(NomicError):
    """Raised when verification operations fail.

    Includes:
    - Syntax checking
    - Protected file verification
    - Formal verification (Z3/Lean)
    - Provenance chain validation
    """

    def __init__(self, check_type: str, reason: str, file_path: str | None = None):
        details = {"check_type": check_type, "reason": reason}
        if file_path:
            details["file_path"] = file_path
        super().__init__(f"Verification '{check_type}' failed: {reason}", details)
        self.check_type = check_type
        self.reason = reason
        self.file_path = file_path


class NomicTimeoutError(NomicError):
    """Raised when Nomic operations exceed time limits.

    Includes:
    - Phase timeouts
    - Agent call timeouts
    - Debate round timeouts
    """

    def __init__(self, operation: str, timeout_seconds: float):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            {"operation": operation, "timeout_seconds": timeout_seconds},
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# ============================================================================
# Checkpoint Errors
# ============================================================================


class CheckpointError(AragoraError):
    """Base exception for checkpoint operations."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""

    def __init__(self, checkpoint_id: str):
        super().__init__(f"Checkpoint not found: {checkpoint_id}", {"checkpoint_id": checkpoint_id})
        self.checkpoint_id = checkpoint_id


class CheckpointCorruptedError(CheckpointError):
    """Raised when checkpoint data is corrupted."""

    def __init__(self, checkpoint_id: str, reason: str):
        super().__init__(
            f"Checkpoint corrupted: {checkpoint_id} - {reason}",
            {"checkpoint_id": checkpoint_id, "reason": reason},
        )
        self.checkpoint_id = checkpoint_id
        self.reason = reason


class CheckpointSaveError(CheckpointError):
    """Raised when saving a checkpoint fails."""

    def __init__(self, checkpoint_id: str, reason: str):
        super().__init__(
            f"Failed to save checkpoint {checkpoint_id}: {reason}",
            {"checkpoint_id": checkpoint_id, "reason": reason},
        )
        self.checkpoint_id = checkpoint_id
        self.reason = reason


# ============================================================================
# Convergence Errors
# ============================================================================


class ConvergenceError(AragoraError):
    """Base exception for convergence detection errors."""

    pass


class ConvergenceBackendError(ConvergenceError):
    """Raised when a convergence backend fails."""

    def __init__(self, backend_name: str, reason: str):
        super().__init__(
            f"Convergence backend '{backend_name}' failed: {reason}",
            {"backend_name": backend_name, "reason": reason},
        )
        self.backend_name = backend_name
        self.reason = reason


class ConvergenceThresholdError(ConvergenceError):
    """Raised when convergence threshold is invalid."""

    def __init__(self, threshold: float, reason: str):
        super().__init__(
            f"Invalid convergence threshold {threshold}: {reason}",
            {"threshold": threshold, "reason": reason},
        )
        self.threshold = threshold
        self.reason = reason


# ============================================================================
# Cache Errors
# ============================================================================


class CacheError(AragoraError):
    """Base exception for cache operations."""

    pass


class CacheKeyError(CacheError):
    """Raised when a cache key is invalid."""

    def __init__(self, key: str, reason: str):
        super().__init__(f"Invalid cache key '{key}': {reason}", {"key": key, "reason": reason})
        self.key = key
        self.reason = reason


class CacheCapacityError(CacheError):
    """Raised when cache capacity is exceeded."""

    def __init__(self, current_size: int, max_size: int):
        super().__init__(
            f"Cache capacity exceeded: {current_size}/{max_size}",
            {"current_size": current_size, "max_size": max_size},
        )
        self.current_size = current_size
        self.max_size = max_size


# ============================================================================
# Streaming Errors
# ============================================================================


class StreamingError(AragoraError):
    """Base exception for streaming-related errors."""

    pass


class WebSocketError(StreamingError):
    """Raised when WebSocket connection fails."""

    def __init__(self, reason: str, code: int | None = None):
        super().__init__(f"WebSocket error: {reason}", {"reason": reason, "code": code})
        self.reason = reason
        self.code = code


class StreamConnectionError(StreamingError):
    """Raised when stream connection is lost."""

    def __init__(self, stream_id: str, reason: str):
        super().__init__(
            f"Stream connection lost for {stream_id}: {reason}",
            {"stream_id": stream_id, "reason": reason},
        )
        self.stream_id = stream_id
        self.reason = reason


class StreamTimeoutError(StreamingError):
    """Raised when stream operation times out."""

    def __init__(self, stream_id: str, timeout_seconds: float):
        super().__init__(
            f"Stream {stream_id} timed out after {timeout_seconds}s",
            {"stream_id": stream_id, "timeout_seconds": timeout_seconds},
        )
        self.stream_id = stream_id
        self.timeout_seconds = timeout_seconds


# ============================================================================
# Evidence Errors
# ============================================================================


class EvidenceError(AragoraError):
    """Base exception for evidence-related errors."""

    pass


class EvidenceParseError(EvidenceError):
    """Raised when evidence parsing fails."""

    def __init__(self, source: str, reason: str):
        super().__init__(
            f"Failed to parse evidence from {source}: {reason}",
            {"source": source, "reason": reason},
        )
        self.source = source
        self.reason = reason


class EvidenceNotFoundError(EvidenceError):
    """Raised when requested evidence cannot be found."""

    def __init__(self, evidence_id: str):
        super().__init__(f"Evidence not found: {evidence_id}", {"evidence_id": evidence_id})
        self.evidence_id = evidence_id


# ============================================================================
# Verification Errors
# ============================================================================


class VerificationError(AragoraError):
    """Base exception for formal verification errors."""

    pass


class Z3NotAvailableError(VerificationError):
    """Raised when Z3 solver is not available."""

    def __init__(self) -> None:
        super().__init__("Z3 solver not available. Install with: pip install z3-solver")


class VerificationTimeoutError(VerificationError):
    """Raised when verification times out."""

    def __init__(self, timeout_ms: int):
        super().__init__(f"Verification timed out after {timeout_ms}ms", {"timeout_ms": timeout_ms})
        self.timeout_ms = timeout_ms


# ============================================================================
# Notification Errors
# ============================================================================


class NotificationError(AragoraError):
    """Base exception for notification delivery errors."""

    pass


class SlackNotificationError(NotificationError):
    """Raised when Slack webhook or API call fails."""

    def __init__(self, message: str, status_code: int | None = None, error_code: str | None = None):
        details: dict[str, int | str] = {}
        if status_code is not None:
            details["status_code"] = status_code
        if error_code is not None:
            details["error_code"] = error_code
        super().__init__(message, details)
        self.status_code = status_code
        self.error_code = error_code


class WebhookDeliveryError(NotificationError):
    """Raised when a generic webhook delivery fails."""

    def __init__(self, webhook_url: str, status_code: int, message: str):
        super().__init__(
            f"Webhook delivery failed: {message}",
            {"webhook_url": webhook_url, "status_code": status_code},
        )
        self.webhook_url = webhook_url
        self.status_code = status_code


# ============================================================================
# Document Processing Errors
# ============================================================================


class DocumentProcessingError(AragoraError):
    """Base exception for document processing errors."""

    pass


class DocumentParseError(DocumentProcessingError):
    """Raised when document parsing fails."""

    def __init__(
        self, document_id: str | None, reason: str, original_error: Exception | None = None
    ):
        details = {"reason": reason}
        if document_id:
            details["document_id"] = document_id
        super().__init__(f"Failed to parse document: {reason}", details)
        self.document_id = document_id
        self.reason = reason
        self.original_error = original_error


class DocumentChunkError(DocumentProcessingError):
    """Raised when document chunking fails."""

    def __init__(
        self, document_id: str | None, reason: str, original_error: Exception | None = None
    ):
        details = {"reason": reason}
        if document_id:
            details["document_id"] = document_id
        super().__init__(f"Failed to chunk document: {reason}", details)
        self.document_id = document_id
        self.reason = reason
        self.original_error = original_error


# ============================================================================
# Connector Errors - Usage Guide
# ============================================================================
# For connector-specific errors (HTTP connectors, API connectors), import from
# aragora.connectors.exceptions which provides the full exception hierarchy with:
#   - is_retryable flag for retry decisions
#   - retry_after for backoff timing
#   - connector_name for error attribution
#   - classify_exception() utility for converting generic exceptions
#   - connector_error_handler context manager
#
# Example:
#   from aragora.connectors.exceptions import (
#       ConnectorError, ConnectorTimeoutError, ConnectorRateLimitError,
#       classify_exception, connector_error_handler,
#   )
#
#   # Using context manager
#   async with connector_error_handler("github"):
#       response = await client.get(url)
#
# Available exceptions:
#   ConnectorError (base), ConnectorAuthError, ConnectorRateLimitError,
#   ConnectorTimeoutError, ConnectorNetworkError, ConnectorAPIError,
#   ConnectorValidationError, ConnectorNotFoundError, ConnectorQuotaError,
#   ConnectorParseError


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


# ============================================================================
# Unified Exception Hierarchy
# ============================================================================
# All domain-specific exception base classes inherit from AragoraError:
#
#   AragoraError (base)
#   ├── DebateError, ValidationError, StorageError, MemoryError, etc. (this file)
#   ├── ConnectorError (aragora.connectors.exceptions)
#   │   └── ConnectorAPIError, ConnectorTimeoutError, ConnectorRateLimitError, ...
#   ├── ControlPlaneError (aragora.control_plane.exceptions)
#   │   └── TaskNotFoundError, PolicyConflictError, ResourceQuotaExceededError, ...
#   ├── AgentError (aragora.agents.errors.exceptions)
#   │   └── AgentTimeoutError, AgentRateLimitError, CLIAgentError, ...
#   └── HandlerError (aragora.server.handlers.exceptions)
#       └── HandlerNotFoundError, HandlerValidationError, HandlerAuthorizationError, ...
#
# This enables catching all Aragora exceptions with a single handler:
#   try:
#       await some_operation()
#   except AragoraError as e:
#       logger.error(f"Operation failed: {e}", extra=e.details)
#
# Re-exports for unified imports (import from this module for convenience):

# Connector exceptions (re-exports for unified imports)
try:
    from aragora.connectors.exceptions import (  # noqa: F401
        ConnectorAPIError,
        ConnectorAuthError,
        ConnectorConfigError,
        ConnectorError,
        ConnectorNetworkError,
        ConnectorNotFoundError,
        ConnectorParseError,
        ConnectorQuotaError,
        ConnectorRateLimitError,
        ConnectorTimeoutError,
        ConnectorValidationError,
    )
except ImportError:
    pass  # Optional: connectors module may not be installed

# Control plane exceptions (re-exports for unified imports)
try:
    from aragora.control_plane.exceptions import (  # noqa: F401
        AgentOverloadedError,
        AgentUnavailableError,
        ControlPlaneError,
        InvalidTaskStateError,
        PolicyConflictError,
        PolicyEvaluationError,
        PolicyNotFoundError,
        RateLimitExceededError as ControlPlaneRateLimitError,
        RegionRoutingError,
        RegionUnavailableError,
        ResourceQuotaExceededError,
        SchedulerConnectionError,
        TaskClaimError,
        TaskNotFoundError,
        TaskTimeoutError,
    )
except ImportError:
    pass  # Optional: control_plane module may not be installed

# Handler exceptions (re-exports for unified imports)
try:
    from aragora.server.handlers.exceptions import (  # noqa: F401
        HandlerAuthorizationError,
        HandlerConflictError,
        HandlerDatabaseError,
        HandlerError,
        HandlerExternalServiceError,
        HandlerNotFoundError,
        HandlerRateLimitError,
        HandlerValidationError,
    )
except ImportError:
    pass  # Optional: server module may not be installed
