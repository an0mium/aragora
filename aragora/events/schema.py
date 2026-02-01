"""
Event Schema Validation.

Provides type-safe schema definitions for StreamEvent.data payloads,
enabling validation, documentation, and safer event handling.

The StreamEvent.data field is an untyped dict by design (flexibility),
but this module adds optional validation for events that need it.

Features:
- Type-safe payload definitions using dataclasses
- Schema registry for event type -> payload mapping
- Validation with detailed error messages
- Coercion for common type mismatches
- Schema documentation generation

Usage:
    from aragora.events.schema import (
        EventSchemaRegistry,
        validate_event,
        DebateStartPayload,
    )

    # Validate an event
    errors = validate_event(event)
    if errors:
        logger.warning("Invalid event: %s", errors)

    # Create typed payload
    payload = DebateStartPayload(
        debate_id="d-123",
        question="What is the best approach?",
        agents=["claude", "gpt-4"],
    )
    event = StreamEvent(type=StreamEventType.DEBATE_START, data=payload.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

from aragora.events.types import StreamEventType

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@dataclass
class EventPayload:
    """Base class for typed event payloads."""

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create payload from dictionary with validation."""
        # Filter to only known fields
        known_fields = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class ValidationError:
    """A validation error for an event payload."""

    field: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.field}: {self.message} (got {type(self.value).__name__}: {self.value!r})"
        return f"{self.field}: {self.message}"


# ---------------------------------------------------------------------------
# Debate Event Payloads
# ---------------------------------------------------------------------------


@dataclass
class DebateStartPayload(EventPayload):
    """Payload for DEBATE_START events."""

    debate_id: str
    question: str
    agents: list[str] = field(default_factory=list)
    domain: str = ""
    rounds: int = 3
    consensus_type: str = "majority"
    user_id: str | None = None
    org_id: str | None = None


@dataclass
class DebateEndPayload(EventPayload):
    """Payload for DEBATE_END events."""

    debate_id: str
    consensus_reached: bool = False
    final_answer: str | None = None
    confidence: float = 0.0
    rounds_used: int = 0
    duration_seconds: float = 0.0
    winner_agent: str | None = None


@dataclass
class RoundStartPayload(EventPayload):
    """Payload for ROUND_START events."""

    debate_id: str
    round_number: int
    phase: str = "proposal"


@dataclass
class AgentMessagePayload(EventPayload):
    """Payload for AGENT_MESSAGE events."""

    debate_id: str
    agent: str
    message: str
    round_number: int = 0
    role: str = "participant"
    confidence: float | None = None
    citations: list[str] = field(default_factory=list)


@dataclass
class CritiquePayload(EventPayload):
    """Payload for CRITIQUE events."""

    debate_id: str
    critic_agent: str
    target_agent: str
    critique: str
    round_number: int = 0
    severity: str = "medium"  # low, medium, high


@dataclass
class VotePayload(EventPayload):
    """Payload for VOTE events."""

    debate_id: str
    voter_agent: str
    voted_for: str
    confidence: float = 0.0
    reason: str | None = None


@dataclass
class ConsensusPayload(EventPayload):
    """Payload for CONSENSUS events."""

    debate_id: str
    consensus_reached: bool
    answer: str | None = None
    confidence: float = 0.0
    supporting_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token Streaming Payloads
# ---------------------------------------------------------------------------


@dataclass
class TokenStartPayload(EventPayload):
    """Payload for TOKEN_START events."""

    debate_id: str
    agent: str
    task_id: str = ""


@dataclass
class TokenDeltaPayload(EventPayload):
    """Payload for TOKEN_DELTA events."""

    debate_id: str
    agent: str
    delta: str
    task_id: str = ""


@dataclass
class TokenEndPayload(EventPayload):
    """Payload for TOKEN_END events."""

    debate_id: str
    agent: str
    task_id: str = ""
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Knowledge Mound Payloads
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeIndexedPayload(EventPayload):
    """Payload for KNOWLEDGE_INDEXED events."""

    document_id: str
    chunk_count: int = 0
    source: str = ""
    tenant_id: str | None = None


@dataclass
class KnowledgeQueriedPayload(EventPayload):
    """Payload for KNOWLEDGE_QUERIED events."""

    query: str
    result_count: int = 0
    latency_ms: float = 0.0
    tenant_id: str | None = None


@dataclass
class KnowledgeStalePayload(EventPayload):
    """Payload for KNOWLEDGE_STALE events."""

    document_id: str
    staleness_score: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Error Event Payloads
# ---------------------------------------------------------------------------


@dataclass
class ErrorPayload(EventPayload):
    """Payload for ERROR events."""

    error_type: str
    message: str
    debate_id: str | None = None
    agent: str | None = None
    trace_id: str | None = None
    recoverable: bool = True


# ---------------------------------------------------------------------------
# Agent Event Payloads
# ---------------------------------------------------------------------------


@dataclass
class AgentEloUpdatedPayload(EventPayload):
    """Payload for AGENT_ELO_UPDATED events."""

    agent: str
    old_elo: float
    new_elo: float
    change: float
    debate_id: str | None = None
    domain: str = ""


@dataclass
class AgentFallbackTriggeredPayload(EventPayload):
    """Payload for AGENT_FALLBACK_TRIGGERED events."""

    agent: str
    fallback_to: str
    reason: str
    debate_id: str | None = None


# ---------------------------------------------------------------------------
# Schema Registry
# ---------------------------------------------------------------------------


# Mapping of event types to their payload classes
EVENT_SCHEMAS: dict[StreamEventType, type[EventPayload]] = {
    StreamEventType.DEBATE_START: DebateStartPayload,
    StreamEventType.DEBATE_END: DebateEndPayload,
    StreamEventType.ROUND_START: RoundStartPayload,
    StreamEventType.AGENT_MESSAGE: AgentMessagePayload,
    StreamEventType.CRITIQUE: CritiquePayload,
    StreamEventType.VOTE: VotePayload,
    StreamEventType.CONSENSUS: ConsensusPayload,
    StreamEventType.TOKEN_START: TokenStartPayload,
    StreamEventType.TOKEN_DELTA: TokenDeltaPayload,
    StreamEventType.TOKEN_END: TokenEndPayload,
    StreamEventType.KNOWLEDGE_INDEXED: KnowledgeIndexedPayload,
    StreamEventType.KNOWLEDGE_QUERIED: KnowledgeQueriedPayload,
    StreamEventType.KNOWLEDGE_STALE: KnowledgeStalePayload,
    StreamEventType.ERROR: ErrorPayload,
    StreamEventType.AGENT_ELO_UPDATED: AgentEloUpdatedPayload,
    StreamEventType.AGENT_FALLBACK_TRIGGERED: AgentFallbackTriggeredPayload,
}

# Required fields for each payload type
REQUIRED_FIELDS: dict[type[EventPayload], list[str]] = {
    DebateStartPayload: ["debate_id", "question"],
    DebateEndPayload: ["debate_id"],
    RoundStartPayload: ["debate_id", "round_number"],
    AgentMessagePayload: ["debate_id", "agent", "message"],
    CritiquePayload: ["debate_id", "critic_agent", "target_agent", "critique"],
    VotePayload: ["debate_id", "voter_agent", "voted_for"],
    ConsensusPayload: ["debate_id", "consensus_reached"],
    TokenStartPayload: ["debate_id", "agent"],
    TokenDeltaPayload: ["debate_id", "agent", "delta"],
    TokenEndPayload: ["debate_id", "agent"],
    KnowledgeIndexedPayload: ["document_id"],
    KnowledgeQueriedPayload: ["query"],
    KnowledgeStalePayload: ["document_id"],
    ErrorPayload: ["error_type", "message"],
    AgentEloUpdatedPayload: ["agent", "old_elo", "new_elo", "change"],
    AgentFallbackTriggeredPayload: ["agent", "fallback_to", "reason"],
}


class EventSchemaRegistry:
    """Registry for event schemas with validation support."""

    def __init__(self, strict: bool = False):
        """Initialize the schema registry.

        Args:
            strict: If True, unknown event types cause validation errors
        """
        self._schemas = dict(EVENT_SCHEMAS)
        self._strict = strict

    def register(self, event_type: StreamEventType, payload_class: type[EventPayload]) -> None:
        """Register a schema for an event type.

        Args:
            event_type: The event type
            payload_class: The payload dataclass
        """
        self._schemas[event_type] = payload_class

    def get_schema(self, event_type: StreamEventType) -> type[EventPayload] | None:
        """Get the schema for an event type."""
        return self._schemas.get(event_type)

    def validate(
        self,
        event_type: StreamEventType,
        data: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate event data against its schema.

        Args:
            event_type: The event type
            data: The event data to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[ValidationError] = []

        schema = self._schemas.get(event_type)
        if schema is None:
            if self._strict:
                errors.append(
                    ValidationError(
                        field="_type",
                        message=f"No schema registered for event type: {event_type.value}",
                    )
                )
            return errors

        # Check required fields
        required = REQUIRED_FIELDS.get(schema, [])
        for field_name in required:
            if field_name not in data or data[field_name] is None:
                errors.append(
                    ValidationError(
                        field=field_name,
                        message="Required field is missing",
                    )
                )

        # Check field types
        for field_info in schema.__dataclass_fields__.values():
            if field_info.name not in data:
                continue

            value = data[field_info.name]
            expected_type = field_info.type

            # Handle optional types
            if hasattr(expected_type, "__origin__"):
                # Handle Optional[T] and list[T]
                origin = getattr(expected_type, "__origin__", None)
                if origin is list and not isinstance(value, list):
                    errors.append(
                        ValidationError(
                            field=field_info.name,
                            message="Expected list",
                            value=value,
                        )
                    )
                continue

            # Basic type checking
            if expected_type is str and not isinstance(value, str):
                errors.append(
                    ValidationError(
                        field=field_info.name,
                        message="Expected string",
                        value=value,
                    )
                )
            elif expected_type is int and not isinstance(value, (int, float)):
                errors.append(
                    ValidationError(
                        field=field_info.name,
                        message="Expected integer",
                        value=value,
                    )
                )
            elif expected_type is float and not isinstance(value, (int, float)):
                errors.append(
                    ValidationError(
                        field=field_info.name,
                        message="Expected number",
                        value=value,
                    )
                )
            elif expected_type is bool and not isinstance(value, bool):
                errors.append(
                    ValidationError(
                        field=field_info.name,
                        message="Expected boolean",
                        value=value,
                    )
                )

        return errors

    def coerce(
        self,
        event_type: StreamEventType,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Coerce event data to match schema types.

        Attempts to convert values to expected types (e.g., string "3" to int 3).

        Args:
            event_type: The event type
            data: The event data to coerce

        Returns:
            Coerced data (new dict, original unchanged)
        """
        schema = self._schemas.get(event_type)
        if schema is None:
            return dict(data)

        result = dict(data)

        for field_info in schema.__dataclass_fields__.values():
            if field_info.name not in result:
                continue

            value = result[field_info.name]
            expected_type = field_info.type

            try:
                if expected_type is int and isinstance(value, str):
                    result[field_info.name] = int(value)
                elif expected_type is float and isinstance(value, str):
                    result[field_info.name] = float(value)
                elif expected_type is bool and isinstance(value, str):
                    result[field_info.name] = value.lower() in ("true", "1", "yes")
                elif expected_type is str and not isinstance(value, str):
                    result[field_info.name] = str(value)
            except (ValueError, TypeError):
                pass  # Keep original value if coercion fails

        return result

    def get_all_schemas(self) -> dict[StreamEventType, type[EventPayload]]:
        """Get all registered schemas."""
        return dict(self._schemas)


# ---------------------------------------------------------------------------
# Global instance and convenience functions
# ---------------------------------------------------------------------------

_registry: EventSchemaRegistry | None = None


def get_schema_registry() -> EventSchemaRegistry:
    """Get or create the global schema registry."""
    global _registry
    if _registry is None:
        _registry = EventSchemaRegistry()
    return _registry


def reset_schema_registry() -> None:
    """Reset the global schema registry (for testing)."""
    global _registry
    _registry = None


def validate_event(
    event_type: StreamEventType,
    data: dict[str, Any],
) -> list[ValidationError]:
    """Validate event data against its schema.

    Convenience function for validation.

    Args:
        event_type: The event type
        data: The event data

    Returns:
        List of validation errors (empty if valid)
    """
    return get_schema_registry().validate(event_type, data)


def validate_event_log_errors(
    event_type: StreamEventType,
    data: dict[str, Any],
) -> bool:
    """Validate event and log any errors.

    Args:
        event_type: The event type
        data: The event data

    Returns:
        True if valid, False otherwise
    """
    errors = validate_event(event_type, data)
    if errors:
        logger.warning(
            "Event validation failed for %s: %s",
            event_type.value,
            "; ".join(str(e) for e in errors),
        )
        return False
    return True


__all__ = [
    # Payloads
    "AgentEloUpdatedPayload",
    "AgentFallbackTriggeredPayload",
    "AgentMessagePayload",
    "ConsensusPayload",
    "CritiquePayload",
    "DebateEndPayload",
    "DebateStartPayload",
    "ErrorPayload",
    "EventPayload",
    "EventSchemaRegistry",
    "KnowledgeIndexedPayload",
    "KnowledgeQueriedPayload",
    "KnowledgeStalePayload",
    "RoundStartPayload",
    "TokenDeltaPayload",
    "TokenEndPayload",
    "TokenStartPayload",
    "ValidationError",
    "VotePayload",
    # Functions
    "get_schema_registry",
    "reset_schema_registry",
    "validate_event",
    "validate_event_log_errors",
]
