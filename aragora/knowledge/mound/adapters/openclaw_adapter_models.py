"""
OpenClaw adapter data models.

Dataclasses and enums used by the OpenClawAdapter for representing
actions, patterns, sync results, and reverse flow state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ActionStatus(str, Enum):
    """Status of an OpenClaw action execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PatternType(str, Enum):
    """Types of patterns extracted from OpenClaw actions."""

    SUCCESS_PATTERN = "success"
    FAILURE_PATTERN = "failure"
    TIMEOUT_PATTERN = "timeout"
    RESOURCE_PATTERN = "resource"
    CAPABILITY_PATTERN = "capability"
    WORKFLOW_PATTERN = "workflow"


@dataclass
class OpenClawKnowledgeItem:
    """Knowledge item representing an OpenClaw action and its outcome.

    Captures the essential information from an OpenClaw action execution
    that should be stored in the Knowledge Mound for future reference
    and pattern extraction.

    Attributes:
        action_id: Unique identifier for the OpenClaw action.
        result: Outcome of the action (success, failure, etc.).
        context: Session and execution context from OpenClaw.
        debate_id: Associated debate ID if action was triggered by debate.
        workspace_id: Workspace where the action was executed.
        tenant_id: Tenant identifier for multi-tenant isolation.
        capabilities_used: List of capabilities used in the action.
        execution_time_ms: Execution duration in milliseconds.
        output: Action output or result content.
        error: Error message if action failed.
        metadata: Additional metadata from OpenClaw.
        created_at: Timestamp when the action was executed.
    """

    action_id: str
    result: ActionStatus
    context: dict[str, Any]
    debate_id: str | None = None
    workspace_id: str = "default"
    tenant_id: str = "default"
    capabilities_used: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "result": self.result.value,
            "context": self.context,
            "debate_id": self.debate_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "capabilities_used": self.capabilities_used,
            "execution_time_ms": self.execution_time_ms,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenClawKnowledgeItem:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            action_id=data["action_id"],
            result=ActionStatus(data.get("result", "pending")),
            context=data.get("context", {}),
            debate_id=data.get("debate_id"),
            workspace_id=data.get("workspace_id", "default"),
            tenant_id=data.get("tenant_id", "default"),
            capabilities_used=data.get("capabilities_used", []),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            output=data.get("output", ""),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


@dataclass
class ActionPattern:
    """Pattern extracted from OpenClaw action executions.

    Represents a learned pattern from analyzing multiple action outcomes,
    used to improve future action planning and execution.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        pattern_type: Type of pattern (success, failure, etc.).
        description: Human-readable description of the pattern.
        success_rate: Observed success rate for this pattern (0.0-1.0).
        observation_count: Number of times this pattern was observed.
        capabilities_involved: Capabilities associated with this pattern.
        context_signature: Characteristic context features.
        recommendation: Action recommendation based on this pattern.
        confidence: Confidence in this pattern (0.0-1.0).
        first_observed_at: When this pattern was first detected.
        last_observed_at: When this pattern was last observed.
        contributing_actions: Action IDs that contributed to this pattern.
        metadata: Additional pattern metadata.
    """

    pattern_id: str
    pattern_type: PatternType
    description: str
    success_rate: float = 0.0
    observation_count: int = 0
    capabilities_involved: list[str] = field(default_factory=list)
    context_signature: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    confidence: float = 0.5
    first_observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contributing_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "success_rate": self.success_rate,
            "observation_count": self.observation_count,
            "capabilities_involved": self.capabilities_involved,
            "context_signature": self.context_signature,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "first_observed_at": self.first_observed_at.isoformat(),
            "last_observed_at": self.last_observed_at.isoformat(),
            "contributing_actions": self.contributing_actions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionPattern:
        """Create from dictionary."""
        first_observed = data.get("first_observed_at")
        if isinstance(first_observed, str):
            first_observed = datetime.fromisoformat(first_observed.replace("Z", "+00:00"))
        elif first_observed is None:
            first_observed = datetime.now(timezone.utc)

        last_observed = data.get("last_observed_at")
        if isinstance(last_observed, str):
            last_observed = datetime.fromisoformat(last_observed.replace("Z", "+00:00"))
        elif last_observed is None:
            last_observed = datetime.now(timezone.utc)

        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data.get("pattern_type", "success")),
            description=data.get("description", ""),
            success_rate=data.get("success_rate", 0.0),
            observation_count=data.get("observation_count", 0),
            capabilities_involved=data.get("capabilities_involved", []),
            context_signature=data.get("context_signature", {}),
            recommendation=data.get("recommendation", ""),
            confidence=data.get("confidence", 0.5),
            first_observed_at=first_observed,
            last_observed_at=last_observed,
            contributing_actions=data.get("contributing_actions", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SyncResult:
    """Result of a sync operation between OpenClaw and Knowledge Mound.

    Attributes:
        items_synced: Number of items successfully synced.
        items_skipped: Number of items skipped (e.g., duplicates).
        items_failed: Number of items that failed to sync.
        errors: List of error messages.
        duration_ms: Duration of the sync operation in milliseconds.
        direction: Sync direction ('forward' or 'reverse').
        metadata: Additional sync metadata.
    """

    items_synced: int = 0
    items_skipped: int = 0
    items_failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    direction: str = "forward"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items_synced": self.items_synced,
            "items_skipped": self.items_skipped,
            "items_failed": self.items_failed,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "direction": self.direction,
            "metadata": self.metadata,
        }


# =============================================================================
# Reverse Flow Dataclasses (KM -> OpenClaw)
# =============================================================================


@dataclass
class KMContextUpdate:
    """Context update to push from KM to OpenClaw.

    Represents insights and knowledge from the KM that should be
    pushed to OpenClaw to improve action planning and execution.

    Attributes:
        update_id: Unique identifier for this update.
        context_type: Type of context being updated.
        content: The actual context content.
        relevance_score: How relevant this context is (0.0-1.0).
        source_debate_id: Source debate if applicable.
        source_items: KM item IDs that contributed to this update.
        priority: Priority for applying this update.
        expires_at: When this context update expires.
        metadata: Additional update metadata.
    """

    update_id: str
    context_type: str
    content: dict[str, Any]
    relevance_score: float = 0.5
    source_debate_id: str | None = None
    source_items: list[str] = field(default_factory=list)
    priority: float = 0.5
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "update_id": self.update_id,
            "context_type": self.context_type,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "source_debate_id": self.source_debate_id,
            "source_items": self.source_items,
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class TaskPrioritizationUpdate:
    """Task prioritization update from debate decisions.

    Represents a debate decision that should influence OpenClaw's
    task prioritization and execution ordering.

    Attributes:
        task_id: OpenClaw task identifier.
        debate_id: Debate that produced this prioritization.
        original_priority: Original priority before update.
        new_priority: Updated priority based on debate.
        reason: Reason for the priority change.
        confidence: Confidence in this prioritization (0.0-1.0).
        applied: Whether this update was applied.
        applied_at: When the update was applied.
        metadata: Additional prioritization metadata.
    """

    task_id: str
    debate_id: str
    original_priority: float
    new_priority: float
    reason: str = ""
    confidence: float = 0.7
    applied: bool = False
    applied_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "debate_id": self.debate_id,
            "original_priority": self.original_priority,
            "new_priority": self.new_priority,
            "reason": self.reason,
            "confidence": self.confidence,
            "applied": self.applied,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "metadata": self.metadata,
        }


@dataclass
class KMValidationResult:
    """Result of validating OpenClaw actions against KM knowledge.

    Represents feedback from KM that can improve OpenClaw's
    action planning and execution strategies.

    Attributes:
        action_id: The validated action ID.
        km_confidence: KM's confidence in the action outcome (0.0-1.0).
        cross_debate_utility: Utility across multiple debates (0.0-1.0).
        validation_count: Number of validations performed.
        was_contradicted: Whether KM found contradicting evidence.
        was_supported: Whether KM found supporting evidence.
        recommendation: Recommendation for future actions.
        patterns_matched: Patterns that matched this action.
        metadata: Additional validation metadata.
    """

    action_id: str
    km_confidence: float = 0.5
    cross_debate_utility: float = 0.0
    validation_count: int = 1
    was_contradicted: bool = False
    was_supported: bool = False
    recommendation: str = "keep"  # keep, boost, demote, review
    patterns_matched: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "km_confidence": self.km_confidence,
            "cross_debate_utility": self.cross_debate_utility,
            "validation_count": self.validation_count,
            "was_contradicted": self.was_contradicted,
            "was_supported": self.was_supported,
            "recommendation": self.recommendation,
            "patterns_matched": self.patterns_matched,
            "metadata": self.metadata,
        }


@dataclass
class OpenClawKMSyncResult:
    """Result of batch sync from KM validations to OpenClaw.

    Attributes:
        actions_analyzed: Number of actions analyzed.
        actions_updated: Number of actions updated.
        patterns_extracted: Number of patterns extracted.
        context_updates_pushed: Number of context updates pushed.
        prioritization_updates: Number of task prioritizations updated.
        errors: List of error messages.
        duration_ms: Duration of the sync operation.
        metadata: Additional sync metadata.
    """

    actions_analyzed: int = 0
    actions_updated: int = 0
    patterns_extracted: int = 0
    context_updates_pushed: int = 0
    prioritization_updates: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "actions_analyzed": self.actions_analyzed,
            "actions_updated": self.actions_updated,
            "patterns_extracted": self.patterns_extracted,
            "context_updates_pushed": self.context_updates_pushed,
            "prioritization_updates": self.prioritization_updates,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


__all__ = [
    "ActionStatus",
    "PatternType",
    "OpenClawKnowledgeItem",
    "ActionPattern",
    "SyncResult",
    "KMContextUpdate",
    "TaskPrioritizationUpdate",
    "KMValidationResult",
    "OpenClawKMSyncResult",
]
