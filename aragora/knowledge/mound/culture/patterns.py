"""
Pattern types for the Culture Accumulator.

Defines the data structures for reasoning patterns and decision heuristics
that are accumulated from organizational experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class PatternType(str, Enum):
    """Types of accumulated patterns."""

    DECISION_HEURISTIC = "decision_heuristic"  # Rules of thumb
    REASONING_CHAIN = "reasoning_chain"  # Common inference paths
    ERROR_PATTERN = "error_pattern"  # Past mistakes to avoid
    SUCCESS_PATTERN = "success_pattern"  # What works well
    COLLABORATION_PATTERN = "collaboration"  # Multi-agent dynamics
    DOMAIN_NORM = "domain_norm"  # Vertical-specific norms
    CONSENSUS_PATTERN = "consensus"  # How consensus was reached
    DISSENT_PATTERN = "dissent"  # Valuable minority opinions


@dataclass
class ReasoningPattern:
    """
    A learned reasoning pattern from organizational experience.

    Represents emergent knowledge about HOW to reason,
    not just WHAT facts are true. These patterns help
    guide future debates and decision-making.
    """

    id: str
    pattern_type: PatternType
    name: str
    description: str

    # Pattern content
    trigger_conditions: list[str]  # When to apply this pattern
    reasoning_steps: list[str]  # How to reason
    expected_outcomes: list[str]  # What to expect

    # Confidence and validation
    confidence: float = 0.5
    success_count: int = 0
    failure_count: int = 0
    last_applied: Optional[datetime] = None

    # Provenance
    derived_from_debates: list[str] = field(default_factory=list)
    contributing_agents: list[str] = field(default_factory=list)

    # Scope
    verticals: list[str] = field(default_factory=list)  # Empty = all
    workspace_id: Optional[str] = None
    topics: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate from outcomes."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def reliability_score(self) -> float:
        """
        Confidence weighted by usage.

        Patterns that have been used successfully many times
        are more reliable than untested patterns.
        """
        usage = self.success_count + self.failure_count
        usage_weight = min(1.0, usage / 10.0)  # Max weight at 10+ uses
        return self.confidence * self.success_rate * usage_weight

    @property
    def is_mature(self) -> bool:
        """Check if pattern has enough usage to be considered mature."""
        return (self.success_count + self.failure_count) >= 5

    def record_outcome(self, success: bool) -> None:
        """Record the outcome of applying this pattern."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_applied = datetime.now()
        self.updated_at = datetime.now()

        # Bayesian update of confidence
        self._update_confidence(success)

    def _update_confidence(self, success: bool) -> None:
        """Bayesian update of confidence based on outcome."""
        # Simple beta-distribution inspired update
        alpha = self.success_count + 1
        beta = self.failure_count + 1
        self.confidence = alpha / (alpha + beta)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "reasoning_steps": self.reasoning_steps,
            "expected_outcomes": self.expected_outcomes,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_applied": self.last_applied.isoformat() if self.last_applied else None,
            "derived_from_debates": self.derived_from_debates,
            "contributing_agents": self.contributing_agents,
            "verticals": self.verticals,
            "workspace_id": self.workspace_id,
            "topics": self.topics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningPattern:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            pattern_type=PatternType(data["pattern_type"]),
            name=data["name"],
            description=data["description"],
            trigger_conditions=data.get("trigger_conditions", []),
            reasoning_steps=data.get("reasoning_steps", []),
            expected_outcomes=data.get("expected_outcomes", []),
            confidence=data.get("confidence", 0.5),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            last_applied=datetime.fromisoformat(data["last_applied"]) if data.get("last_applied") else None,
            derived_from_debates=data.get("derived_from_debates", []),
            contributing_agents=data.get("contributing_agents", []),
            verticals=data.get("verticals", []),
            workspace_id=data.get("workspace_id"),
            topics=data.get("topics", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class DecisionHeuristic:
    """
    A rule of thumb for making decisions.

    Shorter-form than ReasoningPattern, used for quick guidance
    without requiring full reasoning chains.
    """

    id: str
    rule: str  # The heuristic statement
    applies_when: str  # Condition for application
    confidence: float
    supporting_evidence: list[str]
    exceptions: list[str] = field(default_factory=list)
    verticals: list[str] = field(default_factory=list)
    workspace_id: Optional[str] = None
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "rule": self.rule,
            "applies_when": self.applies_when,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "exceptions": self.exceptions,
            "verticals": self.verticals,
            "workspace_id": self.workspace_id,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionHeuristic:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            rule=data["rule"],
            applies_when=data.get("applies_when", ""),
            confidence=data.get("confidence", 0.5),
            supporting_evidence=data.get("supporting_evidence", []),
            exceptions=data.get("exceptions", []),
            verticals=data.get("verticals", []),
            workspace_id=data.get("workspace_id"),
            usage_count=data.get("usage_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )
