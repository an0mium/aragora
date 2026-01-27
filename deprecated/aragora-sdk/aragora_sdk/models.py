"""
Aragora SDK Models.

Pydantic models for API requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConsensusStatus(str, Enum):
    """Status of consensus in a debate."""

    PENDING = "pending"
    REACHED = "reached"
    NOT_REACHED = "not_reached"
    PARTIAL = "partial"


class DebateStatus(str, Enum):
    """Status of a debate."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Agent(BaseModel):
    """Agent information."""

    name: str
    provider: str | None = None
    model: str | None = None
    persona: str | None = None


class Position(BaseModel):
    """A position or argument in the debate."""

    content: str
    agent: str
    round: int
    timestamp: datetime | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """A critique of a position."""

    content: str
    agent: str
    target_position: str | None = None
    round: int
    severity: str | None = None  # "minor", "major", "critical"


class Vote(BaseModel):
    """An agent's vote on the final position."""

    agent: str
    position: str
    confidence: float = 1.0
    reasoning: str | None = None


class DissentingOpinion(BaseModel):
    """A dissenting opinion from the debate."""

    agent: str
    position: str
    reasoning: str
    concerns: list[str] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    """Result of consensus detection."""

    status: ConsensusStatus
    position: str | None = None
    confidence: float = 0.0
    supporting_agents: list[str] = Field(default_factory=list)
    dissenting_agents: list[str] = Field(default_factory=list)


class DecisionReceipt(BaseModel):
    """
    Audit-ready decision receipt from a debate.

    Contains the full record of the debate process for compliance purposes.
    """

    id: str
    task: str
    created_at: datetime
    completed_at: datetime | None = None

    # Debate configuration
    rounds: int
    agents: list[Agent]
    personas: list[str] = Field(default_factory=list)

    # Results
    consensus: ConsensusResult
    final_position: str | None = None
    dissenting_opinions: list[DissentingOpinion] = Field(default_factory=list)

    # Audit trail
    positions: list[Position] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    votes: list[Vote] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksum: str | None = None  # SHA-256 of content for integrity


class ReviewResult(BaseModel):
    """Result of a design/spec review."""

    id: str
    task: str
    status: DebateStatus
    created_at: datetime
    completed_at: datetime | None = None

    # Core results
    consensus: ConsensusResult
    final_position: str | None = None
    dissenting_opinions: list[DissentingOpinion] = Field(default_factory=list)

    # Detailed findings
    findings: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)

    # Full receipt for audit purposes
    decision_receipt: DecisionReceipt | None = None

    # Usage
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class ReviewRequest(BaseModel):
    """Request to review a design or specification."""

    spec: str = Field(..., description="Design specification or document to review")
    personas: list[str] = Field(
        default_factory=lambda: ["security", "performance"],
        description="Personas to use for review (e.g., 'sox', 'pci_dss', 'hipaa')",
    )
    rounds: int = Field(default=3, ge=1, le=10, description="Number of debate rounds")
    task: str | None = Field(
        default=None,
        description="Optional task description (defaults to 'Review the provided specification')",
    )
    agents: list[str] | None = Field(
        default=None,
        description="Specific agents to use (defaults to automatic selection)",
    )
    include_receipt: bool = Field(
        default=True,
        description="Include full decision receipt in response",
    )


class UsageInfo(BaseModel):
    """Usage information for billing."""

    debates_used: int = 0
    debates_limit: int = 0
    debates_remaining: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    period_start: datetime | None = None
    period_end: datetime | None = None


class HealthStatus(BaseModel):
    """Server health status."""

    status: str
    version: str
    uptime_seconds: float = 0.0
    active_debates: int = 0
