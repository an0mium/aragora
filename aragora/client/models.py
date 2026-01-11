"""
Pydantic models for Aragora API responses.

These models provide type-safe representations of API responses
for use with the AragoraClient SDK.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DebateStatus(str, Enum):
    """Status of a debate."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConsensusType(str, Enum):
    """Type of consensus mechanism."""
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    HYBRID = "hybrid"


class AgentMessage(BaseModel):
    """A message from an agent during debate."""
    agent_id: str
    content: str
    round: int
    timestamp: Optional[datetime] = None
    token_count: Optional[int] = None


class Vote(BaseModel):
    """A vote cast by an agent."""
    agent_id: str
    position: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class ConsensusResult(BaseModel):
    """Result of consensus detection."""
    reached: bool
    agreement: float = Field(ge=0.0, le=1.0)
    final_answer: Optional[str] = None
    dissenting_agents: list[str] = Field(default_factory=list)
    votes: list[Vote] = Field(default_factory=list)


class DebateRound(BaseModel):
    """A single round of debate."""
    round_number: int
    messages: list[AgentMessage] = Field(default_factory=list)
    critiques: list[AgentMessage] = Field(default_factory=list)


class Debate(BaseModel):
    """A debate result."""
    debate_id: str
    task: str
    status: DebateStatus
    agents: list[str] = Field(default_factory=list)
    rounds: list[DebateRound] = Field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebateCreateRequest(BaseModel):
    """Request to create a new debate."""
    task: str
    agents: list[str] = Field(default_factory=lambda: ["anthropic-api", "openai-api"])
    rounds: int = Field(default=3, ge=1, le=10)
    consensus: ConsensusType = ConsensusType.MAJORITY
    context: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DebateCreateResponse(BaseModel):
    """Response from creating a debate."""
    debate_id: str
    status: DebateStatus
    task: str


class AgentProfile(BaseModel):
    """Profile of an AI agent."""
    agent_id: str
    name: str
    provider: str
    elo_rating: int = 1500
    matches_played: int = 0
    win_rate: float = 0.0
    available: bool = True
    capabilities: list[str] = Field(default_factory=list)


class LeaderboardEntry(BaseModel):
    """An entry in the leaderboard."""
    rank: int
    agent_id: str
    elo_rating: int
    matches_played: int
    win_rate: float
    recent_trend: str = "stable"  # "up", "down", "stable"


class GauntletVerdict(str, Enum):
    """Verdict from a gauntlet run."""
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class Finding(BaseModel):
    """A finding from gauntlet analysis."""
    severity: str  # "critical", "high", "medium", "low"
    category: str
    title: str
    description: str
    location: Optional[str] = None
    mitigation: Optional[str] = None


class GauntletReceipt(BaseModel):
    """Decision receipt from gauntlet run."""
    receipt_id: str
    verdict: GauntletVerdict
    risk_score: float = Field(ge=0.0, le=1.0)
    findings: list[Finding] = Field(default_factory=list)
    summary: str
    created_at: datetime
    input_hash: str
    persona: str


class GauntletRunRequest(BaseModel):
    """Request to run gauntlet analysis."""
    input_content: str
    input_type: str = "text"  # "text", "policy", "code"
    persona: str = "security"
    profile: str = "default"  # "quick", "default", "thorough"


class GauntletRunResponse(BaseModel):
    """Response from starting a gauntlet run."""
    gauntlet_id: str
    status: str
    estimated_duration: Optional[int] = None


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    components: dict[str, str] = Field(default_factory=dict)


class APIError(BaseModel):
    """API error response."""
    error: str
    code: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
