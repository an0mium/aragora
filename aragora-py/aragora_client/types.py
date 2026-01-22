"""Type definitions for the Aragora SDK."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DebateStatus(str, Enum):
    """Status of a debate."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationStatus(str, Enum):
    """Status of a verification result."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    ERROR = "error"


class ConsensusResult(BaseModel):
    """Result of consensus detection."""

    reached: bool
    conclusion: str | None = None
    confidence: float = 0.0
    supporting_agents: list[str] = Field(default_factory=list)
    dissenting_agents: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class AgentMessage(BaseModel):
    """A message from an agent during debate."""

    agent_id: str
    content: str
    round_number: int
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class Debate(BaseModel):
    """A debate instance."""

    id: str
    task: str
    status: DebateStatus
    agents: list[str]
    rounds: list[list[AgentMessage]] = Field(default_factory=list)
    consensus: ConsensusResult | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphBranch(BaseModel):
    """A branch in a graph debate."""

    id: str
    parent_id: str | None = None
    approach: str
    agents: list[str]
    rounds: list[list[AgentMessage]] = Field(default_factory=list)
    consensus: ConsensusResult | None = None
    divergence_score: float = 0.0


class GraphDebate(BaseModel):
    """A graph debate with branching."""

    id: str
    task: str
    status: DebateStatus
    branches: list[GraphBranch] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatrixConclusion(BaseModel):
    """Conclusions from a matrix debate."""

    universal: list[str] = Field(default_factory=list)
    conditional: dict[str, list[str]] = Field(default_factory=dict)
    contradictions: list[str] = Field(default_factory=list)


class MatrixScenario(BaseModel):
    """A scenario in a matrix debate."""

    name: str
    parameters: dict[str, Any]
    is_baseline: bool = False
    consensus: ConsensusResult | None = None


class MatrixDebate(BaseModel):
    """A matrix debate across scenarios."""

    id: str
    task: str
    status: DebateStatus
    scenarios: list[MatrixScenario] = Field(default_factory=list)
    conclusions: MatrixConclusion | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentProfile(BaseModel):
    """Profile of an agent."""

    id: str
    name: str
    provider: str
    elo_rating: float = 1500.0
    matches_played: int = 0
    win_rate: float = 0.5
    specialties: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of formal verification."""

    status: VerificationStatus
    claim: str
    formal_translation: str | None = None
    proof: str | None = None
    counterexample: str | None = None
    backend: str
    duration_ms: int = 0


class GauntletFinding(BaseModel):
    """A finding from gauntlet validation."""

    severity: str
    category: str
    description: str
    location: str | None = None
    suggestion: str | None = None


class GauntletReceipt(BaseModel):
    """Receipt from gauntlet validation."""

    id: str
    score: float
    findings: list[GauntletFinding] = Field(default_factory=list)
    persona: str
    created_at: datetime
    hash: str | None = None


class MemoryTierStats(BaseModel):
    """Statistics for a memory tier."""

    tier: str
    entries: int
    size_bytes: int
    hit_rate: float
    avg_age_seconds: float


class MemoryAnalytics(BaseModel):
    """Analytics for the memory system."""

    total_entries: int
    total_size_bytes: int
    learning_velocity: float
    tiers: list[MemoryTierStats] = Field(default_factory=list)
    period_days: int


class HealthStatus(BaseModel):
    """Server health status."""

    status: str
    version: str
    uptime_seconds: float
    components: dict[str, str] = Field(default_factory=dict)


class DebateEvent(BaseModel):
    """A WebSocket event from a debate."""

    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    loop_id: str | None = None


class ScorerInfo(BaseModel):
    """Information about a scorer plugin."""

    name: str
    description: str


class TeamSelectorInfo(BaseModel):
    """Information about a team selector plugin."""

    name: str
    description: str


class RoleAssignerInfo(BaseModel):
    """Information about a role assigner plugin."""

    name: str
    description: str


class SelectionPlugins(BaseModel):
    """Available selection plugins."""

    scorers: list[ScorerInfo] = Field(default_factory=list)
    team_selectors: list[TeamSelectorInfo] = Field(default_factory=list)
    role_assigners: list[RoleAssignerInfo] = Field(default_factory=list)


class AgentScore(BaseModel):
    """Score for an agent."""

    name: str
    score: float
    elo_rating: float
    breakdown: dict[str, float] = Field(default_factory=dict)


class TeamMember(BaseModel):
    """A member of a selected team."""

    name: str
    role: str
    score: float


class TeamSelection(BaseModel):
    """Result of team selection."""

    agents: list[TeamMember] = Field(default_factory=list)
    expected_quality: float
    diversity_score: float
    rationale: str


# Request/Response types for API calls
class CreateDebateRequest(BaseModel):
    """Request to create a debate."""

    task: str
    agents: list[str] | None = None
    max_rounds: int = 5
    consensus_threshold: float = 0.8
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateGraphDebateRequest(BaseModel):
    """Request to create a graph debate."""

    task: str
    agents: list[str] | None = None
    max_rounds: int = 5
    branch_threshold: float = 0.5
    max_branches: int = 10


class CreateMatrixDebateRequest(BaseModel):
    """Request to create a matrix debate."""

    task: str
    scenarios: list[dict[str, Any]]
    agents: list[str] | None = None
    max_rounds: int = 3


class VerifyClaimRequest(BaseModel):
    """Request to verify a claim."""

    claim: str
    backend: str = "z3"
    timeout: int = 30


class RunGauntletRequest(BaseModel):
    """Request to run gauntlet validation."""

    input_content: str
    input_type: str = "spec"
    persona: str = "security"


class ScoreAgentsRequest(BaseModel):
    """Request to score agents."""

    task_description: str
    primary_domain: str | None = None
    scorer: str | None = None


class SelectTeamRequest(BaseModel):
    """Request to select a team."""

    task_description: str
    min_agents: int = 2
    max_agents: int = 5
    diversity_preference: float = 0.5
    quality_priority: float = 0.5
    scorer: str | None = None
    team_selector: str | None = None
    role_assigner: str | None = None
