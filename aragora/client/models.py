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


# =============================================================================
# Graph Debates Models
# =============================================================================


class GraphDebateNode(BaseModel):
    """A node in the graph debate."""
    node_id: str
    content: str
    agent_id: str
    node_type: str  # "proposal", "critique", "synthesis"
    parent_id: Optional[str] = None
    round: int = 0


class GraphDebateBranch(BaseModel):
    """A branch in the graph debate."""
    branch_id: str
    name: str
    nodes: list[GraphDebateNode] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    is_main: bool = False


class GraphDebateCreateRequest(BaseModel):
    """Request to create a graph debate."""
    task: str
    agents: list[str] = Field(default_factory=lambda: ["anthropic-api", "openai-api"])
    max_rounds: int = Field(default=5, ge=1, le=20)
    branch_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_branches: int = Field(default=5, ge=1, le=20)


class GraphDebateCreateResponse(BaseModel):
    """Response from creating a graph debate."""
    debate_id: str
    status: str
    task: str


class GraphDebate(BaseModel):
    """A graph-structured debate result."""
    debate_id: str
    task: str
    status: DebateStatus
    agents: list[str] = Field(default_factory=list)
    branches: list[GraphDebateBranch] = Field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =============================================================================
# Matrix Debates Models
# =============================================================================


class MatrixScenario(BaseModel):
    """A scenario configuration for matrix debates."""
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    is_baseline: bool = False


class MatrixScenarioResult(BaseModel):
    """Result from a single scenario in matrix debate."""
    scenario_name: str
    consensus: Optional[ConsensusResult] = None
    key_findings: list[str] = Field(default_factory=list)
    differences_from_baseline: list[str] = Field(default_factory=list)


class MatrixConclusion(BaseModel):
    """Conclusions from matrix debate analysis."""
    universal: list[str] = Field(default_factory=list)  # True across all scenarios
    conditional: dict[str, list[str]] = Field(default_factory=dict)  # Scenario-dependent
    contradictions: list[str] = Field(default_factory=list)  # Conflicting conclusions


class MatrixDebateCreateRequest(BaseModel):
    """Request to create a matrix debate."""
    task: str
    agents: list[str] = Field(default_factory=lambda: ["anthropic-api", "openai-api"])
    scenarios: list[MatrixScenario] = Field(default_factory=list)
    max_rounds: int = Field(default=3, ge=1, le=10)


class MatrixDebateCreateResponse(BaseModel):
    """Response from creating a matrix debate."""
    matrix_id: str
    status: str
    task: str
    scenario_count: int


class MatrixDebate(BaseModel):
    """A matrix debate result with parallel scenarios."""
    matrix_id: str
    task: str
    status: DebateStatus
    agents: list[str] = Field(default_factory=list)
    scenarios: list[MatrixScenarioResult] = Field(default_factory=list)
    conclusions: Optional[MatrixConclusion] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =============================================================================
# Verification Models
# =============================================================================


class VerificationStatus(str, Enum):
    """Status of a verification attempt."""
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    ERROR = "error"


class VerificationBackend(str, Enum):
    """Verification backend type."""
    Z3 = "z3"
    LEAN = "lean"
    COQ = "coq"


class VerifyClaimRequest(BaseModel):
    """Request to verify a claim."""
    claim: str
    context: Optional[str] = None
    backend: str = "z3"  # z3, lean, coq
    timeout: int = Field(default=30, ge=1, le=300)


class VerifyClaimResponse(BaseModel):
    """Response from claim verification."""
    status: VerificationStatus
    claim: str
    formal_translation: Optional[str] = None
    proof: Optional[str] = None
    counterexample: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: int = 0


class VerificationBackendStatus(BaseModel):
    """Status of a verification backend."""
    name: str
    available: bool
    version: Optional[str] = None


class VerifyStatusResponse(BaseModel):
    """Response from verification status check."""
    available: bool
    backends: list[VerificationBackendStatus] = Field(default_factory=list)


# =============================================================================
# Memory Analytics Models
# =============================================================================


class MemoryTierStats(BaseModel):
    """Statistics for a memory tier."""
    tier_name: str
    entry_count: int = 0
    avg_access_frequency: float = 0.0
    promotion_rate: float = 0.0
    demotion_rate: float = 0.0
    hit_rate: float = 0.0


class MemoryRecommendation(BaseModel):
    """A recommendation for memory optimization."""
    type: str  # "promotion", "cleanup", "rebalance"
    description: str
    impact: str  # "high", "medium", "low"


class MemoryAnalyticsResponse(BaseModel):
    """Response from memory analytics endpoint."""
    tiers: list[MemoryTierStats] = Field(default_factory=list)
    total_entries: int = 0
    learning_velocity: float = 0.0
    promotion_effectiveness: float = 0.0
    recommendations: list[MemoryRecommendation] = Field(default_factory=list)
    period_days: int = 30


class MemorySnapshotResponse(BaseModel):
    """Response from taking a memory snapshot."""
    snapshot_id: str
    timestamp: datetime
    success: bool


# =============================================================================
# Replay Models
# =============================================================================


class ReplaySummary(BaseModel):
    """Summary of a debate replay."""
    replay_id: str
    debate_id: str
    task: str
    created_at: datetime
    duration_seconds: int = 0
    agent_count: int = 0
    round_count: int = 0


class ReplayEvent(BaseModel):
    """An event in a replay timeline."""
    event_type: str
    timestamp: datetime
    agent_id: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Replay(BaseModel):
    """Full replay of a debate."""
    replay_id: str
    debate_id: str
    task: str
    agents: list[str] = Field(default_factory=list)
    events: list[ReplayEvent] = Field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    created_at: datetime
    duration_seconds: int = 0
