"""
Pydantic models for Aragora API responses.

These models provide type-safe representations of API responses
for use with the AragoraClient SDK.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class DebateStatus(str, Enum):
    """Status of a debate.

    Canonical values: pending, running, completed, failed, cancelled, paused
    Legacy values (still supported): created, in_progress, starting, active, concluded
    """

    # Canonical SDK values
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

    # Legacy values (kept for backwards compatibility)
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    STARTING = "starting"

    @classmethod
    def _missing_(cls, value: object) -> "DebateStatus | None":
        """Handle legacy server status values.

        Maps internal server statuses (active, concluded, archived) to
        canonical SDK statuses. This provides tolerance for API response
        variations.
        """
        if not isinstance(value, str):
            return None

        legacy_map = {
            "active": cls.RUNNING,
            "concluded": cls.COMPLETED,
            "archived": cls.COMPLETED,
        }
        return legacy_map.get(value.lower())


class ConsensusType(str, Enum):
    """Type of consensus mechanism."""

    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"
    HYBRID = "hybrid"


class AgentMessage(BaseModel):
    """A message from an agent during debate."""

    agent_id: str = Field(validation_alias=AliasChoices("agent_id", "agent"))
    content: str
    round: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("round", "round_number")
    )
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
    agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    final_answer: Optional[str] = None
    conclusion: Optional[str] = None
    supporting_agents: list[str] = Field(default_factory=list)
    dissenting_agents: list[str] = Field(default_factory=list)
    votes: list[Vote] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_fields(self) -> "ConsensusResult":
        if self.agreement is None and self.confidence is not None:
            self.agreement = self.confidence
        if self.confidence is None and self.agreement is not None:
            self.confidence = self.agreement
        if self.final_answer is None and self.conclusion is not None:
            self.final_answer = self.conclusion
        if self.conclusion is None and self.final_answer is not None:
            self.conclusion = self.final_answer
        return self


class DebateRound(BaseModel):
    """A single round of debate."""

    round_number: int = Field(validation_alias=AliasChoices("round_number", "round"))
    messages: list[AgentMessage] = Field(default_factory=list)
    critiques: list[AgentMessage] = Field(default_factory=list)


class Debate(BaseModel):
    """A debate result."""

    debate_id: str = Field(validation_alias=AliasChoices("debate_id", "id"))
    task: str
    status: DebateStatus
    agents: list[str] = Field(default_factory=list)
    rounds: list[DebateRound] = Field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    consensus_proof: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rounds", mode="before")
    @classmethod
    def _coerce_rounds(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, int):
            return []
        return value

    @model_validator(mode="after")
    def _derive_consensus(self) -> "Debate":
        if self.consensus is None and self.consensus_proof:
            proof = self.consensus_proof or {}
            vote_breakdown = proof.get("vote_breakdown") or {}
            supporting = [agent for agent, agreed in vote_breakdown.items() if agreed]
            dissenting = [agent for agent, agreed in vote_breakdown.items() if not agreed]
            self.consensus = ConsensusResult(
                reached=bool(proof.get("reached", False)),
                agreement=proof.get("confidence"),
                confidence=proof.get("confidence"),
                final_answer=proof.get("final_answer"),
                conclusion=proof.get("final_answer"),
                supporting_agents=supporting,
                dissenting_agents=dissenting,
            )
        return self


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
    status: Optional[DebateStatus] = None
    task: Optional[str] = None


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

    severity: str = "medium"  # "critical", "high", "medium", "low"
    category: str = "general"
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    mitigation: Optional[str] = None
    suggestion: Optional[str] = None

    @model_validator(mode="after")
    def _normalize_fields(self) -> "Finding":
        if self.title is None and self.description:
            self.title = self.description
        if self.description is None and self.title:
            self.description = self.title
        if self.mitigation is None and self.suggestion:
            self.mitigation = self.suggestion
        return self


class GauntletReceipt(BaseModel):
    """Decision receipt from gauntlet run."""

    receipt_id: Optional[str] = None
    gauntlet_id: Optional[str] = None
    verdict: Optional[GauntletVerdict | str] = None
    status: Optional[str] = None
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    findings: list[Finding] = Field(default_factory=list)
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    input_hash: Optional[str] = None
    persona: Optional[str] = None

    @field_validator("findings", mode="before")
    @classmethod
    def _coerce_findings(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            normalized: list[Any] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(
                        {
                            "severity": "low",
                            "category": "general",
                            "title": item,
                            "description": item,
                        }
                    )
                else:
                    normalized.append(item)
            return normalized
        return value

    @model_validator(mode="after")
    def _sync_scores(self) -> "GauntletReceipt":
        if self.risk_score is None and self.score is not None:
            self.risk_score = self.score
        if self.score is None and self.risk_score is not None:
            self.score = self.risk_score
        return self


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


# =============================================================================
# Document Models
# =============================================================================


class DocumentStatus(str, Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditType(str, Enum):
    """Type of document audit."""

    SECURITY = "security"
    COMPLIANCE = "compliance"
    CONSISTENCY = "consistency"
    QUALITY = "quality"


class FindingSeverity(str, Enum):
    """Severity level for audit findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Document(BaseModel):
    """A document in the system."""

    id: str
    filename: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A chunk of a processed document."""

    id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response from uploading a document."""

    document_id: str
    filename: str
    status: DocumentStatus = DocumentStatus.PENDING
    message: str = ""


class BatchUploadResponse(BaseModel):
    """Response from batch upload."""

    job_id: str
    document_count: int
    status: str
    message: str = ""


class BatchJobStatus(BaseModel):
    """Status of a batch processing job."""

    job_id: str
    status: str
    progress: float = 0.0
    document_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class BatchJobResults(BaseModel):
    """Results of a completed batch job."""

    job_id: str
    documents: list[Document] = Field(default_factory=list)
    failed: list[dict[str, Any]] = Field(default_factory=list)


class DocumentContext(BaseModel):
    """LLM-ready context from document chunks."""

    document_id: str
    total_tokens: int
    context: str
    chunks_used: int
    truncated: bool = False


class ProcessingStats(BaseModel):
    """Document processing statistics."""

    total_documents: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    total_chunks: int = 0
    total_tokens: int = 0


class SupportedFormats(BaseModel):
    """Supported document formats."""

    formats: list[str] = Field(default_factory=list)
    mime_types: list[str] = Field(default_factory=list)


# =============================================================================
# Audit Session Models
# =============================================================================


class AuditSessionStatus(str, Enum):
    """Status of an audit session."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AuditFinding(BaseModel):
    """A finding from a document audit."""

    id: str = ""
    session_id: str
    document_id: str = ""
    chunk_id: str = ""
    audit_type: AuditType
    category: str
    severity: FindingSeverity
    confidence: float = 0.0
    title: str
    description: str
    evidence_text: str = ""
    evidence_location: str = ""
    recommendation: str = ""
    found_by: str = ""
    created_at: Optional[datetime] = None


class AuditSession(BaseModel):
    """An audit session."""

    id: str
    document_ids: list[str] = Field(default_factory=list)
    audit_types: list[AuditType] = Field(default_factory=list)
    status: AuditSessionStatus = AuditSessionStatus.PENDING
    progress: float = 0.0
    finding_count: int = 0
    model: str = "gemini-1.5-flash"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class AuditSessionCreateRequest(BaseModel):
    """Request to create an audit session."""

    document_ids: list[str]
    audit_types: list[str] = Field(
        default_factory=lambda: ["security", "compliance", "consistency", "quality"]
    )
    model: str = "gemini-1.5-flash"
    options: dict[str, Any] = Field(default_factory=dict)


class AuditSessionCreateResponse(BaseModel):
    """Response from creating an audit session."""

    session_id: str
    status: AuditSessionStatus = AuditSessionStatus.PENDING
    document_count: int = 0
    audit_types: list[str] = Field(default_factory=list)


class AuditReportFormat(str, Enum):
    """Format for audit reports."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class AuditReport(BaseModel):
    """Audit report data."""

    session_id: str
    format: AuditReportFormat = AuditReportFormat.JSON
    content: str
    generated_at: datetime
