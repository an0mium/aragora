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


# =============================================================================
# Enterprise Audit Models
# =============================================================================


class FindingWorkflowStatus(str, Enum):
    """Workflow status for audit findings."""

    OPEN = "open"
    TRIAGING = "triaging"
    INVESTIGATING = "investigating"
    REMEDIATING = "remediating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ACCEPTED_RISK = "accepted_risk"
    DUPLICATE = "duplicate"


class AuditPreset(BaseModel):
    """Audit preset configuration for specific industries/use cases."""

    name: str
    description: str
    audit_types: list[str] = Field(default_factory=list)
    custom_rules_count: int = 0
    consensus_threshold: float = 0.8
    agents: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AuditPresetDetail(BaseModel):
    """Detailed audit preset with custom rules."""

    name: str
    description: str
    audit_types: list[str] = Field(default_factory=list)
    custom_rules: list[dict[str, Any]] = Field(default_factory=list)
    consensus_threshold: float = 0.8
    agents: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AuditTypeCapabilities(BaseModel):
    """Capabilities of an audit type."""

    supports_chunk_analysis: bool = True
    supports_cross_document: bool = False
    requires_llm: bool = True


class AuditTypeInfo(BaseModel):
    """Information about a registered audit type."""

    id: str
    display_name: str
    description: str
    version: str = "1.0.0"
    capabilities: AuditTypeCapabilities = Field(default_factory=AuditTypeCapabilities)


class FindingWorkflowEvent(BaseModel):
    """An event in a finding's workflow history."""

    id: str
    event_type: str  # "state_change", "comment", "assignment", "priority_change"
    timestamp: datetime
    user_id: str
    user_name: str = ""
    comment: Optional[str] = None
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FindingWorkflowData(BaseModel):
    """Full workflow data for a finding."""

    finding_id: str
    current_state: FindingWorkflowStatus = FindingWorkflowStatus.OPEN
    assigned_to: Optional[str] = None
    priority: int = 3  # 1=Critical, 2=High, 3=Medium, 4=Low, 5=Lowest
    due_date: Optional[datetime] = None
    history: list[FindingWorkflowEvent] = Field(default_factory=list)


class QuickAuditResult(BaseModel):
    """Result from a quick audit run."""

    session_id: str
    preset_used: str
    document_count: int
    total_findings: int
    findings_by_severity: dict[str, int] = Field(default_factory=dict)
    critical_findings: list[AuditFinding] = Field(default_factory=list)
    high_findings: list[AuditFinding] = Field(default_factory=list)


# =============================================================================
# Extended Agent Models
# =============================================================================


class AgentCalibration(BaseModel):
    """Calibration scores for an agent."""

    agent: str
    overall_score: float = Field(ge=0.0, le=1.0)
    domain_scores: dict[str, float] = Field(default_factory=dict)
    confidence_accuracy: float = 0.0
    last_calibrated: Optional[datetime] = None
    sample_size: int = 0


class AgentPerformance(BaseModel):
    """Performance statistics for an agent."""

    agent: str
    win_rate: float = Field(ge=0.0, le=1.0)
    loss_rate: float = Field(ge=0.0, le=1.0)
    draw_rate: float = Field(ge=0.0, le=1.0)
    elo_trend: list[float] = Field(default_factory=list)
    elo_change_30d: float = 0.0
    avg_confidence: float = 0.0
    avg_round_duration_ms: int = 0
    total_debates: int = 0
    recent_results: list[dict[str, Any]] = Field(default_factory=list)


class HeadToHeadStats(BaseModel):
    """Head-to-head statistics between two agents."""

    agent: str
    opponent: str
    total_matchups: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    avg_margin: float = 0.0
    recent_matchups: list[dict[str, Any]] = Field(default_factory=list)
    domain_breakdown: dict[str, dict[str, int]] = Field(default_factory=dict)


class OpponentBriefing(BaseModel):
    """Strategic briefing against an opponent."""

    agent: str
    opponent: str
    opponent_profile: dict[str, Any] = Field(default_factory=dict)
    historical_summary: str = ""
    recommended_strategy: str = ""
    key_insights: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class AgentConsistency(BaseModel):
    """Consistency metrics for an agent."""

    agent: str
    overall_consistency: float = 0.0
    position_stability: float = 0.0
    flip_rate: float = 0.0
    consistency_by_domain: dict[str, float] = Field(default_factory=dict)
    volatility_index: float = 0.0
    sample_size: int = 0


class AgentFlip(BaseModel):
    """A position flip event."""

    flip_id: str
    agent: str
    debate_id: str
    topic: str = ""
    original_position: str = ""
    new_position: str = ""
    flip_reason: Optional[str] = None
    round_number: int = 0
    timestamp: Optional[datetime] = None
    was_justified: bool = False


class AgentNetwork(BaseModel):
    """Agent relationship network."""

    agent: str
    allies: list[dict[str, Any]] = Field(default_factory=list)
    rivals: list[dict[str, Any]] = Field(default_factory=list)
    neutrals: list[str] = Field(default_factory=list)
    cluster_id: Optional[str] = None
    network_position: str = "peripheral"  # central, peripheral, bridge


class AgentMoment(BaseModel):
    """A significant moment for an agent."""

    moment_id: str
    agent: str
    debate_id: str
    type: str  # breakthrough, comeback, decisive_argument, consensus_catalyst, upset
    description: str = ""
    impact_score: float = 0.0
    timestamp: Optional[datetime] = None
    context: dict[str, Any] = Field(default_factory=dict)


class AgentPosition(BaseModel):
    """A position taken by an agent."""

    position_id: str
    agent: str
    debate_id: str
    topic: str = ""
    stance: str = ""
    confidence: float = 0.0
    supporting_evidence: list[str] = Field(default_factory=list)
    round_number: int = 0
    timestamp: Optional[datetime] = None
    was_final: bool = False


class DomainRating(BaseModel):
    """Domain-specific rating for an agent."""

    domain: str
    elo: int = 1500
    matches: int = 0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    last_active: Optional[datetime] = None
    trend: str = "stable"  # rising, stable, falling


# =============================================================================
# Extended Gauntlet Models
# =============================================================================


class GauntletRunStatus(str, Enum):
    """Status of a gauntlet run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GauntletRun(BaseModel):
    """A gauntlet run status."""

    id: str
    name: Optional[str] = None
    status: GauntletRunStatus = GauntletRunStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, int] = Field(default_factory=dict)
    results_summary: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GauntletPersonaCategory(str, Enum):
    """Category of gauntlet persona."""

    ADVERSARIAL = "adversarial"
    EDGE_CASE = "edge_case"
    STRESS = "stress"
    COMPLIANCE = "compliance"
    CUSTOM = "custom"


class GauntletPersona(BaseModel):
    """A gauntlet testing persona."""

    id: str
    name: str
    description: str = ""
    category: GauntletPersonaCategory = GauntletPersonaCategory.CUSTOM
    severity: str = "medium"  # low, medium, high, critical
    tags: list[str] = Field(default_factory=list)
    example_prompts: list[str] = Field(default_factory=list)
    enabled: bool = True


class GauntletResultStatus(str, Enum):
    """Status of a gauntlet result."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class GauntletResult(BaseModel):
    """A single gauntlet scenario result."""

    id: str
    gauntlet_id: str
    scenario: str = ""
    persona: Optional[str] = None
    status: GauntletResultStatus = GauntletResultStatus.PASS
    verdict: str = ""
    confidence: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    duration_ms: int = 0
    debate_id: Optional[str] = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: Optional[datetime] = None


class GauntletHeatmapExtended(BaseModel):
    """Extended heatmap data for a gauntlet run."""

    gauntlet_id: str
    dimensions: dict[str, list[str]] = Field(default_factory=dict)
    matrix: list[list[float]] = Field(default_factory=list)
    overall_risk: float = 0.0
    hotspots: list[dict[str, Any]] = Field(default_factory=list)
    generated_at: Optional[datetime] = None


class GauntletComparison(BaseModel):
    """Comparison between two gauntlet runs."""

    gauntlet_a: str
    gauntlet_b: str
    comparison: dict[str, Any] = Field(default_factory=dict)
    scenario_diffs: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str = "investigate"  # promote, investigate, block
    generated_at: Optional[datetime] = None


# =============================================================================
# Analytics Models
# =============================================================================


class DisagreementAnalytics(BaseModel):
    """Disagreement analytics."""

    period: str = ""
    total_debates: int = 0
    disagreement_rate: float = 0.0
    avg_dissent_count: float = 0.0
    top_disagreement_topics: list[dict[str, Any]] = Field(default_factory=list)
    agent_disagreement_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    persistent_disagreements: list[dict[str, Any]] = Field(default_factory=list)


class RoleRotationAnalytics(BaseModel):
    """Role rotation analytics."""

    period: str = ""
    total_assignments: int = 0
    role_distribution: dict[str, int] = Field(default_factory=dict)
    agent_role_frequency: dict[str, dict[str, int]] = Field(default_factory=dict)
    rotation_fairness_index: float = 0.0
    stuck_agents: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class EarlyStopAnalytics(BaseModel):
    """Early stop analytics."""

    period: str = ""
    total_debates: int = 0
    early_stop_rate: float = 0.0
    avg_rounds_saved: float = 0.0
    early_stop_reasons: dict[str, int] = Field(default_factory=dict)
    confidence_at_stop: dict[str, float] = Field(default_factory=dict)
    false_early_stops: int = 0
    missed_early_stops: int = 0


class ConsensusQualityAnalytics(BaseModel):
    """Consensus quality analytics."""

    period: str = ""
    total_consensuses: int = 0
    quality_distribution: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_agreement_level: float = 0.0
    hollow_consensus_rate: float = 0.0
    contested_consensus_rate: float = 0.0
    consensus_durability: dict[str, int] = Field(default_factory=dict)
    quality_by_topic: dict[str, float] = Field(default_factory=dict)


class RankingStats(BaseModel):
    """Ranking statistics."""

    total_agents: int = 0
    elo_distribution: dict[str, float] = Field(default_factory=dict)
    tier_distribution: dict[str, int] = Field(default_factory=dict)
    top_performers: list[dict[str, Any]] = Field(default_factory=list)
    most_improved: list[dict[str, Any]] = Field(default_factory=list)
    last_updated: Optional[datetime] = None


class MemoryStats(BaseModel):
    """Memory system statistics."""

    total_entries: int = 0
    storage_bytes: int = 0
    tier_counts: dict[str, int] = Field(default_factory=dict)
    consolidation_rate: float = 0.0
    avg_importance: float = 0.0
    cache_hit_rate: float = 0.0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None
    health_status: str = "healthy"  # healthy, degraded, critical


# =============================================================================
# Debate Update and Search Models
# =============================================================================


class DebateUpdateRequest(BaseModel):
    """Request to update a debate."""

    status: Optional[DebateStatus] = None
    metadata: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None
    archived: Optional[bool] = None
    notes: Optional[str] = None


class VerificationReportClaimDetail(BaseModel):
    """Detail for a verified claim."""

    claim: str
    verified: bool
    confidence: float = 0.0
    evidence: Optional[str] = None
    counterevidence: Optional[str] = None


class VerificationReport(BaseModel):
    """Verification report for a debate."""

    debate_id: str
    verified: bool
    verification_method: str = ""
    claims_verified: int = 0
    claims_failed: int = 0
    claims_skipped: int = 0
    claim_details: list[VerificationReportClaimDetail] = Field(default_factory=list)
    overall_confidence: float = 0.0
    verification_duration_ms: int = 0
    generated_at: Optional[datetime] = None


class SearchResult(BaseModel):
    """A search result."""

    type: str  # debate, agent, memory, claim
    id: str
    title: Optional[str] = None
    snippet: str = ""
    score: float = 0.0
    highlights: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class SearchResponse(BaseModel):
    """Search response."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    total_count: int = 0
    facets: dict[str, dict[str, int]] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    took_ms: int = 0


# =============================================================================
# Memory Search Models
# =============================================================================


class MemoryTierType(str, Enum):
    """Memory tier type."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


class MemoryEntry(BaseModel):
    """A memory entry."""

    id: str
    tier: MemoryTierType = MemoryTierType.MEDIUM
    content: str = ""
    importance: float = 0.0
    consolidation_count: int = 0
    source_debate_id: Optional[str] = None
    agent: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchParams(BaseModel):
    """Memory search parameters."""

    query: str
    tiers: Optional[list[MemoryTierType]] = None
    agent: Optional[str] = None
    limit: int = 20
    min_importance: Optional[float] = None
    include_expired: bool = False


class CritiqueEntry(BaseModel):
    """A critique entry from memory."""

    id: str
    debate_id: str
    critic_agent: str
    target_agent: str
    critique: str = ""
    severity: str = "moderate"  # minor, moderate, major
    was_addressed: bool = False
    resolution: Optional[str] = None
    round_number: int = 0
    timestamp: Optional[datetime] = None
    impact_score: float = 0.0
