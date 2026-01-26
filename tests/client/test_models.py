"""
Tests for aragora.client.models - Pydantic models for API responses.

These tests verify:
- Model instantiation with valid data
- Validation constraints (Field constraints)
- Enum values and legacy mapping
- Model validators and field coercion
- Alias handling for backwards compatibility
"""

from datetime import datetime, timezone

import pytest

from aragora.client.models import (
    # Enums
    AuditReportFormat,
    AuditSessionStatus,
    AuditType,
    ConsensusType,
    DebateStatus,
    DocumentStatus,
    FindingSeverity,
    FindingWorkflowStatus,
    GauntletVerdict,
    VerificationBackend,
    VerificationStatus,
    # Core models
    AgentMessage,
    AgentProfile,
    APIError,
    AuditFinding,
    AuditPreset,
    AuditPresetDetail,
    AuditReport,
    AuditSession,
    AuditSessionCreateRequest,
    AuditSessionCreateResponse,
    AuditTypeCapabilities,
    AuditTypeInfo,
    BatchJobResults,
    BatchJobStatus,
    BatchUploadResponse,
    ConsensusResult,
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    DebateRound,
    Document,
    DocumentChunk,
    DocumentContext,
    DocumentUploadResponse,
    Finding,
    GauntletReceipt,
    GauntletRunRequest,
    GauntletRunResponse,
    GraphDebate,
    GraphDebateBranch,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
    GraphDebateNode,
    HealthCheck,
    LeaderboardEntry,
    MatrixConclusion,
    MatrixDebate,
    MatrixDebateCreateRequest,
    MatrixDebateCreateResponse,
    MatrixScenario,
    MatrixScenarioResult,
    MemoryAnalyticsResponse,
    MemoryRecommendation,
    MemorySnapshotResponse,
    MemoryTierStats,
    ProcessingStats,
    Replay,
    ReplayEvent,
    ReplaySummary,
    SupportedFormats,
    VerificationBackendStatus,
    VerifyClaimRequest,
    VerifyClaimResponse,
    VerifyStatusResponse,
    Vote,
)


class TestDebateStatusEnum:
    """Tests for DebateStatus enum."""

    def test_canonical_values(self):
        """Test canonical status values."""
        assert DebateStatus.PENDING.value == "pending"
        assert DebateStatus.RUNNING.value == "running"
        assert DebateStatus.COMPLETED.value == "completed"
        assert DebateStatus.FAILED.value == "failed"
        assert DebateStatus.CANCELLED.value == "cancelled"
        assert DebateStatus.PAUSED.value == "paused"

    def test_legacy_values(self):
        """Test legacy status values still work."""
        assert DebateStatus.CREATED.value == "created"
        assert DebateStatus.IN_PROGRESS.value == "in_progress"
        assert DebateStatus.STARTING.value == "starting"

    def test_missing_handler_maps_legacy_values(self):
        """Test that legacy server values are mapped correctly."""
        assert DebateStatus("active") == DebateStatus.RUNNING
        assert DebateStatus("concluded") == DebateStatus.COMPLETED
        assert DebateStatus("archived") == DebateStatus.COMPLETED

    def test_missing_handler_returns_none_for_unknown(self):
        """Test that unknown values return None from _missing_."""
        result = DebateStatus._missing_("unknown_status")
        assert result is None

    def test_missing_handler_handles_non_string(self):
        """Test that non-string values return None."""
        result = DebateStatus._missing_(123)
        assert result is None


class TestConsensusTypeEnum:
    """Tests for ConsensusType enum."""

    def test_all_values(self):
        """Test all consensus type values."""
        assert ConsensusType.UNANIMOUS.value == "unanimous"
        assert ConsensusType.MAJORITY.value == "majority"
        assert ConsensusType.SUPERMAJORITY.value == "supermajority"
        assert ConsensusType.HYBRID.value == "hybrid"


class TestGauntletVerdictEnum:
    """Tests for GauntletVerdict enum."""

    def test_all_values(self):
        """Test all verdict values."""
        assert GauntletVerdict.APPROVED.value == "approved"
        assert GauntletVerdict.APPROVED_WITH_CONDITIONS.value == "approved_with_conditions"
        assert GauntletVerdict.NEEDS_REVIEW.value == "needs_review"
        assert GauntletVerdict.REJECTED.value == "rejected"


class TestVerificationEnums:
    """Tests for verification-related enums."""

    def test_verification_status(self):
        """Test VerificationStatus values."""
        assert VerificationStatus.VALID.value == "valid"
        assert VerificationStatus.INVALID.value == "invalid"
        assert VerificationStatus.UNKNOWN.value == "unknown"
        assert VerificationStatus.ERROR.value == "error"

    def test_verification_backend(self):
        """Test VerificationBackend values."""
        assert VerificationBackend.Z3.value == "z3"
        assert VerificationBackend.LEAN.value == "lean"
        assert VerificationBackend.COQ.value == "coq"


class TestDocumentEnums:
    """Tests for document-related enums."""

    def test_document_status(self):
        """Test DocumentStatus values."""
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.COMPLETED.value == "completed"
        assert DocumentStatus.FAILED.value == "failed"

    def test_audit_type(self):
        """Test AuditType values."""
        assert AuditType.SECURITY.value == "security"
        assert AuditType.COMPLIANCE.value == "compliance"
        assert AuditType.CONSISTENCY.value == "consistency"
        assert AuditType.QUALITY.value == "quality"

    def test_finding_severity(self):
        """Test FindingSeverity values."""
        assert FindingSeverity.CRITICAL.value == "critical"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.INFO.value == "info"


class TestAuditEnums:
    """Tests for audit-related enums."""

    def test_audit_session_status(self):
        """Test AuditSessionStatus values."""
        assert AuditSessionStatus.PENDING.value == "pending"
        assert AuditSessionStatus.RUNNING.value == "running"
        assert AuditSessionStatus.PAUSED.value == "paused"
        assert AuditSessionStatus.COMPLETED.value == "completed"
        assert AuditSessionStatus.FAILED.value == "failed"
        assert AuditSessionStatus.CANCELLED.value == "cancelled"

    def test_finding_workflow_status(self):
        """Test FindingWorkflowStatus values."""
        assert FindingWorkflowStatus.OPEN.value == "open"
        assert FindingWorkflowStatus.TRIAGING.value == "triaging"
        assert FindingWorkflowStatus.INVESTIGATING.value == "investigating"
        assert FindingWorkflowStatus.REMEDIATING.value == "remediating"
        assert FindingWorkflowStatus.RESOLVED.value == "resolved"
        assert FindingWorkflowStatus.FALSE_POSITIVE.value == "false_positive"
        assert FindingWorkflowStatus.ACCEPTED_RISK.value == "accepted_risk"
        assert FindingWorkflowStatus.DUPLICATE.value == "duplicate"

    def test_audit_report_format(self):
        """Test AuditReportFormat values."""
        assert AuditReportFormat.JSON.value == "json"
        assert AuditReportFormat.MARKDOWN.value == "markdown"
        assert AuditReportFormat.HTML.value == "html"
        assert AuditReportFormat.PDF.value == "pdf"


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_basic_creation(self):
        """Test basic AgentMessage creation."""
        msg = AgentMessage(agent_id="claude", content="Hello world")
        assert msg.agent_id == "claude"
        assert msg.content == "Hello world"
        assert msg.round is None
        assert msg.timestamp is None
        assert msg.token_count is None

    def test_agent_alias(self):
        """Test that 'agent' alias works for agent_id."""
        msg = AgentMessage(agent="gpt-4", content="Test")
        assert msg.agent_id == "gpt-4"

    def test_round_alias(self):
        """Test that 'round_number' alias works for round."""
        msg = AgentMessage(agent_id="claude", content="Test", round_number=3)
        assert msg.round == 3

    def test_full_creation(self):
        """Test AgentMessage with all fields."""
        now = datetime.now(timezone.utc)
        msg = AgentMessage(
            agent_id="claude",
            content="Detailed response",
            round=2,
            timestamp=now,
            token_count=150,
        )
        assert msg.round == 2
        assert msg.timestamp == now
        assert msg.token_count == 150


class TestVote:
    """Tests for Vote model."""

    def test_basic_creation(self):
        """Test basic Vote creation."""
        vote = Vote(agent_id="claude", position="yes", confidence=0.85)
        assert vote.agent_id == "claude"
        assert vote.position == "yes"
        assert vote.confidence == 0.85
        assert vote.reasoning is None

    def test_with_reasoning(self):
        """Test Vote with reasoning."""
        vote = Vote(
            agent_id="gpt-4",
            position="no",
            confidence=0.7,
            reasoning="Based on the evidence...",
        )
        assert vote.reasoning == "Based on the evidence..."

    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            Vote(agent_id="test", position="yes", confidence=1.5)
        with pytest.raises(ValueError):
            Vote(agent_id="test", position="yes", confidence=-0.1)


class TestConsensusResult:
    """Tests for ConsensusResult model."""

    def test_basic_creation(self):
        """Test basic ConsensusResult creation."""
        result = ConsensusResult(reached=True)
        assert result.reached is True
        assert result.agreement is None
        assert result.final_answer is None
        assert result.supporting_agents == []
        assert result.dissenting_agents == []
        assert result.votes == []

    def test_field_sync_agreement_to_confidence(self):
        """Test that agreement syncs to confidence."""
        result = ConsensusResult(reached=True, agreement=0.9)
        assert result.agreement == 0.9
        assert result.confidence == 0.9

    def test_field_sync_confidence_to_agreement(self):
        """Test that confidence syncs to agreement."""
        result = ConsensusResult(reached=True, confidence=0.85)
        assert result.confidence == 0.85
        assert result.agreement == 0.85

    def test_field_sync_final_answer_to_conclusion(self):
        """Test that final_answer syncs to conclusion."""
        result = ConsensusResult(reached=True, final_answer="The answer is 42")
        assert result.final_answer == "The answer is 42"
        assert result.conclusion == "The answer is 42"

    def test_field_sync_conclusion_to_final_answer(self):
        """Test that conclusion syncs to final_answer."""
        result = ConsensusResult(reached=True, conclusion="Agreed on X")
        assert result.conclusion == "Agreed on X"
        assert result.final_answer == "Agreed on X"

    def test_full_consensus(self):
        """Test full consensus with all fields."""
        votes = [
            Vote(agent_id="claude", position="yes", confidence=0.9),
            Vote(agent_id="gpt-4", position="yes", confidence=0.85),
        ]
        result = ConsensusResult(
            reached=True,
            agreement=0.95,
            final_answer="Consensus reached",
            supporting_agents=["claude", "gpt-4"],
            dissenting_agents=[],
            votes=votes,
        )
        assert len(result.votes) == 2
        assert len(result.supporting_agents) == 2


class TestDebateRound:
    """Tests for DebateRound model."""

    def test_basic_creation(self):
        """Test basic DebateRound creation."""
        round_data = DebateRound(round_number=1)
        assert round_data.round_number == 1
        assert round_data.messages == []
        assert round_data.critiques == []

    def test_round_alias(self):
        """Test that 'round' alias works for round_number."""
        round_data = DebateRound(round=2)
        assert round_data.round_number == 2

    def test_with_messages(self):
        """Test DebateRound with messages."""
        messages = [
            AgentMessage(agent_id="claude", content="First message"),
            AgentMessage(agent_id="gpt-4", content="Second message"),
        ]
        round_data = DebateRound(round_number=1, messages=messages)
        assert len(round_data.messages) == 2


class TestDebate:
    """Tests for Debate model."""

    def test_basic_creation(self):
        """Test basic Debate creation."""
        debate = Debate(
            debate_id="test-123",
            task="What is the meaning of life?",
            status=DebateStatus.PENDING,
        )
        assert debate.debate_id == "test-123"
        assert debate.task == "What is the meaning of life?"
        assert debate.status == DebateStatus.PENDING
        assert debate.agents == []
        assert debate.rounds == []

    def test_id_alias(self):
        """Test that 'id' alias works for debate_id."""
        debate = Debate(
            id="alias-test",
            task="Test task",
            status=DebateStatus.RUNNING,
        )
        assert debate.debate_id == "alias-test"

    def test_rounds_coercion_from_none(self):
        """Test that None rounds are coerced to empty list."""
        debate = Debate(
            debate_id="test",
            task="Task",
            status=DebateStatus.PENDING,
            rounds=None,
        )
        assert debate.rounds == []

    def test_rounds_coercion_from_int(self):
        """Test that int rounds are coerced to empty list."""
        debate = Debate(
            debate_id="test",
            task="Task",
            status=DebateStatus.PENDING,
            rounds=3,
        )
        assert debate.rounds == []

    def test_derive_consensus_from_proof(self):
        """Test consensus derivation from consensus_proof."""
        debate = Debate(
            debate_id="test",
            task="Task",
            status=DebateStatus.COMPLETED,
            consensus_proof={
                "reached": True,
                "confidence": 0.9,
                "final_answer": "The answer",
                "vote_breakdown": {"claude": True, "gpt-4": False},
            },
        )
        assert debate.consensus is not None
        assert debate.consensus.reached is True
        assert debate.consensus.confidence == 0.9
        assert debate.consensus.final_answer == "The answer"
        assert "claude" in debate.consensus.supporting_agents
        assert "gpt-4" in debate.consensus.dissenting_agents


class TestDebateCreateRequest:
    """Tests for DebateCreateRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = DebateCreateRequest(task="Test task")
        assert request.task == "Test task"
        assert request.agents == ["anthropic-api", "openai-api"]
        assert request.rounds == 3
        assert request.consensus == ConsensusType.MAJORITY
        assert request.context is None
        assert request.metadata == {}

    def test_custom_values(self):
        """Test custom values."""
        request = DebateCreateRequest(
            task="Custom task",
            agents=["claude", "gpt-4", "gemini"],
            rounds=5,
            consensus=ConsensusType.UNANIMOUS,
            context="Additional context",
            metadata={"key": "value"},
        )
        assert len(request.agents) == 3
        assert request.rounds == 5
        assert request.consensus == ConsensusType.UNANIMOUS

    def test_rounds_bounds(self):
        """Test rounds must be between 1 and 10."""
        with pytest.raises(ValueError):
            DebateCreateRequest(task="Test", rounds=0)
        with pytest.raises(ValueError):
            DebateCreateRequest(task="Test", rounds=11)


class TestAgentProfile:
    """Tests for AgentProfile model."""

    def test_basic_creation(self):
        """Test basic AgentProfile creation."""
        profile = AgentProfile(
            agent_id="claude",
            name="Claude",
            provider="anthropic",
        )
        assert profile.agent_id == "claude"
        assert profile.elo_rating == 1500
        assert profile.matches_played == 0
        assert profile.win_rate == 0.0
        assert profile.available is True
        assert profile.capabilities == []

    def test_full_profile(self):
        """Test full agent profile."""
        profile = AgentProfile(
            agent_id="gpt-4",
            name="GPT-4",
            provider="openai",
            elo_rating=1650,
            matches_played=100,
            win_rate=0.65,
            available=True,
            capabilities=["reasoning", "coding", "analysis"],
        )
        assert profile.elo_rating == 1650
        assert len(profile.capabilities) == 3


class TestLeaderboardEntry:
    """Tests for LeaderboardEntry model."""

    def test_basic_creation(self):
        """Test basic LeaderboardEntry creation."""
        entry = LeaderboardEntry(
            rank=1,
            agent_id="claude",
            elo_rating=1700,
            matches_played=50,
            win_rate=0.72,
        )
        assert entry.rank == 1
        assert entry.recent_trend == "stable"

    def test_with_trend(self):
        """Test entry with custom trend."""
        entry = LeaderboardEntry(
            rank=2,
            agent_id="gpt-4",
            elo_rating=1650,
            matches_played=45,
            win_rate=0.68,
            recent_trend="up",
        )
        assert entry.recent_trend == "up"


class TestFinding:
    """Tests for Finding model."""

    def test_basic_creation(self):
        """Test basic Finding creation."""
        finding = Finding()
        assert finding.severity == "medium"
        assert finding.category == "general"

    def test_title_to_description_sync(self):
        """Test title syncs to description."""
        finding = Finding(title="Security issue")
        assert finding.title == "Security issue"
        assert finding.description == "Security issue"

    def test_description_to_title_sync(self):
        """Test description syncs to title."""
        finding = Finding(description="Found a bug")
        assert finding.description == "Found a bug"
        assert finding.title == "Found a bug"

    def test_mitigation_to_suggestion_sync(self):
        """Test mitigation syncs to suggestion."""
        finding = Finding(suggestion="Fix the bug")
        assert finding.suggestion == "Fix the bug"
        assert finding.mitigation == "Fix the bug"


class TestGauntletReceipt:
    """Tests for GauntletReceipt model."""

    def test_basic_creation(self):
        """Test basic GauntletReceipt creation."""
        receipt = GauntletReceipt()
        assert receipt.receipt_id is None
        assert receipt.findings == []

    def test_score_sync(self):
        """Test risk_score and score sync."""
        receipt = GauntletReceipt(score=0.75)
        assert receipt.score == 0.75
        assert receipt.risk_score == 0.75

        receipt2 = GauntletReceipt(risk_score=0.6)
        assert receipt2.risk_score == 0.6
        assert receipt2.score == 0.6

    def test_findings_coercion_from_strings(self):
        """Test findings coercion from string list."""
        receipt = GauntletReceipt(findings=["Issue 1", "Issue 2"])
        assert len(receipt.findings) == 2
        assert receipt.findings[0].title == "Issue 1"
        assert receipt.findings[0].severity == "low"

    def test_findings_coercion_from_none(self):
        """Test findings coercion from None."""
        receipt = GauntletReceipt(findings=None)
        assert receipt.findings == []


class TestGauntletRunRequest:
    """Tests for GauntletRunRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = GauntletRunRequest(input_content="Test content")
        assert request.input_content == "Test content"
        assert request.input_type == "text"
        assert request.persona == "security"
        assert request.profile == "default"


class TestHealthCheck:
    """Tests for HealthCheck model."""

    def test_basic_creation(self):
        """Test basic HealthCheck creation."""
        health = HealthCheck(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5,
        )
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 3600.5
        assert health.components == {}


class TestAPIError:
    """Tests for APIError model."""

    def test_basic_creation(self):
        """Test basic APIError creation."""
        error = APIError(
            error="Something went wrong",
            code="ERR_500",
        )
        assert error.error == "Something went wrong"
        assert error.code == "ERR_500"
        assert error.details is None
        assert error.suggestion is None


class TestGraphDebateModels:
    """Tests for graph debate models."""

    def test_graph_debate_node(self):
        """Test GraphDebateNode creation."""
        node = GraphDebateNode(
            node_id="node-1",
            content="Initial proposal",
            agent_id="claude",
            node_type="proposal",
        )
        assert node.node_id == "node-1"
        assert node.parent_id is None
        assert node.round == 0

    def test_graph_debate_branch(self):
        """Test GraphDebateBranch creation."""
        branch = GraphDebateBranch(
            branch_id="main",
            name="Main Branch",
        )
        assert branch.nodes == []
        assert branch.is_main is False

    def test_graph_debate_create_request(self):
        """Test GraphDebateCreateRequest defaults."""
        request = GraphDebateCreateRequest(task="Test task")
        assert request.agents == ["anthropic-api", "openai-api"]
        assert request.max_rounds == 5
        assert request.branch_threshold == 0.5
        assert request.max_branches == 5


class TestMatrixDebateModels:
    """Tests for matrix debate models."""

    def test_matrix_scenario(self):
        """Test MatrixScenario creation."""
        scenario = MatrixScenario(name="Baseline")
        assert scenario.parameters == {}
        assert scenario.constraints == []
        assert scenario.is_baseline is False

    def test_matrix_scenario_result(self):
        """Test MatrixScenarioResult creation."""
        result = MatrixScenarioResult(scenario_name="Test")
        assert result.key_findings == []
        assert result.differences_from_baseline == []

    def test_matrix_conclusion(self):
        """Test MatrixConclusion creation."""
        conclusion = MatrixConclusion()
        assert conclusion.universal == []
        assert conclusion.conditional == {}
        assert conclusion.contradictions == []

    def test_matrix_debate_create_request(self):
        """Test MatrixDebateCreateRequest defaults."""
        request = MatrixDebateCreateRequest(task="Test")
        assert request.agents == ["anthropic-api", "openai-api"]
        assert request.scenarios == []
        assert request.max_rounds == 3


class TestVerificationModels:
    """Tests for verification models."""

    def test_verify_claim_request(self):
        """Test VerifyClaimRequest defaults."""
        request = VerifyClaimRequest(claim="2 + 2 = 4")
        assert request.backend == "z3"
        assert request.timeout == 30

    def test_verify_claim_response(self):
        """Test VerifyClaimResponse creation."""
        response = VerifyClaimResponse(
            status=VerificationStatus.VALID,
            claim="2 + 2 = 4",
        )
        assert response.duration_ms == 0
        assert response.proof is None

    def test_verification_backend_status(self):
        """Test VerificationBackendStatus creation."""
        status = VerificationBackendStatus(
            name="z3",
            available=True,
            version="4.12.0",
        )
        assert status.available is True


class TestMemoryModels:
    """Tests for memory analytics models."""

    def test_memory_tier_stats(self):
        """Test MemoryTierStats defaults."""
        stats = MemoryTierStats(tier_name="fast")
        assert stats.entry_count == 0
        assert stats.hit_rate == 0.0

    def test_memory_recommendation(self):
        """Test MemoryRecommendation creation."""
        rec = MemoryRecommendation(
            type="promotion",
            description="Promote frequently accessed items",
            impact="high",
        )
        assert rec.type == "promotion"

    def test_memory_analytics_response(self):
        """Test MemoryAnalyticsResponse defaults."""
        response = MemoryAnalyticsResponse()
        assert response.tiers == []
        assert response.period_days == 30


class TestReplayModels:
    """Tests for replay models."""

    def test_replay_summary(self):
        """Test ReplaySummary creation."""
        now = datetime.now(timezone.utc)
        summary = ReplaySummary(
            replay_id="replay-1",
            debate_id="debate-1",
            task="Test task",
            created_at=now,
        )
        assert summary.duration_seconds == 0
        assert summary.agent_count == 0

    def test_replay_event(self):
        """Test ReplayEvent creation."""
        now = datetime.now(timezone.utc)
        event = ReplayEvent(
            event_type="message",
            timestamp=now,
        )
        assert event.agent_id is None
        assert event.metadata == {}


class TestDocumentModels:
    """Tests for document models."""

    def test_document(self):
        """Test Document creation."""
        now = datetime.now(timezone.utc)
        doc = Document(
            id="doc-1",
            filename="test.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            created_at=now,
        )
        assert doc.status == DocumentStatus.PENDING
        assert doc.chunk_count == 0

    def test_document_chunk(self):
        """Test DocumentChunk creation."""
        chunk = DocumentChunk(
            id="chunk-1",
            document_id="doc-1",
            content="Some text",
            chunk_index=0,
        )
        assert chunk.token_count == 0

    def test_document_upload_response(self):
        """Test DocumentUploadResponse creation."""
        response = DocumentUploadResponse(
            document_id="doc-1",
            filename="test.pdf",
        )
        assert response.status == DocumentStatus.PENDING

    def test_processing_stats(self):
        """Test ProcessingStats defaults."""
        stats = ProcessingStats()
        assert stats.total_documents == 0
        assert stats.total_tokens == 0


class TestAuditModels:
    """Tests for audit models."""

    def test_audit_finding(self):
        """Test AuditFinding creation."""
        finding = AuditFinding(
            session_id="session-1",
            audit_type=AuditType.SECURITY,
            category="authentication",
            severity=FindingSeverity.HIGH,
            title="Weak password policy",
            description="Password requirements are too lenient",
        )
        assert finding.confidence == 0.0

    def test_audit_session(self):
        """Test AuditSession creation."""
        now = datetime.now(timezone.utc)
        session = AuditSession(
            id="session-1",
            created_at=now,
        )
        assert session.status == AuditSessionStatus.PENDING
        assert session.progress == 0.0

    def test_audit_session_create_request(self):
        """Test AuditSessionCreateRequest defaults."""
        request = AuditSessionCreateRequest(document_ids=["doc-1"])
        assert len(request.audit_types) == 4
        assert request.model == "gemini-1.5-flash"

    def test_audit_preset(self):
        """Test AuditPreset creation."""
        preset = AuditPreset(
            name="HIPAA",
            description="Healthcare compliance preset",
        )
        assert preset.consensus_threshold == 0.8

    def test_audit_type_capabilities(self):
        """Test AuditTypeCapabilities defaults."""
        caps = AuditTypeCapabilities()
        assert caps.supports_chunk_analysis is True
        assert caps.supports_cross_document is False
        assert caps.requires_llm is True
