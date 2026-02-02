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
    GauntletPersonaCategory,
    GauntletResultStatus,
    GauntletRunStatus,
    GauntletVerdict,
    VerificationBackend,
    VerificationStatus,
    # Core models
    AgentCalibration,
    AgentConsistency,
    AgentFlip,
    AgentMessage,
    AgentMoment,
    AgentNetwork,
    AgentPerformance,
    AgentPosition,
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
    DomainRating,
    Finding,
    GauntletComparison,
    GauntletHeatmapExtended,
    GauntletPersona,
    GauntletReceipt,
    GauntletResult,
    GauntletRun,
    GauntletRunRequest,
    GauntletRunResponse,
    GraphDebate,
    GraphDebateBranch,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
    GraphDebateNode,
    HeadToHeadStats,
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
    OpponentBriefing,
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
        from aragora.config import DEFAULT_AGENTS, DEFAULT_CONSENSUS, DEFAULT_ROUNDS

        expected_agents = [a.strip() for a in DEFAULT_AGENTS.split(",") if a.strip()]
        expected_consensus = ConsensusType(DEFAULT_CONSENSUS)
        request = DebateCreateRequest(task="Test task")
        assert request.task == "Test task"
        assert request.agents == expected_agents
        assert request.rounds == DEFAULT_ROUNDS
        assert request.consensus == expected_consensus
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
        """Test rounds must be between 1 and MAX_ROUNDS."""
        from aragora.config import MAX_ROUNDS

        with pytest.raises(ValueError):
            DebateCreateRequest(task="Test", rounds=0)
        with pytest.raises(ValueError):
            DebateCreateRequest(task="Test", rounds=MAX_ROUNDS + 1)


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


# =============================================================================
# Extended Agent Models
# =============================================================================


class TestAgentCalibration:
    """Tests for AgentCalibration model."""

    def test_basic_creation(self):
        """Test basic AgentCalibration creation."""
        cal = AgentCalibration(agent="claude", overall_score=0.85)
        assert cal.agent == "claude"
        assert cal.overall_score == 0.85
        assert cal.domain_scores == {}
        assert cal.confidence_accuracy == 0.0
        assert cal.last_calibrated is None
        assert cal.sample_size == 0

    def test_with_all_fields(self):
        """Test AgentCalibration with all optional fields populated."""
        now = datetime.now(timezone.utc)
        cal = AgentCalibration(
            agent="gpt-4",
            overall_score=0.92,
            domain_scores={"security": 0.95, "compliance": 0.88},
            confidence_accuracy=0.78,
            last_calibrated=now,
            sample_size=150,
        )
        assert cal.agent == "gpt-4"
        assert cal.overall_score == 0.92
        assert cal.domain_scores == {"security": 0.95, "compliance": 0.88}
        assert cal.confidence_accuracy == 0.78
        assert cal.last_calibrated == now
        assert cal.sample_size == 150

    def test_default_values(self):
        """Test default values are correct."""
        cal = AgentCalibration(agent="test", overall_score=0.5)
        assert cal.domain_scores == {}
        assert cal.confidence_accuracy == 0.0
        assert cal.sample_size == 0

    def test_score_bounds_valid(self):
        """Test overall_score accepts valid bounds."""
        cal_min = AgentCalibration(agent="test", overall_score=0.0)
        assert cal_min.overall_score == 0.0
        cal_max = AgentCalibration(agent="test", overall_score=1.0)
        assert cal_max.overall_score == 1.0

    def test_score_bounds_invalid(self):
        """Test overall_score rejects invalid values."""
        with pytest.raises(ValueError):
            AgentCalibration(agent="test", overall_score=1.5)
        with pytest.raises(ValueError):
            AgentCalibration(agent="test", overall_score=-0.1)


class TestAgentPerformance:
    """Tests for AgentPerformance model."""

    def test_basic_creation(self):
        """Test basic AgentPerformance creation."""
        perf = AgentPerformance(agent="claude", win_rate=0.65, loss_rate=0.25, draw_rate=0.10)
        assert perf.agent == "claude"
        assert perf.win_rate == 0.65
        assert perf.loss_rate == 0.25
        assert perf.draw_rate == 0.10
        assert perf.elo_trend == []
        assert perf.elo_change_30d == 0.0
        assert perf.total_debates == 0

    def test_with_all_fields(self):
        """Test AgentPerformance with all optional fields populated."""
        perf = AgentPerformance(
            agent="gpt-4",
            win_rate=0.70,
            loss_rate=0.20,
            draw_rate=0.10,
            elo_trend=[1500.0, 1520.0, 1545.0, 1560.0],
            elo_change_30d=60.0,
            avg_confidence=0.85,
            avg_round_duration_ms=1500,
            total_debates=100,
            recent_results=[{"debate_id": "d1", "result": "win"}],
        )
        assert perf.elo_trend == [1500.0, 1520.0, 1545.0, 1560.0]
        assert perf.elo_change_30d == 60.0
        assert perf.avg_confidence == 0.85
        assert perf.avg_round_duration_ms == 1500
        assert perf.total_debates == 100
        assert len(perf.recent_results) == 1

    def test_default_values(self):
        """Test default values are correct."""
        perf = AgentPerformance(agent="test", win_rate=0.5, loss_rate=0.3, draw_rate=0.2)
        assert perf.elo_trend == []
        assert perf.elo_change_30d == 0.0
        assert perf.avg_confidence == 0.0
        assert perf.avg_round_duration_ms == 0
        assert perf.total_debates == 0
        assert perf.recent_results == []

    def test_rate_bounds_valid(self):
        """Test rate fields accept valid bounds."""
        perf = AgentPerformance(agent="test", win_rate=0.0, loss_rate=0.0, draw_rate=1.0)
        assert perf.win_rate == 0.0
        assert perf.loss_rate == 0.0
        assert perf.draw_rate == 1.0

    def test_rate_bounds_invalid(self):
        """Test rate fields reject invalid values."""
        with pytest.raises(ValueError):
            AgentPerformance(agent="test", win_rate=1.5, loss_rate=0.0, draw_rate=0.0)
        with pytest.raises(ValueError):
            AgentPerformance(agent="test", win_rate=0.5, loss_rate=-0.1, draw_rate=0.0)


class TestHeadToHeadStats:
    """Tests for HeadToHeadStats model."""

    def test_basic_creation(self):
        """Test basic HeadToHeadStats creation."""
        stats = HeadToHeadStats(agent="claude", opponent="gpt-4")
        assert stats.agent == "claude"
        assert stats.opponent == "gpt-4"
        assert stats.total_matchups == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0
        assert stats.win_rate == 0.0

    def test_with_all_fields(self):
        """Test HeadToHeadStats with all optional fields populated."""
        stats = HeadToHeadStats(
            agent="claude",
            opponent="gpt-4",
            total_matchups=50,
            wins=28,
            losses=18,
            draws=4,
            win_rate=0.56,
            avg_margin=0.12,
            recent_matchups=[{"debate_id": "d1", "result": "win"}],
            domain_breakdown={
                "security": {"wins": 10, "losses": 5},
                "compliance": {"wins": 8, "losses": 7},
            },
        )
        assert stats.total_matchups == 50
        assert stats.wins == 28
        assert stats.losses == 18
        assert stats.draws == 4
        assert stats.win_rate == 0.56
        assert stats.avg_margin == 0.12
        assert len(stats.recent_matchups) == 1
        assert "security" in stats.domain_breakdown
        assert stats.domain_breakdown["security"]["wins"] == 10

    def test_default_values(self):
        """Test default values are correct."""
        stats = HeadToHeadStats(agent="test", opponent="rival")
        assert stats.total_matchups == 0
        assert stats.win_rate == 0.0
        assert stats.avg_margin == 0.0
        assert stats.recent_matchups == []
        assert stats.domain_breakdown == {}

    def test_nested_domain_breakdown(self):
        """Test nested domain_breakdown dict structure."""
        stats = HeadToHeadStats(
            agent="claude",
            opponent="gpt-4",
            domain_breakdown={
                "coding": {"wins": 15, "losses": 10, "draws": 2},
                "reasoning": {"wins": 12, "losses": 8, "draws": 5},
            },
        )
        assert stats.domain_breakdown["coding"]["wins"] == 15
        assert stats.domain_breakdown["reasoning"]["draws"] == 5


class TestGauntletRun:
    """Tests for GauntletRun model."""

    def test_basic_creation(self):
        """Test basic GauntletRun creation."""
        run = GauntletRun(id="run-123")
        assert run.id == "run-123"
        assert run.name is None
        assert run.status == GauntletRunStatus.PENDING
        assert run.config == {}
        assert run.progress == {}
        assert run.results_summary is None

    def test_with_all_fields(self):
        """Test GauntletRun with all optional fields populated."""
        now = datetime.now(timezone.utc)
        run = GauntletRun(
            id="run-456",
            name="Security Audit Run",
            status=GauntletRunStatus.COMPLETED,
            created_at=now,
            started_at=now,
            completed_at=now,
            config={"persona": "adversarial", "intensity": "high"},
            progress={"completed": 50, "total": 50},
            results_summary={"pass_rate": 0.92, "findings": 4},
            metadata={"version": "1.0"},
        )
        assert run.name == "Security Audit Run"
        assert run.status == GauntletRunStatus.COMPLETED
        assert run.created_at == now
        assert run.progress["completed"] == 50
        assert run.results_summary["pass_rate"] == 0.92

    def test_default_values(self):
        """Test default values are correct."""
        run = GauntletRun(id="test")
        assert run.status == GauntletRunStatus.PENDING
        assert run.config == {}
        assert run.progress == {}
        assert run.metadata == {}

    def test_status_enum_values(self):
        """Test all status enum values work."""
        for status in GauntletRunStatus:
            run = GauntletRun(id="test", status=status)
            assert run.status == status


class TestGauntletPersona:
    """Tests for GauntletPersona model."""

    def test_basic_creation(self):
        """Test basic GauntletPersona creation."""
        persona = GauntletPersona(id="persona-1", name="Adversarial Tester")
        assert persona.id == "persona-1"
        assert persona.name == "Adversarial Tester"
        assert persona.description == ""
        assert persona.category == GauntletPersonaCategory.CUSTOM
        assert persona.severity == "medium"
        assert persona.enabled is True

    def test_with_all_fields(self):
        """Test GauntletPersona with all optional fields populated."""
        persona = GauntletPersona(
            id="persona-2",
            name="Security Auditor",
            description="Tests for security vulnerabilities",
            category=GauntletPersonaCategory.ADVERSARIAL,
            severity="critical",
            tags=["security", "injection", "xss"],
            example_prompts=["Try SQL injection", "Test XSS vectors"],
            enabled=True,
        )
        assert persona.description == "Tests for security vulnerabilities"
        assert persona.category == GauntletPersonaCategory.ADVERSARIAL
        assert persona.severity == "critical"
        assert len(persona.tags) == 3
        assert len(persona.example_prompts) == 2

    def test_default_values(self):
        """Test default values are correct."""
        persona = GauntletPersona(id="test", name="Test")
        assert persona.description == ""
        assert persona.category == GauntletPersonaCategory.CUSTOM
        assert persona.severity == "medium"
        assert persona.tags == []
        assert persona.example_prompts == []
        assert persona.enabled is True

    def test_category_enum_values(self):
        """Test all category enum values work."""
        for category in GauntletPersonaCategory:
            persona = GauntletPersona(id="test", name="Test", category=category)
            assert persona.category == category


class TestGauntletResult:
    """Tests for GauntletResult model."""

    def test_basic_creation(self):
        """Test basic GauntletResult creation."""
        result = GauntletResult(id="result-1", gauntlet_id="gauntlet-123")
        assert result.id == "result-1"
        assert result.gauntlet_id == "gauntlet-123"
        assert result.scenario == ""
        assert result.status == GauntletResultStatus.PASS
        assert result.verdict == ""
        assert result.confidence == 0.0
        assert result.risk_level == "low"

    def test_with_all_fields(self):
        """Test GauntletResult with all optional fields populated."""
        now = datetime.now(timezone.utc)
        result = GauntletResult(
            id="result-2",
            gauntlet_id="gauntlet-456",
            scenario="SQL injection test",
            persona="adversarial",
            status=GauntletResultStatus.FAIL,
            verdict="Vulnerability detected",
            confidence=0.95,
            risk_level="critical",
            duration_ms=1500,
            debate_id="debate-789",
            findings=[{"type": "injection", "severity": "high"}],
            timestamp=now,
        )
        assert result.scenario == "SQL injection test"
        assert result.persona == "adversarial"
        assert result.status == GauntletResultStatus.FAIL
        assert result.verdict == "Vulnerability detected"
        assert result.confidence == 0.95
        assert result.risk_level == "critical"
        assert result.duration_ms == 1500
        assert result.debate_id == "debate-789"
        assert len(result.findings) == 1
        assert result.timestamp == now

    def test_default_values(self):
        """Test default values are correct."""
        result = GauntletResult(id="test", gauntlet_id="test")
        assert result.scenario == ""
        assert result.persona is None
        assert result.status == GauntletResultStatus.PASS
        assert result.verdict == ""
        assert result.confidence == 0.0
        assert result.risk_level == "low"
        assert result.duration_ms == 0
        assert result.debate_id is None
        assert result.findings == []
        assert result.timestamp is None

    def test_confidence_bounds_valid(self):
        """Test confidence accepts valid values."""
        result_min = GauntletResult(id="test", gauntlet_id="test", confidence=0.0)
        assert result_min.confidence == 0.0
        result_max = GauntletResult(id="test", gauntlet_id="test", confidence=0.99)
        assert result_max.confidence == 0.99

    def test_status_enum_values(self):
        """Test all status enum values work."""
        for status in GauntletResultStatus:
            result = GauntletResult(id="test", gauntlet_id="test", status=status)
            assert result.status == status


# =============================================================================
# Medium Priority Extended Agent Models
# =============================================================================


class TestOpponentBriefing:
    """Tests for OpponentBriefing model."""

    def test_basic_creation(self):
        """Test basic OpponentBriefing creation."""
        briefing = OpponentBriefing(agent="claude", opponent="gpt-4")
        assert briefing.agent == "claude"
        assert briefing.opponent == "gpt-4"
        assert briefing.opponent_profile == {}
        assert briefing.historical_summary == ""
        assert briefing.recommended_strategy == ""
        assert briefing.key_insights == []
        assert briefing.confidence == 0.0

    def test_with_all_fields(self):
        """Test OpponentBriefing with all optional fields populated."""
        briefing = OpponentBriefing(
            agent="claude",
            opponent="gpt-4",
            opponent_profile={"elo": 1650, "style": "analytical"},
            historical_summary="Claude has won 60% of debates against GPT-4",
            recommended_strategy="Focus on logical reasoning",
            key_insights=["Tends to concede on ethical issues", "Strong on technical details"],
            confidence=0.85,
        )
        assert briefing.opponent_profile["elo"] == 1650
        assert "60%" in briefing.historical_summary
        assert len(briefing.key_insights) == 2
        assert briefing.confidence == 0.85


class TestAgentConsistency:
    """Tests for AgentConsistency model."""

    def test_basic_creation(self):
        """Test basic AgentConsistency creation."""
        consistency = AgentConsistency(agent="claude")
        assert consistency.agent == "claude"
        assert consistency.overall_consistency == 0.0
        assert consistency.position_stability == 0.0
        assert consistency.flip_rate == 0.0
        assert consistency.consistency_by_domain == {}
        assert consistency.volatility_index == 0.0
        assert consistency.sample_size == 0

    def test_with_all_fields(self):
        """Test AgentConsistency with all optional fields populated."""
        consistency = AgentConsistency(
            agent="gpt-4",
            overall_consistency=0.92,
            position_stability=0.88,
            flip_rate=0.05,
            consistency_by_domain={"security": 0.95, "ethics": 0.80},
            volatility_index=0.12,
            sample_size=200,
        )
        assert consistency.overall_consistency == 0.92
        assert consistency.flip_rate == 0.05
        assert consistency.consistency_by_domain["security"] == 0.95
        assert consistency.sample_size == 200


class TestAgentFlip:
    """Tests for AgentFlip model."""

    def test_basic_creation(self):
        """Test basic AgentFlip creation."""
        flip = AgentFlip(flip_id="flip-1", agent="claude", debate_id="debate-123")
        assert flip.flip_id == "flip-1"
        assert flip.agent == "claude"
        assert flip.debate_id == "debate-123"
        assert flip.topic == ""
        assert flip.original_position == ""
        assert flip.new_position == ""
        assert flip.flip_reason is None
        assert flip.round_number == 0
        assert flip.was_justified is False

    def test_with_all_fields(self):
        """Test AgentFlip with all optional fields populated."""
        now = datetime.now(timezone.utc)
        flip = AgentFlip(
            flip_id="flip-2",
            agent="gpt-4",
            debate_id="debate-456",
            topic="Climate change policy",
            original_position="Against carbon tax",
            new_position="For carbon tax",
            flip_reason="Convinced by economic evidence",
            round_number=3,
            timestamp=now,
            was_justified=True,
        )
        assert flip.topic == "Climate change policy"
        assert flip.original_position == "Against carbon tax"
        assert flip.new_position == "For carbon tax"
        assert flip.flip_reason == "Convinced by economic evidence"
        assert flip.round_number == 3
        assert flip.timestamp == now
        assert flip.was_justified is True


class TestAgentNetwork:
    """Tests for AgentNetwork model."""

    def test_basic_creation(self):
        """Test basic AgentNetwork creation."""
        network = AgentNetwork(agent="claude")
        assert network.agent == "claude"
        assert network.allies == []
        assert network.rivals == []
        assert network.neutrals == []
        assert network.cluster_id is None
        assert network.network_position == "peripheral"

    def test_with_all_fields(self):
        """Test AgentNetwork with all optional fields populated."""
        network = AgentNetwork(
            agent="claude",
            allies=[{"agent": "gemini", "affinity": 0.8}],
            rivals=[{"agent": "grok", "rivalry": 0.6}],
            neutrals=["gpt-4", "mistral"],
            cluster_id="cluster-1",
            network_position="central",
        )
        assert len(network.allies) == 1
        assert len(network.rivals) == 1
        assert len(network.neutrals) == 2
        assert network.cluster_id == "cluster-1"
        assert network.network_position == "central"


class TestAgentMoment:
    """Tests for AgentMoment model."""

    def test_basic_creation(self):
        """Test basic AgentMoment creation."""
        moment = AgentMoment(
            moment_id="moment-1", agent="claude", debate_id="debate-123", type="breakthrough"
        )
        assert moment.moment_id == "moment-1"
        assert moment.agent == "claude"
        assert moment.debate_id == "debate-123"
        assert moment.type == "breakthrough"
        assert moment.description == ""
        assert moment.impact_score == 0.0
        assert moment.timestamp is None
        assert moment.context == {}

    def test_with_all_fields(self):
        """Test AgentMoment with all optional fields populated."""
        now = datetime.now(timezone.utc)
        moment = AgentMoment(
            moment_id="moment-2",
            agent="gpt-4",
            debate_id="debate-456",
            type="decisive_argument",
            description="Presented compelling evidence that swayed consensus",
            impact_score=0.95,
            timestamp=now,
            context={"round": 3, "topic": "AI safety"},
        )
        assert moment.type == "decisive_argument"
        assert "compelling evidence" in moment.description
        assert moment.impact_score == 0.95
        assert moment.context["round"] == 3


class TestAgentPosition:
    """Tests for AgentPosition model."""

    def test_basic_creation(self):
        """Test basic AgentPosition creation."""
        position = AgentPosition(position_id="pos-1", agent="claude", debate_id="debate-123")
        assert position.position_id == "pos-1"
        assert position.agent == "claude"
        assert position.debate_id == "debate-123"
        assert position.topic == ""
        assert position.stance == ""
        assert position.confidence == 0.0
        assert position.supporting_evidence == []
        assert position.round_number == 0
        assert position.was_final is False

    def test_with_all_fields(self):
        """Test AgentPosition with all optional fields populated."""
        now = datetime.now(timezone.utc)
        position = AgentPosition(
            position_id="pos-2",
            agent="gpt-4",
            debate_id="debate-456",
            topic="Universal basic income",
            stance="Supportive with conditions",
            confidence=0.88,
            supporting_evidence=["Economic studies", "Pilot program results"],
            round_number=4,
            timestamp=now,
            was_final=True,
        )
        assert position.topic == "Universal basic income"
        assert position.stance == "Supportive with conditions"
        assert position.confidence == 0.88
        assert len(position.supporting_evidence) == 2
        assert position.was_final is True


class TestDomainRating:
    """Tests for DomainRating model."""

    def test_basic_creation(self):
        """Test basic DomainRating creation."""
        rating = DomainRating(domain="security")
        assert rating.domain == "security"
        assert rating.elo == 1500
        assert rating.matches == 0
        assert rating.win_rate == 0.0
        assert rating.avg_confidence == 0.0
        assert rating.last_active is None
        assert rating.trend == "stable"

    def test_with_all_fields(self):
        """Test DomainRating with all optional fields populated."""
        now = datetime.now(timezone.utc)
        rating = DomainRating(
            domain="compliance",
            elo=1720,
            matches=45,
            win_rate=0.73,
            avg_confidence=0.85,
            last_active=now,
            trend="rising",
        )
        assert rating.elo == 1720
        assert rating.matches == 45
        assert rating.win_rate == 0.73
        assert rating.avg_confidence == 0.85
        assert rating.last_active == now
        assert rating.trend == "rising"


# =============================================================================
# Medium Priority Extended Gauntlet Models
# =============================================================================


class TestGauntletHeatmapExtended:
    """Tests for GauntletHeatmapExtended model."""

    def test_basic_creation(self):
        """Test basic GauntletHeatmapExtended creation."""
        heatmap = GauntletHeatmapExtended(gauntlet_id="gauntlet-123")
        assert heatmap.gauntlet_id == "gauntlet-123"
        assert heatmap.dimensions == {}
        assert heatmap.matrix == []
        assert heatmap.overall_risk == 0.0
        assert heatmap.hotspots == []
        assert heatmap.generated_at is None

    def test_with_all_fields(self):
        """Test GauntletHeatmapExtended with all optional fields populated."""
        now = datetime.now(timezone.utc)
        heatmap = GauntletHeatmapExtended(
            gauntlet_id="gauntlet-456",
            dimensions={
                "personas": ["adversarial", "edge_case"],
                "scenarios": ["injection", "overflow"],
            },
            matrix=[[0.2, 0.8], [0.5, 0.3]],
            overall_risk=0.45,
            hotspots=[{"persona": "adversarial", "scenario": "injection", "risk": 0.8}],
            generated_at=now,
        )
        assert len(heatmap.dimensions["personas"]) == 2
        assert len(heatmap.matrix) == 2
        assert heatmap.overall_risk == 0.45
        assert len(heatmap.hotspots) == 1
        assert heatmap.generated_at == now


class TestGauntletComparison:
    """Tests for GauntletComparison model."""

    def test_basic_creation(self):
        """Test basic GauntletComparison creation."""
        comparison = GauntletComparison(gauntlet_a="gauntlet-1", gauntlet_b="gauntlet-2")
        assert comparison.gauntlet_a == "gauntlet-1"
        assert comparison.gauntlet_b == "gauntlet-2"
        assert comparison.comparison == {}
        assert comparison.scenario_diffs == []
        assert comparison.recommendation == "investigate"
        assert comparison.generated_at is None

    def test_with_all_fields(self):
        """Test GauntletComparison with all optional fields populated."""
        now = datetime.now(timezone.utc)
        comparison = GauntletComparison(
            gauntlet_a="gauntlet-old",
            gauntlet_b="gauntlet-new",
            comparison={
                "pass_rate_delta": 0.05,
                "new_failures": 2,
                "resolved_failures": 3,
            },
            scenario_diffs=[
                {"scenario": "injection", "old_status": "fail", "new_status": "pass"},
                {"scenario": "overflow", "old_status": "pass", "new_status": "fail"},
            ],
            recommendation="promote",
            generated_at=now,
        )
        assert comparison.comparison["pass_rate_delta"] == 0.05
        assert len(comparison.scenario_diffs) == 2
        assert comparison.recommendation == "promote"
        assert comparison.generated_at == now
