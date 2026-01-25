"""Tests for Aragora SDK type definitions."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from aragora_client.types import (
    AgentMessage,
    AgentProfile,
    AgentScore,
    ConsensusResult,
    CreateDebateRequest,
    CreateGraphDebateRequest,
    CreateMatrixDebateRequest,
    Debate,
    DebateEvent,
    DebateStatus,
    GauntletFinding,
    GauntletReceipt,
    GraphBranch,
    GraphDebate,
    HealthStatus,
    MatrixConclusion,
    MatrixDebate,
    MatrixScenario,
    MemoryAnalytics,
    MemoryTierStats,
    RoleAssignerInfo,
    RunGauntletRequest,
    ScoreAgentsRequest,
    ScorerInfo,
    SelectionPlugins,
    SelectTeamRequest,
    TeamMember,
    TeamSelection,
    TeamSelectorInfo,
    VerificationResult,
    VerificationStatus,
    VerifyClaimRequest,
)


class TestDebateStatus:
    """Tests for DebateStatus enum."""

    def test_all_values(self) -> None:
        """Test all enum values exist."""
        assert DebateStatus.PENDING == "pending"
        assert DebateStatus.RUNNING == "running"
        assert DebateStatus.COMPLETED == "completed"
        assert DebateStatus.FAILED == "failed"
        assert DebateStatus.CANCELLED == "cancelled"

    def test_is_string(self) -> None:
        """Test enum values are strings."""
        assert isinstance(DebateStatus.PENDING.value, str)
        assert DebateStatus.PENDING.value == "pending"

    def test_from_string(self) -> None:
        """Test creating enum from string."""
        assert DebateStatus("pending") == DebateStatus.PENDING
        assert DebateStatus("completed") == DebateStatus.COMPLETED

    def test_invalid_value(self) -> None:
        """Test invalid enum value raises error."""
        with pytest.raises(ValueError):
            DebateStatus("invalid")


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_all_values(self) -> None:
        """Test all enum values exist."""
        assert VerificationStatus.VALID == "valid"
        assert VerificationStatus.INVALID == "invalid"
        assert VerificationStatus.UNKNOWN == "unknown"
        assert VerificationStatus.ERROR == "error"

    def test_from_string(self) -> None:
        """Test creating enum from string."""
        assert VerificationStatus("valid") == VerificationStatus.VALID


class TestConsensusResult:
    """Tests for ConsensusResult model."""

    def test_minimal_creation(self) -> None:
        """Test creation with only required fields."""
        result = ConsensusResult(reached=True)
        assert result.reached is True
        assert result.conclusion is None
        assert result.confidence == 0.0
        assert result.supporting_agents == []
        assert result.dissenting_agents == []
        assert result.reasoning is None

    def test_full_creation(self) -> None:
        """Test creation with all fields."""
        result = ConsensusResult(
            reached=True,
            conclusion="The answer is 42",
            confidence=0.95,
            supporting_agents=["claude", "gpt4"],
            dissenting_agents=["gemini"],
            reasoning="Strong agreement on core points",
        )
        assert result.reached is True
        assert result.conclusion == "The answer is 42"
        assert result.confidence == 0.95
        assert result.supporting_agents == ["claude", "gpt4"]
        assert result.dissenting_agents == ["gemini"]

    def test_dict_serialization(self) -> None:
        """Test model serializes to dict."""
        result = ConsensusResult(reached=False, confidence=0.3)
        data = result.model_dump()
        assert data["reached"] is False
        assert data["confidence"] == 0.3


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_creation(self) -> None:
        """Test message creation."""
        now = datetime.now()
        msg = AgentMessage(
            agent_id="claude",
            content="My analysis...",
            round_number=1,
            timestamp=now,
        )
        assert msg.agent_id == "claude"
        assert msg.content == "My analysis..."
        assert msg.round_number == 1
        assert msg.timestamp == now
        assert msg.metadata == {}

    def test_with_metadata(self) -> None:
        """Test message with metadata."""
        msg = AgentMessage(
            agent_id="gpt4",
            content="Response",
            round_number=2,
            timestamp=datetime.now(),
            metadata={"tokens": 150, "latency_ms": 200},
        )
        assert msg.metadata["tokens"] == 150

    def test_required_fields(self) -> None:
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            AgentMessage(
                content="Missing agent_id",
                round_number=1,
                timestamp=datetime.now(),
            )


class TestDebate:
    """Tests for Debate model."""

    def test_minimal_creation(self) -> None:
        """Test debate with minimal fields."""
        now = datetime.now()
        debate = Debate(
            id="debate-123",
            task="What is the meaning of life?",
            status=DebateStatus.PENDING,
            agents=["claude", "gpt4"],
            created_at=now,
            updated_at=now,
        )
        assert debate.id == "debate-123"
        assert debate.task == "What is the meaning of life?"
        assert debate.status == DebateStatus.PENDING
        assert debate.agents == ["claude", "gpt4"]
        assert debate.rounds == []
        assert debate.consensus is None

    def test_with_consensus(self) -> None:
        """Test debate with consensus result."""
        now = datetime.now()
        consensus = ConsensusResult(reached=True, conclusion="42")
        debate = Debate(
            id="debate-456",
            task="Test",
            status=DebateStatus.COMPLETED,
            agents=["claude"],
            consensus=consensus,
            created_at=now,
            updated_at=now,
        )
        assert debate.consensus.reached is True
        assert debate.consensus.conclusion == "42"

    def test_status_from_string(self) -> None:
        """Test status can be set from string."""
        now = datetime.now()
        debate = Debate(
            id="test",
            task="Test",
            status="running",  # type: ignore[arg-type]
            agents=[],
            created_at=now,
            updated_at=now,
        )
        assert debate.status == DebateStatus.RUNNING


class TestGraphBranch:
    """Tests for GraphBranch model."""

    def test_creation(self) -> None:
        """Test branch creation."""
        branch = GraphBranch(
            id="branch-1",
            approach="conservative",
            agents=["claude"],
        )
        assert branch.id == "branch-1"
        assert branch.parent_id is None
        assert branch.approach == "conservative"
        assert branch.divergence_score == 0.0

    def test_with_parent(self) -> None:
        """Test branch with parent reference."""
        branch = GraphBranch(
            id="branch-2",
            parent_id="branch-1",
            approach="aggressive",
            agents=["gpt4"],
            divergence_score=0.7,
        )
        assert branch.parent_id == "branch-1"
        assert branch.divergence_score == 0.7


class TestGraphDebate:
    """Tests for GraphDebate model."""

    def test_creation(self) -> None:
        """Test graph debate creation."""
        now = datetime.now()
        debate = GraphDebate(
            id="graph-123",
            task="Complex decision",
            status=DebateStatus.RUNNING,
            created_at=now,
            updated_at=now,
        )
        assert debate.id == "graph-123"
        assert debate.branches == []

    def test_with_branches(self) -> None:
        """Test graph debate with branches."""
        now = datetime.now()
        branches = [
            GraphBranch(id="b1", approach="A", agents=["claude"]),
            GraphBranch(id="b2", parent_id="b1", approach="A.1", agents=["gpt4"]),
        ]
        debate = GraphDebate(
            id="graph-456",
            task="Test",
            status=DebateStatus.COMPLETED,
            branches=branches,
            created_at=now,
            updated_at=now,
        )
        assert len(debate.branches) == 2
        assert debate.branches[1].parent_id == "b1"


class TestMatrixConclusion:
    """Tests for MatrixConclusion model."""

    def test_defaults(self) -> None:
        """Test default values."""
        conclusion = MatrixConclusion()
        assert conclusion.universal == []
        assert conclusion.conditional == {}
        assert conclusion.contradictions == []

    def test_full_creation(self) -> None:
        """Test with all fields."""
        conclusion = MatrixConclusion(
            universal=["Always true statement"],
            conditional={"scenario_a": ["True in A"]},
            contradictions=["Conflict between X and Y"],
        )
        assert len(conclusion.universal) == 1
        assert "scenario_a" in conclusion.conditional


class TestMatrixScenario:
    """Tests for MatrixScenario model."""

    def test_creation(self) -> None:
        """Test scenario creation."""
        scenario = MatrixScenario(
            name="Base case",
            parameters={"budget": 100000, "timeline": "6 months"},
            is_baseline=True,
        )
        assert scenario.name == "Base case"
        assert scenario.parameters["budget"] == 100000
        assert scenario.is_baseline is True
        assert scenario.consensus is None


class TestMatrixDebate:
    """Tests for MatrixDebate model."""

    def test_creation(self) -> None:
        """Test matrix debate creation."""
        now = datetime.now()
        scenarios = [
            MatrixScenario(name="Base", parameters={}, is_baseline=True),
            MatrixScenario(name="Alternative", parameters={"x": 1}),
        ]
        debate = MatrixDebate(
            id="matrix-123",
            task="Scenario analysis",
            status=DebateStatus.RUNNING,
            scenarios=scenarios,
            created_at=now,
            updated_at=now,
        )
        assert len(debate.scenarios) == 2
        assert debate.conclusions is None


class TestAgentProfile:
    """Tests for AgentProfile model."""

    def test_defaults(self) -> None:
        """Test default values."""
        profile = AgentProfile(
            id="claude",
            name="Claude",
            provider="anthropic",
        )
        assert profile.elo_rating == 1500.0
        assert profile.matches_played == 0
        assert profile.win_rate == 0.5
        assert profile.specialties == []

    def test_full_creation(self) -> None:
        """Test with all fields."""
        profile = AgentProfile(
            id="gpt4",
            name="GPT-4",
            provider="openai",
            elo_rating=1650.5,
            matches_played=100,
            win_rate=0.65,
            specialties=["coding", "math"],
        )
        assert profile.elo_rating == 1650.5
        assert len(profile.specialties) == 2


class TestVerificationResult:
    """Tests for VerificationResult model."""

    def test_valid_result(self) -> None:
        """Test valid verification result."""
        result = VerificationResult(
            status=VerificationStatus.VALID,
            claim="x > 0 implies x + 1 > 1",
            formal_translation="∀x. x > 0 → x + 1 > 1",
            proof="By arithmetic...",
            backend="z3",
            duration_ms=150,
        )
        assert result.status == VerificationStatus.VALID
        assert result.proof is not None
        assert result.counterexample is None

    def test_invalid_result(self) -> None:
        """Test invalid verification with counterexample."""
        result = VerificationResult(
            status=VerificationStatus.INVALID,
            claim="All numbers are positive",
            counterexample="x = -1",
            backend="z3",
        )
        assert result.status == VerificationStatus.INVALID
        assert result.counterexample == "x = -1"


class TestGauntletFinding:
    """Tests for GauntletFinding model."""

    def test_creation(self) -> None:
        """Test finding creation."""
        finding = GauntletFinding(
            severity="high",
            category="security",
            description="SQL injection vulnerability",
            location="auth/login.py:42",
            suggestion="Use parameterized queries",
        )
        assert finding.severity == "high"
        assert finding.category == "security"


class TestGauntletReceipt:
    """Tests for GauntletReceipt model."""

    def test_creation(self) -> None:
        """Test receipt creation."""
        receipt = GauntletReceipt(
            id="receipt-123",
            score=0.85,
            persona="security",
            created_at=datetime.now(),
            hash="abc123",
        )
        assert receipt.id == "receipt-123"
        assert receipt.score == 0.85
        assert receipt.findings == []

    def test_with_findings(self) -> None:
        """Test receipt with findings."""
        findings = [
            GauntletFinding(severity="low", category="style", description="Long line")
        ]
        receipt = GauntletReceipt(
            id="r-456",
            score=0.92,
            findings=findings,
            persona="code_review",
            created_at=datetime.now(),
        )
        assert len(receipt.findings) == 1


class TestMemoryModels:
    """Tests for memory-related models."""

    def test_tier_stats(self) -> None:
        """Test MemoryTierStats creation."""
        stats = MemoryTierStats(
            tier="fast",
            entries=1000,
            size_bytes=1048576,
            hit_rate=0.95,
            avg_age_seconds=30.5,
        )
        assert stats.tier == "fast"
        assert stats.entries == 1000

    def test_analytics(self) -> None:
        """Test MemoryAnalytics creation."""
        tiers = [
            MemoryTierStats(
                tier="fast",
                entries=100,
                size_bytes=1000,
                hit_rate=0.9,
                avg_age_seconds=10,
            )
        ]
        analytics = MemoryAnalytics(
            total_entries=100,
            total_size_bytes=1000,
            learning_velocity=0.5,
            tiers=tiers,
            period_days=7,
        )
        assert analytics.learning_velocity == 0.5
        assert len(analytics.tiers) == 1


class TestHealthStatus:
    """Tests for HealthStatus model."""

    def test_creation(self) -> None:
        """Test health status creation."""
        status = HealthStatus(
            status="healthy",
            version="2.1.14",
            uptime_seconds=86400.0,
            components={"database": "healthy", "cache": "healthy"},
        )
        assert status.status == "healthy"
        assert status.uptime_seconds == 86400.0
        assert len(status.components) == 2


class TestDebateEvent:
    """Tests for DebateEvent model."""

    def test_creation(self) -> None:
        """Test event creation."""
        event = DebateEvent(
            type="agent_message",
            data={"agent": "claude", "content": "Hello"},
        )
        assert event.type == "agent_message"
        assert event.data["agent"] == "claude"
        assert event.loop_id is None

    def test_timestamp_default(self) -> None:
        """Test timestamp has default."""
        event = DebateEvent(type="test")
        assert event.timestamp is not None


class TestSelectionPlugins:
    """Tests for selection plugin models."""

    def test_scorer_info(self) -> None:
        """Test ScorerInfo creation."""
        info = ScorerInfo(name="elo", description="ELO-based scoring")
        assert info.name == "elo"

    def test_team_selector_info(self) -> None:
        """Test TeamSelectorInfo creation."""
        info = TeamSelectorInfo(
            name="diverse", description="Diversity-focused selection"
        )
        assert info.name == "diverse"

    def test_role_assigner_info(self) -> None:
        """Test RoleAssignerInfo creation."""
        info = RoleAssignerInfo(name="balanced", description="Balanced role assignment")
        assert info.name == "balanced"

    def test_selection_plugins(self) -> None:
        """Test SelectionPlugins aggregation."""
        plugins = SelectionPlugins(
            scorers=[ScorerInfo(name="elo", description="ELO")],
            team_selectors=[TeamSelectorInfo(name="div", description="Diverse")],
            role_assigners=[],
        )
        assert len(plugins.scorers) == 1
        assert len(plugins.role_assigners) == 0


class TestTeamModels:
    """Tests for team selection models."""

    def test_agent_score(self) -> None:
        """Test AgentScore creation."""
        score = AgentScore(
            name="claude",
            score=0.85,
            elo_rating=1600,
            breakdown={"accuracy": 0.9, "speed": 0.8},
        )
        assert score.score == 0.85
        assert len(score.breakdown) == 2

    def test_team_member(self) -> None:
        """Test TeamMember creation."""
        member = TeamMember(name="claude", role="lead", score=0.9)
        assert member.role == "lead"

    def test_team_selection(self) -> None:
        """Test TeamSelection creation."""
        agents = [
            TeamMember(name="claude", role="lead", score=0.9),
            TeamMember(name="gpt4", role="critic", score=0.85),
        ]
        selection = TeamSelection(
            agents=agents,
            expected_quality=0.88,
            diversity_score=0.75,
            rationale="Balanced expertise mix",
        )
        assert len(selection.agents) == 2
        assert selection.diversity_score == 0.75


class TestRequestModels:
    """Tests for API request models."""

    def test_create_debate_request(self) -> None:
        """Test CreateDebateRequest."""
        request = CreateDebateRequest(
            task="Analyze this problem",
            agents=["claude", "gpt4"],
            max_rounds=3,
        )
        assert request.task == "Analyze this problem"
        assert request.consensus_threshold == 0.8

    def test_create_debate_request_defaults(self) -> None:
        """Test CreateDebateRequest defaults."""
        request = CreateDebateRequest(task="Test")
        assert request.agents is None
        assert request.max_rounds == 5
        assert request.metadata == {}

    def test_create_graph_debate_request(self) -> None:
        """Test CreateGraphDebateRequest."""
        request = CreateGraphDebateRequest(
            task="Complex analysis",
            branch_threshold=0.6,
            max_branches=5,
        )
        assert request.branch_threshold == 0.6
        assert request.max_branches == 5

    def test_create_matrix_debate_request(self) -> None:
        """Test CreateMatrixDebateRequest."""
        scenarios = [
            {"name": "base", "params": {}},
            {"name": "high_budget", "params": {"budget": 2000000}},
        ]
        request = CreateMatrixDebateRequest(
            task="Budget analysis",
            scenarios=scenarios,
        )
        assert len(request.scenarios) == 2

    def test_verify_claim_request(self) -> None:
        """Test VerifyClaimRequest."""
        request = VerifyClaimRequest(
            claim="x > 0",
            backend="lean",
            timeout=60,
        )
        assert request.backend == "lean"
        assert request.timeout == 60

    def test_verify_claim_request_defaults(self) -> None:
        """Test VerifyClaimRequest defaults."""
        request = VerifyClaimRequest(claim="test")
        assert request.backend == "z3"
        assert request.timeout == 30

    def test_run_gauntlet_request(self) -> None:
        """Test RunGauntletRequest."""
        request = RunGauntletRequest(
            input_content="def foo(): pass",
            input_type="code",
            persona="security",
        )
        assert request.input_type == "code"

    def test_score_agents_request(self) -> None:
        """Test ScoreAgentsRequest."""
        request = ScoreAgentsRequest(
            task_description="Math problem",
            primary_domain="mathematics",
            scorer="elo",
        )
        assert request.primary_domain == "mathematics"

    def test_select_team_request(self) -> None:
        """Test SelectTeamRequest."""
        request = SelectTeamRequest(
            task_description="Code review",
            min_agents=2,
            max_agents=4,
            diversity_preference=0.7,
        )
        assert request.diversity_preference == 0.7
        assert request.quality_priority == 0.5
