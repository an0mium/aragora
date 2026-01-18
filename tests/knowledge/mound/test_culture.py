"""
Tests for culture accumulation and stigmergic coordination.

Tests the CultureAccumulator, ReasoningPattern, DecisionHeuristic,
StigmergicSignal, and StigmergyManager implementations.
"""

import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, AsyncMock

from aragora.knowledge.mound.culture.patterns import (
    DecisionHeuristic,
    PatternType,
    ReasoningPattern,
)
from aragora.knowledge.mound.culture.stigmergy import (
    PheromoneTrail,
    SignalType,
    StigmergicSignal,
    StigmergyManager,
)
from aragora.knowledge.mound.culture.accumulator import (
    CultureAccumulator,
    CultureDocument,
    CultureDocumentCategory,
    DebateObservation,
    OrganizationCultureManager,
)
from aragora.knowledge.mound.types import (
    CulturePattern,
    CulturePatternType,
    CultureProfile,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound._vector_store = None  # No vector store for basic tests
    return mound


@pytest.fixture
def culture_accumulator(mock_mound):
    """Create a CultureAccumulator instance."""
    return CultureAccumulator(mound=mock_mound, min_observations_for_pattern=3)


@pytest.fixture
def stigmergy_manager():
    """Create a StigmergyManager instance."""
    return StigmergyManager()


@pytest.fixture
def org_culture_manager(mock_mound, culture_accumulator):
    """Create an OrganizationCultureManager instance."""
    return OrganizationCultureManager(
        mound=mock_mound,
        culture_accumulator=culture_accumulator,
    )


@pytest.fixture
def sample_debate_result():
    """Create a sample debate result object."""
    @dataclass
    class Proposal:
        agent_type: str
        content: str

    @dataclass
    class Critique:
        type: str
        content: str

    @dataclass
    class DebateResult:
        debate_id: str
        task: str
        proposals: List[Proposal]
        winner: Optional[str]
        consensus_reached: bool
        confidence: float
        rounds_used: int
        critiques: List[Critique]

    return DebateResult(
        debate_id="debate_001",
        task="Review security architecture",
        proposals=[
            Proposal(agent_type="claude", content="Use OAuth2"),
            Proposal(agent_type="gpt4", content="Use JWT"),
        ],
        winner="claude",
        consensus_reached=True,
        confidence=0.85,
        rounds_used=3,
        critiques=[
            Critique(type="security", content="Consider token expiry"),
            Critique(type="scalability", content="Think about load balancing"),
        ],
    )


# ============================================================================
# ReasoningPattern Tests
# ============================================================================


class TestReasoningPattern:
    """Tests for ReasoningPattern dataclass."""

    def test_basic_creation(self):
        """Test creating a basic reasoning pattern."""
        pattern = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.DECISION_HEURISTIC,
            name="Security First",
            description="Always consider security implications first",
            trigger_conditions=["security mentioned", "auth discussion"],
            reasoning_steps=["Identify assets", "Assess threats", "Propose controls"],
            expected_outcomes=["Secure design", "Risk mitigation"],
        )

        assert pattern.id == "rp_001"
        assert pattern.pattern_type == PatternType.DECISION_HEURISTIC
        assert len(pattern.trigger_conditions) == 2

    def test_success_rate_no_usage(self):
        """Test success rate with no usage."""
        pattern = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.SUCCESS_PATTERN,
            name="Test",
            description="Test pattern",
            trigger_conditions=[],
            reasoning_steps=[],
            expected_outcomes=[],
        )

        assert pattern.success_rate == 0.5  # Default

    def test_success_rate_with_outcomes(self):
        """Test success rate calculation."""
        pattern = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.SUCCESS_PATTERN,
            name="Test",
            description="Test pattern",
            trigger_conditions=[],
            reasoning_steps=[],
            expected_outcomes=[],
            success_count=7,
            failure_count=3,
        )

        assert pattern.success_rate == 0.7

    def test_reliability_score(self):
        """Test reliability score calculation."""
        pattern = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.SUCCESS_PATTERN,
            name="Test",
            description="Test pattern",
            trigger_conditions=[],
            reasoning_steps=[],
            expected_outcomes=[],
            confidence=0.8,
            success_count=8,
            failure_count=2,
        )

        # reliability = confidence * success_rate * usage_weight
        # = 0.8 * 0.8 * 1.0 = 0.64
        assert pattern.reliability_score == pytest.approx(0.64)

    def test_is_mature(self):
        """Test maturity check."""
        immature = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.SUCCESS_PATTERN,
            name="Test",
            description="Test",
            trigger_conditions=[],
            reasoning_steps=[],
            expected_outcomes=[],
            success_count=2,
            failure_count=1,
        )
        assert not immature.is_mature

        mature = ReasoningPattern(
            id="rp_002",
            pattern_type=PatternType.SUCCESS_PATTERN,
            name="Test",
            description="Test",
            trigger_conditions=[],
            reasoning_steps=[],
            expected_outcomes=[],
            success_count=4,
            failure_count=1,
        )
        assert mature.is_mature

    def test_record_outcome(self):
        """Test recording outcomes."""
        pattern = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.SUCCESS_PATTERN,
            name="Test",
            description="Test",
            trigger_conditions=[],
            reasoning_steps=[],
            expected_outcomes=[],
        )

        pattern.record_outcome(success=True)
        assert pattern.success_count == 1
        assert pattern.failure_count == 0
        assert pattern.last_applied is not None

        pattern.record_outcome(success=False)
        assert pattern.success_count == 1
        assert pattern.failure_count == 1

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        pattern = ReasoningPattern(
            id="rp_001",
            pattern_type=PatternType.CONSENSUS_PATTERN,
            name="Majority Wins",
            description="Accept majority opinion",
            trigger_conditions=["voting phase"],
            reasoning_steps=["Count votes", "Determine winner"],
            expected_outcomes=["Clear decision"],
            confidence=0.9,
            success_count=5,
            failure_count=1,
            verticals=["software"],
        )

        data = pattern.to_dict()
        restored = ReasoningPattern.from_dict(data)

        assert restored.id == pattern.id
        assert restored.pattern_type == pattern.pattern_type
        assert restored.confidence == pattern.confidence
        assert restored.success_count == pattern.success_count


# ============================================================================
# DecisionHeuristic Tests
# ============================================================================


class TestDecisionHeuristic:
    """Tests for DecisionHeuristic dataclass."""

    def test_basic_creation(self):
        """Test creating a decision heuristic."""
        heuristic = DecisionHeuristic(
            id="dh_001",
            rule="When in doubt, favor security over convenience",
            applies_when="Security trade-offs are discussed",
            confidence=0.85,
            supporting_evidence=["debate_001", "debate_005"],
        )

        assert heuristic.id == "dh_001"
        assert heuristic.confidence == 0.85
        assert len(heuristic.supporting_evidence) == 2

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        heuristic = DecisionHeuristic(
            id="dh_001",
            rule="Test rule",
            applies_when="Test condition",
            confidence=0.75,
            supporting_evidence=["e1"],
            exceptions=["Exception 1"],
            verticals=["legal"],
            usage_count=10,
        )

        data = heuristic.to_dict()
        restored = DecisionHeuristic.from_dict(data)

        assert restored.id == heuristic.id
        assert restored.rule == heuristic.rule
        assert restored.exceptions == heuristic.exceptions
        assert restored.usage_count == heuristic.usage_count


# ============================================================================
# StigmergicSignal Tests
# ============================================================================


class TestStigmergicSignal:
    """Tests for StigmergicSignal dataclass."""

    def test_basic_creation(self):
        """Test creating a stigmergic signal."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.ATTENTION,
            target_id="fact_123",
            target_type="fact",
            content="This needs review",
            emitter_agent_id="claude",
            workspace_id="ws_001",
        )

        assert signal.id == "sig_001"
        assert signal.signal_type == SignalType.ATTENTION
        assert signal.intensity == 1.0

    def test_current_intensity_fresh(self):
        """Test intensity is full when fresh."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.SUCCESS,
            target_id="fact_123",
            target_type="fact",
            content="This worked",
        )

        # Fresh signal should be at full intensity
        assert signal.current_intensity >= 0.95

    def test_current_intensity_decays(self):
        """Test that intensity decays over time."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.WARNING,
            target_id="fact_123",
            target_type="fact",
            content="Caution",
            intensity=1.0,
            decay_rate=0.2,  # 20% per day
        )

        # Simulate 3 days passing
        signal.last_reinforced = datetime.now() - timedelta(days=3)

        # After 3 days at 0.2/day = 60% decay
        # intensity = 1.0 * (1 - 0.6) + bonus = 0.4 + small bonus
        assert 0.3 < signal.current_intensity < 0.6

    def test_is_expired(self):
        """Test signal expiration."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.ATTENTION,
            target_id="fact_123",
            target_type="fact",
            content="Old signal",
            intensity=0.5,
            decay_rate=0.1,
        )

        # Simulate enough time for signal to expire
        signal.last_reinforced = datetime.now() - timedelta(days=30)

        assert signal.is_expired

    def test_reinforce(self):
        """Test signal reinforcement."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.INSIGHT,
            target_id="fact_123",
            target_type="fact",
            content="Good insight",
            intensity=0.7,
            emitter_agent_id="claude",
        )

        signal.reinforce("gpt4")

        assert signal.reinforcement_count == 1
        assert signal.intensity == pytest.approx(0.8)  # +0.1, use approx for float
        assert "gpt4" in signal.reinforcing_agents

    def test_age_days(self):
        """Test age calculation."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.QUESTION,
            target_id="fact_123",
            target_type="fact",
            content="Test",
        )
        signal.created_at = datetime.now() - timedelta(days=5)

        assert 4.9 < signal.age_days < 5.1

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        signal = StigmergicSignal(
            id="sig_001",
            signal_type=SignalType.CONTROVERSY,
            target_id="debate_001",
            target_type="debate",
            content="Agents disagreed strongly",
            intensity=0.9,
            reinforcement_count=3,
            reinforcing_agents=["claude", "gpt4"],
        )

        data = signal.to_dict()
        restored = StigmergicSignal.from_dict(data)

        assert restored.id == signal.id
        assert restored.signal_type == signal.signal_type
        assert restored.reinforcement_count == signal.reinforcement_count


# ============================================================================
# StigmergyManager Tests
# ============================================================================


class TestStigmergyManager:
    """Tests for StigmergyManager."""

    @pytest.mark.asyncio
    async def test_emit_signal(self, stigmergy_manager):
        """Test emitting a new signal."""
        signal_id = await stigmergy_manager.emit_signal(
            signal_type=SignalType.ATTENTION,
            target_id="fact_001",
            target_type="fact",
            content="Needs review",
            agent_id="claude",
            workspace_id="ws_001",
        )

        assert signal_id.startswith("sig_")

        signal = await stigmergy_manager.get_signal(signal_id)
        assert signal is not None
        assert signal.signal_type == SignalType.ATTENTION
        assert signal.emitter_agent_id == "claude"

    @pytest.mark.asyncio
    async def test_emit_reinforces_existing(self, stigmergy_manager):
        """Test that emitting similar signal reinforces existing."""
        # First emission
        signal_id_1 = await stigmergy_manager.emit_signal(
            signal_type=SignalType.WARNING,
            target_id="fact_001",
            target_type="fact",
            content="Caution",
            agent_id="claude",
            workspace_id="ws_001",
        )

        signal_1 = await stigmergy_manager.get_signal(signal_id_1)
        original_count = signal_1.reinforcement_count

        # Second emission on same target (should reinforce)
        signal_id_2 = await stigmergy_manager.emit_signal(
            signal_type=SignalType.WARNING,
            target_id="fact_001",
            target_type="fact",
            content="More caution",
            agent_id="gpt4",
            workspace_id="ws_001",
        )

        # Should return same signal ID
        assert signal_id_2 == signal_id_1

        # Signal should be reinforced
        signal = await stigmergy_manager.get_signal(signal_id_1)
        assert signal.reinforcement_count == original_count + 1
        assert "gpt4" in signal.reinforcing_agents

    @pytest.mark.asyncio
    async def test_get_signals_for_target(self, stigmergy_manager):
        """Test getting signals for a target."""
        # Emit multiple signals for same target
        await stigmergy_manager.emit_signal(
            SignalType.ATTENTION, "fact_001", "fact", "Review needed", "claude", "ws_001"
        )
        await stigmergy_manager.emit_signal(
            SignalType.INSIGHT, "fact_001", "fact", "Found something", "gpt4", "ws_001"
        )

        signals = await stigmergy_manager.get_signals_for_target("fact_001")

        assert len(signals) == 2
        signal_types = {s.signal_type for s in signals}
        assert SignalType.ATTENTION in signal_types
        assert SignalType.INSIGHT in signal_types

    @pytest.mark.asyncio
    async def test_get_signals_for_target_with_filter(self, stigmergy_manager):
        """Test filtering signals by type."""
        await stigmergy_manager.emit_signal(
            SignalType.ATTENTION, "fact_001", "fact", "Review", "claude", "ws_001"
        )
        await stigmergy_manager.emit_signal(
            SignalType.WARNING, "fact_001", "fact", "Caution", "gpt4", "ws_001"
        )

        signals = await stigmergy_manager.get_signals_for_target(
            "fact_001",
            signal_types=[SignalType.WARNING],
        )

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.WARNING

    @pytest.mark.asyncio
    async def test_get_attention_signals(self, stigmergy_manager):
        """Test getting attention-worthy signals."""
        # Emit various signals
        await stigmergy_manager.emit_signal(
            SignalType.ATTENTION, "fact_001", "fact", "Review", "claude", "ws_001"
        )
        await stigmergy_manager.emit_signal(
            SignalType.WARNING, "fact_002", "fact", "Danger", "gpt4", "ws_001"
        )
        await stigmergy_manager.emit_signal(
            SignalType.SUCCESS, "fact_003", "fact", "Worked", "gemini", "ws_001"
        )

        signals = await stigmergy_manager.get_attention_signals("ws_001")

        # Should only return attention/warning/controversy signals
        assert len(signals) == 2
        for signal in signals:
            assert signal.signal_type in [
                SignalType.ATTENTION,
                SignalType.WARNING,
                SignalType.CONTROVERSY,
            ]

    @pytest.mark.asyncio
    async def test_reinforce_signal(self, stigmergy_manager):
        """Test reinforcing an existing signal."""
        signal_id = await stigmergy_manager.emit_signal(
            SignalType.INSIGHT, "fact_001", "fact", "Discovery", "claude", "ws_001"
        )

        result = await stigmergy_manager.reinforce_signal(signal_id, "gpt4")

        assert result is True

        signal = await stigmergy_manager.get_signal(signal_id)
        assert signal.reinforcement_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, stigmergy_manager):
        """Test cleanup of expired signals."""
        # Create a signal and make it expire
        signal_id = await stigmergy_manager.emit_signal(
            SignalType.ATTENTION, "fact_001", "fact", "Old", "claude", "ws_001"
        )

        signal = await stigmergy_manager.get_signal(signal_id)
        signal.intensity = 0.05  # Very low
        signal.last_reinforced = datetime.now() - timedelta(days=30)

        cleaned = await stigmergy_manager.cleanup_expired()

        assert cleaned == 1
        assert await stigmergy_manager.get_signal(signal_id) is None

    def test_get_stats(self, stigmergy_manager):
        """Test getting statistics."""
        stats = stigmergy_manager.get_stats()

        assert "total_signals" in stats
        assert "active_signals" in stats
        assert "by_type" in stats
        assert "trails" in stats


# ============================================================================
# CultureAccumulator Tests
# ============================================================================


class TestCultureAccumulator:
    """Tests for CultureAccumulator."""

    @pytest.mark.asyncio
    async def test_observe_debate(self, culture_accumulator, sample_debate_result):
        """Test observing a debate and extracting patterns."""
        patterns = await culture_accumulator.observe_debate(
            sample_debate_result,
            workspace_id="ws_001",
        )

        assert len(patterns) >= 1
        # Should have extracted some patterns from the debate

    @pytest.mark.asyncio
    async def test_observe_multiple_debates_builds_confidence(
        self, culture_accumulator, sample_debate_result
    ):
        """Test that observing multiple debates increases pattern confidence."""
        # Observe same type of debate multiple times
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(
                sample_debate_result,
                workspace_id="ws_001",
            )

        profile = await culture_accumulator.get_profile("ws_001")

        # Should have patterns with increased confidence
        all_patterns = []
        for patterns in profile.patterns.values():
            all_patterns.extend(patterns)

        # At least one pattern should have higher confidence
        high_conf_patterns = [p for p in all_patterns if p.confidence > 0.3]
        assert len(high_conf_patterns) >= 1

    @pytest.mark.asyncio
    async def test_get_profile_empty(self, culture_accumulator):
        """Test getting profile for workspace with no observations."""
        profile = await culture_accumulator.get_profile("empty_workspace")

        assert profile.workspace_id == "empty_workspace"
        assert profile.total_observations == 0

    @pytest.mark.asyncio
    async def test_get_profile_with_patterns(
        self, culture_accumulator, sample_debate_result
    ):
        """Test getting profile with accumulated patterns."""
        # Observe several debates
        for i in range(3):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(
                sample_debate_result,
                workspace_id="ws_001",
            )

        profile = await culture_accumulator.get_profile("ws_001")

        assert profile.workspace_id == "ws_001"
        assert profile.total_observations >= 1
        assert isinstance(profile.patterns, dict)
        assert isinstance(profile.dominant_traits, dict)

    @pytest.mark.asyncio
    async def test_recommend_agents(self, culture_accumulator, sample_debate_result):
        """Test agent recommendations based on culture."""
        # Build up some patterns
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(
                sample_debate_result,
                workspace_id="ws_001",
            )

        recommendations = await culture_accumulator.recommend_agents(
            task_type="security",
            workspace_id="ws_001",
        )

        # Should return a list of recommended agents
        assert isinstance(recommendations, list)

    def test_infer_domain(self, culture_accumulator):
        """Test domain inference from topic."""
        assert culture_accumulator._infer_domain("Review security architecture") == "security"
        # Keywords may match in priority order - "optimize" matches "performance", "design" matches "architecture"
        assert culture_accumulator._infer_domain("Optimize database queries") in ("database", "performance")
        assert culture_accumulator._infer_domain("Design REST API endpoints") in ("api", "architecture")
        assert culture_accumulator._infer_domain("Random topic") is None
        # More specific tests where keywords are unambiguous
        assert culture_accumulator._infer_domain("Fix SQL query bug") == "database"
        assert culture_accumulator._infer_domain("Run tests and check coverage") == "testing"
        assert culture_accumulator._infer_domain("Call the REST endpoint") == "api"


# ============================================================================
# OrganizationCultureManager Tests
# ============================================================================


class TestOrganizationCultureManager:
    """Tests for OrganizationCultureManager."""

    def test_register_workspace(self, org_culture_manager):
        """Test registering a workspace to an organization."""
        org_culture_manager.register_workspace("ws_001", "org_001")

        assert org_culture_manager._workspace_orgs["ws_001"] == "org_001"

    @pytest.mark.asyncio
    async def test_add_document(self, org_culture_manager):
        """Test adding a culture document."""
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Security First",
            content="We prioritize security in all decisions.",
            created_by="admin",
        )

        assert doc.id.startswith("cd_")
        assert doc.org_id == "org_001"
        assert doc.category == CultureDocumentCategory.VALUES
        assert doc.version == 1
        assert doc.is_active is True

    @pytest.mark.asyncio
    async def test_update_document(self, org_culture_manager):
        """Test updating a culture document creates new version."""
        # Add initial document
        doc = await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Code Review",
            content="All code must be reviewed.",
            created_by="admin",
        )

        # Update it
        updated = await org_culture_manager.update_document(
            doc_id=doc.id,
            org_id="org_001",
            content="All code must be reviewed by at least two people.",
            updated_by="manager",
        )

        assert updated.version == 2
        assert updated.supersedes == doc.id
        assert updated.is_active is True

        # Original should be inactive
        assert doc.is_active is False

    @pytest.mark.asyncio
    async def test_get_documents(self, org_culture_manager):
        """Test getting culture documents."""
        # Add some documents
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Value 1",
            content="Content",
            created_by="admin",
        )
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Practice 1",
            content="Content",
            created_by="admin",
        )

        all_docs = await org_culture_manager.get_documents("org_001")
        assert len(all_docs) == 2

        value_docs = await org_culture_manager.get_documents(
            "org_001", category=CultureDocumentCategory.VALUES
        )
        assert len(value_docs) == 1

    @pytest.mark.asyncio
    async def test_query_culture_keyword(self, org_culture_manager):
        """Test querying culture documents by keyword."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Security Practices",
            content="We use encryption for all sensitive data.",
            created_by="admin",
        )
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Coding Standards",
            content="We follow PEP8 for Python code.",
            created_by="admin",
        )

        results = await org_culture_manager.query_culture("org_001", "encryption")

        assert len(results) >= 1
        assert "encryption" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_promote_pattern_to_culture(
        self, org_culture_manager, culture_accumulator, sample_debate_result
    ):
        """Test promoting a workspace pattern to culture document."""
        # Register workspace
        org_culture_manager.register_workspace("ws_001", "org_001")

        # Build up patterns
        for i in range(5):
            sample_debate_result.debate_id = f"debate_{i:03d}"
            await culture_accumulator.observe_debate(
                sample_debate_result,
                workspace_id="ws_001",
            )

        # Get a pattern to promote
        profile = await culture_accumulator.get_profile("ws_001")
        all_patterns = []
        for patterns in profile.patterns.values():
            all_patterns.extend(patterns)

        if all_patterns:
            pattern = all_patterns[0]

            doc = await org_culture_manager.promote_pattern_to_culture(
                workspace_id="ws_001",
                pattern_id=pattern.id,
                promoted_by="admin",
                title="Learned Pattern",
            )

            assert doc.category == CultureDocumentCategory.LEARNINGS
            assert doc.source_pattern_id == pattern.id
            assert doc.source_workspace_id == "ws_001"

    @pytest.mark.asyncio
    async def test_get_organization_culture(
        self, org_culture_manager, culture_accumulator, sample_debate_result
    ):
        """Test getting complete organization culture."""
        # Setup
        org_culture_manager.register_workspace("ws_001", "org_001")
        org_culture_manager.register_workspace("ws_002", "org_001")

        # Add documents
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Core Values",
            content="Quality and innovation",
            created_by="admin",
        )

        # Build patterns in workspaces
        for ws_id in ["ws_001", "ws_002"]:
            for i in range(2):
                sample_debate_result.debate_id = f"{ws_id}_debate_{i}"
                await culture_accumulator.observe_debate(
                    sample_debate_result,
                    workspace_id=ws_id,
                )

        org_culture = await org_culture_manager.get_organization_culture("org_001")

        assert org_culture.org_id == "org_001"
        assert len(org_culture.documents) >= 1
        assert org_culture.workspace_count == 2

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, org_culture_manager):
        """Test getting relevant context for a task."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="API Design",
            content="All APIs should be RESTful and versioned.",
            created_by="admin",
        )

        # Query with exact keyword match (lowercase)
        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="restful",  # Match content keyword
        )

        assert "Organizational Context" in context
        assert "API Design" in context

    @pytest.mark.asyncio
    async def test_get_relevant_context_empty(self, org_culture_manager):
        """Test getting context when no documents match."""
        await org_culture_manager.add_document(
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Security",
            content="We prioritize security.",
            created_by="admin",
        )

        # Query with non-matching term
        context = await org_culture_manager.get_relevant_context(
            org_id="org_001",
            task="unrelated topic xyz",
        )

        # Should return empty string when no matches
        assert context == ""

    def test_cosine_similarity(self, org_culture_manager):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert org_culture_manager._cosine_similarity(a, b) == 1.0

        c = [0.0, 1.0, 0.0]
        assert org_culture_manager._cosine_similarity(a, c) == 0.0

        d = [0.5, 0.5, 0.0]
        # cos similarity between [1,0,0] and [0.5,0.5,0] = 0.5 / (1 * 0.707) ≈ 0.707
        assert 0.7 < org_culture_manager._cosine_similarity(a, d) < 0.72


# ============================================================================
# PheromoneTrail Tests
# ============================================================================


class TestPheromoneTrail:
    """Tests for PheromoneTrail dataclass."""

    def test_basic_creation(self):
        """Test creating a pheromone trail."""
        trail = PheromoneTrail(
            id="trail_001",
            name="Security Review Path",
            signals=["sig_001", "sig_002", "sig_003"],
            total_intensity=2.5,
            agent_count=3,
            workspace_id="ws_001",
        )

        assert trail.id == "trail_001"
        assert len(trail.signals) == 3

    def test_strength(self):
        """Test trail strength calculation."""
        trail = PheromoneTrail(
            id="trail_001",
            name="Test Trail",
            signals=["s1", "s2"],
            total_intensity=2.0,
            agent_count=4,
            workspace_id="ws_001",
        )

        # strength = total_intensity * (1 + 0.1 * agent_count)
        # = 2.0 * (1 + 0.4) = 2.8
        assert trail.strength == pytest.approx(2.8)


# ============================================================================
# CultureDocument Tests
# ============================================================================


class TestCultureDocument:
    """Tests for CultureDocument dataclass."""

    def test_basic_creation(self):
        """Test creating a culture document."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.VALUES,
            title="Core Values",
            content="We believe in quality.",
            created_by="admin",
        )

        assert doc.id == "cd_001"
        assert doc.category == CultureDocumentCategory.VALUES
        assert doc.version == 1
        assert doc.is_active is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = CultureDocument(
            id="cd_001",
            org_id="org_001",
            category=CultureDocumentCategory.PRACTICES,
            title="Test",
            content="Content",
            created_by="user",
            metadata={"key": "value"},
        )

        d = doc.to_dict()

        assert d["id"] == "cd_001"
        assert d["category"] == "practices"
        assert d["metadata"]["key"] == "value"


# ============================================================================
# DebateObservation Tests
# ============================================================================


class TestDebateObservation:
    """Tests for DebateObservation dataclass."""

    def test_basic_creation(self):
        """Test creating a debate observation."""
        obs = DebateObservation(
            debate_id="debate_001",
            topic="Security review",
            participating_agents=["claude", "gpt4"],
            winning_agents=["claude"],
            rounds_to_consensus=3,
            consensus_reached=True,
            consensus_strength="strong",
            critique_patterns=["security", "scalability"],
            domain="security",
        )

        assert obs.debate_id == "debate_001"
        assert len(obs.participating_agents) == 2
        assert obs.consensus_reached is True
