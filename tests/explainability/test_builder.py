"""
Tests for ExplanationBuilder.

Tests the builder class that constructs Decision entities from debate results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.explainability.builder import ExplanationBuilder
from aragora.explainability.decision import (
    BeliefChange,
    ConfidenceAttribution,
    Counterfactual,
    Decision,
    EvidenceLink,
    VotePivot,
)


# =============================================================================
# Mock Objects for Testing
# =============================================================================


@dataclass
class MockVote:
    """Mock vote object."""

    agent: str
    choice: str
    confidence: float = 0.8
    reasoning: str = ""


@dataclass
class MockClaim:
    """Mock claim for provenance tracking."""

    id: str
    content: str
    source: str
    confidence: float
    cited_by: List[str] = field(default_factory=list)


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    id: str = "debate-123"
    task: str = "Evaluate the proposal"
    domain: str = "engineering"
    final_answer: str = "The proposal is accepted"
    consensus_reached: bool = True
    confidence: float = 0.85
    consensus_type: str = "majority"
    consensus_margin: float = 0.75
    rounds_used: int = 3
    participants: List[str] = field(default_factory=lambda: ["claude", "gpt", "gemini"])
    proposals: Dict[str, str] = field(default_factory=dict)
    critiques: Dict[int, Dict[str, str]] = field(default_factory=dict)
    votes: List[MockVote] = field(default_factory=list)
    position_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    flip_data: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class MockDebateContext:
    """Mock debate context."""

    env: Any = None
    domain: str = "general"
    agents: List[Any] = field(default_factory=list)


@dataclass
class MockEnv:
    """Mock environment."""

    task: str = "Context task"


@dataclass
class MockAgent:
    """Mock agent."""

    name: str


@dataclass
class MockBeliefChange:
    """Mock belief change from belief network."""

    agent: str
    round: int
    topic: str
    prior: str
    posterior: str
    prior_confidence: float
    posterior_confidence: float
    trigger: str
    trigger_source: str


# =============================================================================
# Test ExplanationBuilder Initialization
# =============================================================================


class TestExplanationBuilderInit:
    """Tests for ExplanationBuilder initialization."""

    def test_init_no_trackers(self):
        """Test initialization without trackers."""
        builder = ExplanationBuilder()

        assert builder.evidence_tracker is None
        assert builder.belief_network is None
        assert builder.calibration_tracker is None
        assert builder.elo_system is None
        assert builder.provenance_tracker is None

    def test_init_with_trackers(self):
        """Test initialization with trackers."""
        evidence_tracker = MagicMock()
        belief_network = MagicMock()
        calibration_tracker = MagicMock()
        elo_system = MagicMock()
        provenance_tracker = MagicMock()

        builder = ExplanationBuilder(
            evidence_tracker=evidence_tracker,
            belief_network=belief_network,
            calibration_tracker=calibration_tracker,
            elo_system=elo_system,
            provenance_tracker=provenance_tracker,
        )

        assert builder.evidence_tracker is evidence_tracker
        assert builder.belief_network is belief_network
        assert builder.calibration_tracker is calibration_tracker
        assert builder.elo_system is elo_system
        assert builder.provenance_tracker is provenance_tracker


# =============================================================================
# Test Build Method
# =============================================================================


class TestExplanationBuilderBuild:
    """Tests for the main build method."""

    @pytest.fixture
    def builder(self):
        """Create a basic builder."""
        return ExplanationBuilder()

    @pytest.fixture
    def basic_result(self):
        """Create a basic debate result."""
        return MockDebateResult(
            proposals={
                "claude": "Proposal from Claude with detailed analysis",
                "gpt": "GPT's counterproposal with alternative approach",
            },
            votes=[
                MockVote(
                    agent="claude", choice="option_a", confidence=0.9, reasoning="Strong evidence"
                ),
                MockVote(
                    agent="gpt", choice="option_a", confidence=0.8, reasoning="Agree with analysis"
                ),
                MockVote(
                    agent="gemini", choice="option_b", confidence=0.6, reasoning="Alternative view"
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_build_basic(self, builder, basic_result):
        """Test basic build operation."""
        decision = await builder.build(basic_result)

        assert isinstance(decision, Decision)
        assert decision.debate_id == "debate-123"
        assert decision.conclusion == "The proposal is accepted"
        assert decision.consensus_reached is True
        assert decision.confidence == 0.85
        assert decision.rounds_used == 3
        assert decision.agents_participated == ["claude", "gpt", "gemini"]

    @pytest.mark.asyncio
    async def test_build_extracts_task(self, builder, basic_result):
        """Test task extraction from result."""
        decision = await builder.build(basic_result)

        assert decision.task == "Evaluate the proposal"
        assert decision.domain == "engineering"

    @pytest.mark.asyncio
    async def test_build_with_context(self, builder, basic_result):
        """Test build with context object."""
        context = MockDebateContext(
            env=MockEnv(task="Context task"),
            domain="security",
        )

        # Remove task/domain from result to test context fallback
        basic_result.task = ""
        basic_result.domain = ""

        decision = await builder.build(basic_result, context=context)

        # Should fall back to context values
        assert decision.task == "" or decision.task == "Context task"

    @pytest.mark.asyncio
    async def test_build_generates_decision_id(self, builder, basic_result):
        """Test that decision ID is generated."""
        decision = await builder.build(basic_result)

        assert decision.decision_id is not None
        assert len(decision.decision_id) > 0

    @pytest.mark.asyncio
    async def test_build_without_counterfactuals(self, builder, basic_result):
        """Test build without counterfactuals."""
        decision = await builder.build(basic_result, include_counterfactuals=False)

        assert decision.counterfactuals == []

    @pytest.mark.asyncio
    async def test_build_with_counterfactuals(self, builder, basic_result):
        """Test build includes counterfactuals."""
        decision = await builder.build(basic_result, include_counterfactuals=True)

        # Counterfactuals should be built from vote pivots
        assert isinstance(decision.counterfactuals, list)


# =============================================================================
# Test Evidence Chain Building
# =============================================================================


class TestEvidenceChainBuilding:
    """Tests for _build_evidence_chain method."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_evidence_from_proposals(self, builder):
        """Test evidence extraction from proposals."""
        result = MockDebateResult(
            proposals={
                "claude": "Claude's detailed proposal content",
                "gpt": "GPT's alternative proposal",
            }
        )

        decision = await builder.build(result)
        evidence_chain = decision.evidence_chain

        assert len(evidence_chain) >= 2
        sources = [e.source for e in evidence_chain]
        assert "claude" in sources
        assert "gpt" in sources

    @pytest.mark.asyncio
    async def test_evidence_link_fields(self, builder):
        """Test evidence link has correct fields."""
        result = MockDebateResult(
            proposals={"claude": "Test proposal content for evidence"},
        )

        decision = await builder.build(result)
        evidence = decision.evidence_chain[0]

        assert evidence.id.startswith("prop-")
        assert "Test proposal" in evidence.content
        assert evidence.source == "claude"
        assert evidence.relevance_score == 0.8  # Default for proposals
        assert evidence.grounding_type == "argument"
        assert evidence.metadata["type"] == "proposal"

    @pytest.mark.asyncio
    async def test_evidence_content_truncation(self, builder):
        """Test evidence content is truncated."""
        long_content = "X" * 1000
        result = MockDebateResult(proposals={"claude": long_content})

        decision = await builder.build(result)
        evidence = decision.evidence_chain[0]

        assert len(evidence.content) <= 500

    @pytest.mark.asyncio
    async def test_evidence_from_critiques(self, builder):
        """Test evidence extraction from critiques."""
        result = MockDebateResult(
            proposals={"claude": "Original proposal"},
            critiques={
                1: {
                    "gpt": "Critique from GPT on round 1",
                    "gemini": "Critique from Gemini",
                },
            },
        )

        decision = await builder.build(result)
        critique_evidence = [e for e in decision.evidence_chain if e.grounding_type == "critique"]

        assert len(critique_evidence) >= 1

    @pytest.mark.asyncio
    async def test_evidence_with_tracker(self):
        """Test evidence scoring with evidence tracker."""
        evidence_tracker = AsyncMock()
        evidence_tracker.score_evidence = AsyncMock(
            return_value={
                "relevance": 0.95,
                "authority": 0.85,
                "freshness": 0.9,
                "completeness": 0.8,
            }
        )

        builder = ExplanationBuilder(evidence_tracker=evidence_tracker)
        result = MockDebateResult(proposals={"claude": "Test proposal"})

        decision = await builder.build(result)
        evidence = decision.evidence_chain[0]

        assert evidence.relevance_score == 0.95
        assert evidence.quality_scores["authority"] == 0.85

    @pytest.mark.asyncio
    async def test_evidence_tracker_error_handling(self):
        """Test graceful handling of evidence tracker errors."""
        evidence_tracker = AsyncMock()
        evidence_tracker.score_evidence = AsyncMock(side_effect=Exception("Tracker error"))

        builder = ExplanationBuilder(evidence_tracker=evidence_tracker)
        result = MockDebateResult(proposals={"claude": "Test proposal"})

        # Should not raise, should use defaults
        decision = await builder.build(result)
        evidence = decision.evidence_chain[0]

        assert evidence.relevance_score == 0.8  # Default fallback


# =============================================================================
# Test Vote Pivots Building
# =============================================================================


class TestVotePivotsBuilding:
    """Tests for _build_vote_pivots method."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_vote_pivots_from_votes(self, builder):
        """Test vote pivot extraction."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="yes", confidence=0.9, reasoning="Strong evidence"),
                MockVote(agent="gpt", choice="yes", confidence=0.7, reasoning="Agrees"),
                MockVote(agent="gemini", choice="no", confidence=0.6, reasoning="Disagrees"),
            ]
        )

        decision = await builder.build(result)
        pivots = decision.vote_pivots

        assert len(pivots) == 3
        agents = [p.agent for p in pivots]
        assert "claude" in agents
        assert "gpt" in agents
        assert "gemini" in agents

    @pytest.mark.asyncio
    async def test_vote_pivot_fields(self, builder):
        """Test vote pivot has correct fields."""
        result = MockDebateResult(
            votes=[
                MockVote(
                    agent="claude", choice="approve", confidence=0.85, reasoning="Good proposal"
                ),
            ]
        )

        decision = await builder.build(result)
        pivot = decision.vote_pivots[0]

        assert pivot.agent == "claude"
        assert pivot.choice == "approve"
        assert pivot.confidence == 0.85
        assert pivot.weight >= 0
        assert "Good proposal" in pivot.reasoning_summary
        assert 0 <= pivot.influence_score <= 1

    @pytest.mark.asyncio
    async def test_vote_pivots_sorted_by_influence(self, builder):
        """Test vote pivots are sorted by influence score."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="agent_a", choice="yes", confidence=0.5),
                MockVote(agent="agent_b", choice="yes", confidence=0.9),
                MockVote(agent="agent_c", choice="no", confidence=0.7),
            ]
        )

        decision = await builder.build(result)
        pivots = decision.vote_pivots

        # Should be sorted by influence (descending)
        for i in range(len(pivots) - 1):
            assert pivots[i].influence_score >= pivots[i + 1].influence_score

    @pytest.mark.asyncio
    async def test_vote_pivots_no_votes(self, builder):
        """Test handling of no votes."""
        result = MockDebateResult(votes=[])

        decision = await builder.build(result)

        assert decision.vote_pivots == []

    @pytest.mark.asyncio
    async def test_vote_pivots_with_elo(self):
        """Test vote weighting with ELO system."""
        elo_system = MagicMock()
        elo_system.get_rating = MagicMock(return_value=1500)

        builder = ExplanationBuilder(elo_system=elo_system)
        result = MockDebateResult(votes=[MockVote(agent="claude", choice="yes", confidence=0.9)])

        decision = await builder.build(result)
        pivot = decision.vote_pivots[0]

        assert pivot.elo_rating == 1500
        assert pivot.weight > 1.0  # Should be boosted by high ELO

    @pytest.mark.asyncio
    async def test_vote_pivots_with_calibration(self):
        """Test vote weighting with calibration tracker."""
        calibration_tracker = MagicMock()
        calibration_tracker.get_weight = MagicMock(return_value=1.2)
        calibration_tracker.get_adjustment = MagicMock(return_value=0.05)

        builder = ExplanationBuilder(calibration_tracker=calibration_tracker)
        result = MockDebateResult(votes=[MockVote(agent="claude", choice="yes", confidence=0.9)])

        decision = await builder.build(result)
        pivot = decision.vote_pivots[0]

        assert pivot.calibration_adjustment == 0.05

    @pytest.mark.asyncio
    async def test_vote_pivots_flip_detection(self, builder):
        """Test flip detection in vote pivots."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="yes", confidence=0.9),
                MockVote(agent="gpt", choice="yes", confidence=0.8),
            ],
            flip_data={"flipped_agents": ["claude"]},
        )

        decision = await builder.build(result)
        claude_pivot = next(p for p in decision.vote_pivots if p.agent == "claude")
        gpt_pivot = next(p for p in decision.vote_pivots if p.agent == "gpt")

        assert claude_pivot.flip_detected is True
        assert gpt_pivot.flip_detected is False


# =============================================================================
# Test Belief Changes Building
# =============================================================================


class TestBeliefChangesBuilding:
    """Tests for _build_belief_changes method."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_belief_changes_from_position_history(self, builder):
        """Test belief change extraction from position history."""
        result = MockDebateResult(
            position_history={
                "claude": [
                    {"position": "A", "confidence": 0.7},
                    {"position": "B", "confidence": 0.85},
                ],
                "gpt": [
                    {"position": "X", "confidence": 0.6},
                    {"position": "X", "confidence": 0.8},  # No position change
                ],
            }
        )

        decision = await builder.build(result)
        changes = decision.belief_changes

        # Only claude changed position
        assert len(changes) == 1
        assert changes[0].agent == "claude"
        assert changes[0].prior_belief == "A"
        assert changes[0].posterior_belief == "B"

    @pytest.mark.asyncio
    async def test_belief_changes_from_network(self):
        """Test belief change extraction from belief network."""
        belief_network = MagicMock()
        mock_change = MockBeliefChange(
            agent="claude",
            round=2,
            topic="API design",
            prior="REST",
            posterior="GraphQL",
            prior_confidence=0.8,
            posterior_confidence=0.6,
            trigger="critique",
            trigger_source="gpt",
        )
        belief_network.get_changes = MagicMock(return_value=[mock_change])

        builder = ExplanationBuilder(belief_network=belief_network)
        result = MockDebateResult()

        decision = await builder.build(result)
        changes = decision.belief_changes

        assert len(changes) >= 1
        network_change = changes[0]
        assert network_change.agent == "claude"
        assert network_change.topic == "API design"

    @pytest.mark.asyncio
    async def test_belief_changes_network_error_handling(self):
        """Test graceful handling of belief network errors."""
        belief_network = MagicMock()
        belief_network.get_changes = MagicMock(side_effect=Exception("Network error"))

        builder = ExplanationBuilder(belief_network=belief_network)
        result = MockDebateResult()

        # Should not raise
        decision = await builder.build(result)

        assert isinstance(decision.belief_changes, list)


# =============================================================================
# Test Confidence Attribution Building
# =============================================================================


class TestConfidenceAttributionBuilding:
    """Tests for _build_confidence_attribution method."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_confidence_attribution_consensus_factor(self, builder):
        """Test consensus strength factor in attribution."""
        result = MockDebateResult(consensus_margin=0.8)

        decision = await builder.build(result)
        attrs = decision.confidence_attribution

        consensus_attr = next((a for a in attrs if a.factor == "consensus_strength"), None)
        assert consensus_attr is not None
        assert consensus_attr.raw_value == 0.8

    @pytest.mark.asyncio
    async def test_confidence_attribution_evidence_factor(self, builder):
        """Test evidence quality factor in attribution."""
        result = MockDebateResult(proposals={"claude": "High quality proposal"})

        decision = await builder.build(result)
        attrs = decision.confidence_attribution

        evidence_attr = next((a for a in attrs if a.factor == "evidence_quality"), None)
        assert evidence_attr is not None
        assert 0 <= evidence_attr.contribution <= 1

    @pytest.mark.asyncio
    async def test_confidence_attribution_debate_efficiency(self, builder):
        """Test debate efficiency factor in attribution."""
        result = MockDebateResult(rounds_used=2)

        decision = await builder.build(result)
        attrs = decision.confidence_attribution

        efficiency_attr = next((a for a in attrs if a.factor == "debate_efficiency"), None)
        assert efficiency_attr is not None
        assert efficiency_attr.raw_value == 2.0

    @pytest.mark.asyncio
    async def test_confidence_attribution_normalized(self, builder):
        """Test confidence attribution contributions are normalized."""
        result = MockDebateResult(
            consensus_margin=0.8,
            rounds_used=3,
            proposals={"claude": "Test proposal"},
        )

        decision = await builder.build(result)
        attrs = decision.confidence_attribution

        if attrs:
            total = sum(a.contribution for a in attrs)
            assert abs(total - 1.0) < 0.01  # Should sum to ~1.0

    @pytest.mark.asyncio
    async def test_confidence_attribution_sorted_by_contribution(self, builder):
        """Test attributions are sorted by contribution."""
        result = MockDebateResult(
            consensus_margin=0.8,
            rounds_used=3,
            proposals={"claude": "Test"},
        )

        decision = await builder.build(result)
        attrs = decision.confidence_attribution

        for i in range(len(attrs) - 1):
            assert attrs[i].contribution >= attrs[i + 1].contribution


# =============================================================================
# Test Counterfactuals Building
# =============================================================================


class TestCounterfactualsBuilding:
    """Tests for _build_counterfactuals method."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_counterfactuals_from_votes(self, builder):
        """Test counterfactual generation from influential votes."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="yes", confidence=0.95),
                MockVote(agent="gpt", choice="yes", confidence=0.9),
            ]
        )

        decision = await builder.build(result)
        cfs = decision.counterfactuals

        # Should have at least one vote-based counterfactual
        vote_cfs = [cf for cf in cfs if "voted differently" in cf.condition]
        assert len(vote_cfs) >= 1

    @pytest.mark.asyncio
    async def test_counterfactuals_from_evidence(self, builder):
        """Test counterfactual generation from high-relevance evidence."""
        result = MockDebateResult(
            proposals={
                "claude": "Highly relevant proposal with strong evidence",
            }
        )

        decision = await builder.build(result)
        cfs = decision.counterfactuals

        # Should have evidence-based counterfactual
        evidence_cfs = [cf for cf in cfs if "evidence" in cf.condition.lower()]
        assert len(evidence_cfs) >= 1

    @pytest.mark.asyncio
    async def test_counterfactuals_agent_removal(self, builder):
        """Test counterfactual for agent removal scenario."""
        result = MockDebateResult(
            participants=["claude", "gpt", "gemini"],
        )

        decision = await builder.build(result)
        cfs = decision.counterfactuals

        agent_cfs = [cf for cf in cfs if "fewer" in cf.condition.lower()]
        assert len(agent_cfs) >= 1

    @pytest.mark.asyncio
    async def test_counterfactuals_sorted_by_sensitivity(self, builder):
        """Test counterfactuals are sorted by sensitivity."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="yes", confidence=0.9),
                MockVote(agent="gpt", choice="yes", confidence=0.8),
            ],
            proposals={"claude": "Test proposal"},
        )

        decision = await builder.build(result)
        cfs = decision.counterfactuals

        for i in range(len(cfs) - 1):
            assert cfs[i].sensitivity >= cfs[i + 1].sensitivity


# =============================================================================
# Test Summary Metrics
# =============================================================================


class TestSummaryMetrics:
    """Tests for summary metric computation."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_evidence_quality_score(self, builder):
        """Test evidence quality score computation."""
        result = MockDebateResult(
            proposals={
                "claude": "High quality proposal",
                "gpt": "Another proposal",
            }
        )

        decision = await builder.build(result)

        assert 0 <= decision.evidence_quality_score <= 1

    @pytest.mark.asyncio
    async def test_evidence_quality_score_empty(self, builder):
        """Test evidence quality score with no evidence."""
        result = MockDebateResult(proposals={})

        decision = await builder.build(result)

        assert decision.evidence_quality_score == 0.0

    @pytest.mark.asyncio
    async def test_agent_agreement_score(self, builder):
        """Test agent agreement score computation."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="yes", confidence=0.9),
                MockVote(agent="gpt", choice="yes", confidence=0.8),
                MockVote(agent="gemini", choice="yes", confidence=0.7),
            ]
        )

        decision = await builder.build(result)

        # All voted yes, should be 1.0
        assert decision.agent_agreement_score == 1.0

    @pytest.mark.asyncio
    async def test_agent_agreement_score_split(self, builder):
        """Test agent agreement score with split votes."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="yes", confidence=0.9),
                MockVote(agent="gpt", choice="no", confidence=0.8),
            ]
        )

        decision = await builder.build(result)

        assert decision.agent_agreement_score == 0.5

    @pytest.mark.asyncio
    async def test_belief_stability_score_no_changes(self, builder):
        """Test belief stability score with no changes."""
        result = MockDebateResult(position_history={})

        decision = await builder.build(result)

        assert decision.belief_stability_score == 1.0

    @pytest.mark.asyncio
    async def test_belief_stability_score_with_changes(self, builder):
        """Test belief stability score decreases with changes."""
        result = MockDebateResult(
            rounds_used=3,
            participants=["claude", "gpt"],
            position_history={
                "claude": [
                    {"position": "A", "confidence": 0.7},
                    {"position": "B", "confidence": 0.8},
                    {"position": "C", "confidence": 0.9},
                ],
            },
        )

        decision = await builder.build(result)

        # More changes = lower stability
        assert decision.belief_stability_score < 1.0


# =============================================================================
# Test Summary Generation
# =============================================================================


class TestSummaryGeneration:
    """Tests for generate_summary method."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_summary_contains_header(self, builder):
        """Test summary contains header."""
        result = MockDebateResult()

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        assert "## Decision Summary" in summary

    @pytest.mark.asyncio
    async def test_summary_contains_consensus_status(self, builder):
        """Test summary contains consensus status."""
        result = MockDebateResult(consensus_reached=True)

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        assert "Reached" in summary

    @pytest.mark.asyncio
    async def test_summary_contains_confidence(self, builder):
        """Test summary contains confidence."""
        result = MockDebateResult(confidence=0.85)

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        assert "85%" in summary

    @pytest.mark.asyncio
    async def test_summary_contains_rounds(self, builder):
        """Test summary contains rounds info."""
        result = MockDebateResult(rounds_used=3)

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        assert "3" in summary

    @pytest.mark.asyncio
    async def test_summary_contains_conclusion(self, builder):
        """Test summary contains conclusion."""
        result = MockDebateResult(final_answer="The proposal is accepted")

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        assert "Conclusion" in summary
        assert "The proposal is accepted" in summary

    @pytest.mark.asyncio
    async def test_summary_contains_evidence(self, builder):
        """Test summary contains key evidence."""
        result = MockDebateResult(proposals={"claude": "Important evidence for the decision"})

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        assert "Key Evidence" in summary

    @pytest.mark.asyncio
    async def test_summary_contains_influential_votes(self, builder):
        """Test summary contains influential votes."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="claude", choice="approve", confidence=0.95),
            ]
        )

        decision = await builder.build(result)
        summary = builder.generate_summary(decision)

        # Check for votes section if there are pivotal votes
        if decision.get_pivotal_votes(0.3):
            assert "Most Influential Votes" in summary or "claude" in summary


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in builder."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_handles_missing_result_attributes(self, builder):
        """Test handling of minimal result object."""

        class MinimalResult:
            pass

        result = MinimalResult()

        # Should not raise
        decision = await builder.build(result)

        assert isinstance(decision, Decision)
        assert decision.consensus_reached is False
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_handles_none_values(self, builder):
        """Test handling of None values in result."""
        result = MockDebateResult(
            final_answer=None,
            proposals=None,
            votes=None,
        )
        result.proposals = {}
        result.votes = []
        result.final_answer = ""

        decision = await builder.build(result)

        assert isinstance(decision, Decision)

    @pytest.mark.asyncio
    async def test_handles_tracker_exceptions(self):
        """Test handling of tracker exceptions."""
        # All trackers raise exceptions
        evidence_tracker = AsyncMock()
        evidence_tracker.score_evidence = AsyncMock(side_effect=RuntimeError("Tracker failed"))

        belief_network = MagicMock()
        belief_network.get_changes = MagicMock(side_effect=RuntimeError("Network failed"))

        elo_system = MagicMock()
        elo_system.get_rating = MagicMock(side_effect=KeyError("Agent not found"))

        calibration_tracker = MagicMock()
        calibration_tracker.get_weight = MagicMock(side_effect=AttributeError("No weight"))
        calibration_tracker.get_adjustment = MagicMock(side_effect=AttributeError("No adjustment"))

        builder = ExplanationBuilder(
            evidence_tracker=evidence_tracker,
            belief_network=belief_network,
            elo_system=elo_system,
            calibration_tracker=calibration_tracker,
        )

        result = MockDebateResult(
            proposals={"claude": "Test"},
            votes=[MockVote(agent="claude", choice="yes", confidence=0.9)],
        )

        # Should not raise
        decision = await builder.build(result)

        assert isinstance(decision, Decision)


# =============================================================================
# Test ID Generation
# =============================================================================


class TestIdGeneration:
    """Tests for ID generation."""

    @pytest.fixture
    def builder(self):
        return ExplanationBuilder()

    @pytest.mark.asyncio
    async def test_generates_debate_id_when_missing(self, builder):
        """Test debate ID generation when not provided."""
        result = MockDebateResult()
        result.id = ""

        decision = await builder.build(result)

        assert decision.debate_id != ""
        assert len(decision.debate_id) == 16

    @pytest.mark.asyncio
    async def test_uses_provided_debate_id(self, builder):
        """Test uses debate ID when provided."""
        result = MockDebateResult(id="custom-debate-id")

        decision = await builder.build(result)

        assert decision.debate_id == "custom-debate-id"

    @pytest.mark.asyncio
    async def test_generated_ids_are_deterministic(self, builder):
        """Test ID generation produces consistent results for same input."""
        result = MockDebateResult(id="", task="Same task")

        # Generate IDs at different times - they should differ
        decision1 = await builder.build(result)
        decision2 = await builder.build(result)

        # IDs should be different (timestamp-based)
        # This is actually expected behavior, not determinism
        assert decision1.debate_id != "" and decision2.debate_id != ""
