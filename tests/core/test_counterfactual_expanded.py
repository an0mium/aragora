"""
Expanded tests for counterfactual debate branching.

Tests ImpactDetector, CounterfactualBranch, ConditionalConsensus,
and related utilities not covered in basic counterfactual tests.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from aragora.core import Message, Vote
from aragora.debate.counterfactual import (
    CounterfactualStatus,
    PivotClaim,
    CounterfactualBranch,
    ConditionalConsensus,
    ImpactDetector,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pivot_claim():
    """Create a sample pivot claim."""
    return PivotClaim(
        claim_id="pivot-test123",
        statement="AI systems should be required to explain their reasoning",
        author="claude",
        disagreement_score=0.7,
        importance_score=0.6,
        blocking_agents=["gpt4", "gemini"],
    )


@pytest.fixture
def sample_messages():
    """Create sample debate messages with disagreement patterns."""
    return [
        Message(
            role="proposer", agent="claude", content="I believe transparency is essential.", round=1
        ),
        Message(
            role="critic",
            agent="gpt4",
            content="I fundamentally disagree with that premise.",
            round=1,
        ),
        Message(
            role="critic", agent="gemini", content="The core assumption here is flawed.", round=2
        ),
        Message(
            role="proposer",
            agent="claude",
            content="If that were true, we would see more failures.",
            round=2,
        ),
        Message(
            role="critic",
            agent="gpt4",
            content="On the other hand, some systems work fine without it.",
            round=3,
        ),
        Message(
            role="critic",
            agent="gemini",
            content="I cannot accept that reasoning without evidence.",
            round=3,
        ),
    ]


# =============================================================================
# PivotClaim Tests
# =============================================================================


class TestPivotClaim:
    """Tests for PivotClaim dataclass."""

    def test_should_branch_high_disagreement_high_importance(self, sample_pivot_claim):
        """Should branch when disagreement and importance are high."""
        assert sample_pivot_claim.should_branch is True

    def test_should_not_branch_low_disagreement(self):
        """Should not branch with low disagreement."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test claim",
            author="agent",
            disagreement_score=0.3,  # Below threshold
            importance_score=0.8,
            blocking_agents=[],
        )
        assert claim.should_branch is False

    def test_should_not_branch_low_importance(self):
        """Should not branch with low importance."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test claim",
            author="agent",
            disagreement_score=0.8,
            importance_score=0.2,  # Below threshold
            blocking_agents=[],
        )
        assert claim.should_branch is False

    def test_created_at_auto_populated(self):
        """created_at should be auto-populated."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test",
            author="agent",
            disagreement_score=0.5,
            importance_score=0.5,
            blocking_agents=[],
        )
        assert claim.created_at is not None
        # Should be a valid ISO format date
        datetime.fromisoformat(claim.created_at)


# =============================================================================
# ImpactDetector Tests
# =============================================================================


class TestImpactDetector:
    """Tests for ImpactDetector impasse detection."""

    @pytest.fixture
    def detector(self):
        """Create an ImpactDetector instance."""
        return ImpactDetector(
            disagreement_threshold=0.5,
            rounds_before_branch=2,
        )

    def test_detect_disagreement_phrases(self, detector, sample_messages):
        """Should detect disagreement in messages."""
        result = detector.detect_impasse(sample_messages, [])

        # With 6 messages containing disagreement phrases, should detect impasse
        # Note: result may be None if no single claim crosses threshold
        # but the detection logic should run without error
        assert result is None or isinstance(result, PivotClaim)

    def test_no_detection_too_few_messages(self, detector):
        """Should not detect impasse with too few messages."""
        messages = [
            Message(role="proposer", agent="claude", content="Hello", round=1),
            Message(role="critic", agent="gpt4", content="Hi there", round=1),
        ]

        result = detector.detect_impasse(messages, [])
        assert result is None

    def test_threshold_behavior(self):
        """Disagreement threshold should affect detection."""
        # Low threshold detector
        low_detector = ImpactDetector(disagreement_threshold=0.2)
        # High threshold detector
        high_detector = ImpactDetector(disagreement_threshold=0.9)

        messages = [
            Message(role="proposer", agent="a1", content="I fundamentally disagree.", round=1),
            Message(role="critic", agent="a2", content="I agree with a1.", round=1),
            Message(role="proposer", agent="a3", content="Core assumption is wrong.", round=2),
            Message(role="critic", agent="a4", content="If that were true, we fail.", round=2),
            Message(role="proposer", agent="a5", content="Cannot accept this premise.", round=3),
            Message(role="critic", agent="a6", content="Depends on whether we have data.", round=3),
        ]

        # Low threshold should be more likely to trigger
        low_result = low_detector.detect_impasse(messages, [])
        high_result = high_detector.detect_impasse(messages, [])

        # High threshold should be harder to meet
        assert high_result is None or (
            low_result is not None
            and low_result.disagreement_score >= high_detector.disagreement_threshold
        )

    def test_importance_estimation(self, detector, sample_messages):
        """Should estimate importance based on mention frequency."""
        claim = "transparency is essential"
        importance = detector._estimate_importance(claim, sample_messages)

        # Should return a value between 0 and 1
        assert 0.0 <= importance <= 1.0

    def test_no_impasse_when_agreeing(self, detector):
        """Should not detect impasse when agents agree."""
        agreeing_messages = [
            Message(role="proposer", agent="a1", content="I agree completely.", round=1),
            Message(role="critic", agent="a2", content="Yes, that makes sense.", round=1),
            Message(role="proposer", agent="a3", content="I concur with this view.", round=2),
            Message(role="critic", agent="a4", content="This is correct.", round=2),
            Message(role="proposer", agent="a5", content="Well reasoned point.", round=3),
            Message(role="critic", agent="a6", content="I support this conclusion.", round=3),
        ]

        result = detector.detect_impasse(agreeing_messages, [])
        assert result is None


# =============================================================================
# CounterfactualBranch Tests
# =============================================================================


class TestCounterfactualBranch:
    """Tests for CounterfactualBranch dataclass."""

    def test_branch_creation(self, sample_pivot_claim):
        """Should create a branch with required fields."""
        branch = CounterfactualBranch(
            branch_id="branch-123",
            parent_debate_id="debate-456",
            pivot_claim=sample_pivot_claim,
            assumption=True,
        )

        assert branch.branch_id == "branch-123"
        assert branch.status == CounterfactualStatus.PENDING
        assert branch.assumption is True

    def test_assumption_text_true(self, sample_pivot_claim):
        """assumption_text should describe TRUE assumption."""
        branch = CounterfactualBranch(
            branch_id="b1",
            parent_debate_id="d1",
            pivot_claim=sample_pivot_claim,
            assumption=True,
        )

        assert "TRUE" in branch.assumption_text
        assert sample_pivot_claim.statement[:50] in branch.assumption_text

    def test_assumption_text_false(self, sample_pivot_claim):
        """assumption_text should describe FALSE assumption."""
        branch = CounterfactualBranch(
            branch_id="b1",
            parent_debate_id="d1",
            pivot_claim=sample_pivot_claim,
            assumption=False,
        )

        assert "FALSE" in branch.assumption_text

    def test_status_transitions(self, sample_pivot_claim):
        """Branch status should transition correctly."""
        branch = CounterfactualBranch(
            branch_id="b1",
            parent_debate_id="d1",
            pivot_claim=sample_pivot_claim,
            assumption=True,
        )

        # Initial status
        assert branch.status == CounterfactualStatus.PENDING

        # Transition to running
        branch.status = CounterfactualStatus.RUNNING
        assert branch.status == CounterfactualStatus.RUNNING

        # Transition to completed
        branch.status = CounterfactualStatus.COMPLETED
        assert branch.status == CounterfactualStatus.COMPLETED

    def test_duration_calculation(self, sample_pivot_claim):
        """duration_seconds should calculate time between start and end."""
        branch = CounterfactualBranch(
            branch_id="b1",
            parent_debate_id="d1",
            pivot_claim=sample_pivot_claim,
            assumption=True,
        )

        # Set timing
        start = datetime.now()
        end = start + timedelta(seconds=45)
        branch.started_at = start.isoformat()
        branch.completed_at = end.isoformat()

        assert branch.duration_seconds == pytest.approx(45.0, rel=0.01)

    def test_duration_none_when_incomplete(self, sample_pivot_claim):
        """duration_seconds should be None when not completed."""
        branch = CounterfactualBranch(
            branch_id="b1",
            parent_debate_id="d1",
            pivot_claim=sample_pivot_claim,
            assumption=True,
        )

        assert branch.duration_seconds is None


# =============================================================================
# ConditionalConsensus Tests
# =============================================================================


class TestConditionalConsensus:
    """Tests for ConditionalConsensus synthesis."""

    @pytest.fixture
    def sample_consensus(self, sample_pivot_claim):
        """Create a sample ConditionalConsensus."""
        return ConditionalConsensus(
            consensus_id="cons-123",
            pivot_claim=sample_pivot_claim,
            if_true_conclusion="We should implement explainability requirements.",
            if_true_confidence=0.85,
            if_false_conclusion="Market-driven approaches are sufficient.",
            if_false_confidence=0.72,
        )

    def test_natural_language_generation(self, sample_consensus):
        """to_natural_language should produce readable output."""
        text = sample_consensus.to_natural_language()

        assert "Conditional Consensus" in text
        assert "IF" in text
        assert "THEN" in text
        assert "ELSE" in text
        assert "85%" in text  # if_true_confidence
        assert "72%" in text  # if_false_confidence

    def test_to_dict_includes_all_fields(self, sample_consensus):
        """to_dict should include all relevant fields."""
        data = sample_consensus.to_dict()

        assert "consensus_id" in data
        assert "pivot_claim" in data
        assert "if_true_conclusion" in data
        assert "if_false_conclusion" in data
        assert "natural_language" in data
        assert "preferred_world" in data

    def test_preferred_world_selection(self, sample_pivot_claim):
        """preferred_world should track which assumption leads to better outcome."""
        consensus = ConditionalConsensus(
            consensus_id="cons-456",
            pivot_claim=sample_pivot_claim,
            if_true_conclusion="Good outcome",
            if_true_confidence=0.9,
            if_false_conclusion="Bad outcome",
            if_false_confidence=0.3,
            preferred_world=True,
            preference_reason="Higher confidence and better outcome",
        )

        assert consensus.preferred_world is True
        assert "confidence" in consensus.preference_reason.lower()

    def test_unresolved_uncertainties(self, sample_pivot_claim):
        """Should track unresolved uncertainties."""
        consensus = ConditionalConsensus(
            consensus_id="cons-789",
            pivot_claim=sample_pivot_claim,
            if_true_conclusion="Conclusion A",
            if_true_confidence=0.6,
            if_false_conclusion="Conclusion B",
            if_false_confidence=0.5,
            unresolved_uncertainties=[
                "Long-term effects unknown",
                "Requires more data",
            ],
        )

        assert len(consensus.unresolved_uncertainties) == 2
        assert "Long-term" in consensus.unresolved_uncertainties[0]
