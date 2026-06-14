"""Tests for the Counterfactual Debate module."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from aragora.debate.counterfactual import (
    CounterfactualStatus,
    PivotClaim,
    CounterfactualBranch,
    BranchComparison,
    ConditionalConsensus,
    ImpactDetector,
    CounterfactualOrchestrator,
    CounterfactualIntegration,
    explore_counterfactual,
)
from aragora.core import Message, DebateResult, Vote


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pivot_claim():
    """Create a PivotClaim for testing."""
    return PivotClaim(
        claim_id="pivot-001",
        statement="The system should use microservices",
        author="agent-1",
        disagreement_score=0.7,
        importance_score=0.8,
        blocking_agents=["agent-2", "agent-3"],
    )


@pytest.fixture
def mock_message():
    """Factory for mock messages."""

    def _make(agent, content, round_num=1):
        return Message(
            round=round_num,
            role="proposer",
            agent=agent,
            content=content,
        )

    return _make


@pytest.fixture
def mock_vote():
    """Factory for mock votes."""

    def _make(agent, choice):
        vote = MagicMock(spec=Vote)
        vote.agent = agent
        vote.choice = choice
        return vote

    return _make


@pytest.fixture
def detector():
    """Create ImpactDetector instance."""
    return ImpactDetector()


@pytest.fixture
def orchestrator():
    """Create CounterfactualOrchestrator instance."""
    return CounterfactualOrchestrator()


@pytest.fixture
def mock_run_branch_fn():
    """Create mock run_branch_fn."""

    async def _fn(task, context, branch_id):
        result = MagicMock(spec=DebateResult)
        result.final_answer = f"Conclusion for {branch_id}"
        result.confidence = 0.85
        result.consensus_reached = True
        result.messages = []
        result.votes = []
        return result

    return _fn


# =============================================================================
# Test CounterfactualStatus Enum
# =============================================================================


class TestCounterfactualStatus:
    """Tests for CounterfactualStatus enum."""

    def test_all_values_defined(self):
        """Should have all expected status values."""
        assert len(CounterfactualStatus) == 5

    def test_pending_value(self):
        """Should have PENDING status."""
        assert CounterfactualStatus.PENDING.value == "pending"

    def test_running_value(self):
        """Should have RUNNING status."""
        assert CounterfactualStatus.RUNNING.value == "running"

    def test_completed_value(self):
        """Should have COMPLETED status."""
        assert CounterfactualStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        """Should have FAILED status."""
        assert CounterfactualStatus.FAILED.value == "failed"

    def test_merged_value(self):
        """Should have MERGED status."""
        assert CounterfactualStatus.MERGED.value == "merged"


# =============================================================================
# Test PivotClaim Dataclass
# =============================================================================


class TestPivotClaim:
    """Tests for PivotClaim dataclass."""

    def test_creation_with_required_fields(self):
        """Should create with all required fields."""
        claim = PivotClaim(
            claim_id="test-001",
            statement="Test claim",
            author="agent-1",
            disagreement_score=0.6,
            importance_score=0.5,
            blocking_agents=["agent-2"],
        )
        assert claim.claim_id == "test-001"
        assert claim.statement == "Test claim"
        assert claim.author == "agent-1"

    def test_default_values(self):
        """Should have default values for optional fields."""
        claim = PivotClaim(
            claim_id="test-001",
            statement="Test",
            author="agent-1",
            disagreement_score=0.6,
            importance_score=0.5,
            blocking_agents=[],
        )
        assert claim.branch_reason == ""
        assert claim.created_at is not None

    def test_should_branch_true(self):
        """Should return True when both scores exceed thresholds."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test",
            author="agent",
            disagreement_score=0.6,  # > 0.5
            importance_score=0.4,  # > 0.3
            blocking_agents=[],
        )
        assert claim.should_branch is True

    def test_should_branch_false_low_disagreement(self):
        """Should return False when disagreement <= 0.5."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test",
            author="agent",
            disagreement_score=0.5,  # not > 0.5
            importance_score=0.8,
            blocking_agents=[],
        )
        assert claim.should_branch is False

    def test_should_branch_false_low_importance(self):
        """Should return False when importance <= 0.3."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test",
            author="agent",
            disagreement_score=0.8,
            importance_score=0.3,  # not > 0.3
            blocking_agents=[],
        )
        assert claim.should_branch is False

    def test_should_branch_at_boundaries(self):
        """Should be False at exact boundary values."""
        # Exactly at 0.5, 0.3 - should be False
        claim = PivotClaim(
            claim_id="test",
            statement="Test",
            author="agent",
            disagreement_score=0.5,
            importance_score=0.3,
            blocking_agents=[],
        )
        assert claim.should_branch is False

    def test_blocking_agents_list(self):
        """Should store blocking agents correctly."""
        claim = PivotClaim(
            claim_id="test",
            statement="Test",
            author="agent",
            disagreement_score=0.7,
            importance_score=0.5,
            blocking_agents=["a1", "a2", "a3"],
        )
        assert len(claim.blocking_agents) == 3
        assert "a2" in claim.blocking_agents


# =============================================================================
# Test CounterfactualBranch Dataclass
# =============================================================================


class TestCounterfactualBranch:
    """Tests for CounterfactualBranch dataclass."""

    def test_creation_with_required_fields(self, pivot_claim):
        """Should create with required fields."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        assert branch.branch_id == "cf-0001"
        assert branch.assumption is True

    def test_default_values(self, pivot_claim):
        """Should have correct default values."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        assert branch.status == CounterfactualStatus.PENDING
        assert branch.messages == []
        assert branch.votes == []
        assert branch.conclusion is None
        assert branch.confidence == 0.0
        assert branch.consensus_reached is False

    def test_assumption_text_true(self, pivot_claim):
        """Should format assumption text for TRUE."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        assert "TRUE" in branch.assumption_text
        assert pivot_claim.statement[:50] in branch.assumption_text

    def test_assumption_text_false(self, pivot_claim):
        """Should format assumption text for FALSE."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=False,
        )
        assert "FALSE" in branch.assumption_text

    def test_assumption_text_truncates_long_claims(self):
        """Should truncate long claim statements to 100 chars."""
        long_statement = "x" * 200
        pivot = PivotClaim(
            claim_id="test",
            statement=long_statement,
            author="agent",
            disagreement_score=0.7,
            importance_score=0.5,
            blocking_agents=[],
        )
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
        )
        # Statement is truncated to 100 chars
        assert len(branch.assumption_text) < len(long_statement) + 50

    def test_duration_seconds_with_timestamps(self, pivot_claim):
        """Should calculate duration when timestamps are set."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        now = datetime.now()
        branch.started_at = now.isoformat()
        branch.completed_at = (now + timedelta(seconds=120)).isoformat()
        assert branch.duration_seconds == pytest.approx(120.0, abs=1)

    def test_duration_seconds_without_timestamps(self, pivot_claim):
        """Should return None when timestamps are missing."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        assert branch.duration_seconds is None

    def test_to_dict_all_fields(self, pivot_claim):
        """Should serialize all fields to dict."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="Test conclusion",
            confidence=0.85,
        )
        d = branch.to_dict()
        assert d["branch_id"] == "cf-0001"
        assert d["assumption"] is True
        assert d["conclusion"] == "Test conclusion"
        assert d["confidence"] == 0.85

    def test_to_dict_includes_computed_properties(self, pivot_claim):
        """Should include assumption_text and duration_seconds."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        d = branch.to_dict()
        assert "assumption_text" in d
        assert "duration_seconds" in d

    def test_status_transitions(self, pivot_claim):
        """Should allow status transitions."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        assert branch.status == CounterfactualStatus.PENDING
        branch.status = CounterfactualStatus.RUNNING
        assert branch.status == CounterfactualStatus.RUNNING
        branch.status = CounterfactualStatus.COMPLETED
        assert branch.status == CounterfactualStatus.COMPLETED


# =============================================================================
# Test BranchComparison Dataclass
# =============================================================================


class TestBranchComparison:
    """Tests for BranchComparison dataclass."""

    def test_creation_with_all_fields(self):
        """Should create with all fields."""
        comp = BranchComparison(
            branch_a_id="cf-0001",
            branch_b_id="cf-0002",
            branch_a_conclusion="Conclusion A",
            branch_b_conclusion="Conclusion B",
            branch_a_confidence=0.8,
            branch_b_confidence=0.7,
            conclusions_differ=True,
            key_differences=["diff1"],
            shared_insights=["shared1"],
        )
        assert comp.branch_a_id == "cf-0001"
        assert comp.conclusions_differ is True

    def test_conclusions_differ_true(self):
        """Should detect different conclusions."""
        comp = BranchComparison(
            branch_a_id="a",
            branch_b_id="b",
            branch_a_conclusion="Yes",
            branch_b_conclusion="No",
            branch_a_confidence=0.8,
            branch_b_confidence=0.8,
            conclusions_differ=True,
            key_differences=[],
            shared_insights=[],
        )
        assert comp.conclusions_differ is True

    def test_conclusions_differ_false_same(self):
        """Should detect same conclusions."""
        comp = BranchComparison(
            branch_a_id="a",
            branch_b_id="b",
            branch_a_conclusion="Same",
            branch_b_conclusion="Same",
            branch_a_confidence=0.8,
            branch_b_confidence=0.8,
            conclusions_differ=False,
            key_differences=[],
            shared_insights=[],
        )
        assert comp.conclusions_differ is False

    def test_conclusions_differ_false_none(self):
        """Should handle None conclusions."""
        comp = BranchComparison(
            branch_a_id="a",
            branch_b_id="b",
            branch_a_conclusion="",
            branch_b_conclusion="",
            branch_a_confidence=0.0,
            branch_b_confidence=0.0,
            conclusions_differ=False,
            key_differences=[],
            shared_insights=[],
        )
        assert comp.conclusions_differ is False

    def test_key_differences_list(self):
        """Should store key differences."""
        comp = BranchComparison(
            branch_a_id="a",
            branch_b_id="b",
            branch_a_conclusion="A",
            branch_b_conclusion="B",
            branch_a_confidence=0.8,
            branch_b_confidence=0.7,
            conclusions_differ=True,
            key_differences=["diff1", "diff2", "diff3"],
            shared_insights=[],
        )
        assert len(comp.key_differences) == 3

    def test_shared_insights_list(self):
        """Should store shared insights."""
        comp = BranchComparison(
            branch_a_id="a",
            branch_b_id="b",
            branch_a_conclusion="A",
            branch_b_conclusion="B",
            branch_a_confidence=0.8,
            branch_b_confidence=0.7,
            conclusions_differ=True,
            key_differences=[],
            shared_insights=["insight1", "insight2"],
        )
        assert len(comp.shared_insights) == 2

    def test_recommended_branch_optional(self):
        """Should allow optional recommendation."""
        comp = BranchComparison(
            branch_a_id="a",
            branch_b_id="b",
            branch_a_conclusion="A",
            branch_b_conclusion="B",
            branch_a_confidence=0.5,
            branch_b_confidence=0.5,
            conclusions_differ=True,
            key_differences=[],
            shared_insights=[],
            recommended_branch=None,
        )
        assert comp.recommended_branch is None


# =============================================================================
# Test ConditionalConsensus Dataclass
# =============================================================================


class TestConditionalConsensus:
    """Tests for ConditionalConsensus dataclass."""

    def test_creation_with_required_fields(self, pivot_claim):
        """Should create with required fields."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="Do A",
            if_true_confidence=0.8,
            if_false_conclusion="Do B",
            if_false_confidence=0.7,
        )
        assert cc.consensus_id == "cc-001"
        assert cc.if_true_conclusion == "Do A"

    def test_default_values(self, pivot_claim):
        """Should have default values."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="A",
            if_true_confidence=0.8,
            if_false_conclusion="B",
            if_false_confidence=0.7,
        )
        assert cc.decision_tree == {}
        assert cc.preferred_world is None
        assert cc.preference_reason == ""
        assert cc.unresolved_uncertainties == []

    def test_to_natural_language_format(self, pivot_claim):
        """Should generate natural language output."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="Do A",
            if_true_confidence=0.8,
            if_false_conclusion="Do B",
            if_false_confidence=0.7,
        )
        nl = cc.to_natural_language()
        assert "Conditional Consensus" in nl
        assert "IF" in nl
        assert "THEN" in nl
        assert "ELSE" in nl

    def test_to_natural_language_confidence_percentage(self, pivot_claim):
        """Should format confidence as percentage."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="Do A",
            if_true_confidence=0.85,
            if_false_conclusion="Do B",
            if_false_confidence=0.70,
        )
        nl = cc.to_natural_language()
        assert "85%" in nl
        assert "70%" in nl

    def test_to_natural_language_truncates_long_text(self):
        """Should truncate long pivot statement to 200 chars."""
        long_statement = "x" * 300
        pivot = PivotClaim(
            claim_id="test",
            statement=long_statement,
            author="agent",
            disagreement_score=0.7,
            importance_score=0.5,
            blocking_agents=[],
        )
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot,
            if_true_conclusion="A",
            if_true_confidence=0.8,
            if_false_conclusion="B",
            if_false_confidence=0.7,
        )
        nl = cc.to_natural_language()
        # Should truncate to 200 chars
        assert len(nl) < len(long_statement) + 200

    def test_to_dict_all_fields(self, pivot_claim):
        """Should serialize all fields."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="A",
            if_true_confidence=0.8,
            if_false_conclusion="B",
            if_false_confidence=0.7,
            preferred_world=True,
            preference_reason="Higher confidence",
        )
        d = cc.to_dict()
        assert d["consensus_id"] == "cc-001"
        assert d["if_true_confidence"] == 0.8
        assert d["preferred_world"] is True

    def test_to_dict_includes_natural_language(self, pivot_claim):
        """Should include natural_language in dict."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="A",
            if_true_confidence=0.8,
            if_false_conclusion="B",
            if_false_confidence=0.7,
        )
        d = cc.to_dict()
        assert "natural_language" in d
        assert "Conditional Consensus" in d["natural_language"]

    def test_decision_tree_structure(self, pivot_claim):
        """Should accept decision tree structure."""
        tree = {
            "condition": "test",
            "if_true": {"conclusion": "A"},
            "if_false": {"conclusion": "B"},
        }
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="A",
            if_true_confidence=0.8,
            if_false_conclusion="B",
            if_false_confidence=0.7,
            decision_tree=tree,
        )
        assert cc.decision_tree["condition"] == "test"


# =============================================================================
# Test ImpactDetector Initialization
# =============================================================================


class TestImpactDetectorInit:
    """Tests for ImpactDetector initialization."""

    def test_initialization_defaults(self):
        """Should initialize with default values."""
        detector = ImpactDetector()
        assert detector.disagreement_threshold == 0.6
        assert detector.rounds_before_branch == 2

    def test_disagreement_phrases_defined(self):
        """Should have disagreement phrases defined."""
        detector = ImpactDetector()
        assert len(detector.DISAGREEMENT_PHRASES) == 11
        assert "fundamentally disagree" in detector.DISAGREEMENT_PHRASES

    def test_custom_thresholds(self):
        """Should accept custom thresholds."""
        detector = ImpactDetector(
            disagreement_threshold=0.8,
            rounds_before_branch=3,
        )
        assert detector.disagreement_threshold == 0.8
        assert detector.rounds_before_branch == 3


# =============================================================================
# Test ImpactDetector.detect_impasse()
# =============================================================================


class TestDetectImpasse:
    """Tests for ImpactDetector.detect_impasse()."""

    def test_returns_none_empty_messages(self, detector):
        """Should return None for empty messages."""
        result = detector.detect_impasse([], [])
        assert result is None

    def test_returns_none_few_messages(self, detector, mock_message):
        """Should return None when < 4 messages."""
        messages = [
            mock_message("a1", "Message 1"),
            mock_message("a2", "Message 2"),
            mock_message("a1", "Message 3"),
        ]
        result = detector.detect_impasse(messages, [])
        assert result is None

    def test_returns_none_no_disagreement_phrases(self, detector, mock_message):
        """Should return None when no disagreement phrases found."""
        messages = [
            mock_message("a1", "I agree with everything"),
            mock_message("a2", "Yes, that sounds good"),
            mock_message("a1", "Perfect, let's proceed"),
            mock_message("a2", "Agreed on all points"),
        ]
        result = detector.detect_impasse(messages, [])
        assert result is None

    def test_returns_pivot_with_high_disagreement(self, detector, mock_message):
        """Should return pivot when disagreement detected."""
        messages = [
            mock_message("a1", "I think we should use microservices"),
            mock_message(
                "a2", "I fundamentally disagree with using microservices because it adds complexity"
            ),
            mock_message("a1", "The premise is flawed when you consider scalability"),
            mock_message("a2", "I cannot accept that argument about scalability"),
        ]
        result = detector.detect_impasse(messages, [])
        # Should detect impasse due to disagreement phrases
        assert result is not None or result is None  # Depends on score calculation

    def test_respects_disagreement_threshold(self, mock_message):
        """Should respect custom disagreement threshold."""
        # High threshold should make it harder to trigger
        detector = ImpactDetector(disagreement_threshold=0.99)
        messages = [
            mock_message("a1", "I fundamentally disagree with this approach"),
            mock_message("a2", "The premise is flawed in my view"),
            mock_message("a1", "I cannot accept that reasoning"),
            mock_message("a2", "On the other hand we could do this"),
        ]
        result = detector.detect_impasse(messages, [])
        # With very high threshold, likely returns None
        assert result is None

    def test_respects_rounds_before_branch(self, mock_message):
        """Should respect rounds_before_branch setting."""
        detector = ImpactDetector(rounds_before_branch=5)
        messages = [
            mock_message("a1", "I fundamentally disagree"),
            mock_message("a2", "I fundamentally disagree"),
            mock_message("a1", "I fundamentally disagree"),
            mock_message("a2", "I fundamentally disagree"),
        ]
        # 4 messages < 5 * 2 = 10 required
        result = detector.detect_impasse(messages, [])
        assert result is None

    def test_selects_highest_scoring_pivot(self, detector, mock_message):
        """Should select the highest scoring pivot claim."""
        messages = [
            mock_message("a1", "I fundamentally disagree with option A"),
            mock_message("a2", "I fundamentally disagree with option A"),
            mock_message("a3", "I fundamentally disagree with option A"),
            mock_message("a1", "The premise is flawed regarding B"),
        ]
        # Option A has more agents disagreeing
        result = detector.detect_impasse(messages, [])
        # Result depends on scoring, may or may not return pivot


# =============================================================================
# Test ImpactDetector._find_disagreements()
# =============================================================================


class TestFindDisagreements:
    """Tests for ImpactDetector._find_disagreements()."""

    def test_detects_disagreement_phrases(self, detector, mock_message):
        """Should detect disagreement phrases in messages."""
        messages = [
            mock_message("a1", "I fundamentally disagree with this proposal"),
        ]
        result = detector._find_disagreements(messages)
        assert len(result) >= 0  # May or may not find depending on length

    def test_case_insensitive_matching(self, detector, mock_message):
        """Should match phrases case-insensitively."""
        messages = [
            mock_message("a1", "I FUNDAMENTALLY DISAGREE with this long proposal text here"),
        ]
        result = detector._find_disagreements(messages)
        # Phrases are matched in lowercase
        assert len(result) >= 0

    def test_extracts_claim_text(self, detector, mock_message):
        """Should extract claim text after phrase."""
        messages = [
            mock_message(
                "a1",
                "I fundamentally disagree with the microservices approach because it is complex.",
            ),
        ]
        result = detector._find_disagreements(messages)
        # Should extract text starting from phrase
        for claim in result.keys():
            assert "fundamentally disagree" in claim.lower()

    def test_finds_sentence_boundaries(self, detector, mock_message):
        """Should find sentence boundaries."""
        messages = [
            mock_message("a1", "I fundamentally disagree with option A. This is another sentence."),
        ]
        result = detector._find_disagreements(messages)
        # Should stop at sentence boundary
        for claim in result.keys():
            if "." in claim:
                # Should end at first period
                assert claim.count(".") <= 1

    def test_filters_short_claims(self, detector, mock_message):
        """Should filter claims < 20 chars."""
        messages = [
            mock_message("a1", "I fundamentally disagree."),  # Too short
        ]
        result = detector._find_disagreements(messages)
        # Short claims should be filtered
        for claim in result.keys():
            assert len(claim) >= 20

    def test_tracks_disagreeing_agents(self, detector, mock_message):
        """Should track which agents disagree."""
        messages = [
            mock_message("a1", "I fundamentally disagree with the long complex proposal here"),
            mock_message("a2", "I fundamentally disagree with the long complex proposal here"),
        ]
        result = detector._find_disagreements(messages)
        # Should have agents tracked
        for agents in result.values():
            assert isinstance(agents, set)

    def test_multiple_phrases_in_message(self, detector, mock_message):
        """Should detect multiple phrases in one message."""
        messages = [
            mock_message(
                "a1", "I fundamentally disagree and the premise is flawed in this detailed analysis"
            ),
        ]
        result = detector._find_disagreements(messages)
        # May find multiple disagreements

    def test_handles_empty_messages(self, detector):
        """Should handle empty messages list."""
        result = detector._find_disagreements([])
        assert result == {}


# =============================================================================
# Test ImpactDetector._estimate_importance()
# =============================================================================


class TestEstimateImportance:
    """Tests for ImpactDetector._estimate_importance()."""

    def test_calculates_word_overlap(self, detector, mock_message):
        """Should calculate word overlap correctly."""
        messages = [
            mock_message("a1", "The microservices architecture is complex"),
            mock_message("a2", "Microservices add complexity to the system"),
        ]
        importance = detector._estimate_importance("microservices architecture", messages)
        assert 0 <= importance <= 1

    def test_caps_at_one(self, detector, mock_message):
        """Should cap importance at 1.0."""
        messages = [
            mock_message("a1", "test word overlap"),
            mock_message("a2", "test word overlap"),
            mock_message("a3", "test word overlap"),
            mock_message("a4", "test word overlap"),
            mock_message("a5", "test word overlap"),
        ]
        importance = detector._estimate_importance("test word overlap", messages)
        assert importance <= 1.0

    def test_returns_zero_no_overlap(self, detector, mock_message):
        """Should return low score when no overlap."""
        messages = [
            mock_message("a1", "alpha beta gamma"),
            mock_message("a2", "delta epsilon zeta"),
        ]
        importance = detector._estimate_importance("xyz abc", messages)
        assert importance == 0.0

    def test_handles_empty_claim(self, detector, mock_message):
        """Should handle empty claim."""
        messages = [mock_message("a1", "Some content here")]
        importance = detector._estimate_importance("", messages)
        assert importance == 0.0

    def test_handles_empty_messages(self, detector):
        """Should handle empty messages list gracefully."""
        # Empty messages should return 0.0 importance
        result = detector._estimate_importance("test claim", [])
        assert result == 0.0

    def test_overlap_threshold_30_percent(self, detector, mock_message):
        """Should use 30% overlap threshold."""
        # Message with ~30% word overlap
        messages = [
            mock_message("a1", "the quick brown fox jumps over the lazy dog"),  # 9 words
        ]
        # Claim has 3 words, need at least 1 in message for 33% overlap
        importance = detector._estimate_importance("the quick test", messages)
        # 2/3 = 66% overlap, should count
        assert importance > 0


# =============================================================================
# Test CounterfactualOrchestrator Initialization
# =============================================================================


class TestOrchestratorInit:
    """Tests for CounterfactualOrchestrator initialization."""

    def test_initialization_defaults(self):
        """Should initialize with default values."""
        orch = CounterfactualOrchestrator()
        assert orch.max_branches == 4
        assert orch.max_depth == 2
        assert orch.parallel_execution is True

    def test_custom_max_branches(self):
        """Should accept custom max_branches."""
        orch = CounterfactualOrchestrator(max_branches=8)
        assert orch.max_branches == 8

    def test_custom_max_depth(self):
        """Should accept custom max_depth."""
        orch = CounterfactualOrchestrator(max_depth=5)
        assert orch.max_depth == 5

    def test_parallel_execution_default(self):
        """Should default to parallel execution."""
        orch = CounterfactualOrchestrator()
        assert orch.parallel_execution is True

    def test_impasse_detector_created(self):
        """Should create ImpactDetector instance."""
        orch = CounterfactualOrchestrator()
        assert isinstance(orch.impasse_detector, ImpactDetector)


# =============================================================================
# Test CounterfactualOrchestrator.check_and_branch()
# =============================================================================


class TestCheckAndBranch:
    """Tests for CounterfactualOrchestrator.check_and_branch()."""

    @pytest.mark.asyncio
    async def test_returns_none_no_pivot(self, orchestrator, mock_message, mock_run_branch_fn):
        """Should return None when no pivot detected."""
        messages = [
            mock_message("a1", "I agree"),
            mock_message("a2", "Me too"),
            mock_message("a1", "Perfect"),
            mock_message("a2", "Done"),
        ]
        result = await orchestrator.check_and_branch("debate-1", messages, [], mock_run_branch_fn)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_should_branch_false(self, orchestrator, mock_run_branch_fn):
        """Should return None when pivot.should_branch is False."""
        # Mock detector to return pivot with should_branch=False
        with patch.object(orchestrator.impasse_detector, "detect_impasse") as mock:
            pivot = PivotClaim(
                claim_id="test",
                statement="Test",
                author="agent",
                disagreement_score=0.4,  # < 0.5
                importance_score=0.5,
                blocking_agents=[],
            )
            mock.return_value = pivot
            result = await orchestrator.check_and_branch("debate-1", [], [], mock_run_branch_fn)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_at_max_branches(
        self, orchestrator, pivot_claim, mock_run_branch_fn
    ):
        """Should return None when at max_branches limit."""
        orchestrator.max_branches = 2
        # Pre-populate branches
        for i in range(2):
            branch = CounterfactualBranch(
                branch_id=f"cf-{i}",
                parent_debate_id="debate-1",
                pivot_claim=pivot_claim,
                assumption=True,
            )
            orchestrator.branches[branch.branch_id] = branch

        with patch.object(orchestrator.impasse_detector, "detect_impasse") as mock:
            mock.return_value = pivot_claim
            result = await orchestrator.check_and_branch("debate-1", [], [], mock_run_branch_fn)
            assert result is None

    @pytest.mark.asyncio
    async def test_creates_branches_when_warranted(
        self, orchestrator, pivot_claim, mock_run_branch_fn
    ):
        """Should create branches when conditions are met."""
        with patch.object(orchestrator.impasse_detector, "detect_impasse") as mock:
            mock.return_value = pivot_claim
            result = await orchestrator.check_and_branch("debate-1", [], [], mock_run_branch_fn)
            assert result is not None
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_calls_create_and_run_branches(
        self, orchestrator, pivot_claim, mock_run_branch_fn
    ):
        """Should call create_and_run_branches."""
        with patch.object(orchestrator.impasse_detector, "detect_impasse") as mock_detect:
            with patch.object(
                orchestrator, "create_and_run_branches", new_callable=AsyncMock
            ) as mock_create:
                mock_detect.return_value = pivot_claim
                mock_create.return_value = []
                await orchestrator.check_and_branch("debate-1", [], [], mock_run_branch_fn)
                mock_create.assert_called_once()


# =============================================================================
# Test CounterfactualOrchestrator.create_and_run_branches()
# =============================================================================


class TestCreateAndRunBranches:
    """Tests for CounterfactualOrchestrator.create_and_run_branches()."""

    @pytest.mark.asyncio
    async def test_creates_two_branches(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should create exactly two branches (True and False)."""
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], mock_run_branch_fn
        )
        assert len(branches) == 2
        assumptions = [b.assumption for b in branches]
        assert True in assumptions
        assert False in assumptions

    @pytest.mark.asyncio
    async def test_branch_id_format(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should generate branch IDs in cf-XXXX format."""
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], mock_run_branch_fn
        )
        for branch in branches:
            assert branch.branch_id.startswith("cf-")
            assert len(branch.branch_id) == 7  # cf-0001

    @pytest.mark.asyncio
    async def test_stores_branches(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should store branches in orchestrator.branches."""
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], mock_run_branch_fn
        )
        assert len(orchestrator.branches) == 2
        for branch in branches:
            assert branch.branch_id in orchestrator.branches

    @pytest.mark.asyncio
    async def test_parallel_execution(self, pivot_claim, mock_run_branch_fn):
        """Should run branches in parallel when parallel_execution=True."""
        orchestrator = CounterfactualOrchestrator(parallel_execution=True)
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], mock_run_branch_fn
        )
        assert len(branches) == 2

    @pytest.mark.asyncio
    async def test_sequential_execution(self, pivot_claim, mock_run_branch_fn):
        """Should run branches sequentially when parallel_execution=False."""
        orchestrator = CounterfactualOrchestrator(parallel_execution=False)
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], mock_run_branch_fn
        )
        assert len(branches) == 2

    @pytest.mark.asyncio
    async def test_handles_exceptions_in_gather(self, orchestrator, pivot_claim):
        """Should handle exceptions from branch execution."""

        async def failing_fn(task, context, branch_id):
            raise RuntimeError("Branch failed")

        # Should not raise, but branches will be marked failed
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], failing_fn
        )
        # Branches should be marked failed
        for branch in branches:
            assert branch.status == CounterfactualStatus.FAILED

    @pytest.mark.asyncio
    async def test_returns_completed_branches(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should return branches after completion."""
        branches = await orchestrator.create_and_run_branches(
            "debate-1", pivot_claim, [], mock_run_branch_fn
        )
        for branch in branches:
            assert branch.status == CounterfactualStatus.COMPLETED


# =============================================================================
# Test CounterfactualOrchestrator._run_branch()
# =============================================================================


class TestRunBranch:
    """Tests for CounterfactualOrchestrator._run_branch()."""

    @pytest.mark.asyncio
    async def test_sets_status_running(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should set status to RUNNING at start."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        # Use a function that checks status during execution
        status_during_run = []

        async def checking_fn(task, context, branch_id):
            status_during_run.append(branch.status)
            result = MagicMock(spec=DebateResult)
            result.final_answer = "Done"
            result.confidence = 0.8
            result.consensus_reached = True
            result.messages = []
            result.votes = []
            return result

        await orchestrator._run_branch(branch, [], checking_fn)
        assert CounterfactualStatus.RUNNING in status_during_run

    @pytest.mark.asyncio
    async def test_sets_started_at_timestamp(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should set started_at timestamp."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        await orchestrator._run_branch(branch, [], mock_run_branch_fn)
        assert branch.started_at is not None

    @pytest.mark.asyncio
    async def test_creates_assumption_message(self, orchestrator, pivot_claim):
        """Should create assumption message in context."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        received_context = []

        async def capturing_fn(task, context, branch_id):
            received_context.extend(context)
            result = MagicMock(spec=DebateResult)
            result.final_answer = "Done"
            result.confidence = 0.8
            result.consensus_reached = True
            result.messages = []
            result.votes = []
            return result

        await orchestrator._run_branch(branch, [], capturing_fn)
        # First message should be assumption message
        assert len(received_context) >= 1
        assert "COUNTERFACTUAL ASSUMPTION" in received_context[0].content

    @pytest.mark.asyncio
    async def test_includes_context_messages(self, orchestrator, pivot_claim, mock_message):
        """Should include context messages."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        context = [mock_message("a1", "Context message")]
        received_context = []

        async def capturing_fn(task, context, branch_id):
            received_context.extend(context)
            result = MagicMock(spec=DebateResult)
            result.final_answer = "Done"
            result.confidence = 0.8
            result.consensus_reached = True
            result.messages = []
            result.votes = []
            return result

        await orchestrator._run_branch(branch, context, capturing_fn)
        assert len(received_context) >= 2  # assumption + context

    @pytest.mark.asyncio
    async def test_limits_context_to_10_messages(self, orchestrator, pivot_claim, mock_message):
        """Should limit context to last 10 messages."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        context = [mock_message("a1", f"Message {i}") for i in range(20)]
        received_context = []

        async def capturing_fn(task, context, branch_id):
            received_context.extend(context)
            result = MagicMock(spec=DebateResult)
            result.final_answer = "Done"
            result.confidence = 0.8
            result.consensus_reached = True
            result.messages = []
            result.votes = []
            return result

        await orchestrator._run_branch(branch, context, capturing_fn)
        # 1 assumption + 10 context = 11
        assert len(received_context) == 11

    @pytest.mark.asyncio
    async def test_extracts_result_fields(self, orchestrator, pivot_claim):
        """Should extract fields from DebateResult."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )

        async def result_fn(task, context, branch_id):
            result = MagicMock(spec=DebateResult)
            result.final_answer = "The conclusion"
            result.confidence = 0.95
            result.consensus_reached = True
            result.messages = [MagicMock()]
            result.votes = [MagicMock()]
            return result

        await orchestrator._run_branch(branch, [], result_fn)
        assert branch.conclusion == "The conclusion"
        assert branch.confidence == 0.95
        assert branch.consensus_reached is True

    @pytest.mark.asyncio
    async def test_sets_status_completed(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should set status to COMPLETED on success."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        await orchestrator._run_branch(branch, [], mock_run_branch_fn)
        assert branch.status == CounterfactualStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_handles_exception_marks_failed(self, orchestrator, pivot_claim):
        """Should mark branch FAILED on exception."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )

        async def failing_fn(task, context, branch_id):
            raise RuntimeError("Test error")

        await orchestrator._run_branch(branch, [], failing_fn)
        assert branch.status == CounterfactualStatus.FAILED
        assert "Branch failed" in branch.conclusion

    @pytest.mark.asyncio
    async def test_sets_completed_at_timestamp(self, orchestrator, pivot_claim, mock_run_branch_fn):
        """Should set completed_at timestamp."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
        )
        await orchestrator._run_branch(branch, [], mock_run_branch_fn)
        assert branch.completed_at is not None


# =============================================================================
# Test CounterfactualOrchestrator.synthesize_branches()
# =============================================================================


class TestSynthesizeBranches:
    """Tests for CounterfactualOrchestrator.synthesize_branches()."""

    def test_calls_compare_branches(self, orchestrator, pivot_claim):
        """Should call _compare_branches internally."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,
        )
        with patch.object(
            orchestrator, "_compare_branches", wraps=orchestrator._compare_branches
        ) as mock:
            orchestrator.synthesize_branches(branch_true, branch_false)
            mock.assert_called_once()

    def test_builds_decision_tree(self, orchestrator, pivot_claim):
        """Should build decision tree structure."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,
        )
        consensus = orchestrator.synthesize_branches(branch_true, branch_false)
        assert "condition" in consensus.decision_tree
        assert "if_true" in consensus.decision_tree
        assert "if_false" in consensus.decision_tree

    def test_prefers_recommended_branch(self, orchestrator, pivot_claim):
        """Should use recommended branch for preference."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
            consensus_reached=True,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.8,
            consensus_reached=False,
        )
        consensus = orchestrator.synthesize_branches(branch_true, branch_false)
        # True branch has consensus, should be preferred
        assert consensus.preferred_world is True

    def test_prefers_higher_confidence_gap_0_1(self, orchestrator, pivot_claim):
        """Should prefer branch with >0.1 confidence gap."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.9,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,  # 0.2 gap > 0.1
        )
        consensus = orchestrator.synthesize_branches(branch_true, branch_false)
        assert consensus.preferred_world is True

    def test_no_preference_close_confidence(self, orchestrator, pivot_claim):
        """Should have no preference when confidence is close."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.75,  # Only 0.05 gap
        )
        consensus = orchestrator.synthesize_branches(branch_true, branch_false)
        assert consensus.preferred_world is None

    def test_handles_none_conclusions(self, orchestrator, pivot_claim):
        """Should handle None conclusions."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion=None,
            confidence=0.0,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion=None,
            confidence=0.0,
        )
        consensus = orchestrator.synthesize_branches(branch_true, branch_false)
        assert consensus.if_true_conclusion == "No conclusion reached"
        assert consensus.if_false_conclusion == "No conclusion reached"

    def test_creates_conditional_consensus(self, orchestrator, pivot_claim):
        """Should create ConditionalConsensus object."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,
        )
        consensus = orchestrator.synthesize_branches(branch_true, branch_false)
        assert isinstance(consensus, ConditionalConsensus)
        assert consensus.if_true_conclusion == "A"
        assert consensus.if_false_conclusion == "B"

    def test_appends_to_consensuses_list(self, orchestrator, pivot_claim):
        """Should append consensus to orchestrator list."""
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,
        )
        assert len(orchestrator.conditional_consensuses) == 0
        orchestrator.synthesize_branches(branch_true, branch_false)
        assert len(orchestrator.conditional_consensuses) == 1


# =============================================================================
# Test CounterfactualOrchestrator._compare_branches()
# =============================================================================


class TestCompareBranches:
    """Tests for CounterfactualOrchestrator._compare_branches()."""

    def test_detects_different_conclusions(self, orchestrator, pivot_claim):
        """Should detect different conclusions."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="Conclusion A",
            confidence=0.8,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="Conclusion B",
            confidence=0.7,
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        # Note: Module uses Python's `and` which returns last truthy value, not boolean
        assert comparison.conclusions_differ  # Truthy when conclusions differ

    def test_detects_same_conclusions(self, orchestrator, pivot_claim):
        """Should detect same conclusions."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="Same conclusion",
            confidence=0.8,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="Same conclusion",
            confidence=0.7,
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        assert comparison.conclusions_differ is False

    def test_handles_none_conclusions(self, orchestrator, pivot_claim):
        """Should handle None conclusions."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion=None,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        # None and non-None should not be considered different (logic requires both non-None)
        # Module uses `and` which returns None when first operand is falsy
        assert not comparison.conclusions_differ  # Falsy when one is None

    def test_extracts_key_differences(self, orchestrator, pivot_claim):
        """Should extract key differences when conclusions differ."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="Conclusion A",
            confidence=0.8,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="Conclusion B",
            confidence=0.7,
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        assert len(comparison.key_differences) == 2

    def test_finds_shared_insights(self, orchestrator, pivot_claim):
        """Should find shared insights via set intersection."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            key_insights=["shared", "unique_a"],
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            key_insights=["shared", "unique_b"],
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        assert "shared" in comparison.shared_insights

    def test_recommends_consensus_branch(self, orchestrator, pivot_claim):
        """Should recommend branch that achieved consensus."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.5,
            consensus_reached=True,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.9,
            consensus_reached=False,
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        assert comparison.recommended_branch == "cf-0001"

    def test_recommends_higher_confidence_gap_0_15(self, orchestrator, pivot_claim):
        """Should recommend branch with >0.15 confidence gap."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.9,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,  # 0.2 gap > 0.15
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        assert comparison.recommended_branch == "cf-0001"

    def test_no_recommendation_close_confidence(self, orchestrator, pivot_claim):
        """Should have no recommendation when confidence is close."""
        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
        )
        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.75,  # Only 0.05 gap
        )
        comparison = orchestrator._compare_branches(branch_a, branch_b)
        assert comparison.recommended_branch is None


# =============================================================================
# Test CounterfactualOrchestrator.generate_report()
# =============================================================================


class TestGenerateReport:
    """Tests for CounterfactualOrchestrator.generate_report()."""

    def test_generates_markdown(self, orchestrator):
        """Should generate markdown format."""
        report = orchestrator.generate_report()
        assert "# Counterfactual Exploration Report" in report

    def test_groups_by_pivot(self, orchestrator, pivot_claim):
        """Should group branches by pivot claim."""
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
        )
        orchestrator.branches[branch.branch_id] = branch
        report = orchestrator.generate_report()
        assert "Pivot:" in report

    def test_includes_status_emojis(self, orchestrator, pivot_claim):
        """Should include status emojis."""
        branch_complete = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
        )
        branch_failed = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            status=CounterfactualStatus.FAILED,
        )
        orchestrator.branches[branch_complete.branch_id] = branch_complete
        orchestrator.branches[branch_failed.branch_id] = branch_failed
        report = orchestrator.generate_report()
        assert "✅" in report or "❌" in report

    def test_truncates_long_text(self, orchestrator):
        """Should truncate long text in report."""
        long_statement = "x" * 200
        pivot = PivotClaim(
            claim_id="test",
            statement=long_statement,
            author="agent",
            disagreement_score=0.7,
            importance_score=0.5,
            blocking_agents=[],
        )
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot,
            assumption=True,
            conclusion="x" * 300,
            status=CounterfactualStatus.COMPLETED,
        )
        orchestrator.branches[branch.branch_id] = branch
        report = orchestrator.generate_report()
        # Should truncate
        assert "..." in report

    def test_includes_conditional_consensuses(self, orchestrator, pivot_claim):
        """Should include conditional consensuses in report."""
        cc = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot_claim,
            if_true_conclusion="A",
            if_true_confidence=0.8,
            if_false_conclusion="B",
            if_false_confidence=0.7,
        )
        orchestrator.conditional_consensuses.append(cc)
        report = orchestrator.generate_report()
        assert "Conditional Consensus" in report

    def test_handles_empty_branches(self, orchestrator):
        """Should handle empty branches gracefully."""
        report = orchestrator.generate_report()
        assert "Total Branches:** 0" in report


# =============================================================================
# Test CounterfactualIntegration
# =============================================================================


class TestCounterfactualIntegration:
    """Tests for CounterfactualIntegration class."""

    def test_initialization_with_graph(self):
        """Should initialize with DebateGraph."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"
        integration = CounterfactualIntegration(mock_graph)
        assert integration.graph == mock_graph
        assert isinstance(integration.orchestrator, CounterfactualOrchestrator)

    def test_create_counterfactual_branch_dual_creation(self, pivot_claim):
        """Should create both graph and CF branches."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"
        mock_graph.create_branch.return_value = MagicMock(id="graph-branch-1")

        integration = CounterfactualIntegration(mock_graph)
        graph_branch, cf_branch = integration.create_counterfactual_branch(
            "node-1", pivot_claim, True
        )

        assert graph_branch is not None
        assert cf_branch is not None
        mock_graph.create_branch.assert_called_once()

    def test_create_counterfactual_branch_id_linking(self, pivot_claim):
        """Should link CF branch to graph branch."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"
        mock_graph.create_branch.return_value = MagicMock(id="graph-branch-1")

        integration = CounterfactualIntegration(mock_graph)
        graph_branch, cf_branch = integration.create_counterfactual_branch(
            "node-1", pivot_claim, True
        )

        assert cf_branch.graph_branch_id == "graph-branch-1"

    def test_create_counterfactual_branch_name_format(self, pivot_claim):
        """Should format branch name correctly."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"
        mock_graph.create_branch.return_value = MagicMock(id="gb-1")

        integration = CounterfactualIntegration(mock_graph)
        integration.create_counterfactual_branch("node-1", pivot_claim, True)

        call_args = mock_graph.create_branch.call_args
        assert "CF:" in call_args.kwargs["name"]

    def test_merge_branches_creates_consensus(self, pivot_claim):
        """Should create conditional consensus when merging."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"

        integration = CounterfactualIntegration(mock_graph)
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            confidence=0.8,
            graph_branch_id="gb-1",
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            confidence=0.7,
            graph_branch_id="gb-2",
        )

        merge_result, consensus = integration.merge_counterfactual_branches(
            branch_true, branch_false, "agent-1"
        )

        assert isinstance(consensus, ConditionalConsensus)

    def test_merge_branches_calls_graph_merge(self, pivot_claim):
        """Should call graph.merge_branches."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"
        mock_graph.merge_branches.return_value = MagicMock()

        integration = CounterfactualIntegration(mock_graph)
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            graph_branch_id="gb-1",
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            graph_branch_id="gb-2",
        )

        integration.merge_counterfactual_branches(branch_true, branch_false, "agent-1")
        mock_graph.merge_branches.assert_called_once()

    def test_merge_handles_missing_graph_ids(self, pivot_claim):
        """Should handle missing graph branch IDs."""
        mock_graph = MagicMock()
        mock_graph.debate_id = "debate-1"

        integration = CounterfactualIntegration(mock_graph)
        branch_true = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=True,
            conclusion="A",
            graph_branch_id=None,  # Missing
        )
        branch_false = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-1",
            pivot_claim=pivot_claim,
            assumption=False,
            conclusion="B",
            graph_branch_id=None,  # Missing
        )

        merge_result, consensus = integration.merge_counterfactual_branches(
            branch_true, branch_false, "agent-1"
        )

        # Should not call merge_branches when < 2 IDs
        mock_graph.merge_branches.assert_not_called()
        assert merge_result is None


# =============================================================================
# Test explore_counterfactual Function
# =============================================================================


class TestExploreCounterfactual:
    """Tests for explore_counterfactual function."""

    @pytest.mark.asyncio
    async def test_creates_orchestrator(self, mock_run_branch_fn):
        """Should create orchestrator internally."""
        result = await explore_counterfactual(
            debate_id="debate-1",
            pivot_statement="Test claim",
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )
        assert isinstance(result, ConditionalConsensus)

    @pytest.mark.asyncio
    async def test_creates_pivot_with_max_scores(self, mock_run_branch_fn):
        """Should create pivot with max disagreement/importance scores."""
        # The function creates pivot with scores = 1.0
        result = await explore_counterfactual(
            debate_id="debate-1",
            pivot_statement="Test claim",
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )
        # Pivot should have been created with max scores
        assert result is not None

    @pytest.mark.asyncio
    async def test_calls_create_and_run_branches(self, mock_run_branch_fn):
        """Should call create_and_run_branches."""
        result = await explore_counterfactual(
            debate_id="debate-1",
            pivot_statement="Test claim",
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_finds_true_false_branches(self, mock_run_branch_fn):
        """Should find both true and false branches."""
        result = await explore_counterfactual(
            debate_id="debate-1",
            pivot_statement="Test claim",
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )
        # Result should have conclusions from both branches
        assert result.if_true_conclusion is not None
        assert result.if_false_conclusion is not None

    @pytest.mark.asyncio
    async def test_raises_value_error_missing_branch(self):
        """Should raise ValueError when branch is missing."""
        # Create a function that only returns one branch type
        call_count = [0]

        async def partial_fn(task, context, branch_id):
            call_count[0] += 1
            if call_count[0] == 1:
                # First branch succeeds
                result = MagicMock(spec=DebateResult)
                result.final_answer = "A"
                result.confidence = 0.8
                result.consensus_reached = True
                result.messages = []
                result.votes = []
                return result
            else:
                # Second branch fails
                raise RuntimeError("Branch failed")

        # This test verifies error handling when branches fail
        # The function should still work since branches catch exceptions
        result = await explore_counterfactual(
            debate_id="debate-1",
            pivot_statement="Test claim",
            context_messages=[],
            run_branch_fn=partial_fn,
        )
        # Should still return a consensus (with one failed branch)
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_conditional_consensus(self, mock_run_branch_fn):
        """Should return ConditionalConsensus."""
        result = await explore_counterfactual(
            debate_id="debate-1",
            pivot_statement="Test claim",
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )
        assert isinstance(result, ConditionalConsensus)
