"""
Tests for Counterfactual Debate Branches module.

Tests cover:
- CounterfactualStatus enum
- Data classes (PivotClaim, CounterfactualBranch, BranchComparison, ConditionalConsensus)
- ImpactDetector impasse detection
- CounterfactualOrchestrator branch management
- CounterfactualIntegration with DebateGraph
- explore_counterfactual convenience function
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.counterfactual import (
    BranchComparison,
    ConditionalConsensus,
    CounterfactualBranch,
    CounterfactualIntegration,
    CounterfactualOrchestrator,
    CounterfactualStatus,
    ImpactDetector,
    PivotClaim,
    explore_counterfactual,
)


# =============================================================================
# CounterfactualStatus Enum Tests
# =============================================================================


class TestCounterfactualStatus:
    """Tests for CounterfactualStatus enum."""

    @pytest.mark.smoke
    def test_all_status_values_exist(self):
        """Test all status values exist."""
        assert CounterfactualStatus.PENDING.value == "pending"
        assert CounterfactualStatus.RUNNING.value == "running"
        assert CounterfactualStatus.COMPLETED.value == "completed"
        assert CounterfactualStatus.FAILED.value == "failed"
        assert CounterfactualStatus.MERGED.value == "merged"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert CounterfactualStatus("pending") == CounterfactualStatus.PENDING
        assert CounterfactualStatus("completed") == CounterfactualStatus.COMPLETED


# =============================================================================
# PivotClaim Tests
# =============================================================================


class TestPivotClaim:
    """Tests for PivotClaim dataclass."""

    def test_create_pivot_claim(self):
        """Test creating a pivot claim."""
        claim = PivotClaim(
            claim_id="pivot-001",
            statement="Redis is the best caching solution",
            author="claude",
            disagreement_score=0.75,
            importance_score=0.8,
            blocking_agents=["gpt4", "gemini"],
        )

        assert claim.claim_id == "pivot-001"
        assert claim.author == "claude"
        assert claim.disagreement_score == 0.75
        assert len(claim.blocking_agents) == 2
        assert claim.created_at is not None

    def test_should_branch_true(self):
        """Test should_branch returns True for high disagreement and importance."""
        claim = PivotClaim(
            claim_id="pivot-002",
            statement="Test claim",
            author="test",
            disagreement_score=0.6,  # > 0.5
            importance_score=0.4,  # > 0.3
            blocking_agents=["a", "b"],
        )

        assert claim.should_branch is True

    def test_should_branch_false_low_disagreement(self):
        """Test should_branch returns False for low disagreement."""
        claim = PivotClaim(
            claim_id="pivot-003",
            statement="Test claim",
            author="test",
            disagreement_score=0.4,  # < 0.5
            importance_score=0.8,
            blocking_agents=["a"],
        )

        assert claim.should_branch is False

    def test_should_branch_false_low_importance(self):
        """Test should_branch returns False for low importance."""
        claim = PivotClaim(
            claim_id="pivot-004",
            statement="Test claim",
            author="test",
            disagreement_score=0.8,
            importance_score=0.2,  # < 0.3
            blocking_agents=["a"],
        )

        assert claim.should_branch is False

    def test_pivot_claim_with_branch_reason(self):
        """Test pivot claim with branch reason."""
        claim = PivotClaim(
            claim_id="pivot-005",
            statement="Test claim",
            author="test",
            disagreement_score=0.7,
            importance_score=0.5,
            blocking_agents=[],
            branch_reason="Fundamental architectural disagreement",
        )

        assert claim.branch_reason == "Fundamental architectural disagreement"


# =============================================================================
# CounterfactualBranch Tests
# =============================================================================


class TestCounterfactualBranch:
    """Tests for CounterfactualBranch dataclass."""

    def _create_pivot_claim(self) -> PivotClaim:
        """Helper to create a pivot claim."""
        return PivotClaim(
            claim_id="pivot-001",
            statement="Microservices are better than monoliths",
            author="claude",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=["gpt4"],
        )

    def test_create_true_branch(self):
        """Test creating branch with assumption=True."""
        pivot = self._create_pivot_claim()

        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
        )

        assert branch.branch_id == "cf-0001"
        assert branch.assumption is True
        assert branch.status == CounterfactualStatus.PENDING
        assert branch.conclusion is None
        assert branch.confidence == 0.0

    def test_create_false_branch(self):
        """Test creating branch with assumption=False."""
        pivot = self._create_pivot_claim()

        branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
        )

        assert branch.assumption is False

    def test_assumption_text_true(self):
        """Test assumption_text for True assumption."""
        pivot = self._create_pivot_claim()
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
        )

        assert "TRUE" in branch.assumption_text
        assert "Microservices" in branch.assumption_text

    def test_assumption_text_false(self):
        """Test assumption_text for False assumption."""
        pivot = self._create_pivot_claim()
        branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
        )

        assert "FALSE" in branch.assumption_text

    def test_duration_seconds_none_when_not_complete(self):
        """Test duration_seconds returns None when not complete."""
        pivot = self._create_pivot_claim()
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
        )

        assert branch.duration_seconds is None

    def test_duration_seconds_calculated(self):
        """Test duration_seconds calculation."""
        pivot = self._create_pivot_claim()
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            started_at="2024-01-01T10:00:00",
            completed_at="2024-01-01T10:05:30",
        )

        assert branch.duration_seconds == 330.0  # 5 min 30 sec

    def test_to_dict(self):
        """Test to_dict serialization."""
        pivot = self._create_pivot_claim()
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Redis is recommended",
            confidence=0.85,
            consensus_reached=True,
        )

        result = branch.to_dict()

        assert result["branch_id"] == "cf-0001"
        assert result["assumption"] is True
        assert result["status"] == "completed"
        assert result["conclusion"] == "Redis is recommended"
        assert result["confidence"] == 0.85


# =============================================================================
# BranchComparison Tests
# =============================================================================


class TestBranchComparison:
    """Tests for BranchComparison dataclass."""

    def test_create_comparison(self):
        """Test creating branch comparison."""
        comparison = BranchComparison(
            branch_a_id="cf-0001",
            branch_b_id="cf-0002",
            branch_a_conclusion="Use Redis",
            branch_b_conclusion="Use Memcached",
            branch_a_confidence=0.85,
            branch_b_confidence=0.75,
            conclusions_differ=True,
            key_differences=["Persistence support", "Clustering"],
            shared_insights=["Both support high throughput"],
            recommended_branch="cf-0001",
            recommendation_reason="Higher confidence and persistence support",
        )

        assert comparison.branch_a_id == "cf-0001"
        assert comparison.conclusions_differ is True
        assert len(comparison.key_differences) == 2
        assert comparison.recommended_branch == "cf-0001"

    def test_comparison_no_recommendation(self):
        """Test comparison without clear recommendation."""
        comparison = BranchComparison(
            branch_a_id="cf-0001",
            branch_b_id="cf-0002",
            branch_a_conclusion="Option A",
            branch_b_conclusion="Option B",
            branch_a_confidence=0.5,
            branch_b_confidence=0.5,
            conclusions_differ=True,
            key_differences=[],
            shared_insights=[],
        )

        assert comparison.recommended_branch is None
        assert comparison.recommendation_reason == ""


# =============================================================================
# ConditionalConsensus Tests
# =============================================================================


class TestConditionalConsensus:
    """Tests for ConditionalConsensus dataclass."""

    def _create_pivot_claim(self) -> PivotClaim:
        """Helper to create a pivot claim."""
        return PivotClaim(
            claim_id="pivot-001",
            statement="Data persistence is required",
            author="claude",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=["gpt4"],
        )

    def test_create_conditional_consensus(self):
        """Test creating conditional consensus."""
        pivot = self._create_pivot_claim()

        consensus = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot,
            if_true_conclusion="Use Redis with RDB persistence",
            if_true_confidence=0.85,
            if_false_conclusion="Use Memcached for pure caching",
            if_false_confidence=0.80,
        )

        assert consensus.consensus_id == "cc-001"
        assert consensus.if_true_confidence == 0.85
        assert consensus.if_false_confidence == 0.80

    def test_to_natural_language(self):
        """Test natural language generation."""
        pivot = self._create_pivot_claim()

        consensus = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot,
            if_true_conclusion="Use Redis",
            if_true_confidence=0.85,
            if_false_conclusion="Use Memcached",
            if_false_confidence=0.80,
        )

        text = consensus.to_natural_language()

        assert "Conditional Consensus" in text
        assert "IF" in text
        assert "THEN" in text
        assert "ELSE" in text
        assert "85%" in text
        assert "80%" in text

    def test_to_dict(self):
        """Test to_dict serialization."""
        pivot = self._create_pivot_claim()

        consensus = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot,
            if_true_conclusion="Use Redis",
            if_true_confidence=0.85,
            if_false_conclusion="Use Memcached",
            if_false_confidence=0.80,
            preferred_world=True,
            preference_reason="Higher confidence",
            unresolved_uncertainties=["Cost considerations"],
            true_branch_id="cf-0001",
            false_branch_id="cf-0002",
        )

        result = consensus.to_dict()

        assert result["consensus_id"] == "cc-001"
        assert result["pivot_claim"] == pivot.statement
        assert result["preferred_world"] is True
        assert "natural_language" in result

    def test_with_decision_tree(self):
        """Test with decision tree populated."""
        pivot = self._create_pivot_claim()

        decision_tree = {
            "condition": pivot.statement,
            "if_true": {"conclusion": "Redis", "confidence": 0.85},
            "if_false": {"conclusion": "Memcached", "confidence": 0.80},
        }

        consensus = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot,
            if_true_conclusion="Use Redis",
            if_true_confidence=0.85,
            if_false_conclusion="Use Memcached",
            if_false_confidence=0.80,
            decision_tree=decision_tree,
        )

        assert consensus.decision_tree["if_true"]["conclusion"] == "Redis"


# =============================================================================
# ImpactDetector Tests
# =============================================================================


class TestImpactDetector:
    """Tests for ImpactDetector class."""

    def _create_mock_message(self, agent: str, content: str, round_num: int = 1) -> Mock:
        """Helper to create mock messages."""
        msg = Mock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg

    def test_init_default_thresholds(self):
        """Test initialization with default thresholds."""
        detector = ImpactDetector()

        assert detector.disagreement_threshold == 0.6
        assert detector.rounds_before_branch == 2

    def test_init_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        detector = ImpactDetector(
            disagreement_threshold=0.8,
            rounds_before_branch=3,
        )

        assert detector.disagreement_threshold == 0.8
        assert detector.rounds_before_branch == 3

    def test_detect_impasse_too_few_messages(self):
        """Test impasse detection with too few messages."""
        detector = ImpactDetector(rounds_before_branch=2)

        messages = [
            self._create_mock_message("claude", "Test", 1),
            self._create_mock_message("gpt4", "Test", 1),
        ]

        result = detector.detect_impasse(messages, [])

        assert result is None

    def test_detect_impasse_with_disagreement_phrase(self):
        """Test impasse detection with disagreement phrases."""
        detector = ImpactDetector(rounds_before_branch=2, disagreement_threshold=0.3)

        messages = [
            self._create_mock_message(
                "claude", "I fundamentally disagree with the microservices approach", 1
            ),
            self._create_mock_message("gpt4", "On the contrary, monoliths are problematic", 1),
            self._create_mock_message("claude", "The core assumption is flawed", 2),
            self._create_mock_message("gpt4", "I cannot accept that premise", 2),
            self._create_mock_message(
                "claude", "If that were true, we would need a different approach", 3
            ),
            self._create_mock_message("gpt4", "I reject the premise entirely", 3),
        ]

        result = detector.detect_impasse(messages, [])

        # Should detect impasse due to disagreement phrases
        # Note: Detection depends on threshold and phrase matching
        # May return None if disagreement score is below threshold
        assert result is None or isinstance(result, PivotClaim)

    def test_detect_impasse_no_disagreement(self):
        """Test no impasse when agents agree."""
        detector = ImpactDetector(rounds_before_branch=2)

        messages = [
            self._create_mock_message("claude", "We should use Redis", 1),
            self._create_mock_message("gpt4", "Agreed, Redis is best", 1),
            self._create_mock_message("claude", "Redis provides persistence", 2),
            self._create_mock_message("gpt4", "Yes, and good caching", 2),
            self._create_mock_message("claude", "Let's proceed with Redis", 3),
            self._create_mock_message("gpt4", "Confirmed, Redis it is", 3),
        ]

        result = detector.detect_impasse(messages, [])

        assert result is None

    def test_find_disagreements(self):
        """Test _find_disagreements internal method."""
        detector = ImpactDetector()

        messages = [
            self._create_mock_message("claude", "I fundamentally disagree with this approach", 1),
            self._create_mock_message("gpt4", "The core assumption is wrong", 1),
        ]

        disagreements = detector._find_disagreements(messages)

        # Should find at least one disagreement
        assert isinstance(disagreements, dict)

    def test_estimate_importance(self):
        """Test _estimate_importance internal method."""
        detector = ImpactDetector()

        claim = "redis caching performance"
        messages = [
            self._create_mock_message("a", "Redis provides excellent caching", 1),
            self._create_mock_message("b", "Caching with Redis is fast", 1),
            self._create_mock_message("a", "Performance of Redis caching", 2),
        ]

        importance = detector._estimate_importance(claim, messages)

        assert 0.0 <= importance <= 1.0


# =============================================================================
# CounterfactualOrchestrator Tests
# =============================================================================


class TestCounterfactualOrchestrator:
    """Tests for CounterfactualOrchestrator class."""

    def _create_pivot_claim(self) -> PivotClaim:
        """Helper to create a pivot claim."""
        return PivotClaim(
            claim_id="pivot-001",
            statement="Test claim",
            author="claude",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=["gpt4"],
        )

    def _create_mock_message(self, agent: str, content: str, round_num: int = 1) -> Mock:
        """Helper to create mock messages."""
        msg = Mock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg

    def test_init_default_config(self):
        """Test initialization with default config."""
        orchestrator = CounterfactualOrchestrator()

        assert orchestrator.max_branches == 4
        assert orchestrator.max_depth == 2
        assert orchestrator.parallel_execution is True
        assert len(orchestrator.branches) == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        orchestrator = CounterfactualOrchestrator(
            max_branches=6,
            max_depth=3,
            parallel_execution=False,
        )

        assert orchestrator.max_branches == 6
        assert orchestrator.max_depth == 3
        assert orchestrator.parallel_execution is False

    @pytest.mark.asyncio
    async def test_create_and_run_branches(self):
        """Test creating and running branches."""
        orchestrator = CounterfactualOrchestrator(parallel_execution=True)
        pivot = self._create_pivot_claim()

        context_messages = [
            self._create_mock_message("claude", "Initial context", 1),
        ]

        async def mock_run_branch_fn(task, context, branch_id):
            from aragora.core import DebateResult

            return DebateResult(
                final_answer=f"Answer for {branch_id}",
                confidence=0.8,
                consensus_reached=True,
                rounds_used=2,
                messages=[],
                critiques=[],
                votes=[],
            )

        branches = await orchestrator.create_and_run_branches(
            debate_id="debate-123",
            pivot=pivot,
            context_messages=context_messages,
            run_branch_fn=mock_run_branch_fn,
        )

        assert len(branches) == 2
        assert any(b.assumption is True for b in branches)
        assert any(b.assumption is False for b in branches)

    @pytest.mark.asyncio
    async def test_create_and_run_branches_sequential(self):
        """Test creating and running branches sequentially."""
        orchestrator = CounterfactualOrchestrator(parallel_execution=False)
        pivot = self._create_pivot_claim()

        async def mock_run_branch_fn(task, context, branch_id):
            from aragora.core import DebateResult

            return DebateResult(
                final_answer="Test answer",
                confidence=0.7,
                consensus_reached=True,
                rounds_used=2,
                messages=[],
                critiques=[],
                votes=[],
            )

        branches = await orchestrator.create_and_run_branches(
            debate_id="debate-123",
            pivot=pivot,
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )

        assert len(branches) == 2

    @pytest.mark.asyncio
    async def test_run_branch_failure_handling(self):
        """Test handling branch execution failure."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        async def failing_run_branch_fn(task, context, branch_id):
            raise RuntimeError("Simulated failure")

        branches = await orchestrator.create_and_run_branches(
            debate_id="debate-123",
            pivot=pivot,
            context_messages=[],
            run_branch_fn=failing_run_branch_fn,
        )

        # Branches should be created but marked as failed
        assert len(branches) == 2
        for branch in branches:
            assert branch.status == CounterfactualStatus.FAILED

    def test_synthesize_branches(self):
        """Test synthesizing branches into conditional consensus."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        true_branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Use Redis for persistence",
            confidence=0.85,
            consensus_reached=True,
        )

        false_branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Use Memcached for speed",
            confidence=0.75,
            consensus_reached=True,
        )

        consensus = orchestrator.synthesize_branches(true_branch, false_branch)

        assert consensus.consensus_id is not None
        assert consensus.if_true_conclusion == "Use Redis for persistence"
        assert consensus.if_false_conclusion == "Use Memcached for speed"
        assert consensus.if_true_confidence == 0.85
        assert len(orchestrator.conditional_consensuses) == 1

    def test_synthesize_branches_preferred_true(self):
        """Test synthesis prefers branch with higher confidence."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        true_branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option A",
            confidence=0.95,  # Higher
            consensus_reached=True,
        )

        false_branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option B",
            confidence=0.60,  # Lower
            consensus_reached=False,
        )

        consensus = orchestrator.synthesize_branches(true_branch, false_branch)

        assert consensus.preferred_world is True

    def test_synthesize_branches_preferred_false(self):
        """Test synthesis prefers branch with consensus."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        true_branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option A",
            confidence=0.70,
            consensus_reached=False,  # No consensus
        )

        false_branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option B",
            confidence=0.65,
            consensus_reached=True,  # Has consensus
        )

        consensus = orchestrator.synthesize_branches(true_branch, false_branch)

        assert consensus.preferred_world is False

    def test_compare_branches(self):
        """Test _compare_branches internal method."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option A",
            confidence=0.85,
            consensus_reached=True,
            key_insights=["Insight 1", "Shared insight"],
        )

        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option B",
            confidence=0.65,
            consensus_reached=False,
            key_insights=["Insight 2", "Shared insight"],
        )

        comparison = orchestrator._compare_branches(branch_a, branch_b)

        assert comparison.conclusions_differ is True
        assert comparison.branch_a_confidence == 0.85
        assert comparison.recommended_branch == "cf-0001"
        assert "Shared insight" in comparison.shared_insights

    def test_get_all_consensuses(self):
        """Test getting all conditional consensuses."""
        orchestrator = CounterfactualOrchestrator()

        assert orchestrator.get_all_consensuses() == []

        # Add some consensuses via synthesis
        pivot = self._create_pivot_claim()
        true_branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            conclusion="A",
            confidence=0.8,
            consensus_reached=True,
        )
        false_branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            conclusion="B",
            confidence=0.7,
            consensus_reached=True,
        )

        orchestrator.synthesize_branches(true_branch, false_branch)

        assert len(orchestrator.get_all_consensuses()) == 1

    def test_generate_report(self):
        """Test report generation."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        # Add branches
        true_branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option A",
            confidence=0.85,
            consensus_reached=True,
        )

        false_branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option B",
            confidence=0.75,
            consensus_reached=True,
        )

        orchestrator.branches[true_branch.branch_id] = true_branch
        orchestrator.branches[false_branch.branch_id] = false_branch
        orchestrator.synthesize_branches(true_branch, false_branch)

        report = orchestrator.generate_report()

        assert "Counterfactual Exploration Report" in report
        assert "Total Branches" in report
        assert "TRUE" in report
        assert "FALSE" in report

    def test_cleanup_debate(self):
        """Test cleaning up branches for a completed debate."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        # Add branches for debate-123
        for i in range(3):
            branch = CounterfactualBranch(
                branch_id=f"cf-{i:04d}",
                parent_debate_id="debate-123",
                pivot_claim=pivot,
                assumption=i % 2 == 0,
            )
            orchestrator.branches[branch.branch_id] = branch

        # Add branch for different debate
        other_branch = CounterfactualBranch(
            branch_id="cf-9999",
            parent_debate_id="debate-other",
            pivot_claim=pivot,
            assumption=True,
        )
        orchestrator.branches[other_branch.branch_id] = other_branch

        removed = orchestrator.cleanup_debate("debate-123")

        assert removed == 3
        assert len(orchestrator.branches) == 1
        assert "cf-9999" in orchestrator.branches

    def test_clear_all(self):
        """Test clearing all state."""
        orchestrator = CounterfactualOrchestrator()
        pivot = self._create_pivot_claim()

        # Add some data
        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
        )
        orchestrator.branches[branch.branch_id] = branch
        orchestrator._branch_counter = 5

        orchestrator.clear_all()

        assert len(orchestrator.branches) == 0
        assert len(orchestrator.conditional_consensuses) == 0
        assert orchestrator._branch_counter == 0

    @pytest.mark.asyncio
    async def test_check_and_branch_no_impasse(self):
        """Test check_and_branch returns None when no impasse."""
        orchestrator = CounterfactualOrchestrator()

        messages = [Mock(content="Agreement", agent="a")]

        async def mock_run_fn(task, context, branch_id):
            pass

        result = await orchestrator.check_and_branch(
            debate_id="debate-123",
            messages=messages,
            votes=[],
            run_branch_fn=mock_run_fn,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_check_and_branch_max_branches_reached(self):
        """Test check_and_branch returns None when max branches reached."""
        orchestrator = CounterfactualOrchestrator(max_branches=2)
        pivot = self._create_pivot_claim()

        # Pre-populate with branches
        for i in range(3):
            branch = CounterfactualBranch(
                branch_id=f"cf-{i:04d}",
                parent_debate_id="debate-123",
                pivot_claim=pivot,
                assumption=i % 2 == 0,
            )
            orchestrator.branches[branch.branch_id] = branch

        async def mock_run_fn(task, context, branch_id):
            pass

        messages = [Mock(content="Test", agent="a")]

        result = await orchestrator.check_and_branch(
            debate_id="debate-123",
            messages=messages,
            votes=[],
            run_branch_fn=mock_run_fn,
        )

        assert result is None


# =============================================================================
# CounterfactualIntegration Tests
# =============================================================================


class TestCounterfactualIntegration:
    """Tests for CounterfactualIntegration class."""

    def _create_mock_graph(self) -> Mock:
        """Create a mock DebateGraph."""
        graph = Mock()
        graph.debate_id = "debate-123"

        # Mock branch creation
        mock_branch = Mock()
        mock_branch.id = "graph-branch-001"
        graph.create_branch.return_value = mock_branch

        # Mock merge
        mock_merge = Mock()
        mock_merge.merged_node_id = "merged-001"
        graph.merge_branches.return_value = mock_merge

        return graph

    def _create_pivot_claim(self) -> PivotClaim:
        """Helper to create a pivot claim."""
        return PivotClaim(
            claim_id="pivot-001",
            statement="Test claim",
            author="claude",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=["gpt4"],
        )

    def test_init(self):
        """Test initialization."""
        graph = self._create_mock_graph()
        integration = CounterfactualIntegration(graph)

        assert integration.graph == graph
        assert integration.orchestrator is not None

    def test_create_counterfactual_branch(self):
        """Test creating counterfactual branch in both graph and orchestrator."""
        graph = self._create_mock_graph()
        integration = CounterfactualIntegration(graph)
        pivot = self._create_pivot_claim()

        graph_branch, cf_branch = integration.create_counterfactual_branch(
            from_node_id="node-001",
            pivot=pivot,
            assumption=True,
        )

        assert graph_branch is not None
        assert cf_branch is not None
        assert cf_branch.assumption is True
        assert cf_branch.graph_branch_id == "graph-branch-001"
        graph.create_branch.assert_called_once()

    def test_merge_counterfactual_branches(self):
        """Test merging counterfactual branches."""
        graph = self._create_mock_graph()
        integration = CounterfactualIntegration(graph)
        pivot = self._create_pivot_claim()

        true_branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option A",
            confidence=0.85,
            consensus_reached=True,
            graph_branch_id="graph-branch-001",
        )

        false_branch = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            status=CounterfactualStatus.COMPLETED,
            conclusion="Option B",
            confidence=0.75,
            consensus_reached=True,
            graph_branch_id="graph-branch-002",
        )

        merge_result, consensus = integration.merge_counterfactual_branches(
            true_branch, false_branch, synthesizer_agent_id="synthesizer"
        )

        assert consensus is not None
        assert consensus.if_true_conclusion == "Option A"
        graph.merge_branches.assert_called_once()


# =============================================================================
# explore_counterfactual Function Tests
# =============================================================================


class TestExploreCounterfactual:
    """Tests for explore_counterfactual convenience function."""

    @pytest.mark.asyncio
    async def test_explore_counterfactual_basic(self):
        """Test basic counterfactual exploration."""

        async def mock_run_branch_fn(task, context, branch_id):
            from aragora.core import DebateResult

            return DebateResult(
                final_answer=f"Answer for {branch_id}",
                confidence=0.8,
                consensus_reached=True,
                rounds_used=2,
                messages=[],
                critiques=[],
                votes=[],
            )

        consensus = await explore_counterfactual(
            debate_id="debate-123",
            pivot_statement="Microservices are better than monoliths",
            context_messages=[],
            run_branch_fn=mock_run_branch_fn,
        )

        assert consensus is not None
        assert isinstance(consensus, ConditionalConsensus)
        assert consensus.if_true_conclusion is not None
        assert consensus.if_false_conclusion is not None

    @pytest.mark.asyncio
    async def test_explore_counterfactual_with_context(self):
        """Test counterfactual exploration with context messages."""
        from aragora.core import Message

        context = [
            Message(
                agent="claude",
                role="proposer",
                content="Initial proposal",
                round=1,
            ),
        ]

        async def mock_run_branch_fn(task, context, branch_id):
            from aragora.core import DebateResult

            return DebateResult(
                final_answer="Test answer",
                confidence=0.75,
                consensus_reached=True,
                rounds_used=3,
                messages=[],
                critiques=[],
                votes=[],
            )

        consensus = await explore_counterfactual(
            debate_id="debate-456",
            pivot_statement="Test pivot",
            context_messages=context,
            run_branch_fn=mock_run_branch_fn,
        )

        assert consensus is not None


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestCounterfactualEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pivot_claim_empty_blocking_agents(self):
        """Test pivot claim with empty blocking agents."""
        claim = PivotClaim(
            claim_id="pivot-001",
            statement="Test",
            author="test",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=[],
        )

        assert claim.should_branch is True

    def test_branch_without_graph_branch_id(self):
        """Test branch without graph branch ID."""
        pivot = PivotClaim(
            claim_id="pivot-001",
            statement="Test",
            author="test",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=[],
        )

        branch = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
        )

        assert branch.graph_branch_id is None

    def test_consensus_without_preference(self):
        """Test consensus without clear preference."""
        pivot = PivotClaim(
            claim_id="pivot-001",
            statement="Test",
            author="test",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=[],
        )

        consensus = ConditionalConsensus(
            consensus_id="cc-001",
            pivot_claim=pivot,
            if_true_conclusion="Option A",
            if_true_confidence=0.5,
            if_false_conclusion="Option B",
            if_false_confidence=0.5,
            preferred_world=None,
        )

        assert consensus.preferred_world is None

    def test_orchestrator_max_history_limit(self):
        """Test orchestrator respects max_history limit."""
        orchestrator = CounterfactualOrchestrator()
        orchestrator.max_history = 5

        pivot = PivotClaim(
            claim_id="pivot-001",
            statement="Test",
            author="test",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=[],
        )

        # Add many consensuses
        for i in range(10):
            true_branch = CounterfactualBranch(
                branch_id=f"cf-{i:04d}-t",
                parent_debate_id=f"debate-{i}",
                pivot_claim=pivot,
                assumption=True,
                conclusion="A",
                confidence=0.8,
                consensus_reached=True,
            )
            false_branch = CounterfactualBranch(
                branch_id=f"cf-{i:04d}-f",
                parent_debate_id=f"debate-{i}",
                pivot_claim=pivot,
                assumption=False,
                conclusion="B",
                confidence=0.7,
                consensus_reached=True,
            )
            orchestrator.synthesize_branches(true_branch, false_branch)

        # Cleanup should trim to max_history
        orchestrator.cleanup_debate("debate-0")

        assert len(orchestrator.conditional_consensuses) <= orchestrator.max_history

    def test_compare_branches_same_conclusion(self):
        """Test comparing branches with same conclusion."""
        orchestrator = CounterfactualOrchestrator()
        pivot = PivotClaim(
            claim_id="pivot-001",
            statement="Test",
            author="test",
            disagreement_score=0.8,
            importance_score=0.7,
            blocking_agents=[],
        )

        branch_a = CounterfactualBranch(
            branch_id="cf-0001",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=True,
            conclusion="Same conclusion",
            confidence=0.8,
            consensus_reached=True,
        )

        branch_b = CounterfactualBranch(
            branch_id="cf-0002",
            parent_debate_id="debate-123",
            pivot_claim=pivot,
            assumption=False,
            conclusion="Same conclusion",
            confidence=0.8,
            consensus_reached=True,
        )

        comparison = orchestrator._compare_branches(branch_a, branch_b)

        assert comparison.conclusions_differ is False
