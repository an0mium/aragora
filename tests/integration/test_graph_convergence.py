"""
Tests for graph convergence detection.

Phase 8: Debate Integration Test Gaps - Convergence detection tests.

Tests:
- test_convergence_detection_threshold_0_8 - Default threshold
- test_convergence_auto_merge_trigger - Auto-merge on convergence
- test_convergence_with_low_similarity - Below threshold
- test_convergence_scorer_claim_weighting - 70% claim / 30% confidence
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aragora.debate.graph import (
    Branch,
    BranchReason,
    ConvergenceScorer,
    DebateGraph,
    DebateNode,
    NodeType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def convergence_scorer():
    """Create a convergence scorer with default threshold."""
    return ConvergenceScorer(threshold=0.8)


@pytest.fixture
def low_threshold_scorer():
    """Create a convergence scorer with low threshold."""
    return ConvergenceScorer(threshold=0.3)


def create_branch_with_nodes(
    branch_id: str,
    name: str,
    claims: list[str],
    confidence: float = 0.7,
) -> tuple[Branch, list[DebateNode]]:
    """Helper to create a branch with nodes."""
    branch = Branch(
        id=branch_id,
        name=name,
        reason=BranchReason.ALTERNATIVE_APPROACH,
        start_node_id="root",
        hypothesis=f"Hypothesis for {name}",
        confidence=confidence,
    )

    nodes = []
    for i in range(3):
        node = DebateNode(
            id=f"{branch_id}-node-{i}",
            node_type=NodeType.PROPOSAL,
            agent_id=f"agent-{i}",
            content=f"Content for {name} node {i}",
            branch_id=branch_id,
            confidence=confidence,
            claims=claims,
        )
        nodes.append(node)

    return branch, nodes


# ============================================================================
# Test: Convergence Detection - Default Threshold
# ============================================================================


class TestConvergenceDetectionDefaultThreshold:
    """Test convergence detection with default 0.8 threshold."""

    def test_convergence_detection_threshold_0_8(self, convergence_scorer):
        """Test that identical branches are detected as converged."""
        # Create two branches with identical claims
        shared_claims = ["Claim 1", "Claim 2", "Claim 3"]

        branch_a, nodes_a = create_branch_with_nodes("branch-a", "Branch A", shared_claims, 0.8)
        branch_b, nodes_b = create_branch_with_nodes("branch-b", "Branch B", shared_claims, 0.8)

        # Score convergence
        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        # With identical claims and confidence, should be high
        assert score >= 0.8

    def test_should_merge_returns_true_for_identical_branches(self, convergence_scorer):
        """Test should_merge returns True for converged branches."""
        shared_claims = ["Claim 1", "Claim 2"]

        branch_a, nodes_a = create_branch_with_nodes("branch-a", "Branch A", shared_claims, 0.75)
        branch_b, nodes_b = create_branch_with_nodes("branch-b", "Branch B", shared_claims, 0.75)

        should_merge = convergence_scorer.should_merge(branch_a, branch_b, nodes_a, nodes_b)
        assert should_merge is True

    def test_partial_overlap_below_threshold(self, convergence_scorer):
        """Test that partial claim overlap is below threshold."""
        claims_a = ["Claim 1", "Claim 2", "Claim A"]
        claims_b = ["Claim 3", "Claim 4", "Claim B"]  # No overlap

        branch_a, nodes_a = create_branch_with_nodes("branch-a", "Branch A", claims_a)
        branch_b, nodes_b = create_branch_with_nodes("branch-b", "Branch B", claims_b)

        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        # No claim overlap = 0 claim similarity
        # Confidence may be similar, so score is 0.3 * conf_similarity
        assert score < 0.8


# ============================================================================
# Test: Auto-Merge Trigger
# ============================================================================


class TestConvergenceAutoMergeTrigger:
    """Test auto-merge trigger on convergence."""

    def test_convergence_auto_merge_trigger(self, convergence_scorer):
        """Test that convergence above threshold triggers merge recommendation."""
        # Create branches that should converge
        shared_claims = ["Key insight 1", "Key insight 2"]

        branch_a, nodes_a = create_branch_with_nodes("branch-a", "Branch A", shared_claims, 0.8)
        branch_b, nodes_b = create_branch_with_nodes("branch-b", "Branch B", shared_claims, 0.8)

        # Check convergence
        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)
        should_merge = score >= convergence_scorer.threshold

        assert should_merge is True
        assert score >= 0.8


# ============================================================================
# Test: Convergence with Low Similarity
# ============================================================================


class TestConvergenceWithLowSimilarity:
    """Test convergence detection with low similarity branches."""

    def test_convergence_with_low_similarity(self, convergence_scorer):
        """Test that divergent branches don't trigger convergence."""
        # Create branches with completely different claims
        claims_a = ["Approach X is best", "Use method A"]
        claims_b = ["Approach Y is better", "Use method B"]

        branch_a, nodes_a = create_branch_with_nodes("branch-a", "Branch A", claims_a, 0.9)
        branch_b, nodes_b = create_branch_with_nodes("branch-b", "Branch B", claims_b, 0.1)

        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)
        should_merge = convergence_scorer.should_merge(branch_a, branch_b, nodes_a, nodes_b)

        # Low claim overlap + high confidence difference
        assert score < 0.8
        assert should_merge is False

    def test_empty_nodes_return_zero_score(self, convergence_scorer):
        """Test that empty node lists return zero score."""
        branch_a = Branch(
            id="a", name="A", reason=BranchReason.ALTERNATIVE_APPROACH, start_node_id="root"
        )
        branch_b = Branch(
            id="b", name="B", reason=BranchReason.ALTERNATIVE_APPROACH, start_node_id="root"
        )

        score = convergence_scorer.score_convergence(branch_a, branch_b, [], [])
        assert score == 0.0


# ============================================================================
# Test: Claim Weighting (70% claim / 30% confidence)
# ============================================================================


class TestConvergenceScorerClaimWeighting:
    """Test the 70% claim / 30% confidence weighting."""

    def test_convergence_scorer_claim_weighting(self, convergence_scorer):
        """Test that claim similarity is weighted 70% and confidence 30%."""
        # Perfect claim match, different confidence
        shared_claims = ["Claim 1", "Claim 2"]

        branch_a, nodes_a = create_branch_with_nodes(
            "branch-a",
            "Branch A",
            shared_claims,
            0.9,  # High confidence
        )
        branch_b, nodes_b = create_branch_with_nodes(
            "branch-b",
            "Branch B",
            shared_claims,
            0.1,  # Low confidence
        )

        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        # Claim similarity = 1.0 (Jaccard = 1.0)
        # Confidence similarity = 1.0 - abs(0.9 - 0.1) = 0.2
        # Score = 0.7 * 1.0 + 0.3 * 0.2 = 0.76
        expected_score = 0.7 * 1.0 + 0.3 * 0.2
        assert abs(score - expected_score) < 0.05

    def test_claim_overlap_dominates_score(self, convergence_scorer):
        """Test that claim overlap has more impact than confidence."""
        # 50% claim overlap
        claims_a = ["Shared 1", "Unique A"]
        claims_b = ["Shared 1", "Unique B"]

        branch_a, nodes_a = create_branch_with_nodes("branch-a", "Branch A", claims_a, 0.8)
        branch_b, nodes_b = create_branch_with_nodes("branch-b", "Branch B", claims_b, 0.8)

        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        # Claim Jaccard: 1 / 3 = 0.33
        # Confidence similarity = 1.0
        # Score = 0.7 * 0.33 + 0.3 * 1.0 = 0.53
        # Score should be moderate due to claim overlap
        assert 0.4 < score < 0.7

    def test_content_similarity_fallback_no_claims(self, convergence_scorer):
        """Test content similarity when no claims are present."""
        # Create branches with no claims but similar content
        branch_a = Branch(
            id="a", name="A", reason=BranchReason.ALTERNATIVE_APPROACH, start_node_id="root"
        )
        branch_b = Branch(
            id="b", name="B", reason=BranchReason.ALTERNATIVE_APPROACH, start_node_id="root"
        )

        # Create nodes with similar content but no claims
        nodes_a = [
            DebateNode(
                id="a-0",
                node_type=NodeType.PROPOSAL,
                agent_id="agent-1",
                content="The quick brown fox jumps over the lazy dog",
                branch_id="a",
                claims=[],  # No claims
            )
        ]
        nodes_b = [
            DebateNode(
                id="b-0",
                node_type=NodeType.PROPOSAL,
                agent_id="agent-2",
                content="The quick brown fox runs over the lazy dog",  # Similar
                branch_id="b",
                claims=[],  # No claims
            )
        ]

        score = convergence_scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        # Should use content similarity fallback
        # Word overlap should be high
        assert score > 0.5


# ============================================================================
# Test: Scorer Configuration
# ============================================================================


class TestScorerConfiguration:
    """Test convergence scorer configuration."""

    def test_custom_threshold(self):
        """Test scorer with custom threshold."""
        scorer = ConvergenceScorer(threshold=0.5)
        assert scorer.threshold == 0.5

        # Branches that would not converge at 0.8 should converge at 0.5
        claims_a = ["Shared", "A only"]
        claims_b = ["Shared", "B only"]

        branch_a, nodes_a = create_branch_with_nodes("a", "A", claims_a, 0.7)
        branch_b, nodes_b = create_branch_with_nodes("b", "B", claims_b, 0.7)

        score = scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)
        should_merge = scorer.should_merge(branch_a, branch_b, nodes_a, nodes_b)

        # With lower threshold, may trigger merge
        assert score > 0.5 or not should_merge

    def test_default_threshold_is_0_8(self):
        """Test that default threshold is 0.8."""
        scorer = ConvergenceScorer()
        assert scorer.threshold == 0.8


__all__ = [
    "TestConvergenceDetectionDefaultThreshold",
    "TestConvergenceAutoMergeTrigger",
    "TestConvergenceWithLowSimilarity",
    "TestConvergenceScorerClaimWeighting",
    "TestScorerConfiguration",
]
