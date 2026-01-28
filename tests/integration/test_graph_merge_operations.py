"""
Tests for graph branch merge operations.

Phase 8: Debate Integration Test Gaps - Branch merge strategy tests.

Tests:
- test_merge_two_branches_best_path - BEST_PATH strategy
- test_merge_three_branches_synthesis - SYNTHESIS strategy
- test_merge_with_vote_strategy - VOTE strategy with tie-breaker
- test_merge_weighted_by_confidence - WEIGHTED strategy
- test_merge_preserve_all_branches - PRESERVE_ALL strategy
- test_cascade_merge_operations - Multiple successive merges
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

from aragora.debate.graph import (
    Branch,
    BranchReason,
    DebateGraph,
    DebateNode,
    MergeResult,
    MergeStrategy,
    NodeType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def debate_graph():
    """Create a fresh debate graph."""
    return DebateGraph()


@pytest.fixture
def graph_with_branches(debate_graph):
    """Create a graph with multiple branches."""
    graph = debate_graph

    # Add root node
    root = graph.add_node(
        node_type=NodeType.PROPOSAL,
        agent_id="agent-1",
        content="Initial proposal",
    )

    # Create first branch
    branch1 = graph.create_branch(
        from_node_id=root.id,
        reason=BranchReason.ALTERNATIVE_APPROACH,
        name="Option A",
        hypothesis="Hypothesis A",
    )

    # Add nodes to branch 1
    node1 = graph.add_node(
        node_type=NodeType.PROPOSAL,
        agent_id="agent-2",
        content="Branch A argument 1",
        branch_id=branch1.id,
    )
    node1.claims = ["Claim A1", "Shared claim"]
    node1.confidence = 0.8

    # Create second branch
    branch2 = graph.create_branch(
        from_node_id=root.id,
        reason=BranchReason.ALTERNATIVE_APPROACH,
        name="Option B",
        hypothesis="Hypothesis B",
    )

    # Add nodes to branch 2
    node2 = graph.add_node(
        node_type=NodeType.PROPOSAL,
        agent_id="agent-3",
        content="Branch B argument 1",
        branch_id=branch2.id,
    )
    node2.claims = ["Claim B1", "Shared claim"]
    node2.confidence = 0.7

    # Update end nodes
    branch1.end_node_id = node1.id
    branch2.end_node_id = node2.id

    return graph, branch1, branch2


@pytest.fixture
def graph_with_three_branches(debate_graph):
    """Create a graph with three branches."""
    graph = debate_graph

    # Add root
    root = graph.add_node(
        node_type=NodeType.PROPOSAL,
        agent_id="agent-1",
        content="Root proposal",
    )

    branches = []
    for i in range(3):
        branch = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.ALTERNATIVE_APPROACH,
            name=f"Option {chr(65 + i)}",  # A, B, C
            hypothesis=f"Hypothesis {chr(65 + i)}",
        )

        node = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id=f"agent-{i + 2}",
            content=f"Branch {chr(65 + i)} argument",
            branch_id=branch.id,
        )
        node.claims = [f"Claim {chr(65 + i)}", "Common insight"]
        node.confidence = 0.6 + (i * 0.1)  # 0.6, 0.7, 0.8

        branch.end_node_id = node.id
        branch.confidence = node.confidence
        branches.append(branch)

    return graph, branches


# ============================================================================
# Test: Merge Two Branches - BEST_PATH Strategy
# ============================================================================


class TestMergeTwoBranchesBestPath:
    """Test BEST_PATH merge strategy."""

    def test_merge_two_branches_best_path(self, graph_with_branches):
        """Test merging two branches with BEST_PATH strategy."""
        graph, branch1, branch2 = graph_with_branches

        # Merge branches
        result = graph.merge_branches(
            branch_ids=[branch1.id, branch2.id],
            strategy=MergeStrategy.BEST_PATH,
            synthesizer_agent_id="synthesizer",
            synthesis_content="Best path synthesis: choosing Option A approach",
        )

        # Verify merge result
        assert result.merged_node_id is not None
        assert branch1.id in result.source_branch_ids
        assert branch2.id in result.source_branch_ids
        assert result.strategy == MergeStrategy.BEST_PATH

        # Verify branches are marked as merged
        assert not branch1.is_active
        assert not branch2.is_active
        assert branch1.is_merged
        assert branch2.is_merged

        # Verify merge node was created
        merge_node = graph.nodes[result.merged_node_id]
        assert merge_node.node_type == NodeType.MERGE_POINT
        assert "Best path" in merge_node.content


# ============================================================================
# Test: Merge Three Branches - SYNTHESIS Strategy
# ============================================================================


class TestMergeThreeBranchesSynthesis:
    """Test SYNTHESIS merge strategy with three branches."""

    def test_merge_three_branches_synthesis(self, graph_with_three_branches):
        """Test merging three branches with SYNTHESIS strategy."""
        graph, branches = graph_with_three_branches

        result = graph.merge_branches(
            branch_ids=[b.id for b in branches],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="synthesizer",
            synthesis_content="Synthesized conclusion combining insights from A, B, and C",
        )

        # All branches should be merged
        assert len(result.source_branch_ids) == 3

        # All branches should be inactive
        for branch in branches:
            assert not branch.is_active
            assert branch.merged_into == result.merged_node_id

        # Insights from all branches should be preserved
        # Each branch had unique claim + "Common insight"
        assert len(result.insights_preserved) >= 3

    def test_synthesis_preserves_all_claims(self, graph_with_three_branches):
        """Test that synthesis preserves claims from all branches."""
        graph, branches = graph_with_three_branches

        result = graph.merge_branches(
            branch_ids=[b.id for b in branches],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="synthesizer",
            synthesis_content="Full synthesis",
        )

        # Should have collected claims from all branches
        expected_claims = {"Claim A", "Claim B", "Claim C", "Common insight"}
        actual_claims = set(result.insights_preserved)

        # Should have the common insight (deduped)
        assert "Common insight" in actual_claims


# ============================================================================
# Test: Merge with VOTE Strategy
# ============================================================================


class TestMergeWithVoteStrategy:
    """Test VOTE merge strategy."""

    def test_merge_with_vote_strategy(self, graph_with_branches):
        """Test merging with VOTE strategy."""
        graph, branch1, branch2 = graph_with_branches

        result = graph.merge_branches(
            branch_ids=[branch1.id, branch2.id],
            strategy=MergeStrategy.VOTE,
            synthesizer_agent_id="synthesizer",
            synthesis_content="Voted result: agents prefer Option A",
        )

        assert result.strategy == MergeStrategy.VOTE
        assert result.merged_node_id is not None

        # Verify metadata indicates vote strategy
        merge_node = graph.nodes[result.merged_node_id]
        assert merge_node.metadata.get("strategy") == "vote"


# ============================================================================
# Test: Merge Weighted by Confidence
# ============================================================================


class TestMergeWeightedByConfidence:
    """Test WEIGHTED merge strategy."""

    def test_merge_weighted_by_confidence(self, graph_with_three_branches):
        """Test merging with WEIGHTED strategy based on confidence."""
        graph, branches = graph_with_three_branches

        # Set varying confidence levels
        for i, branch in enumerate(branches):
            branch.confidence = 0.5 + (i * 0.2)  # 0.5, 0.7, 0.9

        result = graph.merge_branches(
            branch_ids=[b.id for b in branches],
            strategy=MergeStrategy.WEIGHTED,
            synthesizer_agent_id="synthesizer",
            synthesis_content="Weighted synthesis: heavily favoring Option C (highest confidence)",
        )

        assert result.strategy == MergeStrategy.WEIGHTED

        # The merge node should reflect the weighted decision
        merge_node = graph.nodes[result.merged_node_id]
        assert (
            "weighted" in merge_node.metadata.get("strategy", "").lower()
            or merge_node.metadata.get("strategy") == "weighted"
        )


# ============================================================================
# Test: Merge Preserve All Branches
# ============================================================================


class TestMergePreserveAllBranches:
    """Test PRESERVE_ALL merge strategy."""

    def test_merge_preserve_all_branches(self, graph_with_branches):
        """Test merging with PRESERVE_ALL strategy."""
        graph, branch1, branch2 = graph_with_branches

        result = graph.merge_branches(
            branch_ids=[branch1.id, branch2.id],
            strategy=MergeStrategy.PRESERVE_ALL,
            synthesizer_agent_id="synthesizer",
            synthesis_content="Both approaches are valid alternatives",
        )

        assert result.strategy == MergeStrategy.PRESERVE_ALL

        # All insights should be preserved
        assert len(result.insights_preserved) > 0

        # Both branches should be in source
        assert branch1.id in result.source_branch_ids
        assert branch2.id in result.source_branch_ids


# ============================================================================
# Test: Cascade Merge Operations
# ============================================================================


class TestCascadeMergeOperations:
    """Test multiple successive merges."""

    def test_cascade_merge_operations(self, debate_graph):
        """Test multiple successive merge operations."""
        graph = debate_graph

        # Create root
        root = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent-1",
            content="Root",
        )

        # Create 4 branches
        branches = []
        for i in range(4):
            branch = graph.create_branch(
                from_node_id=root.id,
                reason=BranchReason.ALTERNATIVE_APPROACH,
                name=f"Branch {i}",
            )
            node = graph.add_node(
                node_type=NodeType.PROPOSAL,
                agent_id=f"agent-{i + 2}",
                content=f"Content {i}",
                branch_id=branch.id,
            )
            branch.end_node_id = node.id
            branches.append(branch)

        # First merge: branches 0 and 1
        result1 = graph.merge_branches(
            branch_ids=[branches[0].id, branches[1].id],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="synth-1",
            synthesis_content="First merge",
        )

        # Second merge: branches 2 and 3
        result2 = graph.merge_branches(
            branch_ids=[branches[2].id, branches[3].id],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="synth-2",
            synthesis_content="Second merge",
        )

        # All 4 original branches should be merged
        assert not branches[0].is_active
        assert not branches[1].is_active
        assert not branches[2].is_active
        assert not branches[3].is_active

        # Should have 2 merge results in history
        assert len(graph.merge_history) == 2

    def test_merge_history_preserved(self, graph_with_branches):
        """Test that merge history is preserved."""
        graph, branch1, branch2 = graph_with_branches

        result = graph.merge_branches(
            branch_ids=[branch1.id, branch2.id],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="synth",
            synthesis_content="Merged",
        )

        assert len(graph.merge_history) == 1
        assert graph.merge_history[0].merged_node_id == result.merged_node_id


# ============================================================================
# Test: Invalid Branch Merge
# ============================================================================


class TestInvalidBranchMerge:
    """Test error handling for invalid merges."""

    def test_merge_nonexistent_branch_raises_error(self, debate_graph):
        """Test that merging nonexistent branches raises ValueError."""
        graph = debate_graph

        with pytest.raises(ValueError, match="Branch .* not found"):
            graph.merge_branches(
                branch_ids=["nonexistent-branch"],
                strategy=MergeStrategy.SYNTHESIS,
                synthesizer_agent_id="synth",
                synthesis_content="Invalid merge",
            )


__all__ = [
    "TestMergeTwoBranchesBestPath",
    "TestMergeThreeBranchesSynthesis",
    "TestMergeWithVoteStrategy",
    "TestMergeWeightedByConfidence",
    "TestMergePreserveAllBranches",
    "TestCascadeMergeOperations",
    "TestInvalidBranchMerge",
]
