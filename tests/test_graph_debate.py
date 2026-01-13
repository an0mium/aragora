"""Tests for the Graph-Based Debate Engine."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from aragora.debate.graph import (
    NodeType,
    BranchReason,
    MergeStrategy,
    DebateNode,
    Branch,
    MergeResult,
    BranchPolicy,
    ConvergenceScorer,
    DebateGraph,
    GraphReplayBuilder,
    GraphDebateOrchestrator,
)


# ============================================================================
# NodeType and BranchReason tests
# ============================================================================


class TestEnums:
    """Tests for enum types."""

    def test_node_types(self):
        """Verify all node types are defined."""
        assert NodeType.ROOT.value == "root"
        assert NodeType.PROPOSAL.value == "proposal"
        assert NodeType.CRITIQUE.value == "critique"
        assert NodeType.SYNTHESIS.value == "synthesis"
        assert NodeType.BRANCH_POINT.value == "branch_point"
        assert NodeType.MERGE_POINT.value == "merge_point"
        assert NodeType.COUNTERFACTUAL.value == "counterfactual"
        assert NodeType.CONCLUSION.value == "conclusion"

    def test_branch_reasons(self):
        """Verify all branch reasons are defined."""
        assert BranchReason.HIGH_DISAGREEMENT.value == "high_disagreement"
        assert BranchReason.ALTERNATIVE_APPROACH.value == "alternative_approach"
        assert BranchReason.COUNTERFACTUAL_EXPLORATION.value == "counterfactual_exploration"

    def test_merge_strategies(self):
        """Verify all merge strategies are defined."""
        assert MergeStrategy.BEST_PATH.value == "best_path"
        assert MergeStrategy.SYNTHESIS.value == "synthesis"
        assert MergeStrategy.VOTE.value == "vote"


# ============================================================================
# DebateNode tests
# ============================================================================


class TestDebateNode:
    """Tests for DebateNode dataclass."""

    def test_node_creation(self):
        """Verify node can be created with all fields."""
        node = DebateNode(
            id="node-1",
            node_type=NodeType.PROPOSAL,
            agent_id="agent-1",
            content="Test proposal content",
            confidence=0.8,
            claims=["claim1", "claim2"],
        )

        assert node.id == "node-1"
        assert node.node_type == NodeType.PROPOSAL
        assert node.agent_id == "agent-1"
        assert node.confidence == 0.8
        assert len(node.claims) == 2

    def test_node_hash_deterministic(self):
        """Verify node hash is deterministic."""
        timestamp = datetime.now()
        node1 = DebateNode(
            id="1",
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="content",
            timestamp=timestamp,
        )
        node2 = DebateNode(
            id="2",
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="content",
            timestamp=timestamp,
        )

        assert node1.hash() == node2.hash()

    def test_node_to_dict_roundtrip(self):
        """Verify node survives to_dict/from_dict roundtrip."""
        node = DebateNode(
            id="roundtrip",
            node_type=NodeType.CRITIQUE,
            agent_id="agent-1",
            content="Test content",
            confidence=0.7,
            claims=["claim1"],
            parent_ids=["parent-1"],
        )

        data = node.to_dict()
        restored = DebateNode.from_dict(data)

        assert restored.id == node.id
        assert restored.node_type == node.node_type
        assert restored.agent_id == node.agent_id
        assert restored.content == node.content
        assert restored.confidence == node.confidence
        assert restored.claims == node.claims


# ============================================================================
# Branch tests
# ============================================================================


class TestBranch:
    """Tests for Branch dataclass."""

    def test_branch_creation(self):
        """Verify branch can be created."""
        branch = Branch(
            id="branch-1",
            name="Alternative",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="node-1",
            hypothesis="What if we approach differently?",
        )

        assert branch.id == "branch-1"
        assert branch.name == "Alternative"
        assert branch.reason == BranchReason.HIGH_DISAGREEMENT
        assert branch.is_active is True
        assert branch.is_merged is False

    def test_branch_to_dict_roundtrip(self):
        """Verify branch survives serialization."""
        branch = Branch(
            id="test",
            name="Test Branch",
            reason=BranchReason.UNCERTAINTY,
            start_node_id="node-1",
            confidence=0.6,
        )

        data = branch.to_dict()
        restored = Branch.from_dict(data)

        assert restored.id == branch.id
        assert restored.reason == branch.reason
        assert restored.confidence == branch.confidence


# ============================================================================
# BranchPolicy tests
# ============================================================================


class TestBranchPolicy:
    """Tests for BranchPolicy."""

    def test_default_policy(self):
        """Verify default policy values."""
        policy = BranchPolicy()

        assert policy.disagreement_threshold == 0.6
        assert policy.max_branches == 4
        assert policy.max_depth == 5

    def test_should_branch_high_disagreement(self):
        """Verify branching on high disagreement."""
        policy = BranchPolicy(disagreement_threshold=0.5)

        should, reason = policy.should_branch(
            disagreement=0.7,
            uncertainty=0.3,
            current_branches=1,
            current_depth=1,
        )

        assert should is True
        assert reason == BranchReason.HIGH_DISAGREEMENT

    def test_should_branch_high_uncertainty(self):
        """Verify branching on high uncertainty."""
        policy = BranchPolicy(uncertainty_threshold=0.6)

        should, reason = policy.should_branch(
            disagreement=0.3,
            uncertainty=0.8,
            current_branches=1,
            current_depth=1,
        )

        assert should is True
        assert reason == BranchReason.UNCERTAINTY

    def test_should_not_branch_at_max(self):
        """Verify no branching when at max branches."""
        policy = BranchPolicy(max_branches=2)

        should, reason = policy.should_branch(
            disagreement=0.9,
            uncertainty=0.9,
            current_branches=2,
            current_depth=1,
        )

        assert should is False
        assert reason is None

    def test_should_not_branch_at_max_depth(self):
        """Verify no branching when at max depth."""
        policy = BranchPolicy(max_depth=3)

        should, reason = policy.should_branch(
            disagreement=0.9,
            uncertainty=0.9,
            current_branches=1,
            current_depth=3,
        )

        assert should is False
        assert reason is None


# ============================================================================
# ConvergenceScorer tests
# ============================================================================


class TestConvergenceScorer:
    """Tests for ConvergenceScorer."""

    def test_identical_claims_converge(self):
        """Verify branches with same claims have high convergence."""
        scorer = ConvergenceScorer(threshold=0.8)

        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="1")

        nodes_a = [
            DebateNode(
                id="a1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="content",
                claims=["claim1", "claim2"],
            )
        ]
        nodes_b = [
            DebateNode(
                id="b1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="content",
                claims=["claim1", "claim2"],
            )
        ]

        score = scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        assert score >= 0.7  # High convergence

    def test_different_claims_diverge(self):
        """Verify branches with different claims have low convergence."""
        scorer = ConvergenceScorer()

        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="1")

        nodes_a = [
            DebateNode(
                id="a1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="content",
                claims=["claim1", "claim2"],
            )
        ]
        nodes_b = [
            DebateNode(
                id="b1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="content",
                claims=["claim3", "claim4"],
            )
        ]

        score = scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        assert score < 0.5  # Low convergence

    def test_empty_nodes_zero_convergence(self):
        """Verify empty node lists return zero convergence."""
        scorer = ConvergenceScorer()

        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="1")

        score = scorer.score_convergence(branch_a, branch_b, [], [])

        assert score == 0.0

    def test_should_merge_above_threshold(self):
        """Verify should_merge returns True above threshold."""
        scorer = ConvergenceScorer(threshold=0.5)

        branch_a = Branch(id="a", name="A", reason=BranchReason.UNCERTAINTY, start_node_id="1")
        branch_b = Branch(id="b", name="B", reason=BranchReason.UNCERTAINTY, start_node_id="1")

        nodes = [
            DebateNode(
                id="1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="same content",
                claims=["same claim"],
            )
        ]

        should = scorer.should_merge(branch_a, branch_b, nodes, nodes)

        assert should is True


# ============================================================================
# DebateGraph tests
# ============================================================================


class TestDebateGraph:
    """Tests for DebateGraph."""

    def test_graph_creation(self):
        """Verify graph can be created."""
        graph = DebateGraph()

        assert graph.debate_id is not None
        assert graph.main_branch_id == "main"
        assert "main" in graph.branches
        assert len(graph.nodes) == 0

    def test_add_root_node(self):
        """Verify root node can be added."""
        graph = DebateGraph()

        node = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="What is the best approach?",
        )

        assert graph.root_id == node.id
        assert node.id in graph.nodes
        assert node.branch_id == "main"

    def test_add_child_node(self):
        """Verify child node is linked to parent."""
        graph = DebateGraph()

        parent = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        child = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent-1",
            content="Proposal",
            parent_id=parent.id,
        )

        assert parent.id in child.parent_ids
        assert child.id in parent.child_ids

    def test_create_branch(self):
        """Verify branch creation from node."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        branch = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.HIGH_DISAGREEMENT,
            name="Alternative",
            hypothesis="What if X?",
        )

        assert branch.id in graph.branches
        assert branch.start_node_id == root.id
        assert branch.hypothesis == "What if X?"

    def test_create_branch_invalid_node(self):
        """Verify branch creation fails with invalid node."""
        graph = DebateGraph()

        with pytest.raises(ValueError):
            graph.create_branch(
                from_node_id="nonexistent",
                reason=BranchReason.UNCERTAINTY,
                name="Test",
            )

    def test_merge_branches(self):
        """Verify branches can be merged."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        # Create two branches
        branch_a = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.HIGH_DISAGREEMENT,
            name="Branch A",
        )
        branch_b = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.ALTERNATIVE_APPROACH,
            name="Branch B",
        )

        # Add nodes to each branch
        node_a = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent-1",
            content="Path A",
            parent_id=root.id,
            branch_id=branch_a.id,
        )
        node_b = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent-2",
            content="Path B",
            parent_id=root.id,
            branch_id=branch_b.id,
        )

        # Merge
        result = graph.merge_branches(
            branch_ids=[branch_a.id, branch_b.id],
            strategy=MergeStrategy.SYNTHESIS,
            synthesizer_agent_id="system",
            synthesis_content="Merged insights from both paths",
        )

        assert result.merged_node_id in graph.nodes
        assert graph.branches[branch_a.id].is_merged is True
        assert graph.branches[branch_b.id].is_merged is True
        assert len(result.source_branch_ids) == 2

    def test_get_branch_nodes(self):
        """Verify getting nodes for a branch."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        branch = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.UNCERTAINTY,
            name="Test",
        )

        node1 = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="Node 1",
            parent_id=root.id,
            branch_id=branch.id,
        )

        branch_nodes = graph.get_branch_nodes(branch.id)

        assert len(branch_nodes) == 1
        assert branch_nodes[0].id == node1.id

    def test_get_leaf_nodes(self):
        """Verify getting leaf nodes."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        child = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="Child",
            parent_id=root.id,
        )

        leaves = graph.get_leaf_nodes()

        assert len(leaves) == 1
        assert leaves[0].id == child.id

    def test_get_path_to_node(self):
        """Verify getting path from root to node."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        child1 = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="Child 1",
            parent_id=root.id,
        )

        child2 = graph.add_node(
            node_type=NodeType.CRITIQUE,
            agent_id="agent",
            content="Child 2",
            parent_id=child1.id,
        )

        path = graph.get_path_to_node(child2.id)

        assert len(path) == 3
        assert path[0].id == root.id
        assert path[1].id == child1.id
        assert path[2].id == child2.id

    def test_graph_serialization(self):
        """Verify graph survives to_dict/from_dict roundtrip."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="Child",
            parent_id=root.id,
        )

        data = graph.to_dict()
        restored = DebateGraph.from_dict(data)

        assert restored.debate_id == graph.debate_id
        assert len(restored.nodes) == len(graph.nodes)
        assert restored.root_id == graph.root_id


# ============================================================================
# GraphReplayBuilder tests
# ============================================================================


class TestGraphReplayBuilder:
    """Tests for GraphReplayBuilder."""

    def test_replay_branch(self):
        """Verify branch replay in order."""
        graph = DebateGraph()

        root = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        branch = graph.create_branch(
            from_node_id=root.id,
            reason=BranchReason.UNCERTAINTY,
            name="Test",
        )

        graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="agent",
            content="First",
            parent_id=root.id,
            branch_id=branch.id,
        )

        graph.add_node(
            node_type=NodeType.CRITIQUE,
            agent_id="agent",
            content="Second",
            parent_id=root.id,
            branch_id=branch.id,
        )

        builder = GraphReplayBuilder(graph)
        nodes = builder.replay_branch(branch.id)

        assert len(nodes) == 2

    def test_generate_summary(self):
        """Verify summary generation."""
        graph = DebateGraph()

        graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        builder = GraphReplayBuilder(graph)
        summary = builder.generate_summary()

        assert "debate_id" in summary
        assert "total_nodes" in summary
        assert summary["total_nodes"] == 1
        assert "agents" in summary


# ============================================================================
# GraphDebateOrchestrator tests
# ============================================================================


class TestGraphDebateOrchestrator:
    """Tests for GraphDebateOrchestrator."""

    def test_orchestrator_creation(self):
        """Verify orchestrator can be created."""
        agents = [Mock(name="Agent1"), Mock(name="Agent2")]
        orchestrator = GraphDebateOrchestrator(agents)

        assert len(orchestrator.agents) == 2
        assert orchestrator.graph is not None

    @pytest.mark.asyncio
    async def test_run_debate_no_agent_fn(self):
        """Verify debate runs without agent function (just creates root)."""
        agents = [Mock(name="Agent1")]
        orchestrator = GraphDebateOrchestrator(agents)

        graph = await orchestrator.run_debate(
            task="What is the best approach?",
            max_rounds=3,
        )

        assert graph.root_id is not None
        assert len(graph.nodes) == 1

    @pytest.mark.asyncio
    async def test_run_debate_with_mock_agents(self):
        """Verify debate runs with mock agent function."""
        agent1 = Mock()
        agent1.name = "Agent1"
        agent2 = Mock()
        agent2.name = "Agent2"

        orchestrator = GraphDebateOrchestrator([agent1, agent2])

        async def mock_run_agent(agent, prompt, context):
            return f"Response from {agent.name}. I am 70% confident."

        graph = await orchestrator.run_debate(
            task="Test task",
            max_rounds=2,
            run_agent_fn=mock_run_agent,
        )

        assert len(graph.nodes) >= 2  # Root + at least one response

    @pytest.mark.asyncio
    async def test_run_debate_with_callbacks(self):
        """Verify callbacks are invoked."""
        agent = Mock()
        agent.name = "TestAgent"

        orchestrator = GraphDebateOrchestrator([agent])

        nodes_created = []

        async def mock_run_agent(agent, prompt, context):
            return "Response with 80% confidence"

        def on_node(node):
            nodes_created.append(node)

        graph = await orchestrator.run_debate(
            task="Test",
            max_rounds=1,
            run_agent_fn=mock_run_agent,
            on_node=on_node,
        )

        assert len(nodes_created) >= 1  # At least root node

    def test_evaluate_disagreement_high_variance(self):
        """Verify high variance leads to high disagreement."""
        orchestrator = GraphDebateOrchestrator([])

        responses = [
            ("agent1", "Very confident", 0.9),
            ("agent2", "Very uncertain", 0.1),
        ]

        disagreement, alternative = orchestrator.evaluate_disagreement(responses)

        assert disagreement > 0.5
        assert alternative is not None

    def test_evaluate_disagreement_low_variance(self):
        """Verify low variance leads to low disagreement."""
        orchestrator = GraphDebateOrchestrator([])

        responses = [
            ("agent1", "Same view", 0.7),
            ("agent2", "Same view", 0.75),
        ]

        disagreement, alternative = orchestrator.evaluate_disagreement(responses)

        assert disagreement < 0.3
        assert alternative is None

    def test_extract_confidence_explicit(self):
        """Verify explicit confidence extraction."""
        orchestrator = GraphDebateOrchestrator([])

        response = "I believe this is correct. Confidence: 85%"
        confidence = orchestrator._extract_confidence(response)

        assert confidence == 0.85

    def test_extract_confidence_from_tone(self):
        """Verify confidence extraction from tone words."""
        orchestrator = GraphDebateOrchestrator([])

        high_conf = "I am definitely certain this is correct."
        low_conf = "Maybe this could possibly be true, perhaps."

        assert orchestrator._extract_confidence(high_conf) > 0.7
        assert orchestrator._extract_confidence(low_conf) < 0.5

    def test_extract_claims_numbered(self):
        """Verify numbered claims extraction."""
        orchestrator = GraphDebateOrchestrator([])

        response = """My key points:
1. First important claim
2. Second important claim
3. Third important claim"""

        claims = orchestrator._extract_claims(response)

        assert len(claims) >= 1  # At least one claim extracted
        assert any("claim" in c.lower() for c in claims)

    def test_extract_claims_bullet(self):
        """Verify bullet point claims extraction."""
        orchestrator = GraphDebateOrchestrator([])

        response = """Key observations:
- First bullet claim
- Second bullet claim"""

        claims = orchestrator._extract_claims(response)

        assert len(claims) >= 1  # At least one claim extracted
        assert any("bullet" in c.lower() or "claim" in c.lower() for c in claims)

    def test_build_context(self):
        """Verify context building from nodes."""
        orchestrator = GraphDebateOrchestrator([])

        nodes = [
            DebateNode(
                id="1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent1",
                content="First content",
            ),
            DebateNode(
                id="2",
                node_type=NodeType.CRITIQUE,
                agent_id="agent2",
                content="Second content",
            ),
        ]

        context = orchestrator._build_context(nodes)

        assert "agent1" in context
        assert "agent2" in context
        assert "First content" in context

    def test_synthesize_branches(self):
        """Verify branch synthesis."""
        orchestrator = GraphDebateOrchestrator([])

        nodes_a = [
            DebateNode(
                id="a1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="Conclusion A",
                claims=["claim1", "common"],
            ),
        ]
        nodes_b = [
            DebateNode(
                id="b1",
                node_type=NodeType.PROPOSAL,
                agent_id="agent",
                content="Conclusion B",
                claims=["claim2", "common"],
            ),
        ]

        synthesis = orchestrator._synthesize_branches(nodes_a, nodes_b)

        assert "Synthesis" in synthesis
        assert "common" in synthesis or "Agreed" in synthesis

    def test_create_final_synthesis_single_leaf(self):
        """Verify final synthesis with single leaf."""
        orchestrator = GraphDebateOrchestrator([])

        leaves = [
            DebateNode(
                id="1",
                node_type=NodeType.CONCLUSION,
                agent_id="agent",
                content="Final answer",
            ),
        ]

        result = orchestrator._create_final_synthesis(leaves)

        assert "Conclusion" in result
        assert "Final answer" in result

    def test_create_final_synthesis_multiple_leaves(self):
        """Verify final synthesis with multiple leaves."""
        orchestrator = GraphDebateOrchestrator([])

        leaves = [
            DebateNode(
                id="1",
                node_type=NodeType.CONCLUSION,
                agent_id="agent",
                content="Path 1 conclusion",
                confidence=0.8,
                claims=["claim1"],
            ),
            DebateNode(
                id="2",
                node_type=NodeType.CONCLUSION,
                agent_id="agent",
                content="Path 2 conclusion",
                confidence=0.7,
                claims=["claim2"],
            ),
        ]

        result = orchestrator._create_final_synthesis(leaves)

        assert "Path 1" in result
        assert "Path 2" in result
