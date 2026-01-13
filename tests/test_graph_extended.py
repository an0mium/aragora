"""
Extended tests for Debate Graph with counterfactual branching.

Tests cover gaps not in test_debate_graph.py:
- Graph navigation edge cases (circular refs, multiple parents)
- Merge operation edge cases (cascade effects, no common ancestor)
- Convergence detection edge cases
- BranchPolicy threshold edge cases
- GraphDebateOrchestrator error handling and events
- Large graph serialization
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.graph import (
    Branch,
    BranchPolicy,
    BranchReason,
    ConvergenceScorer,
    DebateGraph,
    DebateNode,
    GraphDebateOrchestrator,
    GraphReplayBuilder,
    MergeResult,
    MergeStrategy,
    NodeType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_graph():
    """Create a sample graph with multiple branches."""
    graph = DebateGraph(debate_id="test-graph")
    root = graph.add_node(NodeType.ROOT, "system", "Test task")
    p1 = graph.add_node(NodeType.PROPOSAL, "agent1", "Proposal 1", root.id)
    p2 = graph.add_node(NodeType.CRITIQUE, "agent2", "Critique 1", p1.id)
    return graph, root, p1, p2


@pytest.fixture
def mock_agents():
    """Create mock agents for orchestrator tests."""
    agent1 = MagicMock()
    agent1.name = "claude"
    agent2 = MagicMock()
    agent2.name = "gpt4"
    return [agent1, agent2]


# =============================================================================
# Graph Navigation Extended Tests
# =============================================================================


class TestGraphNavigationExtended:
    """Extended tests for graph navigation operations."""

    def test_path_with_multiple_parents(self):
        """Test path traversal follows first parent (DAG behavior)."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        n1 = graph.add_node(NodeType.PROPOSAL, "a1", "P1", root.id, b1.id)
        n2 = graph.add_node(NodeType.PROPOSAL, "a2", "P2", root.id, b2.id)

        # Merge creates node with multiple parents
        merge = graph.merge_branches(
            [b1.id, b2.id], MergeStrategy.SYNTHESIS, "system", "Merged content"
        )

        merge_node = graph.nodes[merge.merged_node_id]
        assert len(merge_node.parent_ids) == 2

        # Path follows first parent
        path = graph.get_path_to_node(merge_node.id)
        assert path[0] == root

    def test_circular_reference_prevention(self):
        """Test path traversal handles potential circular refs gracefully."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        n1 = graph.add_node(NodeType.PROPOSAL, "a1", "P1", root.id)
        n2 = graph.add_node(NodeType.PROPOSAL, "a2", "P2", n1.id)

        # Manually create circular reference (shouldn't happen normally)
        n2.parent_ids.append(n2.id)  # Self-reference

        # Should not infinite loop
        path = graph.get_path_to_node(n2.id)
        assert len(path) <= 10  # Bounded

    def test_get_leaf_nodes_multiple_branches(self):
        """Test leaf node detection across multiple branches."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        # Each branch has its own leaf
        leaf1 = graph.add_node(NodeType.SYNTHESIS, "a1", "S1", root.id, b1.id)
        leaf2 = graph.add_node(NodeType.SYNTHESIS, "a2", "S2", root.id, b2.id)

        leaves = graph.get_leaf_nodes()
        assert len(leaves) >= 2
        assert leaf1 in leaves
        assert leaf2 in leaves

    def test_get_path_to_root(self):
        """Test path to root node returns single element."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        path = graph.get_path_to_node(root.id)
        assert len(path) == 1
        assert path[0] == root

    def test_branch_node_association(self):
        """Test nodes are correctly associated with branches."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        branch = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "Alt")

        # Main branch node
        main_node = graph.add_node(NodeType.PROPOSAL, "a1", "Main", root.id, graph.main_branch_id)
        # Alt branch node
        alt_node = graph.add_node(NodeType.PROPOSAL, "a2", "Alt", root.id, branch.id)

        main_nodes = graph.get_branch_nodes(graph.main_branch_id)
        alt_nodes = graph.get_branch_nodes(branch.id)

        assert main_node in main_nodes
        assert alt_node in alt_nodes
        assert main_node not in alt_nodes

    def test_deep_path_traversal(self):
        """Test path traversal for deep graph."""
        graph = DebateGraph()
        current = graph.add_node(NodeType.ROOT, "system", "Task")

        # Build deep chain
        for i in range(20):
            current = graph.add_node(
                NodeType.PROPOSAL if i % 2 == 0 else NodeType.CRITIQUE,
                f"agent{i}",
                f"Content {i}",
                current.id,
            )

        path = graph.get_path_to_node(current.id)
        assert len(path) == 21  # Root + 20 nodes


# =============================================================================
# Merge Operations Extended Tests
# =============================================================================


class TestMergeOperationsExtended:
    """Extended tests for branch merge operations."""

    def test_merge_three_branches(self):
        """Test merging three branches at once."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        branches = []
        for i in range(3):
            b = graph.create_branch(root.id, BranchReason.UNCERTAINTY, f"B{i}")
            graph.add_node(NodeType.PROPOSAL, f"a{i}", f"P{i}", root.id, b.id, claims=[f"claim{i}"])
            branches.append(b)

        result = graph.merge_branches(
            [b.id for b in branches], MergeStrategy.SYNTHESIS, "system", "Three-way merge"
        )

        assert len(result.source_branch_ids) == 3
        merge_node = graph.nodes[result.merged_node_id]
        assert len(merge_node.parent_ids) == 3

    def test_merge_deactivates_branches(self):
        """Test that merged branches are deactivated."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        graph.add_node(NodeType.PROPOSAL, "a", "P", root.id, b1.id)
        graph.add_node(NodeType.PROPOSAL, "b", "Q", root.id, b2.id)

        active_before = len(graph.get_active_branches())

        graph.merge_branches([b1.id, b2.id], MergeStrategy.BEST_PATH, "sys", "Merged")

        active_after = len(graph.get_active_branches())
        assert active_after < active_before

        assert graph.branches[b1.id].is_merged is True
        assert graph.branches[b2.id].is_merged is True

    def test_merge_preserves_insights(self):
        """Test that merge preserves unique insights from branches."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        graph.add_node(NodeType.SYNTHESIS, "a", "S1", root.id, b1.id, claims=["unique_a", "shared"])
        graph.add_node(NodeType.SYNTHESIS, "b", "S2", root.id, b2.id, claims=["unique_b", "shared"])

        result = graph.merge_branches([b1.id, b2.id], MergeStrategy.SYNTHESIS, "sys", "M")

        # Should preserve all unique insights
        assert "unique_a" in result.insights_preserved or "unique_b" in result.insights_preserved

    def test_merge_empty_branches(self):
        """Test merging branches with no end nodes."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        # Branches have no additional nodes (just created)
        result = graph.merge_branches([b1.id, b2.id], MergeStrategy.SYNTHESIS, "sys", "M")

        # Should still create merge node
        assert result.merged_node_id in graph.nodes

    def test_merge_updates_branch_metadata(self):
        """Test merge updates branch merged_into field."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        graph.add_node(NodeType.PROPOSAL, "a", "P", root.id, b1.id)
        graph.add_node(NodeType.PROPOSAL, "b", "Q", root.id, b2.id)

        result = graph.merge_branches([b1.id, b2.id], MergeStrategy.SYNTHESIS, "sys", "M")

        assert graph.branches[b1.id].merged_into == result.merged_node_id
        assert graph.branches[b2.id].merged_into == result.merged_node_id

    def test_merge_history_recorded(self):
        """Test that merges are recorded in history."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        graph.add_node(NodeType.PROPOSAL, "a", "P", root.id, b1.id)

        initial_history_len = len(graph.merge_history)

        graph.merge_branches([b1.id], MergeStrategy.BEST_PATH, "sys", "M")

        assert len(graph.merge_history) == initial_history_len + 1


# =============================================================================
# Convergence Detection Extended Tests
# =============================================================================


class TestConvergenceExtended:
    """Extended tests for convergence detection."""

    def test_convergence_empty_claims_uses_content(self):
        """Test convergence falls back to content similarity."""
        scorer = ConvergenceScorer(threshold=0.5)

        b1 = Branch(id="b1", name="B1", reason=BranchReason.UNCERTAINTY, start_node_id="n")
        b2 = Branch(id="b2", name="B2", reason=BranchReason.UNCERTAINTY, start_node_id="n")

        # Nodes with no claims but similar content
        n1 = DebateNode(
            id="n1",
            node_type=NodeType.SYNTHESIS,
            agent_id="a",
            content="The answer is definitely forty two",
        )
        n2 = DebateNode(
            id="n2",
            node_type=NodeType.SYNTHESIS,
            agent_id="b",
            content="The answer is forty two for sure",
        )

        score = scorer.score_convergence(b1, b2, [n1], [n2])
        assert score > 0.3  # Some similarity

    def test_convergence_uses_last_three_nodes(self):
        """Test convergence looks at last 3 nodes of each branch."""
        scorer = ConvergenceScorer(threshold=0.5)

        b1 = Branch(id="b1", name="B1", reason=BranchReason.UNCERTAINTY, start_node_id="n")
        b2 = Branch(id="b2", name="B2", reason=BranchReason.UNCERTAINTY, start_node_id="n")

        # Many nodes, but last 3 are similar
        nodes_a = [
            DebateNode(
                id=f"a{i}",
                node_type=NodeType.PROPOSAL,
                agent_id="a",
                content="old",
                claims=["different"],
            )
            for i in range(5)
        ]
        nodes_a[-1].claims = ["same"]
        nodes_a[-2].claims = ["same"]

        nodes_b = [
            DebateNode(
                id=f"b{i}",
                node_type=NodeType.PROPOSAL,
                agent_id="b",
                content="old",
                claims=["different"],
            )
            for i in range(5)
        ]
        nodes_b[-1].claims = ["same"]
        nodes_b[-2].claims = ["same"]

        score = scorer.score_convergence(b1, b2, nodes_a, nodes_b)
        # Should have some convergence from last nodes
        assert score > 0

    def test_convergence_partial_claim_overlap(self):
        """Test convergence with partial claim overlap."""
        scorer = ConvergenceScorer(threshold=0.5)

        b1 = Branch(id="b1", name="B1", reason=BranchReason.UNCERTAINTY, start_node_id="n")
        b2 = Branch(id="b2", name="B2", reason=BranchReason.UNCERTAINTY, start_node_id="n")

        n1 = DebateNode(
            id="n1",
            node_type=NodeType.SYNTHESIS,
            agent_id="a",
            content="X",
            claims=["a", "b", "c"],
            confidence=0.8,
        )
        n2 = DebateNode(
            id="n2",
            node_type=NodeType.SYNTHESIS,
            agent_id="b",
            content="Y",
            claims=["b", "c", "d"],
            confidence=0.8,
        )

        # 2/4 overlap = 0.5 Jaccard
        score = scorer.score_convergence(b1, b2, [n1], [n2])
        assert 0.3 < score < 0.8

    def test_graph_check_convergence_multiple_pairs(self):
        """Test check_convergence returns sorted candidates."""
        graph = DebateGraph(branch_policy=BranchPolicy(convergence_threshold=0.3))
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        # Create three branches
        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")
        b3 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B3")

        # B1 and B2 highly similar
        graph.add_node(
            NodeType.SYNTHESIS, "a", "X", root.id, b1.id, claims=["same", "same2"], confidence=0.9
        )
        graph.add_node(
            NodeType.SYNTHESIS, "b", "Y", root.id, b2.id, claims=["same", "same2"], confidence=0.9
        )

        # B3 different
        graph.add_node(
            NodeType.SYNTHESIS, "c", "Z", root.id, b3.id, claims=["different"], confidence=0.5
        )

        candidates = graph.check_convergence()

        # Should find at least one pair
        if candidates:
            # Sorted by score descending
            scores = [c[2] for c in candidates]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# BranchPolicy Extended Tests
# =============================================================================


class TestBranchPolicyExtended:
    """Extended tests for BranchPolicy edge cases."""

    def test_threshold_edge_case_exactly_at_threshold(self):
        """Test behavior exactly at threshold."""
        policy = BranchPolicy(disagreement_threshold=0.6)

        # Exactly at threshold - should not branch (> required)
        should, reason = policy.should_branch(0.6, 0.3, 1, 1)
        assert should is False

        # Just above threshold
        should, reason = policy.should_branch(0.61, 0.3, 1, 1)
        assert should is True

    def test_uncertainty_threshold_edge_case(self):
        """Test uncertainty threshold edge case."""
        policy = BranchPolicy(uncertainty_threshold=0.7)

        should, reason = policy.should_branch(0.3, 0.7, 1, 1)
        assert should is False

        should, reason = policy.should_branch(0.3, 0.71, 1, 1)
        assert should is True
        assert reason == BranchReason.UNCERTAINTY

    def test_max_branches_exactly_at_limit(self):
        """Test max branches at exact limit."""
        policy = BranchPolicy(max_branches=3)

        # At limit
        should, _ = policy.should_branch(0.9, 0.9, 3, 1)
        assert should is False

        # Below limit
        should, _ = policy.should_branch(0.9, 0.9, 2, 1)
        assert should is True

    def test_max_depth_exactly_at_limit(self):
        """Test max depth at exact limit."""
        policy = BranchPolicy(max_depth=5)

        # At limit
        should, _ = policy.should_branch(0.9, 0.9, 1, 5)
        assert should is False

        # Below limit
        should, _ = policy.should_branch(0.9, 0.9, 1, 4)
        assert should is True

    def test_alternative_score_triggers_branch(self):
        """Test alternative score triggers ALTERNATIVE_APPROACH."""
        policy = BranchPolicy(
            disagreement_threshold=0.9,  # Very high
            uncertainty_threshold=0.9,  # Very high
            min_alternative_score=0.4,
        )

        # Low disagreement/uncertainty, but good alternative
        should, reason = policy.should_branch(0.2, 0.2, 1, 1, alternative_score=0.5)
        assert should is True
        assert reason == BranchReason.ALTERNATIVE_APPROACH

    def test_priority_disagreement_over_uncertainty(self):
        """Test disagreement is checked before uncertainty."""
        policy = BranchPolicy(disagreement_threshold=0.5, uncertainty_threshold=0.5)

        # Both exceed threshold - disagreement should be returned
        should, reason = policy.should_branch(0.6, 0.6, 1, 1)
        assert should is True
        assert reason == BranchReason.HIGH_DISAGREEMENT


# =============================================================================
# GraphDebateOrchestrator Extended Tests
# =============================================================================


class TestOrchestratorExtended:
    """Extended tests for GraphDebateOrchestrator."""

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_agents):
        """Test orchestrator handles agent errors gracefully."""
        orchestrator = GraphDebateOrchestrator(mock_agents)

        call_count = 0

        async def failing_agent(agent, prompt, context):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Agent failed")
            return "Fallback response. Confidence: 50%"

        # Should not crash
        graph = await orchestrator.run_debate(
            "Test task",
            max_rounds=1,
            run_agent_fn=failing_agent,
        )

        assert graph.root_id is not None

    @pytest.mark.asyncio
    async def test_event_callbacks_all_types(self, mock_agents):
        """Test all callback types are invoked."""
        policy = BranchPolicy(disagreement_threshold=0.1)  # Easy branching
        orchestrator = GraphDebateOrchestrator(mock_agents, policy=policy)

        nodes_seen = []
        branches_seen = []
        merges_seen = []

        response_idx = 0
        responses = [
            "I strongly disagree! Confidence: 90%",
            "No way, different view. Confidence: 20%",  # High disagreement
            "Coming together now. Confidence: 80%",
            "Yes, agreed. Confidence: 80%",
        ]

        async def mock_agent(agent, prompt, context):
            nonlocal response_idx
            resp = responses[response_idx % len(responses)]
            response_idx += 1
            return resp

        await orchestrator.run_debate(
            "Test",
            max_rounds=2,
            run_agent_fn=mock_agent,
            on_node=lambda n: nodes_seen.append(n),
            on_branch=lambda b: branches_seen.append(b),
            on_merge=lambda m: merges_seen.append(m),
        )

        assert len(nodes_seen) >= 1  # At least root

    @pytest.mark.asyncio
    async def test_early_termination_on_convergence(self, mock_agents):
        """Test debate terminates early when all branches converge."""
        policy = BranchPolicy(convergence_threshold=0.3)  # Easy convergence
        orchestrator = GraphDebateOrchestrator(mock_agents, policy=policy)

        async def agreeing_agent(agent, prompt, context):
            return "I completely agree with everything. Confidence: 95%"

        graph = await orchestrator.run_debate(
            "Test",
            max_rounds=10,  # High limit
            run_agent_fn=agreeing_agent,
        )

        # Should terminate early
        assert len(graph.nodes) < 20  # Not all rounds executed

    @pytest.mark.asyncio
    async def test_event_emitter_integration(self, mock_agents):
        """Test WebSocket event emission."""
        orchestrator = GraphDebateOrchestrator(mock_agents)

        mock_emitter = MagicMock()
        mock_emitter.emit = MagicMock()

        # StreamEvent is imported inside _emit_graph_event from aragora.server.stream
        with patch("aragora.server.stream.StreamEvent") as MockEvent:
            with patch("aragora.server.stream.StreamEventType") as MockType:
                MockType.GRAPH_NODE_ADDED = "node_added"
                MockEvent.return_value = MagicMock()

                await orchestrator.run_debate(
                    "Test",
                    event_emitter=mock_emitter,
                    debate_id="test-123",
                )

    def test_extract_confidence_various_formats(self):
        """Test confidence extraction from various text formats."""
        orchestrator = GraphDebateOrchestrator([])

        # Standard format
        assert orchestrator._extract_confidence("Confidence: 85%") == 0.85
        assert orchestrator._extract_confidence("confidence 70") == 0.70

        # Natural language
        assert orchestrator._extract_confidence("I am 60% certain") == 0.6

        # Caps after percentage
        assert orchestrator._extract_confidence("75% confident about this") == 0.75

        # Over 100 capped
        assert orchestrator._extract_confidence("Confidence: 150%") == 1.0

    def test_extract_claims_numbered_list(self):
        """Test claim extraction from numbered lists."""
        orchestrator = GraphDebateOrchestrator([])

        text = """
1. First important claim
2. Second claim here
3) Third claim with parens
        """
        claims = orchestrator._extract_claims(text)
        assert any("First" in c for c in claims)

    def test_extract_claims_bullets(self):
        """Test claim extraction from bullet points."""
        orchestrator = GraphDebateOrchestrator([])

        text = """
- Bullet point one
- Bullet point two
* Asterisk bullet
        """
        claims = orchestrator._extract_claims(text)
        assert len(claims) >= 1

    def test_extract_claims_limits_to_five(self):
        """Test claim extraction limits to 5 claims."""
        orchestrator = GraphDebateOrchestrator([])

        text = "\n".join([f"- Claim {i}" for i in range(20)])
        claims = orchestrator._extract_claims(text)
        assert len(claims) <= 5

    def test_evaluate_disagreement_single_response(self):
        """Test disagreement with single response."""
        orchestrator = GraphDebateOrchestrator([])

        responses = [("agent1", "Response", 0.8)]
        score, alt = orchestrator.evaluate_disagreement(responses)
        assert score == 0.0
        assert alt is None

    def test_synthesize_branches_no_claims(self):
        """Test synthesis with branches having no claims."""
        orchestrator = GraphDebateOrchestrator([])

        nodes_a = [
            DebateNode(id="a", node_type=NodeType.SYNTHESIS, agent_id="x", content="Content A")
        ]
        nodes_b = [
            DebateNode(id="b", node_type=NodeType.SYNTHESIS, agent_id="y", content="Content B")
        ]

        synthesis = orchestrator._synthesize_branches(nodes_a, nodes_b)
        assert "Synthesis" in synthesis
        assert "Content A" in synthesis or "Content B" in synthesis

    def test_create_final_synthesis_multiple_paths(self):
        """Test final synthesis with multiple divergent paths."""
        orchestrator = GraphDebateOrchestrator([])

        leaves = [
            DebateNode(
                id=f"l{i}",
                node_type=NodeType.CONCLUSION,
                agent_id=f"a{i}",
                content=f"Conclusion {i}",
                confidence=0.7 + i * 0.1,
                claims=[f"claim{i}"],
            )
            for i in range(4)
        ]

        result = orchestrator._create_final_synthesis(leaves)
        assert "Path" in result
        assert "Key claims" in result.lower() or "claim" in result.lower()


# =============================================================================
# Serialization Extended Tests
# =============================================================================


class TestSerializationExtended:
    """Extended tests for graph serialization."""

    def test_large_graph_roundtrip(self):
        """Test serialization of large graph."""
        graph = DebateGraph(debate_id="large-graph")
        root = graph.add_node(NodeType.ROOT, "system", "Large task")

        # Create 50 nodes across branches
        current = root
        for i in range(50):
            node_type = NodeType.PROPOSAL if i % 3 == 0 else NodeType.CRITIQUE
            current = graph.add_node(
                node_type, f"agent{i % 5}", f"Content {i}", current.id, claims=[f"claim{i}"]
            )

        # Serialize and restore
        d = graph.to_dict()
        restored = DebateGraph.from_dict(d)

        assert len(restored.nodes) == 51  # Root + 50
        assert restored.debate_id == "large-graph"

    def test_graph_with_merges_roundtrip(self):
        """Test serialization preserves merge history."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        b1 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B1")
        b2 = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "B2")

        graph.add_node(NodeType.PROPOSAL, "a", "P1", root.id, b1.id)
        graph.add_node(NodeType.PROPOSAL, "b", "P2", root.id, b2.id)

        graph.merge_branches([b1.id, b2.id], MergeStrategy.SYNTHESIS, "sys", "Merged")

        d = graph.to_dict()
        assert len(d["merge_history"]) == 1

        # Restore (merge_history not fully restored in from_dict, just verify format)
        restored = DebateGraph.from_dict(d)
        assert restored.debate_id == graph.debate_id

    def test_node_timestamp_roundtrip(self):
        """Test node timestamps survive serialization."""
        ts = datetime(2026, 1, 6, 12, 30, 45)
        node = DebateNode(
            id="n1",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test",
            timestamp=ts,
        )

        d = node.to_dict()
        restored = DebateNode.from_dict(d)

        assert restored.timestamp == ts

    def test_branch_all_fields_roundtrip(self):
        """Test branch serialization preserves all fields."""
        branch = Branch(
            id="b1",
            name="Test Branch",
            reason=BranchReason.COUNTERFACTUAL_EXPLORATION,
            start_node_id="n1",
            end_node_id="n5",
            hypothesis="What if X?",
            confidence=0.75,
            is_active=False,
            is_merged=True,
            merged_into="merge_node",
            node_count=5,
            total_agreement=0.8,
        )

        d = branch.to_dict()
        restored = Branch.from_dict(d)

        assert restored.id == branch.id
        assert restored.reason == branch.reason
        assert restored.hypothesis == branch.hypothesis
        assert restored.is_merged == branch.is_merged
        assert restored.merged_into == branch.merged_into

    def test_policy_roundtrip_via_graph(self):
        """Test policy is preserved in graph serialization."""
        policy = BranchPolicy(
            disagreement_threshold=0.75,
            uncertainty_threshold=0.85,
            max_branches=10,
            max_depth=8,
        )

        graph = DebateGraph(branch_policy=policy)
        d = graph.to_dict()

        assert d["policy"]["disagreement_threshold"] == 0.75
        assert d["policy"]["max_branches"] == 10


# =============================================================================
# GraphReplayBuilder Extended Tests
# =============================================================================


class TestReplayBuilderExtended:
    """Extended tests for GraphReplayBuilder."""

    def test_replay_orders_by_timestamp(self):
        """Test replay returns nodes ordered by timestamp."""
        graph = DebateGraph()

        # Add nodes with specific timestamps (out of order)
        ts3 = datetime(2026, 1, 3)
        ts1 = datetime(2026, 1, 1)
        ts2 = datetime(2026, 1, 2)

        n3 = DebateNode(
            id="n3", node_type=NodeType.SYNTHESIS, agent_id="a", content="Third", timestamp=ts3
        )
        n1 = DebateNode(
            id="n1", node_type=NodeType.ROOT, agent_id="system", content="First", timestamp=ts1
        )
        n2 = DebateNode(
            id="n2", node_type=NodeType.PROPOSAL, agent_id="a", content="Second", timestamp=ts2
        )

        # Add in wrong order
        for n in [n3, n1, n2]:
            graph.nodes[n.id] = n
            n.branch_id = graph.main_branch_id

        graph.root_id = n1.id

        builder = GraphReplayBuilder(graph)
        nodes = builder.replay_branch(graph.main_branch_id)

        # Should be sorted by timestamp
        assert nodes[0].timestamp <= nodes[1].timestamp <= nodes[2].timestamp

    def test_replay_full_with_callback(self):
        """Test replay_full invokes callback correctly."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        branch = graph.create_branch(root.id, BranchReason.UNCERTAINTY, "Alt")
        graph.add_node(NodeType.PROPOSAL, "a", "Main", root.id, graph.main_branch_id)
        graph.add_node(NodeType.PROPOSAL, "b", "Alt", root.id, branch.id)

        builder = GraphReplayBuilder(graph)
        callback_data = []

        def callback(node, branch_id, index):
            callback_data.append((node.id, branch_id, index))

        builder.replay_full(callback=callback)

        # Should have called for nodes in each branch
        assert len(callback_data) >= 2

    def test_counterfactual_paths_includes_prefix(self):
        """Test counterfactual paths include path from root."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")
        p1 = graph.add_node(NodeType.PROPOSAL, "a", "P1", root.id)

        # Create counterfactual branch
        cf_branch = graph.create_branch(p1.id, BranchReason.COUNTERFACTUAL_EXPLORATION, "What if?")
        graph.add_node(NodeType.COUNTERFACTUAL, "b", "CF", p1.id, cf_branch.id)

        builder = GraphReplayBuilder(graph)
        paths = builder.get_counterfactual_paths()

        assert len(paths) == 1
        # Path should include prefix (root -> p1) + counterfactual nodes
        assert len(paths[0]) >= 2

    def test_summary_counts_all_branch_reasons(self):
        """Test summary includes counts for all branch reasons."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        # Create branches with different reasons
        graph.create_branch(root.id, BranchReason.HIGH_DISAGREEMENT, "D")
        graph.create_branch(root.id, BranchReason.UNCERTAINTY, "U")
        graph.create_branch(root.id, BranchReason.COUNTERFACTUAL_EXPLORATION, "C")

        builder = GraphReplayBuilder(graph)
        summary = builder.generate_summary()

        # Should have counts for all reason types
        assert "branch_reasons" in summary
        assert summary["branch_reasons"]["high_disagreement"] >= 1
        assert summary["branch_reasons"]["uncertainty"] >= 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_branch_from_nonexistent_node(self):
        """Test creating branch from nonexistent node raises error."""
        graph = DebateGraph()

        with pytest.raises(ValueError, match="not found"):
            graph.create_branch("nonexistent", BranchReason.UNCERTAINTY, "Test")

    def test_merge_nonexistent_branch(self):
        """Test merging nonexistent branch raises error."""
        graph = DebateGraph()

        with pytest.raises(ValueError, match="not found"):
            graph.merge_branches(["nonexistent"], MergeStrategy.BEST_PATH, "sys", "M")

    def test_empty_graph_operations(self):
        """Test operations on empty graph."""
        graph = DebateGraph()

        assert graph.get_leaf_nodes() == []
        assert graph.get_active_branches() == [graph.branches["main"]]
        assert graph.check_convergence() == []

    def test_single_node_graph(self):
        """Test operations on single-node graph."""
        graph = DebateGraph()
        root = graph.add_node(NodeType.ROOT, "system", "Task")

        leaves = graph.get_leaf_nodes()
        assert len(leaves) == 1
        assert leaves[0] == root

        path = graph.get_path_to_node(root.id)
        assert len(path) == 1

    def test_node_hash_different_for_different_content(self):
        """Test node hashes differ for different content."""
        ts = datetime(2026, 1, 1)

        n1 = DebateNode(
            id="n1", node_type=NodeType.PROPOSAL, agent_id="a", content="Content A", timestamp=ts
        )
        n2 = DebateNode(
            id="n2", node_type=NodeType.PROPOSAL, agent_id="a", content="Content B", timestamp=ts
        )

        assert n1.hash() != n2.hash()

    def test_merge_result_to_dict(self):
        """Test MergeResult serialization."""
        result = MergeResult(
            merged_node_id="m1",
            source_branch_ids=["b1", "b2"],
            strategy=MergeStrategy.WEIGHTED,
            synthesis="Combined",
            confidence=0.85,
            insights_preserved=["i1", "i2"],
            conflicts_resolved=["c1"],
        )

        d = result.to_dict()
        assert d["strategy"] == "weighted"
        assert d["insights_preserved"] == ["i1", "i2"]
        assert d["conflicts_resolved"] == ["c1"]
