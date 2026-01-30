"""
Tests for the graph-based debate orchestrator module.

Tests cover:
1. Graph-based debate flow
2. Graph construction from debate state
3. Node operations
4. Cycle detection in debate flow
5. Node/edge weight calculations
6. Path finding for argument chains
7. Visualization export formats
8. Integration with standard orchestrator
9. Performance with complex debates
10. Error handling edge cases
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.graph import (
    Branch,
    BranchPolicy as GraphBranchPolicy,
    BranchReason,
    ConvergenceScorer,
    DebateGraph,
    DebateNode,
    GraphReplayBuilder,
    MergeResult,
    MergeStrategy,
    NodeType,
)
from aragora.debate.graph_orchestrator import (
    BranchPolicy,
    GraphBranch,
    GraphDebateOrchestrator,
    GraphDebateResult,
    GraphNode,
)


# ==============================================================================
# Mock Agent for Testing
# ==============================================================================


class MockGraphAgent:
    """Mock agent for graph debate testing."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "Test response with 85% confidence",
        confidence: float = 0.85,
    ):
        self.name = name
        self.response = response
        self.confidence = confidence
        self.generate_calls = 0

    async def generate(self, prompt: str, context: Any = None) -> str:
        self.generate_calls += 1
        return self.response


class DisagreeingAgent:
    """Agent that disagrees with others to trigger branching."""

    def __init__(
        self,
        name: str = "disagreeing-agent",
        disagreement_markers: list[str] | None = None,
    ):
        self.name = name
        self.disagreement_markers = disagreement_markers or [
            "However, I disagree",
            "On the contrary",
            "This is incorrect",
        ]
        self._call_count = 0
        self.generate_calls = 0

    async def generate(self, prompt: str, context: Any = None) -> str:
        self.generate_calls += 1
        self._call_count += 1
        marker = self.disagreement_markers[self._call_count % len(self.disagreement_markers)]
        return f"{marker}: Alternative viewpoint {self._call_count}"


class ConvergingAgent:
    """Agent that converges responses over time."""

    def __init__(self, name: str = "converging-agent", shared_claim: str = "API response"):
        self.name = name
        self.shared_claim = shared_claim
        self.generate_calls = 0

    async def generate(self, prompt: str, context: Any = None) -> str:
        self.generate_calls += 1
        return f"I agree that {self.shared_claim} is correct. Confidence: 90%"


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_agents() -> list[MockGraphAgent]:
    """Create a list of mock agents."""
    return [
        MockGraphAgent(name="claude", response="Claude's proposal with 80% confidence"),
        MockGraphAgent(name="gpt4", response="GPT-4's proposal with 85% confidence"),
        MockGraphAgent(name="gemini", response="Gemini's proposal with 75% confidence"),
    ]


@pytest.fixture
def branch_policy() -> BranchPolicy:
    """Create a default branch policy."""
    return BranchPolicy(
        min_disagreement=0.5,
        max_branches=4,
        auto_merge=True,
        merge_strategy=MergeStrategy.SYNTHESIS,
        min_confidence_to_branch=0.3,
        convergence_threshold=0.8,
    )


@pytest.fixture
def orchestrator(mock_agents: list[MockGraphAgent], branch_policy: BranchPolicy):
    """Create a GraphDebateOrchestrator instance."""
    return GraphDebateOrchestrator(agents=mock_agents, policy=branch_policy)


@pytest.fixture
def debate_graph() -> DebateGraph:
    """Create a DebateGraph with some nodes."""
    graph = DebateGraph(debate_id="test-graph")

    # Add root node
    root = graph.add_node(
        node_type=NodeType.ROOT,
        agent_id="system",
        content="Test debate topic",
        confidence=1.0,
    )

    # Add proposal nodes
    node1 = graph.add_node(
        node_type=NodeType.PROPOSAL,
        agent_id="claude",
        content="First proposal with claims",
        parent_id=root.id,
        claims=["claim1", "claim2"],
        confidence=0.8,
    )

    node2 = graph.add_node(
        node_type=NodeType.CRITIQUE,
        agent_id="gpt4",
        content="Critique of proposal",
        parent_id=node1.id,
        claims=["claim2", "claim3"],
        confidence=0.75,
    )

    return graph


# ==============================================================================
# 1. Graph-based debate flow tests
# ==============================================================================


class TestGraphDebateFlow:
    """Tests for graph-based debate flow."""

    @pytest.mark.asyncio
    async def test_basic_debate_flow(self, orchestrator: GraphDebateOrchestrator):
        """Test basic debate flow completes successfully."""
        result = await orchestrator.run_debate(
            task="Design a rate limiter",
            max_rounds=2,
        )

        assert result is not None
        assert isinstance(result, GraphDebateResult)
        assert result.task == "Design a rate limiter"
        assert result.total_rounds > 0

    @pytest.mark.asyncio
    async def test_debate_with_custom_run_agent_fn(self, orchestrator: GraphDebateOrchestrator):
        """Test debate with custom agent run function."""
        call_count = 0

        async def custom_run_agent(agent, prompt, context):
            nonlocal call_count
            call_count += 1
            return f"Custom response from {agent.name}"

        result = await orchestrator.run_debate(
            task="Test task",
            max_rounds=1,
            run_agent_fn=custom_run_agent,
        )

        assert result is not None
        assert call_count > 0

    @pytest.mark.asyncio
    async def test_debate_with_context(self, orchestrator: GraphDebateOrchestrator):
        """Test debate with provided context."""
        context = {"background": "API design principles"}

        async def context_aware_run_agent(agent, prompt, context):
            return f"Response considering {context.get('background', 'none')}"

        result = await orchestrator.run_debate(
            task="Design API",
            max_rounds=1,
            run_agent_fn=context_aware_run_agent,
            context=context,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_events_emitted(self, mock_agents: list[MockGraphAgent]):
        """Test that debate events are emitted correctly."""
        events = []

        def event_callback(event_type: str, data: dict):
            events.append((event_type, data))

        orchestrator = GraphDebateOrchestrator(
            agents=mock_agents,
            event_callback=event_callback,
        )

        await orchestrator.run_debate(task="Test task", max_rounds=1)

        event_types = [e[0] for e in events]
        assert "debate_start" in event_types
        assert "round_start" in event_types
        assert "debate_end" in event_types


# ==============================================================================
# 2. Graph construction from debate state tests
# ==============================================================================


class TestGraphConstruction:
    """Tests for graph construction from debate state."""

    def test_graph_result_contains_nodes(self, debate_graph: DebateGraph):
        """Test that the result contains all graph nodes."""
        assert len(debate_graph.nodes) == 3

        node_types = [n.node_type for n in debate_graph.nodes.values()]
        assert NodeType.ROOT in node_types
        assert NodeType.PROPOSAL in node_types
        assert NodeType.CRITIQUE in node_types

    def test_graph_result_contains_branches(self, debate_graph: DebateGraph):
        """Test that the result contains branches."""
        # Main branch exists by default
        assert "main" in debate_graph.branches
        assert debate_graph.branches["main"].is_active

    @pytest.mark.asyncio
    async def test_result_serialization(self, orchestrator: GraphDebateOrchestrator):
        """Test that GraphDebateResult serializes correctly."""
        result = await orchestrator.run_debate(task="Test", max_rounds=1)

        serialized = result.to_dict()

        assert "id" in serialized
        assert "task" in serialized
        assert "nodes" in serialized
        assert "branches" in serialized
        assert isinstance(serialized["nodes"], list)
        assert isinstance(serialized["branches"], list)


# ==============================================================================
# 3. Node operations tests
# ==============================================================================


class TestNodeOperations:
    """Tests for node operations in debate graph."""

    def test_add_node_to_graph(self, debate_graph: DebateGraph):
        """Test adding a node to the graph."""
        initial_count = len(debate_graph.nodes)

        node = debate_graph.add_node(
            node_type=NodeType.SYNTHESIS,
            agent_id="claude",
            content="Synthesis content",
            confidence=0.9,
        )

        assert len(debate_graph.nodes) == initial_count + 1
        assert node.id in debate_graph.nodes

    def test_node_parent_child_relationship(self, debate_graph: DebateGraph):
        """Test parent-child relationships between nodes."""
        parent_node = debate_graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Parent content",
        )

        child_node = debate_graph.add_node(
            node_type=NodeType.CRITIQUE,
            agent_id="gpt4",
            content="Child content",
            parent_id=parent_node.id,
        )

        assert child_node.parent_ids == [parent_node.id]
        assert child_node.id in debate_graph.nodes[parent_node.id].child_ids

    def test_graph_node_from_debate_node(self):
        """Test GraphNode.from_debate_node conversion."""
        debate_node = DebateNode(
            id="test-id",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test content",
            branch_id="main",
            confidence=0.85,
            claims=["claim1"],
        )

        graph_node = GraphNode.from_debate_node(debate_node, round_num=2)

        assert graph_node.id == "test-id"
        assert graph_node.agent == "claude"
        assert graph_node.round_num == 2
        assert graph_node.node_type == "proposal"
        assert graph_node.confidence == 0.85

    def test_graph_node_serialization(self):
        """Test GraphNode serialization to dict."""
        node = GraphNode(
            id="node-1",
            content="Test content",
            agent="claude",
            round_num=1,
            branch_id="main",
            confidence=0.8,
            metadata={"key": "value"},
        )

        serialized = node.to_dict()

        assert serialized["id"] == "node-1"
        assert serialized["agent"] == "claude"
        assert serialized["confidence"] == 0.8
        assert "timestamp" in serialized


# ==============================================================================
# 4. Cycle detection in debate flow tests
# ==============================================================================


class TestCycleDetection:
    """Tests for cycle detection in debate graph."""

    def test_path_to_node_avoids_cycles(self, debate_graph: DebateGraph):
        """Test that get_path_to_node handles cycles safely."""
        # Get a leaf node
        leaf_nodes = debate_graph.get_leaf_nodes()
        assert len(leaf_nodes) > 0

        # Get path to leaf - should not infinite loop
        path = debate_graph.get_path_to_node(leaf_nodes[0].id)
        assert len(path) > 0

        # Check no duplicates (cycle detection working)
        node_ids = [n.id for n in path]
        assert len(node_ids) == len(set(node_ids))

    def test_graph_with_potential_cycle(self):
        """Test graph handles potential cycle creation safely."""
        graph = DebateGraph(debate_id="cycle-test")

        node1 = graph.add_node(
            node_type=NodeType.ROOT,
            agent_id="system",
            content="Root",
        )

        node2 = graph.add_node(
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Proposal",
            parent_id=node1.id,
        )

        # Path should work correctly
        path = graph.get_path_to_node(node2.id)
        assert len(path) == 2
        assert path[0].id == node1.id
        assert path[1].id == node2.id


# ==============================================================================
# 5. Node/edge weight calculations tests
# ==============================================================================


class TestWeightCalculations:
    """Tests for node and edge weight calculations."""

    def test_disagreement_calculation(self, orchestrator: GraphDebateOrchestrator):
        """Test disagreement score calculation."""
        responses = {
            "claude": "I agree with the approach",
            "gpt4": "However, I disagree strongly",
            "gemini": "This is incorrect because...",
        }

        disagreement = orchestrator._calculate_disagreement(responses)

        assert 0.0 <= disagreement <= 1.0
        # Should have some disagreement due to markers
        assert disagreement > 0.0

    def test_no_disagreement_with_agreeing_responses(self, orchestrator: GraphDebateOrchestrator):
        """Test disagreement is low when all agents agree."""
        responses = {
            "claude": "This approach is correct",
            "gpt4": "This approach is correct",
        }

        disagreement = orchestrator._calculate_disagreement(responses)

        # Should be zero or very low
        assert disagreement < 0.3

    def test_confidence_extraction(self):
        """Test confidence extraction from responses."""
        # Using the graph.py orchestrator's method
        from aragora.debate.graph import GraphDebateOrchestrator as GraphOrc

        policy = GraphBranchPolicy()
        orch = GraphOrc(agents=[], policy=policy)

        # Test explicit confidence
        conf = orch._extract_confidence("I am 85% confident")
        assert conf == 0.85

        # Test high confidence words
        conf = orch._extract_confidence("I am certain this is correct")
        assert conf == 0.8

        # Test low confidence words
        conf = orch._extract_confidence("Perhaps this might work")
        assert conf == 0.4


# ==============================================================================
# 6. Path finding for argument chains tests
# ==============================================================================


class TestPathFinding:
    """Tests for path finding in debate graph."""

    def test_get_path_to_node(self, debate_graph: DebateGraph):
        """Test getting path from root to a node."""
        leaf_nodes = debate_graph.get_leaf_nodes()
        assert len(leaf_nodes) > 0

        path = debate_graph.get_path_to_node(leaf_nodes[0].id)

        assert len(path) > 0
        assert path[0].node_type == NodeType.ROOT

    def test_get_path_nonexistent_node(self, debate_graph: DebateGraph):
        """Test getting path to nonexistent node returns empty list."""
        path = debate_graph.get_path_to_node("nonexistent-id")
        assert path == []

    def test_get_branch_nodes(self, debate_graph: DebateGraph):
        """Test getting all nodes in a branch."""
        main_branch_nodes = debate_graph.get_branch_nodes("main")

        assert len(main_branch_nodes) > 0
        # All nodes should be in main branch
        for node in main_branch_nodes:
            assert node.branch_id == "main"

    def test_get_leaf_nodes(self, debate_graph: DebateGraph):
        """Test getting leaf nodes (no children)."""
        leaf_nodes = debate_graph.get_leaf_nodes()

        assert len(leaf_nodes) >= 1
        # Leaf nodes have no children
        for node in leaf_nodes:
            assert len(node.child_ids) == 0


# ==============================================================================
# 7. Visualization export formats tests
# ==============================================================================


class TestVisualizationExport:
    """Tests for visualization export formats."""

    def test_graph_to_dict(self, debate_graph: DebateGraph):
        """Test graph serialization to dictionary."""
        serialized = debate_graph.to_dict()

        assert "debate_id" in serialized
        assert "nodes" in serialized
        assert "branches" in serialized
        assert "policy" in serialized
        assert serialized["debate_id"] == "test-graph"

    def test_graph_from_dict_roundtrip(self, debate_graph: DebateGraph):
        """Test graph serialization and deserialization roundtrip."""
        serialized = debate_graph.to_dict()
        restored = DebateGraph.from_dict(serialized)

        assert restored.debate_id == debate_graph.debate_id
        assert len(restored.nodes) == len(debate_graph.nodes)
        assert len(restored.branches) == len(debate_graph.branches)

    def test_node_to_dict(self):
        """Test DebateNode serialization."""
        node = DebateNode(
            id="test-node",
            node_type=NodeType.PROPOSAL,
            agent_id="claude",
            content="Test content",
            confidence=0.9,
            claims=["claim1", "claim2"],
        )

        serialized = node.to_dict()

        assert serialized["id"] == "test-node"
        assert serialized["node_type"] == "proposal"
        assert serialized["agent_id"] == "claude"
        assert "hash" in serialized

    def test_merge_result_to_dict(self):
        """Test MergeResult serialization."""
        result = MergeResult(
            merged_node_id="merge-1",
            source_branch_ids=["branch-a", "branch-b"],
            strategy=MergeStrategy.SYNTHESIS,
            synthesis="Combined insights",
            confidence=0.85,
            insights_preserved=["insight1", "insight2"],
        )

        serialized = result.to_dict()

        assert serialized["merged_node_id"] == "merge-1"
        assert serialized["strategy"] == "synthesis"
        assert len(serialized["source_branch_ids"]) == 2


# ==============================================================================
# 8. Integration with standard orchestrator tests
# ==============================================================================


class TestOrchestratorIntegration:
    """Tests for integration with standard orchestrator patterns."""

    @pytest.mark.asyncio
    async def test_event_emitter_integration(self, mock_agents: list[MockGraphAgent]):
        """Test integration with event emitter."""
        mock_emitter = MagicMock()
        mock_emitter.emit = MagicMock()

        orchestrator = GraphDebateOrchestrator(agents=mock_agents)

        await orchestrator.run_debate(
            task="Test",
            max_rounds=1,
            event_emitter=mock_emitter,
        )

        # Event emitter should have been called
        assert mock_emitter.emit.called

    @pytest.mark.asyncio
    async def test_policy_integration(self, mock_agents: list[MockGraphAgent]):
        """Test that branch policy is respected."""
        strict_policy = BranchPolicy(
            min_disagreement=0.9,  # Very high threshold
            max_branches=2,
            auto_merge=False,
        )

        orchestrator = GraphDebateOrchestrator(
            agents=mock_agents,
            policy=strict_policy,
        )

        result = await orchestrator.run_debate(task="Test", max_rounds=2)

        # With high threshold, shouldn't create many branches
        assert len(result.branches) <= 2

    @pytest.mark.asyncio
    async def test_synthesis_generation(self, mock_agents: list[MockGraphAgent]):
        """Test that final synthesis is generated."""
        orchestrator = GraphDebateOrchestrator(agents=mock_agents)

        result = await orchestrator.run_debate(task="Design API", max_rounds=2)

        # Synthesis should be generated
        assert result.synthesis is not None


# ==============================================================================
# 9. Performance with complex debates tests
# ==============================================================================


class TestPerformance:
    """Tests for performance with complex debates."""

    @pytest.mark.asyncio
    async def test_many_agents_performance(self):
        """Test performance with many agents."""
        agents = [MockGraphAgent(name=f"agent-{i}", response=f"Response {i}") for i in range(10)]

        orchestrator = GraphDebateOrchestrator(agents=agents)

        start = time.time()
        result = await orchestrator.run_debate(task="Complex task", max_rounds=2)
        elapsed = time.time() - start

        assert result is not None
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_many_rounds_performance(self, mock_agents: list[MockGraphAgent]):
        """Test performance with many debate rounds."""
        orchestrator = GraphDebateOrchestrator(agents=mock_agents)

        start = time.time()
        result = await orchestrator.run_debate(task="Extended debate", max_rounds=5)
        elapsed = time.time() - start

        assert result is not None
        assert result.total_rounds <= 5

    def test_graph_cache_invalidation(self, debate_graph: DebateGraph):
        """Test that caching works correctly on graph operations."""
        # First call populates cache
        leaf1 = debate_graph.get_leaf_nodes()

        # Add a new node
        new_node = debate_graph.add_node(
            node_type=NodeType.SYNTHESIS,
            agent_id="test",
            content="New node",
        )

        # Cache should be invalidated, new call should include new node
        leaf2 = debate_graph.get_leaf_nodes()

        # New node should be in leaf nodes (since it has no children)
        leaf_ids = [n.id for n in leaf2]
        assert new_node.id in leaf_ids


# ==============================================================================
# 10. Error handling edge cases tests
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    @pytest.mark.asyncio
    async def test_empty_agents_list(self):
        """Test handling of empty agents list."""
        orchestrator = GraphDebateOrchestrator(agents=[])

        result = await orchestrator.run_debate(task="Test", max_rounds=1)

        # Should not crash, but result may be minimal
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_exception_propagates(self):
        """Test that agent exceptions propagate (not silently swallowed)."""

        class FailingAgent:
            name = "failing"

            async def generate(self, prompt: str, context: Any = None) -> str:
                raise RuntimeError("Agent failed")

        orchestrator = GraphDebateOrchestrator(agents=[FailingAgent()])

        # Exception should propagate - this is expected behavior
        # (caller should handle agent failures)
        with pytest.raises(RuntimeError, match="Agent failed"):
            await orchestrator.run_debate(task="Test", max_rounds=1)

    @pytest.mark.asyncio
    async def test_agent_with_run_agent_fn_exception_handling(self):
        """Test handling agent exceptions via custom run_agent_fn."""
        agents = [MockGraphAgent(name="test")]
        orchestrator = GraphDebateOrchestrator(agents=agents)

        call_count = 0

        async def run_agent_with_fallback(agent, prompt, context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First call fails")
            return f"Fallback response from {agent.name}"

        # With a custom run_agent_fn that handles errors, should work
        # Here we simulate the pattern where run_agent_fn handles its own errors
        async def safe_run_agent(agent, prompt, context):
            try:
                return await run_agent_with_fallback(agent, prompt, context)
            except RuntimeError:
                return f"Error handled for {agent.name}"

        result = await orchestrator.run_debate(
            task="Test",
            max_rounds=1,
            run_agent_fn=safe_run_agent,
        )
        assert result is not None

    def test_invalid_branch_creation(self, debate_graph: DebateGraph):
        """Test creating branch from nonexistent node raises error."""
        with pytest.raises(ValueError):
            debate_graph.create_branch(
                from_node_id="nonexistent-id",
                reason=BranchReason.HIGH_DISAGREEMENT,
                name="test-branch",
            )

    def test_invalid_branch_merge(self, debate_graph: DebateGraph):
        """Test merging nonexistent branches raises error."""
        with pytest.raises(ValueError):
            debate_graph.merge_branches(
                branch_ids=["nonexistent-a", "nonexistent-b"],
                strategy=MergeStrategy.SYNTHESIS,
                synthesizer_agent_id="test",
                synthesis_content="Test",
            )

    def test_event_callback_exception_handling(self, mock_agents: list[MockGraphAgent]):
        """Test that event callback exceptions are handled gracefully."""

        def failing_callback(event_type: str, data: dict):
            raise RuntimeError("Callback failed")

        orchestrator = GraphDebateOrchestrator(
            agents=mock_agents,
            event_callback=failing_callback,
        )

        # Should not crash despite callback failure
        # The callback failure is logged but doesn't stop the debate
        orchestrator._emit_event("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_zero_rounds(self, mock_agents: list[MockGraphAgent]):
        """Test handling of zero rounds."""
        orchestrator = GraphDebateOrchestrator(agents=mock_agents)

        result = await orchestrator.run_debate(task="Test", max_rounds=0)

        assert result is not None
        assert result.total_rounds == 0


# ==============================================================================
# Additional tests for BranchPolicy
# ==============================================================================


class TestBranchPolicy:
    """Tests for BranchPolicy decision making."""

    def test_should_branch_high_disagreement(self):
        """Test branching trigger on high disagreement."""
        policy = BranchPolicy(min_disagreement=0.5, max_branches=4)

        should_branch, reason = policy.should_branch(
            disagreement=0.7,
            current_branches=1,
        )

        assert should_branch is True
        assert reason == BranchReason.HIGH_DISAGREEMENT

    def test_should_branch_max_branches_reached(self):
        """Test no branching when max branches reached."""
        policy = BranchPolicy(max_branches=2)

        should_branch, reason = policy.should_branch(
            disagreement=0.9,
            current_branches=2,
        )

        assert should_branch is False
        assert reason is None

    def test_should_branch_alternative_confidence(self):
        """Test branching on alternative confidence."""
        policy = BranchPolicy(
            min_disagreement=0.9,  # High threshold
            min_confidence_to_branch=0.4,
        )

        should_branch, reason = policy.should_branch(
            disagreement=0.3,  # Low disagreement
            current_branches=1,
            alternative_confidence=0.5,  # Above threshold
        )

        assert should_branch is True
        assert reason == BranchReason.ALTERNATIVE_APPROACH


# ==============================================================================
# Tests for GraphBranch
# ==============================================================================


class TestGraphBranch:
    """Tests for GraphBranch dataclass."""

    def test_graph_branch_creation(self):
        """Test creating a GraphBranch."""
        branch = GraphBranch(
            id="branch-1",
            name="Alternative Path",
            start_node_id="node-1",
            hypothesis="What if we try X instead?",
            is_active=True,
            is_merged=False,
        )

        assert branch.id == "branch-1"
        assert branch.name == "Alternative Path"
        assert branch.is_active is True

    def test_graph_branch_to_dict(self):
        """Test GraphBranch serialization."""
        branch = GraphBranch(
            id="branch-1",
            name="Test Branch",
            start_node_id="node-1",
        )

        serialized = branch.to_dict()

        assert serialized["id"] == "branch-1"
        assert serialized["name"] == "Test Branch"
        assert serialized["is_active"] is True


# ==============================================================================
# Tests for GraphReplayBuilder
# ==============================================================================


class TestGraphReplayBuilder:
    """Tests for GraphReplayBuilder functionality."""

    def test_replay_branch(self, debate_graph: DebateGraph):
        """Test replaying a branch."""
        builder = GraphReplayBuilder(debate_graph)

        nodes = builder.replay_branch("main")

        assert len(nodes) > 0
        # Nodes should be sorted by timestamp
        for i in range(len(nodes) - 1):
            assert nodes[i].timestamp <= nodes[i + 1].timestamp

    def test_replay_branch_with_callback(self, debate_graph: DebateGraph):
        """Test replaying branch with callback."""
        builder = GraphReplayBuilder(debate_graph)
        callback_calls = []

        def callback(node: DebateNode, index: int):
            callback_calls.append((node.id, index))

        nodes = builder.replay_branch("main", callback=callback)

        assert len(callback_calls) == len(nodes)

    def test_replay_full(self, debate_graph: DebateGraph):
        """Test replaying full graph."""
        builder = GraphReplayBuilder(debate_graph)

        result = builder.replay_full()

        assert "main" in result
        assert len(result["main"]) > 0

    def test_generate_summary(self, debate_graph: DebateGraph):
        """Test generating graph summary."""
        builder = GraphReplayBuilder(debate_graph)

        summary = builder.generate_summary()

        assert "debate_id" in summary
        assert "total_nodes" in summary
        assert "total_branches" in summary
        assert "active_branches" in summary
        assert "agents" in summary


# ==============================================================================
# Tests for ConvergenceScorer
# ==============================================================================


class TestConvergenceScorer:
    """Tests for ConvergenceScorer functionality."""

    def test_score_convergence_with_shared_claims(self):
        """Test convergence scoring with shared claims."""
        scorer = ConvergenceScorer(threshold=0.7)

        branch_a = Branch(
            id="a",
            name="A",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="1",
        )
        branch_b = Branch(
            id="b",
            name="B",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="2",
        )

        nodes_a = [
            DebateNode(
                id="n1",
                node_type=NodeType.PROPOSAL,
                agent_id="claude",
                content="Content A",
                claims=["claim1", "claim2"],
                confidence=0.8,
            )
        ]
        nodes_b = [
            DebateNode(
                id="n2",
                node_type=NodeType.PROPOSAL,
                agent_id="gpt4",
                content="Content B",
                claims=["claim1", "claim3"],
                confidence=0.7,
            )
        ]

        score = scorer.score_convergence(branch_a, branch_b, nodes_a, nodes_b)

        assert 0.0 <= score <= 1.0

    def test_score_convergence_empty_nodes(self):
        """Test convergence scoring with empty nodes."""
        scorer = ConvergenceScorer()

        branch_a = Branch(
            id="a",
            name="A",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="1",
        )
        branch_b = Branch(
            id="b",
            name="B",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="2",
        )

        score = scorer.score_convergence(branch_a, branch_b, [], [])

        assert score == 0.0

    def test_should_merge(self):
        """Test should_merge decision."""
        scorer = ConvergenceScorer(threshold=0.5)

        branch_a = Branch(
            id="a",
            name="A",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="1",
        )
        branch_b = Branch(
            id="b",
            name="B",
            reason=BranchReason.HIGH_DISAGREEMENT,
            start_node_id="2",
        )

        # Same claims should merge
        nodes_a = [
            DebateNode(
                id="n1",
                node_type=NodeType.PROPOSAL,
                agent_id="claude",
                content="Same content",
                claims=["claim1"],
                confidence=0.8,
            )
        ]
        nodes_b = [
            DebateNode(
                id="n2",
                node_type=NodeType.PROPOSAL,
                agent_id="gpt4",
                content="Same content",
                claims=["claim1"],
                confidence=0.8,
            )
        ]

        result = scorer.should_merge(branch_a, branch_b, nodes_a, nodes_b)

        # With same claims and confidence, should merge
        assert result is True
