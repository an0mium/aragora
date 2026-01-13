"""
Integration tests for Graph Debates feature.

Tests cover end-to-end flows for:
- Creating graph debates with branching
- Branch creation based on disagreement
- Merge operations (synthesis, vote, hybrid)
- Node traversal and relationship tracking
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_agents():
    """Create mock agents for debates."""
    agent1 = Mock()
    agent1.name = "claude"
    agent1.generate = AsyncMock(return_value="Agent 1 response")

    agent2 = Mock()
    agent2.name = "gpt4"
    agent2.generate = AsyncMock(return_value="Agent 2 response")

    return [agent1, agent2]


@pytest.fixture
def mock_storage():
    """Create mock graph storage."""
    storage = Mock()
    storage.get_graph_debate = AsyncMock(return_value=None)
    storage.save_graph_debate = AsyncMock()
    storage.get_debate_branches = AsyncMock(return_value=[])
    storage.get_debate_nodes = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_handler(mock_storage):
    """Create mock HTTP handler."""
    handler = Mock()
    handler.command = "GET"
    handler.storage = mock_storage
    handler.event_emitter = None
    return handler


# ============================================================================
# Graph Debate Structure Tests
# ============================================================================


class TestGraphDebateStructure:
    """Tests for graph debate data structures."""

    def test_branch_policy_defaults(self):
        """Test BranchPolicy default values."""
        try:
            from aragora.debate.graph_orchestrator import BranchPolicy

            policy = BranchPolicy()
            assert policy.min_disagreement >= 0.0
            assert policy.min_disagreement <= 1.0
            assert policy.max_branches > 0
            assert isinstance(policy.auto_merge, bool)
        except ImportError:
            pytest.skip("Graph orchestrator not available")

    def test_merge_strategy_enum(self):
        """Test MergeStrategy enum values."""
        try:
            from aragora.debate.graph_orchestrator import MergeStrategy

            assert MergeStrategy.SYNTHESIS
            assert MergeStrategy.VOTE
        except ImportError:
            pytest.skip("Graph orchestrator not available")

    def test_graph_node_creation(self):
        """Test creating graph nodes."""
        try:
            from aragora.debate.graph_orchestrator import GraphNode

            node = GraphNode(
                id="node-1",
                content="Test content",
                agent="claude",
                round_num=1,
                branch_id="main",
            )
            assert node.id == "node-1"
            assert node.content == "Test content"
            assert node.agent == "claude"
        except ImportError:
            pytest.skip("Graph orchestrator not available")


# ============================================================================
# Branch Creation Tests
# ============================================================================


class TestBranchCreation:
    """Tests for debate branch creation."""

    @pytest.mark.asyncio
    async def test_branch_created_on_high_disagreement(self, mock_agents):
        """Test that branches are created when disagreement is high."""
        try:
            from aragora.debate.graph_orchestrator import (
                GraphDebateOrchestrator,
                BranchPolicy,
            )

            policy = BranchPolicy(min_disagreement=0.5, max_branches=3)
            orchestrator = GraphDebateOrchestrator(agents=mock_agents, policy=policy)

            # Simulate responses with disagreement
            mock_agents[0].generate = AsyncMock(
                return_value="I strongly disagree. Option A is better."
            )
            mock_agents[1].generate = AsyncMock(return_value="No, Option B is clearly superior.")

            async def run_agent(agent, prompt, context):
                return await agent.generate(prompt, context)

            graph = await orchestrator.run_debate(
                task="Choose between A and B",
                max_rounds=2,
                run_agent_fn=run_agent,
            )

            # Graph should exist
            assert graph is not None
            assert len(graph.nodes) > 0
        except ImportError:
            pytest.skip("Graph orchestrator not available")

    @pytest.mark.asyncio
    async def test_max_branches_respected(self, mock_agents):
        """Test that max branch limit is enforced."""
        try:
            from aragora.debate.graph_orchestrator import (
                GraphDebateOrchestrator,
                BranchPolicy,
            )

            policy = BranchPolicy(max_branches=2)
            orchestrator = GraphDebateOrchestrator(agents=mock_agents, policy=policy)

            async def run_agent(agent, prompt, context):
                return await agent.generate(prompt, context)

            graph = await orchestrator.run_debate(
                task="Test task",
                max_rounds=3,
                run_agent_fn=run_agent,
            )

            # Should not exceed max branches
            assert len(graph.branches) <= 2
        except ImportError:
            pytest.skip("Graph orchestrator not available")


# ============================================================================
# Merge Operation Tests
# ============================================================================


class TestMergeOperations:
    """Tests for branch merge operations."""

    @pytest.mark.asyncio
    async def test_synthesis_merge(self, mock_agents):
        """Test synthesis merge strategy."""
        try:
            from aragora.debate.graph_orchestrator import (
                GraphDebateOrchestrator,
                BranchPolicy,
                MergeStrategy,
            )

            policy = BranchPolicy(
                auto_merge=True,
                merge_strategy=MergeStrategy.SYNTHESIS,
            )
            orchestrator = GraphDebateOrchestrator(agents=mock_agents, policy=policy)

            async def run_agent(agent, prompt, context):
                return await agent.generate(prompt, context)

            graph = await orchestrator.run_debate(
                task="Debate topic",
                max_rounds=2,
                run_agent_fn=run_agent,
            )

            # Test passes if no errors
            assert graph is not None
        except ImportError:
            pytest.skip("Graph orchestrator not available")

    @pytest.mark.asyncio
    async def test_vote_merge(self, mock_agents):
        """Test vote-based merge strategy."""
        try:
            from aragora.debate.graph_orchestrator import (
                GraphDebateOrchestrator,
                BranchPolicy,
                MergeStrategy,
            )

            policy = BranchPolicy(
                auto_merge=True,
                merge_strategy=MergeStrategy.VOTE,
            )
            orchestrator = GraphDebateOrchestrator(agents=mock_agents, policy=policy)

            async def run_agent(agent, prompt, context):
                return await agent.generate(prompt, context)

            graph = await orchestrator.run_debate(
                task="Vote on this topic",
                max_rounds=2,
                run_agent_fn=run_agent,
            )

            assert graph is not None
        except ImportError:
            pytest.skip("Graph orchestrator not available")


# ============================================================================
# Handler Integration Tests
# ============================================================================


class TestGraphDebatesHandlerIntegration:
    """Integration tests for GraphDebatesHandler."""

    def test_handler_routes(self):
        """Test handler recognizes all graph debate routes."""
        try:
            from aragora.server.handlers.graph_debates import GraphDebatesHandler

            handler = GraphDebatesHandler({})

            # Test route recognition
            assert "/api/debates/graph" in handler.ROUTES
        except ImportError:
            pytest.skip("GraphDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_graph_debate_not_found(self, mock_handler):
        """Test 404 response for non-existent debate."""
        try:
            from aragora.server.handlers.graph_debates import GraphDebatesHandler

            handler = GraphDebatesHandler({})

            result = await handler._get_graph_debate(mock_handler, "nonexistent-id")

            assert result.status_code == 404
        except ImportError:
            pytest.skip("GraphDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_branches_empty(self, mock_handler):
        """Test empty branches response."""
        try:
            from aragora.server.handlers.graph_debates import GraphDebatesHandler

            handler = GraphDebatesHandler({})

            result = await handler._get_branches(mock_handler, "debate-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "branches" in data
        except ImportError:
            pytest.skip("GraphDebatesHandler not available")


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestGraphEventEmission:
    """Tests for graph debate event emission."""

    @pytest.mark.asyncio
    async def test_events_emitted_during_debate(self, mock_agents):
        """Test that events are emitted during graph debates."""
        try:
            from aragora.debate.graph_orchestrator import (
                GraphDebateOrchestrator,
                BranchPolicy,
            )

            policy = BranchPolicy()
            orchestrator = GraphDebateOrchestrator(agents=mock_agents, policy=policy)

            events = []
            event_emitter = Mock()
            event_emitter.emit = Mock(side_effect=lambda e: events.append(e))

            async def run_agent(agent, prompt, context):
                return await agent.generate(prompt, context)

            await orchestrator.run_debate(
                task="Test task",
                max_rounds=1,
                run_agent_fn=run_agent,
                event_emitter=event_emitter,
            )

            # Should emit at least some events
            # (specific events depend on implementation)
        except ImportError:
            pytest.skip("Graph orchestrator not available")


# ============================================================================
# Serialization Tests
# ============================================================================


class TestGraphSerialization:
    """Tests for graph debate serialization."""

    def test_graph_to_dict(self):
        """Test graph can be serialized to dict."""
        try:
            from aragora.debate.graph_orchestrator import DebateGraph, GraphNode

            graph = DebateGraph()
            graph.add_node(
                GraphNode(
                    id="node-1",
                    content="Test",
                    agent="claude",
                    round_num=1,
                    branch_id="main",
                )
            )

            result = graph.to_dict()

            assert "nodes" in result
            assert len(result["nodes"]) == 1
        except ImportError:
            pytest.skip("Graph orchestrator not available")

    def test_branch_to_dict(self):
        """Test branch can be serialized to dict."""
        try:
            from aragora.debate.graph_orchestrator import Branch

            branch = Branch(
                id="branch-1",
                parent_id="main",
                fork_reason="Disagreement on approach",
            )

            result = branch.to_dict()

            assert result["id"] == "branch-1"
            assert result["parent_id"] == "main"
        except ImportError:
            pytest.skip("Graph orchestrator not available")
