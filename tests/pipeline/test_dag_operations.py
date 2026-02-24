"""Tests for DAGOperationsCoordinator."""

from __future__ import annotations

import pytest

from aragora.canvas.stages import PipelineStage, StageEdgeType
from aragora.pipeline.dag_operations import DAGOperationsCoordinator, DAGOperationResult
from aragora.pipeline.universal_node import UniversalEdge, UniversalGraph, UniversalNode


def _make_graph() -> UniversalGraph:
    """Create a test graph with some nodes."""
    graph = UniversalGraph(id="test-graph", name="Test")
    graph.add_node(UniversalNode(
        id="idea-1",
        stage=PipelineStage.IDEAS,
        node_subtype="concept",
        label="Build a rate limiter",
        description="Implement token bucket rate limiting for API gateway",
    ))
    graph.add_node(UniversalNode(
        id="idea-2",
        stage=PipelineStage.IDEAS,
        node_subtype="concept",
        label="Add caching layer",
        description="Add Redis caching for frequently accessed endpoints",
    ))
    graph.add_node(UniversalNode(
        id="orch-1",
        stage=PipelineStage.ORCHESTRATION,
        node_subtype="agent_task",
        label="Execute rate limiter",
        description="Agent task to implement rate limiter",
        parent_ids=["idea-1"],
    ))
    return graph


class TestDAGOperationsCoordinator:
    """Test DAGOperationsCoordinator methods."""

    @pytest.mark.asyncio
    async def test_decompose_node_not_found(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)
        result = await coord.decompose_node("nonexistent")
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_decompose_node_creates_children(self, monkeypatch):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)

        # Mock TaskDecomposer at the import target
        from aragora.nomic.task_decomposer import SubTask as RealSubTask

        class MockDecomposition:
            should_decompose = True
            complexity_score = 7
            subtasks = [
                RealSubTask(
                    id="sub1",
                    title="Design rate limiter",
                    description="Design the token bucket algorithm",
                ),
                RealSubTask(
                    id="sub2",
                    title="Implement rate limiter",
                    description="Code the rate limiter middleware",
                ),
            ]

        class MockDecomposer:
            def analyze(self, task, **kwargs):
                return MockDecomposition()

        monkeypatch.setattr(
            "aragora.nomic.task_decomposer.TaskDecomposer",
            MockDecomposer,
        )

        result = await coord.decompose_node("idea-1")
        assert result.success
        assert len(result.created_nodes) == 2
        assert "2 subtasks" in result.message

        # Verify children are in graph
        for nid in result.created_nodes:
            assert nid in graph.nodes
            child = graph.nodes[nid]
            assert "idea-1" in child.parent_ids

    @pytest.mark.asyncio
    async def test_cluster_ideas(self):
        graph = UniversalGraph(id="cluster-test")
        coord = DAGOperationsCoordinator(graph)

        result = await coord.cluster_ideas([
            "Build a rate limiter for the API",
            "Implement API rate limiting with tokens",
            "Create user documentation",
        ])
        assert result.success
        assert len(result.created_nodes) >= 3  # At least 3 idea nodes
        assert result.metadata["cluster_count"] >= 1

        # Check nodes were added to graph
        ideas_in_graph = [
            n for n in graph.nodes.values()
            if n.stage == PipelineStage.IDEAS
        ]
        assert len(ideas_in_graph) >= 3

    @pytest.mark.asyncio
    async def test_cluster_ideas_empty(self):
        graph = UniversalGraph(id="empty-test")
        coord = DAGOperationsCoordinator(graph)

        result = await coord.cluster_ideas([])
        assert not result.success
        assert "No ideas" in result.message

    @pytest.mark.asyncio
    async def test_assign_agents_not_found(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)

        # assign_agents with nonexistent node_ids just skips them
        result = await coord.assign_agents(["nonexistent"])
        assert result.success
        assert result.metadata["assignments"] == {}

    @pytest.mark.asyncio
    async def test_execute_node_wrong_stage(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)

        result = await coord.execute_node("idea-1")
        assert not result.success
        assert "orchestration" in result.message.lower()

    @pytest.mark.asyncio
    async def test_find_precedents_not_found(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)

        result = await coord.find_precedents("nonexistent")
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_debate_node_not_found(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)

        result = await coord.debate_node("nonexistent")
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_prioritize_no_children(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)

        result = await coord.prioritize_children("idea-2")
        assert result.success
        assert "No children" in result.message

    @pytest.mark.asyncio
    async def test_auto_flow(self):
        graph = UniversalGraph(id="flow-test")
        coord = DAGOperationsCoordinator(graph)

        result = await coord.auto_flow([
            "Build a rate limiter",
            "Add response caching",
            "Create API documentation",
        ])
        assert result.success
        assert result.metadata["ideas"] == 3
        assert result.metadata["goals"] >= 1

        # Verify graph has nodes across multiple stages
        stages = {n.stage for n in graph.nodes.values()}
        assert PipelineStage.IDEAS in stages
        assert PipelineStage.GOALS in stages

    @pytest.mark.asyncio
    async def test_save_called_on_operations(self):
        """Verify _save is called when store is provided."""
        graph = UniversalGraph(id="save-test")
        mock_store = type("MockStore", (), {"update": lambda self, g: None})()
        coord = DAGOperationsCoordinator(graph, store=mock_store)

        # cluster_ideas should call _save
        save_called = []
        mock_store.update = lambda g: save_called.append(True)

        await coord.cluster_ideas(["test idea"])
        assert len(save_called) > 0


class TestDAGOperationResult:
    def test_defaults(self):
        r = DAGOperationResult(success=True, message="ok")
        assert r.created_nodes == []
        assert r.metadata == {}

    def test_with_data(self):
        r = DAGOperationResult(
            success=True,
            message="done",
            created_nodes=["a", "b"],
            metadata={"count": 2},
        )
        assert len(r.created_nodes) == 2
        assert r.metadata["count"] == 2
