"""Tests for DAGOperationsCoordinator.

Covers: node CRUD, debate_node, decompose_node, prioritize_children,
assign_agents, execute_node, find_precedents, cluster_ideas, auto_flow,
_save persistence, and error handling (ImportError, RuntimeError fallbacks).
"""

from __future__ import annotations

import builtins
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestPrioritizeChildren:
    """Tests for DAGOperationsCoordinator.prioritize_children."""

    def _graph_with_children(self) -> UniversalGraph:
        """Create a graph with a parent and 3 children."""
        graph = UniversalGraph(id="prio-test", name="Prioritization")
        graph.add_node(UniversalNode(
            id="parent",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Improve API performance",
            description="Parent goal for performance improvements",
        ))
        for i in range(1, 4):
            graph.add_node(UniversalNode(
                id=f"child-{i}",
                stage=PipelineStage.ACTIONS,
                node_subtype="task",
                label=f"Task {i}",
                description=f"Subtask {i} of parent",
                parent_ids=["parent"],
            ))
        return graph

    @pytest.mark.asyncio
    async def test_prioritize_not_found(self):
        graph = _make_graph()
        coord = DAGOperationsCoordinator(graph)
        result = await coord.prioritize_children("nonexistent")
        assert not result.success
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_prioritize_success(self, monkeypatch):
        """prioritize_children applies MetaPlanner-sourced priority to children."""
        graph = self._graph_with_children()
        coord = DAGOperationsCoordinator(graph)

        class _FakeGoal:
            def __init__(self, priority, impact):
                self.priority = priority
                self.estimated_impact = impact

        class _FakePlanner:
            def __init__(self, **_kw):
                pass

            async def prioritize_work(self, objective, **_kw):
                return [
                    _FakeGoal(priority=1, impact=0.9),
                    _FakeGoal(priority=2, impact=0.7),
                    _FakeGoal(priority=3, impact=0.4),
                ]

        monkeypatch.setattr(
            "aragora.nomic.meta_planner.MetaPlanner",
            _FakePlanner,
        )

        result = await coord.prioritize_children("parent")
        assert result.success
        assert "3" in result.message

        # Check priorities were applied to child nodes
        priorities = result.metadata["priorities"]
        assert priorities == [1, 2, 3]

        # Verify node data was updated
        child1 = graph.nodes["child-1"]
        assert child1.data["priority"] == 1
        assert child1.data["estimated_impact"] == 0.9

    @pytest.mark.asyncio
    async def test_prioritize_import_error(self, monkeypatch):
        """prioritize_children returns failure when MetaPlanner unavailable."""
        graph = self._graph_with_children()
        coord = DAGOperationsCoordinator(graph)

        import builtins
        real_import = builtins.__import__

        def _block_meta_planner(name, *args, **kwargs):
            if name == "aragora.nomic.meta_planner":
                raise ImportError("MetaPlanner not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_meta_planner)
        result = await coord.prioritize_children("parent")
        assert not result.success
        assert "not available" in result.message.lower()

    @pytest.mark.asyncio
    async def test_prioritize_runtime_error(self, monkeypatch):
        """prioritize_children handles MetaPlanner runtime failures."""
        graph = self._graph_with_children()
        coord = DAGOperationsCoordinator(graph)

        class _FailingPlanner:
            def __init__(self, **_kw):
                pass

            async def prioritize_work(self, objective, **_kw):
                raise RuntimeError("LLM API down")

        monkeypatch.setattr(
            "aragora.nomic.meta_planner.MetaPlanner",
            _FailingPlanner,
        )

        result = await coord.prioritize_children("parent")
        assert not result.success
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_prioritize_fewer_goals_than_children(self, monkeypatch):
        """When MetaPlanner returns fewer goals, remaining children get index-based priority."""
        graph = self._graph_with_children()
        coord = DAGOperationsCoordinator(graph)

        class _FakeGoal:
            def __init__(self, priority, impact):
                self.priority = priority
                self.estimated_impact = impact

        class _PartialPlanner:
            def __init__(self, **_kw):
                pass

            async def prioritize_work(self, objective, **_kw):
                # Return only 1 goal for 3 children
                return [_FakeGoal(priority=1, impact=0.95)]

        monkeypatch.setattr(
            "aragora.nomic.meta_planner.MetaPlanner",
            _PartialPlanner,
        )

        result = await coord.prioritize_children("parent")
        assert result.success
        priorities = result.metadata["priorities"]
        assert priorities[0] == 1       # From MetaPlanner
        assert priorities[1] == 2       # Fallback index
        assert priorities[2] == 3       # Fallback index

    @pytest.mark.asyncio
    async def test_prioritize_saves_graph(self, monkeypatch):
        """prioritize_children persists changes via _save."""
        graph = self._graph_with_children()
        save_calls = []
        mock_store = type("MockStore", (), {"update": lambda self, g: save_calls.append(True)})()
        coord = DAGOperationsCoordinator(graph, store=mock_store)

        class _FakeGoal:
            def __init__(self, priority, impact):
                self.priority = priority
                self.estimated_impact = impact

        class _FakePlanner:
            def __init__(self, **_kw):
                pass

            async def prioritize_work(self, objective, **_kw):
                return [_FakeGoal(1, 0.8), _FakeGoal(2, 0.6), _FakeGoal(3, 0.3)]

        monkeypatch.setattr("aragora.nomic.meta_planner.MetaPlanner", _FakePlanner)

        result = await coord.prioritize_children("parent")
        assert result.success
        assert len(save_calls) > 0
