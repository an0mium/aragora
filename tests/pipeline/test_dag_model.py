"""Tests for pipeline DAG visualization models."""

from __future__ import annotations

from aragora.canvas.stages import PipelineStage, StageEdgeType
from aragora.pipeline.dag_model import PipelineLiveUpdate
from aragora.pipeline.graph_store import GraphStore
from aragora.pipeline.universal_node import UniversalEdge, UniversalGraph, UniversalNode


def _make_graph() -> UniversalGraph:
    graph = UniversalGraph(id="dag-graph", name="DAG Graph")
    graph.add_node(
        UniversalNode(
            id="idea-1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Seed idea",
            execution_status="succeeded",
        )
    )
    graph.add_node(
        UniversalNode(
            id="goal-1",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Turn idea into goal",
            parent_ids=["idea-1"],
            source_stage=PipelineStage.IDEAS,
            execution_status="in_progress",
            data={"assigned_agent": "codex"},
        )
    )
    graph.add_node(
        UniversalNode(
            id="action-1",
            stage=PipelineStage.ACTIONS,
            node_subtype="task",
            label="Implement the goal",
            parent_ids=["goal-1"],
            source_stage=PipelineStage.GOALS,
        )
    )
    graph.add_edge(
        UniversalEdge(
            id="edge-1",
            source_id="idea-1",
            target_id="goal-1",
            edge_type=StageEdgeType.DERIVED_FROM,
            label="derived_from",
        )
    )
    return graph


def test_snapshot_builds_stage_summaries_and_dependencies() -> None:
    graph = _make_graph()

    snapshot = graph.to_dag_snapshot()

    assert snapshot.graph_id == "dag-graph"
    assert snapshot.stage_status == {
        "ideas": "complete",
        "principles": "pending",
        "goals": "in_progress",
        "actions": "pending",
        "orchestration": "pending",
    }
    assert snapshot.dependency_map() == {
        "action-1": ["goal-1"],
        "goal-1": ["idea-1"],
    }
    action_dependency = next(
        dependency for dependency in snapshot.dependencies if dependency.target_id == "action-1"
    )
    assert action_dependency.implicit is True
    assert action_dependency.edge_type == StageEdgeType.DERIVED_FROM

    actions_stage = next(stage for stage in snapshot.stages if stage.stage == PipelineStage.ACTIONS)
    assert actions_stage.node_ids == ["action-1"]
    assert actions_stage.dependency_stage_ids == ["goals"]
    assert actions_stage.status_counts == {"pending": 1}

    react_flow = snapshot.to_react_flow()
    assert {node["id"] for node in react_flow["nodes"]} == {"idea-1", "goal-1", "action-1"}
    assert len(react_flow["edges"]) == 2


def test_snapshot_applies_status_updates() -> None:
    snapshot = _make_graph().to_dag_snapshot()

    snapshot.apply_live_update(
        PipelineLiveUpdate.from_status_change(
            pipeline_id="dag-graph",
            node_id="action-1",
            stage=PipelineStage.ACTIONS,
            status="succeeded",
            assigned_agent="openai",
            output_preview="implemented",
        )
    )

    assert snapshot.runtime["action-1"].execution_status == "succeeded"
    assert snapshot.runtime["action-1"].assigned_agent == "openai"
    assert snapshot.stage_status["actions"] == "complete"
    assert len(snapshot.live_updates) == 1

    action_node = next(node for node in snapshot.nodes if node["id"] == "action-1")
    assert action_node["execution_status"] == "succeeded"
    assert action_node["metadata"]["execution_status"] == "succeeded"
    assert action_node["metadata"]["assigned_agent"] == "openai"
    assert action_node["metadata"]["output_preview"] == "implemented"


def test_snapshot_applies_node_additions() -> None:
    snapshot = _make_graph().to_dag_snapshot()
    orchestration_node = UniversalNode(
        id="orch-1",
        stage=PipelineStage.ORCHESTRATION,
        node_subtype="agent_task",
        label="Execute action",
        parent_ids=["action-1"],
        source_stage=PipelineStage.ACTIONS,
        execution_status="submitted",
    )

    snapshot.apply_live_update(
        PipelineLiveUpdate.from_node_added(
            pipeline_id="dag-graph",
            node=orchestration_node,
            dependency={
                "id": "edge-2",
                "source_id": "action-1",
                "target_id": "orch-1",
                "edge_type": StageEdgeType.EXECUTES.value,
                "source_stage": PipelineStage.ACTIONS.value,
                "target_stage": PipelineStage.ORCHESTRATION.value,
                "label": "executes",
            },
        )
    )

    assert "orch-1" in {node["id"] for node in snapshot.nodes}
    assert snapshot.stage_status["orchestration"] == "pending"
    assert snapshot.dependency_map()["orch-1"] == ["action-1"]
    orchestration_stage = next(
        stage for stage in snapshot.stages if stage.stage == PipelineStage.ORCHESTRATION
    )
    assert orchestration_stage.node_count == 1
    assert orchestration_stage.dependency_stage_ids == ["actions"]


def test_graph_store_returns_dag_snapshot(tmp_path) -> None:
    store = GraphStore(db_path=str(tmp_path / "pipeline_graphs.db"))
    graph = _make_graph()
    store.create(graph)

    snapshot = store.get_dag_snapshot(graph.id)

    assert snapshot is not None
    assert snapshot.graph_id == graph.id
    assert snapshot.integrity_hash == graph.integrity_hash()
    assert snapshot.stage_status["ideas"] == "complete"
