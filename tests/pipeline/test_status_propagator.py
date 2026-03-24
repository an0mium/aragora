"""Tests for execution-state propagation across the universal DAG."""

from __future__ import annotations

from aragora.canvas.stages import PipelineStage
from aragora.pipeline.status_propagator import StatusPropagator
from aragora.pipeline.universal_node import UniversalGraph, UniversalNode


def _make_graph() -> UniversalGraph:
    graph = UniversalGraph(id="prop-test")
    graph.add_node(
        UniversalNode(
            id="idea-1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Idea",
        )
    )
    graph.add_node(
        UniversalNode(
            id="goal-1",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Goal",
            parent_ids=["idea-1"],
            source_stage=PipelineStage.IDEAS,
        )
    )
    graph.add_node(
        UniversalNode(
            id="orch-1",
            stage=PipelineStage.ORCHESTRATION,
            node_subtype="human_gate",
            label="Review",
            parent_ids=["goal-1"],
            source_stage=PipelineStage.GOALS,
        )
    )
    return graph


def test_propagate_status_sets_top_level_and_metadata() -> None:
    graph = _make_graph()
    propagator = StatusPropagator(graph)

    updated = propagator.propagate_status("orch-1", "failed")

    assert updated == ["orch-1", "goal-1", "idea-1"]
    assert graph.nodes["orch-1"].execution_status == "failed"
    assert graph.nodes["orch-1"].metadata["execution_status"] == "failed"
    assert graph.nodes["goal-1"].execution_status == "failed"
    assert graph.nodes["idea-1"].execution_status == "failed"


def test_propagate_status_supports_awaiting_human() -> None:
    graph = _make_graph()
    propagator = StatusPropagator(graph)

    updated = propagator.propagate_status("orch-1", "awaiting_human")

    assert updated == ["orch-1", "goal-1", "idea-1"]
    assert graph.nodes["goal-1"].execution_status == "awaiting_human"
    assert graph.nodes["idea-1"].execution_status == "awaiting_human"
