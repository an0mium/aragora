"""Tests for UniversalNode, UniversalEdge, and UniversalGraph."""

from __future__ import annotations

import time
import uuid

import pytest

from aragora.canvas.stages import (
    PipelineStage,
    StageEdgeType,
    StageTransition,
    content_hash,
)
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
)


# ── UniversalNode ──────────────────────────────────────────────────────


class TestUniversalNode:
    def test_auto_content_hash(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Test Idea",
        )
        assert node.content_hash == content_hash("Test Idea")

    def test_explicit_content_hash_preserved(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Test",
            content_hash="custom123",
        )
        assert node.content_hash == "custom123"

    def test_validate_subtype_valid(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Test",
        )
        assert node.validate_subtype() is True

    def test_validate_subtype_invalid(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS,
            node_subtype="agent_task", label="Test",
        )
        assert node.validate_subtype() is False

    @pytest.mark.parametrize("stage,subtype", [
        (PipelineStage.IDEAS, "concept"),
        (PipelineStage.IDEAS, "cluster"),
        (PipelineStage.IDEAS, "hypothesis"),
        (PipelineStage.GOALS, "goal"),
        (PipelineStage.GOALS, "risk"),
        (PipelineStage.ACTIONS, "task"),
        (PipelineStage.ACTIONS, "epic"),
        (PipelineStage.ORCHESTRATION, "agent_task"),
        (PipelineStage.ORCHESTRATION, "debate"),
    ])
    def test_validate_subtype_all_stages(self, stage, subtype):
        node = UniversalNode(
            id="n1", stage=stage, node_subtype=subtype, label="Test",
        )
        assert node.validate_subtype() is True

    def test_to_dict_roundtrip(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.GOALS,
            node_subtype="goal", label="Build API",
            description="Build the REST API",
            position_x=100, position_y=200,
            parent_ids=["idea-1"],
            source_stage=PipelineStage.IDEAS,
            confidence=0.8,
            data={"priority": "high"},
            metadata={"key": "val"},
        )
        d = node.to_dict()
        restored = UniversalNode.from_dict(d)
        assert restored.id == node.id
        assert restored.stage == node.stage
        assert restored.node_subtype == node.node_subtype
        assert restored.label == node.label
        assert restored.description == node.description
        assert restored.position_x == node.position_x
        assert restored.parent_ids == node.parent_ids
        assert restored.source_stage == PipelineStage.IDEAS
        assert restored.confidence == 0.8
        assert restored.data == {"priority": "high"}

    def test_to_react_flow_node(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Test Idea",
            position_x=50, position_y=100,
        )
        rf = node.to_react_flow_node()
        assert rf["id"] == "n1"
        assert rf["type"] == "ideasNode"
        assert rf["position"] == {"x": 50, "y": 100}
        assert rf["data"]["label"] == "Test Idea"
        assert rf["data"]["stage"] == "ideas"
        assert rf["data"]["subtype"] == "concept"
        assert "color" in rf["data"]

    def test_source_stage_none_serialization(self):
        node = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="T",
        )
        d = node.to_dict()
        assert d["source_stage"] is None
        restored = UniversalNode.from_dict(d)
        assert restored.source_stage is None


# ── UniversalEdge ──────────────────────────────────────────────────────


class TestUniversalEdge:
    def test_to_dict_roundtrip(self):
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2",
            edge_type=StageEdgeType.SUPPORTS,
            label="supports", weight=0.8,
            cross_stage=True,
            data={"custom": "data"},
        )
        d = edge.to_dict()
        restored = UniversalEdge.from_dict(d)
        assert restored.id == edge.id
        assert restored.source_id == "n1"
        assert restored.target_id == "n2"
        assert restored.edge_type == StageEdgeType.SUPPORTS
        assert restored.weight == 0.8
        assert restored.cross_stage is True

    def test_to_react_flow_edge(self):
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2",
            edge_type=StageEdgeType.DERIVED_FROM,
            cross_stage=True,
        )
        rf = edge.to_react_flow_edge()
        assert rf["id"] == "e1"
        assert rf["source"] == "n1"
        assert rf["target"] == "n2"
        assert rf["animated"] is True  # cross-stage
        assert rf["type"] == "smoothstep"

    def test_intra_stage_edge(self):
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2",
            edge_type=StageEdgeType.SUPPORTS,
            cross_stage=False,
        )
        rf = edge.to_react_flow_edge()
        assert rf["animated"] is False
        assert rf["type"] == "default"


# ── UniversalGraph ─────────────────────────────────────────────────────


class TestUniversalGraph:
    def _make_graph(self):
        graph = UniversalGraph(id="g1", name="Test Pipeline")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="Idea 1")
        n2 = UniversalNode(
            id="n2", stage=PipelineStage.GOALS, node_subtype="goal", label="Goal 1",
            parent_ids=["n1"], source_stage=PipelineStage.IDEAS,
        )
        n3 = UniversalNode(
            id="n3", stage=PipelineStage.ACTIONS, node_subtype="task", label="Task 1",
            parent_ids=["n2"], source_stage=PipelineStage.GOALS,
        )
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        return graph

    def test_add_node(self):
        graph = UniversalGraph(id="g1")
        node = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="T")
        graph.add_node(node)
        assert "n1" in graph.nodes

    def test_add_edge(self):
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2",
            edge_type=StageEdgeType.DERIVED_FROM,
        )
        graph.add_edge(edge)
        assert "e1" in graph.edges
        assert graph.edges["e1"].cross_stage is True

    def test_add_edge_same_stage(self):
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        n2 = UniversalNode(id="n2", stage=PipelineStage.IDEAS, node_subtype="insight", label="B")
        graph.add_node(n1)
        graph.add_node(n2)
        edge = UniversalEdge(id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.SUPPORTS)
        graph.add_edge(edge)
        assert graph.edges["e1"].cross_stage is False

    def test_add_edge_invalid_nodes(self):
        graph = UniversalGraph(id="g1")
        edge = UniversalEdge(id="e1", source_id="bad1", target_id="bad2", edge_type=StageEdgeType.SUPPORTS)
        graph.add_edge(edge)
        assert "e1" not in graph.edges  # silently skipped

    def test_remove_node_cascades_edges(self):
        graph = self._make_graph()
        edge = UniversalEdge(id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM)
        graph.add_edge(edge)
        assert "e1" in graph.edges
        graph.remove_node("n1")
        assert "n1" not in graph.nodes
        assert "e1" not in graph.edges

    def test_remove_edge(self):
        graph = self._make_graph()
        edge = UniversalEdge(id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM)
        graph.add_edge(edge)
        removed = graph.remove_edge("e1")
        assert removed is not None
        assert "e1" not in graph.edges

    def test_get_stage(self):
        graph = self._make_graph()
        ideas = graph.get_stage(PipelineStage.IDEAS)
        assert len(ideas) == 1
        assert ideas[0].id == "n1"

    def test_get_cross_stage_edges(self):
        graph = self._make_graph()
        e1 = UniversalEdge(id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM)
        e2 = UniversalEdge(id="e2", source_id="n2", target_id="n3", edge_type=StageEdgeType.IMPLEMENTS)
        graph.add_edge(e1)
        graph.add_edge(e2)
        cross = graph.get_cross_stage_edges()
        assert len(cross) == 2

    def test_get_provenance_chain(self):
        graph = self._make_graph()
        chain = graph.get_provenance_chain("n3")
        ids = [n.id for n in chain]
        assert "n3" in ids
        assert "n2" in ids
        assert "n1" in ids

    def test_provenance_chain_no_cycles(self):
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A", parent_ids=["n2"])
        n2 = UniversalNode(id="n2", stage=PipelineStage.IDEAS, node_subtype="concept", label="B", parent_ids=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)
        chain = graph.get_provenance_chain("n1")
        assert len(chain) == 2

    def test_integrity_hash(self):
        graph = self._make_graph()
        h1 = graph.integrity_hash()
        assert len(h1) == 16
        # Adding a node changes hash
        n4 = UniversalNode(id="n4", stage=PipelineStage.IDEAS, node_subtype="concept", label="New")
        graph.add_node(n4)
        h2 = graph.integrity_hash()
        assert h1 != h2

    def test_to_react_flow_all(self):
        graph = self._make_graph()
        rf = graph.to_react_flow()
        assert len(rf["nodes"]) == 3
        assert len(rf["edges"]) == 0  # no edges added

    def test_to_react_flow_filtered(self):
        graph = self._make_graph()
        rf = graph.to_react_flow(stage_filter=PipelineStage.IDEAS)
        assert len(rf["nodes"]) == 1
        assert rf["nodes"][0]["data"]["stage"] == "ideas"

    def test_to_dict_roundtrip(self):
        graph = self._make_graph()
        e1 = UniversalEdge(id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM)
        graph.add_edge(e1)
        graph.transitions.append(StageTransition(
            id="t1", from_stage=PipelineStage.IDEAS, to_stage=PipelineStage.GOALS,
        ))
        d = graph.to_dict()
        restored = UniversalGraph.from_dict(d)
        assert restored.id == "g1"
        assert len(restored.nodes) == 3
        assert len(restored.edges) == 1
        assert len(restored.transitions) == 1
        assert restored.transitions[0].from_stage == PipelineStage.IDEAS

    def test_empty_graph(self):
        graph = UniversalGraph(id="empty")
        assert graph.integrity_hash()
        assert graph.to_react_flow() == {"nodes": [], "edges": []}
        assert graph.get_stage(PipelineStage.IDEAS) == []
        assert graph.get_cross_stage_edges() == []
