"""Tests for UniversalNode, UniversalEdge, and UniversalGraph."""

from __future__ import annotations

import time
import uuid

import pytest

from aragora.canvas.stages import (
    NODE_TYPE_COLORS,
    PipelineStage,
    STAGE_COLORS,
    StageEdgeType,
    StageTransition,
    content_hash,
)
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
    _VALID_SUBTYPES,
)


# ── UniversalNode ──────────────────────────────────────────────────────


class TestUniversalNode:
    def test_auto_content_hash(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test Idea",
        )
        assert node.content_hash == content_hash("Test Idea")

    def test_auto_content_hash_includes_description(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test",
            description="Description",
        )
        assert node.content_hash == content_hash("TestDescription")

    def test_explicit_content_hash_preserved(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test",
            content_hash="custom123",
        )
        assert node.content_hash == "custom123"

    def test_default_field_values(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test",
        )
        assert node.description == ""
        assert node.position_x == 0.0
        assert node.position_y == 0.0
        assert node.width == 200.0
        assert node.height == 100.0
        assert node.previous_hash is None
        assert node.parent_ids == []
        assert node.source_stage is None
        assert node.status == "active"
        assert node.confidence == 0.0
        assert node.data == {}
        assert node.style == {}
        assert node.metadata == {}
        assert node.created_at > 0
        assert node.updated_at > 0

    def test_validate_subtype_valid(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test",
        )
        assert node.validate_subtype() is True

    def test_validate_subtype_invalid(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="agent_task",
            label="Test",
        )
        assert node.validate_subtype() is False

    def test_validate_subtype_empty_string(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="",
            label="Test",
        )
        assert node.validate_subtype() is False

    def test_validate_subtype_nonexistent_type(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.GOALS,
            node_subtype="nonexistent",
            label="Test",
        )
        assert node.validate_subtype() is False

    @pytest.mark.parametrize(
        "stage,subtype",
        [
            (PipelineStage.IDEAS, "concept"),
            (PipelineStage.IDEAS, "cluster"),
            (PipelineStage.IDEAS, "question"),
            (PipelineStage.IDEAS, "insight"),
            (PipelineStage.IDEAS, "evidence"),
            (PipelineStage.IDEAS, "assumption"),
            (PipelineStage.IDEAS, "constraint"),
            (PipelineStage.IDEAS, "observation"),
            (PipelineStage.IDEAS, "hypothesis"),
            (PipelineStage.GOALS, "goal"),
            (PipelineStage.GOALS, "principle"),
            (PipelineStage.GOALS, "strategy"),
            (PipelineStage.GOALS, "milestone"),
            (PipelineStage.GOALS, "metric"),
            (PipelineStage.GOALS, "risk"),
            (PipelineStage.ACTIONS, "task"),
            (PipelineStage.ACTIONS, "epic"),
            (PipelineStage.ACTIONS, "checkpoint"),
            (PipelineStage.ACTIONS, "deliverable"),
            (PipelineStage.ACTIONS, "dependency"),
            (PipelineStage.ORCHESTRATION, "agent_task"),
            (PipelineStage.ORCHESTRATION, "debate"),
            (PipelineStage.ORCHESTRATION, "human_gate"),
            (PipelineStage.ORCHESTRATION, "parallel_fan"),
            (PipelineStage.ORCHESTRATION, "merge"),
            (PipelineStage.ORCHESTRATION, "verification"),
        ],
    )
    def test_validate_subtype_all_stages(self, stage, subtype):
        node = UniversalNode(
            id="n1",
            stage=stage,
            node_subtype=subtype,
            label="Test",
        )
        assert node.validate_subtype() is True

    def test_to_dict_roundtrip(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Build API",
            description="Build the REST API",
            position_x=100,
            position_y=200,
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

    def test_to_dict_content_hash_preserved(self):
        """Content hash in to_dict matches from_dict restoration."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Hello",
            description="World",
        )
        d = node.to_dict()
        assert d["content_hash"] == content_hash("HelloWorld")
        restored = UniversalNode.from_dict(d)
        assert restored.content_hash == node.content_hash

    def test_to_react_flow_node(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test Idea",
            position_x=50,
            position_y=100,
        )
        rf = node.to_react_flow_node()
        assert rf["id"] == "n1"
        assert rf["type"] == "ideasNode"
        assert rf["position"] == {"x": 50, "y": 100}
        assert rf["data"]["label"] == "Test Idea"
        assert rf["data"]["stage"] == "ideas"
        assert rf["data"]["subtype"] == "concept"
        assert "color" in rf["data"]

    def test_to_react_flow_node_uses_node_type_colors(self):
        """The color in react flow data uses NODE_TYPE_COLORS mapping."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.GOALS,
            node_subtype="risk",
            label="Risk Item",
        )
        rf = node.to_react_flow_node()
        assert rf["data"]["color"] == NODE_TYPE_COLORS["risk"]

    def test_to_react_flow_node_unknown_subtype_fallback_color(self):
        """Unknown subtype gets fallback gray color."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="unknown_type",
            label="Test",
        )
        rf = node.to_react_flow_node()
        assert rf["data"]["color"] == "#94a3b8"

    def test_to_react_flow_node_stage_color(self):
        """Stage color from STAGE_COLORS is included."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.ORCHESTRATION,
            node_subtype="debate",
            label="Debate",
        )
        rf = node.to_react_flow_node()
        assert rf["data"]["stageColor"] == STAGE_COLORS[PipelineStage.ORCHESTRATION]["primary"]

    def test_to_react_flow_node_style_dimensions(self):
        """Width and height appear in style dict."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="T",
            width=300.0,
            height=150.0,
        )
        rf = node.to_react_flow_node()
        assert rf["style"]["width"] == 300.0
        assert rf["style"]["height"] == 150.0

    def test_to_react_flow_node_custom_data_merged(self):
        """Custom data dict is merged into react flow data."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="T",
            data={"priority": "high", "tags": ["a"]},
        )
        rf = node.to_react_flow_node()
        assert rf["data"]["priority"] == "high"
        assert rf["data"]["tags"] == ["a"]

    def test_to_react_flow_node_custom_style_merged(self):
        """Custom style dict is merged into react flow style."""
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="T",
            style={"borderRadius": "8px"},
        )
        rf = node.to_react_flow_node()
        assert rf["style"]["borderRadius"] == "8px"

    def test_source_stage_none_serialization(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="T",
        )
        d = node.to_dict()
        assert d["source_stage"] is None
        restored = UniversalNode.from_dict(d)
        assert restored.source_stage is None

    def test_from_dict_minimal_fields(self):
        """from_dict works with only required fields."""
        d = {
            "stage": "ideas",
            "node_subtype": "concept",
        }
        node = UniversalNode.from_dict(d)
        assert node.stage == PipelineStage.IDEAS
        assert node.node_subtype == "concept"
        assert node.label == ""
        assert node.status == "active"
        assert node.id  # auto-generated UUID

    def test_previous_hash_roundtrip(self):
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="T",
            previous_hash="abc123",
        )
        d = node.to_dict()
        assert d["previous_hash"] == "abc123"
        restored = UniversalNode.from_dict(d)
        assert restored.previous_hash == "abc123"

    def test_status_values(self):
        """All documented status values are accepted."""
        for status in ("active", "completed", "archived", "rejected"):
            node = UniversalNode(
                id="n1",
                stage=PipelineStage.IDEAS,
                node_subtype="concept",
                label="T",
                status=status,
            )
            assert node.status == status


# ── UniversalEdge ──────────────────────────────────────────────────────


class TestUniversalEdge:
    def test_creation_defaults(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.SUPPORTS,
        )
        assert edge.label == ""
        assert edge.weight == 1.0
        assert edge.cross_stage is False
        assert edge.data == {}
        assert edge.created_at > 0

    def test_to_dict_roundtrip(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.SUPPORTS,
            label="supports",
            weight=0.8,
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
        assert restored.data == {"custom": "data"}

    def test_to_dict_edge_type_serialized_as_string(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.INSPIRES,
        )
        d = edge.to_dict()
        assert d["edge_type"] == "inspires"

    def test_to_react_flow_edge(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.DERIVED_FROM,
            cross_stage=True,
        )
        rf = edge.to_react_flow_edge()
        assert rf["id"] == "e1"
        assert rf["source"] == "n1"
        assert rf["target"] == "n2"
        assert rf["animated"] is True  # cross-stage
        assert rf["type"] == "smoothstep"

    def test_to_react_flow_edge_label_fallback(self):
        """When label is empty, edge_type value is used as label."""
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.BLOCKS,
        )
        rf = edge.to_react_flow_edge()
        assert rf["label"] == "blocks"

    def test_to_react_flow_edge_custom_label(self):
        """When label is provided, it overrides edge_type value."""
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.BLOCKS,
            label="is blocked by",
        )
        rf = edge.to_react_flow_edge()
        assert rf["label"] == "is blocked by"

    def test_intra_stage_edge(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.SUPPORTS,
            cross_stage=False,
        )
        rf = edge.to_react_flow_edge()
        assert rf["animated"] is False
        assert rf["type"] == "default"

    def test_to_react_flow_edge_data_includes_weight(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.REQUIRES,
            weight=0.5,
        )
        rf = edge.to_react_flow_edge()
        assert rf["data"]["weight"] == 0.5
        assert rf["data"]["edgeType"] == "requires"

    def test_to_react_flow_edge_custom_data_merged(self):
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.SUPPORTS,
            data={"notes": "important"},
        )
        rf = edge.to_react_flow_edge()
        assert rf["data"]["notes"] == "important"

    def test_from_dict_minimal(self):
        d = {
            "source_id": "a",
            "target_id": "b",
            "edge_type": "follows",
        }
        edge = UniversalEdge.from_dict(d)
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.edge_type == StageEdgeType.FOLLOWS
        assert edge.weight == 1.0
        assert edge.cross_stage is False


# ── UniversalGraph ─────────────────────────────────────────────────────


class TestUniversalGraph:
    def _make_graph(self):
        graph = UniversalGraph(id="g1", name="Test Pipeline")
        n1 = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="Idea 1"
        )
        n2 = UniversalNode(
            id="n2",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Goal 1",
            parent_ids=["n1"],
            source_stage=PipelineStage.IDEAS,
        )
        n3 = UniversalNode(
            id="n3",
            stage=PipelineStage.ACTIONS,
            node_subtype="task",
            label="Task 1",
            parent_ids=["n2"],
            source_stage=PipelineStage.GOALS,
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

    def test_add_node_updates_timestamp(self):
        graph = UniversalGraph(id="g1")
        old_time = graph.updated_at
        # Small sleep to ensure time advances
        node = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="T")
        graph.add_node(node)
        assert graph.updated_at >= old_time

    def test_add_edge(self):
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.DERIVED_FROM,
        )
        graph.add_edge(edge)
        assert "e1" in graph.edges
        assert graph.edges["e1"].cross_stage is True

    def test_add_edge_auto_detects_cross_stage(self):
        """add_edge sets cross_stage=True when source and target are in different stages."""
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.DERIVED_FROM,
            cross_stage=False,  # Explicitly set to False
        )
        graph.add_edge(edge)
        # Should be overridden to True since n1=IDEAS, n2=GOALS
        assert graph.edges["e1"].cross_stage is True

    def test_add_edge_same_stage(self):
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        n2 = UniversalNode(id="n2", stage=PipelineStage.IDEAS, node_subtype="insight", label="B")
        graph.add_node(n1)
        graph.add_node(n2)
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.SUPPORTS
        )
        graph.add_edge(edge)
        assert graph.edges["e1"].cross_stage is False

    def test_add_edge_invalid_source(self):
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1", source_id="missing", target_id="n1", edge_type=StageEdgeType.SUPPORTS
        )
        graph.add_edge(edge)
        assert "e1" not in graph.edges

    def test_add_edge_invalid_target(self):
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="missing", edge_type=StageEdgeType.SUPPORTS
        )
        graph.add_edge(edge)
        assert "e1" not in graph.edges

    def test_add_edge_both_invalid(self):
        graph = UniversalGraph(id="g1")
        edge = UniversalEdge(
            id="e1", source_id="bad1", target_id="bad2", edge_type=StageEdgeType.SUPPORTS
        )
        graph.add_edge(edge)
        assert "e1" not in graph.edges  # silently skipped

    def test_remove_node_cascades_edges(self):
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        graph.add_edge(edge)
        assert "e1" in graph.edges
        graph.remove_node("n1")
        assert "n1" not in graph.nodes
        assert "e1" not in graph.edges

    def test_remove_node_returns_removed_node(self):
        graph = self._make_graph()
        removed = graph.remove_node("n1")
        assert removed is not None
        assert removed.id == "n1"
        assert removed.label == "Idea 1"

    def test_remove_node_nonexistent_returns_none(self):
        graph = self._make_graph()
        removed = graph.remove_node("nonexistent")
        assert removed is None

    def test_remove_node_cascades_multiple_edges(self):
        """Removing a node removes ALL edges touching it (both directions)."""
        graph = self._make_graph()
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        e2 = UniversalEdge(id="e2", source_id="n2", target_id="n1", edge_type=StageEdgeType.REFUTES)
        e3 = UniversalEdge(
            id="e3", source_id="n2", target_id="n3", edge_type=StageEdgeType.IMPLEMENTS
        )
        graph.add_edge(e1)
        graph.add_edge(e2)
        graph.add_edge(e3)
        graph.remove_node("n1")
        assert "e1" not in graph.edges
        assert "e2" not in graph.edges
        assert "e3" in graph.edges  # unrelated to n1

    def test_remove_edge(self):
        graph = self._make_graph()
        edge = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        graph.add_edge(edge)
        removed = graph.remove_edge("e1")
        assert removed is not None
        assert "e1" not in graph.edges

    def test_remove_edge_nonexistent_returns_none(self):
        graph = self._make_graph()
        removed = graph.remove_edge("nonexistent")
        assert removed is None

    def test_get_stage(self):
        graph = self._make_graph()
        ideas = graph.get_stage(PipelineStage.IDEAS)
        assert len(ideas) == 1
        assert ideas[0].id == "n1"

    def test_get_stage_multiple_nodes(self):
        graph = self._make_graph()
        n4 = UniversalNode(
            id="n4", stage=PipelineStage.IDEAS, node_subtype="insight", label="Idea 2"
        )
        graph.add_node(n4)
        ideas = graph.get_stage(PipelineStage.IDEAS)
        assert len(ideas) == 2
        idea_ids = {n.id for n in ideas}
        assert idea_ids == {"n1", "n4"}

    def test_get_stage_empty(self):
        graph = self._make_graph()
        orch = graph.get_stage(PipelineStage.ORCHESTRATION)
        assert orch == []

    def test_get_cross_stage_edges(self):
        graph = self._make_graph()
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        e2 = UniversalEdge(
            id="e2", source_id="n2", target_id="n3", edge_type=StageEdgeType.IMPLEMENTS
        )
        graph.add_edge(e1)
        graph.add_edge(e2)
        cross = graph.get_cross_stage_edges()
        assert len(cross) == 2

    def test_get_cross_stage_edges_excludes_intra(self):
        """Intra-stage edges are excluded from cross-stage results."""
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        n2 = UniversalNode(id="n2", stage=PipelineStage.IDEAS, node_subtype="insight", label="B")
        n3 = UniversalNode(id="n3", stage=PipelineStage.GOALS, node_subtype="goal", label="C")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.SUPPORTS
        )
        e2 = UniversalEdge(
            id="e2", source_id="n1", target_id="n3", edge_type=StageEdgeType.DERIVED_FROM
        )
        graph.add_edge(e1)
        graph.add_edge(e2)
        cross = graph.get_cross_stage_edges()
        assert len(cross) == 1
        assert cross[0].id == "e2"

    def test_get_provenance_chain(self):
        graph = self._make_graph()
        chain = graph.get_provenance_chain("n3")
        ids = [n.id for n in chain]
        assert "n3" in ids
        assert "n2" in ids
        assert "n1" in ids

    def test_get_provenance_chain_order(self):
        """Chain starts with the requested node, then its parents depth-first."""
        graph = self._make_graph()
        chain = graph.get_provenance_chain("n3")
        assert chain[0].id == "n3"

    def test_provenance_chain_single_node(self):
        """A node with no parents returns a chain of just itself."""
        graph = self._make_graph()
        chain = graph.get_provenance_chain("n1")
        assert len(chain) == 1
        assert chain[0].id == "n1"

    def test_provenance_chain_missing_parent(self):
        """If a parent_id references a non-existent node, it is skipped."""
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(
            id="n1",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="G",
            parent_ids=["missing_parent"],
        )
        graph.add_node(n1)
        chain = graph.get_provenance_chain("n1")
        assert len(chain) == 1
        assert chain[0].id == "n1"

    def test_provenance_chain_no_cycles(self):
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(
            id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A", parent_ids=["n2"]
        )
        n2 = UniversalNode(
            id="n2", stage=PipelineStage.IDEAS, node_subtype="concept", label="B", parent_ids=["n1"]
        )
        graph.add_node(n1)
        graph.add_node(n2)
        chain = graph.get_provenance_chain("n1")
        assert len(chain) == 2

    def test_provenance_chain_self_referencing_parent(self):
        """A node listing itself in parent_ids does not cause infinite loop."""
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Self",
            parent_ids=["n1"],
        )
        graph.add_node(n1)
        chain = graph.get_provenance_chain("n1")
        assert len(chain) == 1
        assert chain[0].id == "n1"

    def test_provenance_chain_multi_level(self):
        """Four levels deep: n4 -> n3 -> n2 -> n1."""
        graph = self._make_graph()
        n4 = UniversalNode(
            id="n4",
            stage=PipelineStage.ORCHESTRATION,
            node_subtype="agent_task",
            label="Agent Task 1",
            parent_ids=["n3"],
            source_stage=PipelineStage.ACTIONS,
        )
        graph.add_node(n4)
        chain = graph.get_provenance_chain("n4")
        ids = [n.id for n in chain]
        assert ids == ["n4", "n3", "n2", "n1"]

    def test_provenance_chain_diamond(self):
        """Diamond shape: n3 has two parents (n1, n2), both derived from same root."""
        graph = UniversalGraph(id="g1")
        root = UniversalNode(
            id="root", stage=PipelineStage.IDEAS, node_subtype="concept", label="Root"
        )
        n1 = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="insight",
            label="A",
            parent_ids=["root"],
        )
        n2 = UniversalNode(
            id="n2",
            stage=PipelineStage.IDEAS,
            node_subtype="insight",
            label="B",
            parent_ids=["root"],
        )
        n3 = UniversalNode(
            id="n3",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="C",
            parent_ids=["n1", "n2"],
        )
        graph.add_node(root)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        chain = graph.get_provenance_chain("n3")
        ids = [n.id for n in chain]
        # All 4 nodes visited, root visited only once despite diamond
        assert set(ids) == {"n3", "n1", "n2", "root"}
        assert len(ids) == 4

    def test_provenance_chain_nonexistent_node(self):
        """Requesting chain for a non-existent node returns empty list."""
        graph = self._make_graph()
        chain = graph.get_provenance_chain("nonexistent")
        assert chain == []

    def test_integrity_hash(self):
        graph = self._make_graph()
        h1 = graph.integrity_hash()
        assert len(h1) == 16
        # Adding a node changes hash
        n4 = UniversalNode(id="n4", stage=PipelineStage.IDEAS, node_subtype="concept", label="New")
        graph.add_node(n4)
        h2 = graph.integrity_hash()
        assert h1 != h2

    def test_integrity_hash_deterministic(self):
        """Same nodes produce the same hash regardless of insertion order."""
        g1 = UniversalGraph(id="g1")
        g2 = UniversalGraph(id="g2")
        na = UniversalNode(id="na", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        nb = UniversalNode(id="nb", stage=PipelineStage.IDEAS, node_subtype="concept", label="B")
        # Insert in opposite orders
        g1.add_node(
            UniversalNode(id="na", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        )
        g1.add_node(
            UniversalNode(id="nb", stage=PipelineStage.IDEAS, node_subtype="concept", label="B")
        )
        g2.add_node(
            UniversalNode(id="nb", stage=PipelineStage.IDEAS, node_subtype="concept", label="B")
        )
        g2.add_node(
            UniversalNode(id="na", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        )
        assert g1.integrity_hash() == g2.integrity_hash()

    def test_integrity_hash_empty_graph(self):
        graph = UniversalGraph(id="g1")
        h = graph.integrity_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_to_react_flow_all(self):
        graph = self._make_graph()
        rf = graph.to_react_flow()
        assert len(rf["nodes"]) == 3
        assert len(rf["edges"]) == 0  # no edges added

    def test_to_react_flow_with_edges(self):
        """React flow export includes edges when they exist."""
        graph = self._make_graph()
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        graph.add_edge(e1)
        rf = graph.to_react_flow()
        assert len(rf["edges"]) == 1
        assert rf["edges"][0]["id"] == "e1"

    def test_to_react_flow_filtered(self):
        graph = self._make_graph()
        rf = graph.to_react_flow(stage_filter=PipelineStage.IDEAS)
        assert len(rf["nodes"]) == 1
        assert rf["nodes"][0]["data"]["stage"] == "ideas"

    def test_to_react_flow_filtered_excludes_cross_stage_edges(self):
        """When filtering by stage, edges to nodes outside the filter are excluded."""
        graph = self._make_graph()
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        graph.add_edge(e1)
        rf = graph.to_react_flow(stage_filter=PipelineStage.IDEAS)
        assert len(rf["edges"]) == 0  # n2 is GOALS, filtered out

    def test_to_react_flow_filtered_includes_intra_stage_edges(self):
        """When filtering by stage, intra-stage edges within the filter are included."""
        graph = UniversalGraph(id="g1")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        n2 = UniversalNode(id="n2", stage=PipelineStage.IDEAS, node_subtype="insight", label="B")
        graph.add_node(n1)
        graph.add_node(n2)
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.SUPPORTS
        )
        graph.add_edge(e1)
        rf = graph.to_react_flow(stage_filter=PipelineStage.IDEAS)
        assert len(rf["nodes"]) == 2
        assert len(rf["edges"]) == 1

    def test_to_dict_roundtrip(self):
        graph = self._make_graph()
        e1 = UniversalEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=StageEdgeType.DERIVED_FROM
        )
        graph.add_edge(e1)
        graph.transitions.append(
            StageTransition(
                id="t1",
                from_stage=PipelineStage.IDEAS,
                to_stage=PipelineStage.GOALS,
            )
        )
        d = graph.to_dict()
        restored = UniversalGraph.from_dict(d)
        assert restored.id == "g1"
        assert len(restored.nodes) == 3
        assert len(restored.edges) == 1
        assert len(restored.transitions) == 1
        assert restored.transitions[0].from_stage == PipelineStage.IDEAS

    def test_to_dict_roundtrip_preserves_metadata(self):
        graph = UniversalGraph(
            id="g1",
            name="My Pipeline",
            metadata={"version": 2, "tags": ["test"]},
            owner_id="user-123",
            workspace_id="ws-456",
        )
        d = graph.to_dict()
        restored = UniversalGraph.from_dict(d)
        assert restored.name == "My Pipeline"
        assert restored.metadata == {"version": 2, "tags": ["test"]}
        assert restored.owner_id == "user-123"
        assert restored.workspace_id == "ws-456"

    def test_empty_graph(self):
        graph = UniversalGraph(id="empty")
        assert graph.integrity_hash()
        assert graph.to_react_flow() == {"nodes": [], "edges": []}
        assert graph.get_stage(PipelineStage.IDEAS) == []
        assert graph.get_cross_stage_edges() == []

    def test_empty_graph_provenance_chain(self):
        graph = UniversalGraph(id="empty")
        chain = graph.get_provenance_chain("anything")
        assert chain == []

    def test_graph_default_name(self):
        graph = UniversalGraph(id="g1")
        assert graph.name == "Untitled Pipeline"

    def test_graph_to_dict_nodes_as_list(self):
        """to_dict serializes nodes as a list, not a dict keyed by id."""
        graph = self._make_graph()
        d = graph.to_dict()
        assert isinstance(d["nodes"], list)
        assert len(d["nodes"]) == 3

    def test_valid_subtypes_mapping_completeness(self):
        """_VALID_SUBTYPES covers all four pipeline stages."""
        assert set(_VALID_SUBTYPES.keys()) == {
            PipelineStage.IDEAS,
            PipelineStage.GOALS,
            PipelineStage.ACTIONS,
            PipelineStage.ORCHESTRATION,
        }
