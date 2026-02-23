"""Tests for GraphStore (SQLite persistence for UniversalGraph)."""

from __future__ import annotations

import tempfile
import os

import pytest

from aragora.canvas.stages import PipelineStage, StageEdgeType, StageTransition
from aragora.pipeline.graph_store import GraphStore
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_graphs.db")
    return GraphStore(db_path=db_path)


@pytest.fixture
def sample_graph():
    graph = UniversalGraph(
        id="test-graph-1",
        name="Test Pipeline",
        owner_id="user-1",
        workspace_id="ws-1",
        metadata={"version": 1},
    )
    n1 = UniversalNode(
        id="n1",
        stage=PipelineStage.IDEAS,
        node_subtype="concept",
        label="Idea A",
        description="First idea",
        confidence=0.9,
    )
    n2 = UniversalNode(
        id="n2",
        stage=PipelineStage.GOALS,
        node_subtype="goal",
        label="Goal from A",
        parent_ids=["n1"],
        source_stage=PipelineStage.IDEAS,
        confidence=0.7,
    )
    n3 = UniversalNode(
        id="n3",
        stage=PipelineStage.ACTIONS,
        node_subtype="task",
        label="Task for Goal",
        parent_ids=["n2"],
        source_stage=PipelineStage.GOALS,
    )
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    e1 = UniversalEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type=StageEdgeType.DERIVED_FROM,
        label="derived",
    )
    graph.add_edge(e1)

    graph.transitions.append(
        StageTransition(
            id="t1",
            from_stage=PipelineStage.IDEAS,
            to_stage=PipelineStage.GOALS,
            status="approved",
            confidence=0.8,
        )
    )

    return graph


class TestGraphStoreCRUD:
    def test_create_and_get(self, store, sample_graph):
        store.create(sample_graph)
        loaded = store.get(sample_graph.id)
        assert loaded is not None
        assert loaded.id == sample_graph.id
        assert loaded.name == "Test Pipeline"
        assert loaded.owner_id == "user-1"
        assert loaded.workspace_id == "ws-1"
        assert len(loaded.nodes) == 3
        assert len(loaded.edges) == 1
        assert len(loaded.transitions) == 1

    def test_get_nonexistent(self, store):
        assert store.get("no-such-id") is None

    def test_list_empty(self, store):
        result = store.list()
        assert result == []

    def test_list_with_graphs(self, store, sample_graph):
        store.create(sample_graph)
        result = store.list()
        assert len(result) == 1
        assert result[0]["id"] == sample_graph.id
        assert result[0]["node_count"] == 3

    def test_list_filter_by_owner(self, store, sample_graph):
        store.create(sample_graph)
        assert len(store.list(owner_id="user-1")) == 1
        assert len(store.list(owner_id="user-2")) == 0

    def test_list_filter_by_workspace(self, store, sample_graph):
        store.create(sample_graph)
        assert len(store.list(workspace_id="ws-1")) == 1
        assert len(store.list(workspace_id="ws-other")) == 0

    def test_update(self, store, sample_graph):
        store.create(sample_graph)
        sample_graph.name = "Updated Name"
        sample_graph.metadata["version"] = 2
        store.update(sample_graph)
        loaded = store.get(sample_graph.id)
        assert loaded.name == "Updated Name"
        assert loaded.metadata["version"] == 2

    def test_delete(self, store, sample_graph):
        store.create(sample_graph)
        assert store.delete(sample_graph.id) is True
        assert store.get(sample_graph.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("no-id") is False


class TestGraphStoreNodes:
    def test_add_node(self, store, sample_graph):
        store.create(sample_graph)
        new_node = UniversalNode(
            id="n4",
            stage=PipelineStage.ORCHESTRATION,
            node_subtype="agent_task",
            label="Execute",
        )
        store.add_node(sample_graph.id, new_node)
        loaded = store.get(sample_graph.id)
        assert "n4" in loaded.nodes

    def test_remove_node(self, store, sample_graph):
        store.create(sample_graph)
        store.remove_node(sample_graph.id, "n1")
        loaded = store.get(sample_graph.id)
        assert "n1" not in loaded.nodes
        # Edge referencing n1 should be cleaned up
        edge_node_ids = set()
        for e in loaded.edges.values():
            edge_node_ids.add(e.source_id)
            edge_node_ids.add(e.target_id)
        assert "n1" not in edge_node_ids

    def test_query_nodes_all(self, store, sample_graph):
        store.create(sample_graph)
        nodes = store.query_nodes(sample_graph.id)
        assert len(nodes) == 3

    def test_query_nodes_by_stage(self, store, sample_graph):
        store.create(sample_graph)
        ideas = store.query_nodes(sample_graph.id, stage=PipelineStage.IDEAS)
        assert len(ideas) == 1
        assert ideas[0].id == "n1"

    def test_query_nodes_by_subtype(self, store, sample_graph):
        store.create(sample_graph)
        goals = store.query_nodes(sample_graph.id, subtype="goal")
        assert len(goals) == 1
        assert goals[0].id == "n2"

    def test_query_nodes_combined_filter(self, store, sample_graph):
        store.create(sample_graph)
        result = store.query_nodes(sample_graph.id, stage=PipelineStage.IDEAS, subtype="concept")
        assert len(result) == 1


class TestGraphStoreProvenance:
    def test_provenance_chain(self, store, sample_graph):
        store.create(sample_graph)
        chain = store.get_provenance_chain(sample_graph.id, "n3")
        ids = [n.id for n in chain]
        assert "n3" in ids
        assert "n2" in ids
        assert "n1" in ids

    def test_provenance_chain_root(self, store, sample_graph):
        store.create(sample_graph)
        chain = store.get_provenance_chain(sample_graph.id, "n1")
        assert len(chain) == 1
        assert chain[0].id == "n1"

    def test_provenance_chain_nonexistent(self, store, sample_graph):
        store.create(sample_graph)
        chain = store.get_provenance_chain(sample_graph.id, "no-node")
        assert chain == []


class TestGraphStoreDataIntegrity:
    def test_node_data_preserved(self, store):
        graph = UniversalGraph(id="g-data")
        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="With Data",
            data={"key1": "val1", "nested": {"a": 1}},
            style={"color": "red"},
            metadata={"source": "test"},
            previous_hash="prev123",
            parent_ids=["p1", "p2"],
            source_stage=PipelineStage.GOALS,
        )
        graph.add_node(node)
        store.create(graph)
        loaded = store.get("g-data")
        n = loaded.nodes["n1"]
        assert n.data == {"key1": "val1", "nested": {"a": 1}}
        assert n.style == {"color": "red"}
        assert n.metadata == {"source": "test"}
        assert n.previous_hash == "prev123"
        assert n.parent_ids == ["p1", "p2"]
        assert n.source_stage == PipelineStage.GOALS

    def test_edge_data_preserved(self, store):
        graph = UniversalGraph(id="g-edge")
        n1 = UniversalNode(id="n1", stage=PipelineStage.IDEAS, node_subtype="concept", label="A")
        n2 = UniversalNode(id="n2", stage=PipelineStage.GOALS, node_subtype="goal", label="B")
        graph.add_node(n1)
        graph.add_node(n2)
        edge = UniversalEdge(
            id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=StageEdgeType.DERIVED_FROM,
            weight=0.95,
            label="test edge",
            data={"custom": True},
        )
        graph.add_edge(edge)
        store.create(graph)
        loaded = store.get("g-edge")
        e = loaded.edges["e1"]
        assert e.edge_type == StageEdgeType.DERIVED_FROM
        assert e.weight == 0.95
        assert e.label == "test edge"
        assert e.cross_stage is True
        assert e.data == {"custom": True}

    def test_transition_data_preserved(self, store, sample_graph):
        store.create(sample_graph)
        loaded = store.get(sample_graph.id)
        t = loaded.transitions[0]
        assert t.id == "t1"
        assert t.from_stage == PipelineStage.IDEAS
        assert t.to_stage == PipelineStage.GOALS
        assert t.status == "approved"
        assert t.confidence == 0.8

    def test_multiple_graphs(self, store, sample_graph):
        store.create(sample_graph)
        graph2 = UniversalGraph(id="g2", name="Second")
        n = UniversalNode(id="n-g2", stage=PipelineStage.IDEAS, node_subtype="concept", label="X")
        graph2.add_node(n)
        store.create(graph2)

        assert len(store.list()) == 2
        g1 = store.get(sample_graph.id)
        g2 = store.get("g2")
        assert len(g1.nodes) == 3
        assert len(g2.nodes) == 1
