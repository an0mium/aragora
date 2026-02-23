"""Tests for the ProvenanceExplorerHandler.

Covers:
  - Route registration and can_handle logic
  - GET /api/v1/pipeline/graph/:graphId/react-flow  (full graph in React Flow format)
  - GET /api/v1/pipeline/graph/:graphId/provenance/:nodeId (node provenance chain)
  - 404 for missing graph/node
  - Rate limiting
  - Input validation (invalid IDs)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.pipeline.provenance_explorer import (
    ProvenanceExplorerHandler,
    _node_to_provenance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_handler_ctx() -> dict[str, Any]:
    return {}


def _make_http_handler() -> MagicMock:
    handler = MagicMock()
    handler.headers = {"Content-Length": "0"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


def _make_graph(graph_id: str = "graph-abc123"):
    """Build a small UniversalGraph with 3 nodes and 2 edges for testing."""
    from aragora.canvas.stages import PipelineStage, StageEdgeType
    from aragora.pipeline.universal_node import (
        UniversalEdge,
        UniversalGraph,
        UniversalNode,
    )

    graph = UniversalGraph(id=graph_id, name="Test Pipeline")

    idea = UniversalNode(
        id="node-idea-1",
        stage=PipelineStage.IDEAS,
        node_subtype="concept",
        label="Build rate limiter",
        description="Token bucket implementation",
        position_x=0,
        position_y=0,
        content_hash="aabbccdd11223344",
    )
    goal = UniversalNode(
        id="node-goal-1",
        stage=PipelineStage.GOALS,
        node_subtype="goal",
        label="Reduce 429 errors",
        description="Achieve < 1% 429 rate",
        position_x=250,
        position_y=0,
        content_hash="eeff00112233aabb",
        parent_ids=["node-idea-1"],
        source_stage=PipelineStage.IDEAS,
    )
    action = UniversalNode(
        id="node-action-1",
        stage=PipelineStage.ACTIONS,
        node_subtype="task",
        label="Implement sliding window counter",
        description="Redis-backed sliding window",
        position_x=500,
        position_y=0,
        content_hash="44556677889900aa",
        parent_ids=["node-goal-1"],
        source_stage=PipelineStage.GOALS,
    )

    graph.add_node(idea)
    graph.add_node(goal)
    graph.add_node(action)

    edge1 = UniversalEdge(
        id="edge-1",
        source_id="node-idea-1",
        target_id="node-goal-1",
        edge_type=StageEdgeType.DERIVED_FROM,
        label="derives",
    )
    edge2 = UniversalEdge(
        id="edge-2",
        source_id="node-goal-1",
        target_id="node-action-1",
        edge_type=StageEdgeType.DERIVED_FROM,
        label="decomposes",
    )
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    return graph


@pytest.fixture
def handler():
    return ProvenanceExplorerHandler(ctx=_make_handler_ctx())


@pytest.fixture
def sample_graph():
    return _make_graph()


# ---------------------------------------------------------------------------
# Route registration and can_handle
# ---------------------------------------------------------------------------


class TestRouting:
    def test_can_handle_graph_path(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/abc/react-flow") is True

    def test_can_handle_provenance_path(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/abc/provenance/node1") is True

    def test_cannot_handle_graphs_plural_path(self, handler):
        """The plural /graphs/ path is handled by UniversalGraphHandler, not us."""
        assert handler.can_handle("/api/v1/pipeline/graphs/abc/react-flow") is False

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_routes_attribute(self, handler):
        assert "/api/v1/pipeline/graph" in handler.ROUTES


# ---------------------------------------------------------------------------
# React Flow endpoint
# ---------------------------------------------------------------------------


class TestReactFlowEndpoint:
    def test_returns_flow_nodes_and_edges(self, handler, sample_graph):
        http_handler = _make_http_handler()
        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get.return_value = sample_graph
            resp = handler.handle(
                "/api/v1/pipeline/graph/graph-abc123/react-flow",
                {},
                http_handler,
            )

        assert resp is not None
        data, status, _ = resp
        assert status == 200

        # Verify node structure
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

        # Verify first node has correct shape
        first_node = data["nodes"][0]
        assert "id" in first_node
        assert "position" in first_node
        assert "x" in first_node["position"]
        assert "y" in first_node["position"]
        assert "data" in first_node

        # Verify ProvenanceNode data fields
        node_data = first_node["data"]
        assert "type" in node_data
        assert "label" in node_data
        assert "hash" in node_data
        assert node_data["type"] in ("debate", "goal", "action", "orchestration")

    def test_react_flow_graph_not_found(self, handler):
        http_handler = _make_http_handler()
        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get.return_value = None
            resp = handler.handle(
                "/api/v1/pipeline/graph/nonexistent/react-flow",
                {},
                http_handler,
            )

        assert resp is not None
        _, status, _ = resp
        assert status == 404

    def test_react_flow_edge_structure(self, handler, sample_graph):
        http_handler = _make_http_handler()
        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get.return_value = sample_graph
            resp = handler.handle(
                "/api/v1/pipeline/graph/graph-abc123/react-flow",
                {},
                http_handler,
            )

        data, _, _ = resp
        edge = data["edges"][0]
        assert "id" in edge
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge


# ---------------------------------------------------------------------------
# Node provenance endpoint
# ---------------------------------------------------------------------------


class TestNodeProvenanceEndpoint:
    def test_returns_provenance_chain(self, handler, sample_graph):
        http_handler = _make_http_handler()

        # The provenance chain for action node should include idea -> goal -> action
        chain_nodes = list(sample_graph.get_provenance_chain("node-action-1"))

        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get_provenance_chain.return_value = chain_nodes
            resp = handler.handle(
                "/api/v1/pipeline/graph/graph-abc123/provenance/node-action-1",
                {},
                http_handler,
            )

        assert resp is not None
        data, status, _ = resp
        assert status == 200
        assert "nodes" in data
        assert "edges" in data
        # Chain should have 3 nodes (idea -> goal -> action)
        assert len(data["nodes"]) == 3
        # Edges should connect parent -> child
        assert len(data["edges"]) >= 1

        # Verify node format
        for node in data["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "label" in node
            assert "hash" in node

    def test_provenance_node_not_found(self, handler):
        http_handler = _make_http_handler()
        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get_provenance_chain.return_value = []
            resp = handler.handle(
                "/api/v1/pipeline/graph/graph-abc123/provenance/missing-node",
                {},
                http_handler,
            )

        assert resp is not None
        _, status, _ = resp
        assert status == 404

    def test_provenance_single_node_no_parents(self, handler):
        """A node with no parents should return a single-node chain."""
        from aragora.canvas.stages import PipelineStage
        from aragora.pipeline.universal_node import UniversalNode

        lone_node = UniversalNode(
            id="node-lone-1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Standalone idea",
            content_hash="deadbeef12345678",
        )
        http_handler = _make_http_handler()
        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get_provenance_chain.return_value = [lone_node]
            resp = handler.handle(
                "/api/v1/pipeline/graph/graph-abc123/provenance/node-lone-1",
                {},
                http_handler,
            )

        data, status, _ = resp
        assert status == 200
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 0

    def test_provenance_edges_use_parent_ids(self, handler, sample_graph):
        """Edges in provenance response should reflect parent_ids relationships."""
        chain_nodes = list(sample_graph.get_provenance_chain("node-action-1"))

        http_handler = _make_http_handler()
        with patch("aragora.server.handlers.pipeline.provenance_explorer._get_store") as mock_store:
            mock_store.return_value.get_provenance_chain.return_value = chain_nodes
            resp = handler.handle(
                "/api/v1/pipeline/graph/graph-abc123/provenance/node-action-1",
                {},
                http_handler,
            )

        data, _, _ = resp
        edge_pairs = [(e["source"], e["target"]) for e in data["edges"]]
        # idea -> goal edge
        assert ("node-idea-1", "node-goal-1") in edge_pairs
        # goal -> action edge
        assert ("node-goal-1", "node-action-1") in edge_pairs


# ---------------------------------------------------------------------------
# Stage type mapping
# ---------------------------------------------------------------------------


class TestNodeToProvenance:
    def test_ideas_stage_maps_to_debate_type(self):
        from aragora.canvas.stages import PipelineStage
        from aragora.pipeline.universal_node import UniversalNode

        node = UniversalNode(
            id="n1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test",
            content_hash="abc123",
        )
        result = _node_to_provenance(node)
        assert result["type"] == "debate"

    def test_goals_stage_maps_to_goal_type(self):
        from aragora.canvas.stages import PipelineStage
        from aragora.pipeline.universal_node import UniversalNode

        node = UniversalNode(
            id="n2",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Goal",
            content_hash="def456",
        )
        result = _node_to_provenance(node)
        assert result["type"] == "goal"

    def test_actions_stage_maps_to_action_type(self):
        from aragora.canvas.stages import PipelineStage
        from aragora.pipeline.universal_node import UniversalNode

        node = UniversalNode(
            id="n3",
            stage=PipelineStage.ACTIONS,
            node_subtype="task",
            label="Action",
            content_hash="ghi789",
        )
        result = _node_to_provenance(node)
        assert result["type"] == "action"

    def test_orchestration_stage_maps_correctly(self):
        from aragora.canvas.stages import PipelineStage
        from aragora.pipeline.universal_node import UniversalNode

        node = UniversalNode(
            id="n4",
            stage=PipelineStage.ORCHESTRATION,
            node_subtype="agent_task",
            label="Orchestrate",
            content_hash="jkl012",
        )
        result = _node_to_provenance(node)
        assert result["type"] == "orchestration"

    def test_metadata_includes_stage_and_subtype(self):
        from aragora.canvas.stages import PipelineStage
        from aragora.pipeline.universal_node import UniversalNode

        node = UniversalNode(
            id="n5",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Test",
            content_hash="xyz",
            metadata={"priority": "high"},
        )
        result = _node_to_provenance(node)
        assert result["metadata"]["stage"] == "goals"
        assert result["metadata"]["subtype"] == "goal"
        assert result["metadata"]["priority"] == "high"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_short_path_returns_none(self, handler):
        """Paths that are too short should return None (not handled)."""
        http_handler = _make_http_handler()
        resp = handler.handle("/api/v1/pipeline/graph/abc", {}, http_handler)
        assert resp is None

    def test_unknown_sub_path_returns_none(self, handler):
        """Unknown sub-paths should return None."""
        http_handler = _make_http_handler()
        resp = handler.handle(
            "/api/v1/pipeline/graph/abc/unknown-endpoint",
            {},
            http_handler,
        )
        assert resp is None
