"""Tests for UniversalGraphHandler PATCH (node update) and execute endpoints.

Covers:
  PATCH /api/v1/pipeline/graphs/:id/nodes/:node_id  Update node properties
  POST  /api/v1/pipeline/graphs/:id/execute/:node_id  Trigger debate on node
  404 for missing graph/node
  Error handling for execution failures
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.universal_node import UniversalEdge, UniversalGraph, UniversalNode
from aragora.server.handlers.pipeline.universal_graph import (
    UniversalGraphHandler,
    _graph_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(ctx: dict[str, Any] | None = None) -> UniversalGraphHandler:
    return UniversalGraphHandler(ctx=ctx or {})


def _make_http_handler(
    body: dict[str, Any] | None = None,
    client_ip: str = "127.0.0.1",
) -> MagicMock:
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {"Content-Length": "0"}
    if body is not None:
        raw = json.dumps(body).encode()
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile.read.return_value = raw
    else:
        handler.rfile.read.return_value = b"{}"
        handler.headers = {"Content-Length": "2"}
    return handler


def _body(result) -> dict[str, Any]:
    """Extract JSON body from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw) if raw else {}
    if isinstance(result, tuple):
        return result[0] if isinstance(result[0], dict) else json.loads(result[0])
    return {}


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


def _mock_graph(graph_id: str = "graph-abc123", name: str = "Test Pipeline"):
    """Create a mock UniversalGraph."""
    graph = MagicMock(spec=UniversalGraph)
    graph.id = graph_id
    graph.name = name
    graph.nodes = {}
    graph.edges = {}
    graph.owner_id = "user-1"
    graph.workspace_id = "ws-1"
    graph.metadata = {}
    graph.created_at = 1000.0
    graph.updated_at = 1000.0
    graph.to_dict.return_value = {
        "id": graph_id,
        "name": name,
        "nodes": [],
        "edges": [],
        "transitions": [],
        "metadata": {},
    }
    return graph


def _mock_node(node_id: str = "node-abc123"):
    """Create a mock UniversalNode with mutable attributes."""
    node = MagicMock(spec=UniversalNode)
    node.id = node_id
    node.label = "Test Node"
    node.description = "A test node"
    node.status = "active"
    node.position_x = 0.0
    node.position_y = 0.0
    node.confidence = 0.5
    node.data = {}
    node.metadata = {}
    node.to_dict.return_value = {
        "id": node_id,
        "stage": "ideas",
        "node_subtype": "concept",
        "label": "Test Node",
        "status": "active",
        "position_x": 0.0,
        "position_y": 0.0,
    }
    return node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _graph_limiter._buckets.clear()
    yield
    _graph_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def _bypass_check_permission(request, monkeypatch):
    """Bypass _check_permission on UniversalGraphHandler for all tests."""
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        return
    monkeypatch.setattr(
        UniversalGraphHandler,
        "_check_permission",
        lambda self, handler, permission: None,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the module-level _store between tests."""
    import aragora.server.handlers.pipeline.universal_graph as mod

    mod._store = None
    yield
    mod._store = None


@pytest.fixture
def mock_store():
    """Create a mock graph store."""
    store = MagicMock()
    store.create = MagicMock()
    store.list = MagicMock(return_value=[])
    store.get = MagicMock(return_value=None)
    store.update = MagicMock()
    store.delete = MagicMock(return_value=True)
    store.add_node = MagicMock()
    store.remove_node = MagicMock()
    store.query_nodes = MagicMock(return_value=[])
    store.get_provenance_chain = MagicMock(return_value=[])
    return store


@pytest.fixture
def patched_store(mock_store):
    """Patch _get_store to return the mock store."""
    with patch(
        "aragora.server.handlers.pipeline.universal_graph._get_store",
        return_value=mock_store,
    ):
        yield mock_store


# ===========================================================================
# PATCH /api/v1/pipeline/graphs/:id/nodes/:node_id  (update node)
# ===========================================================================


class TestUpdateNode:
    """Tests for PATCH node property updates."""

    def test_update_label(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"label": "New Label"},
            http,
        )
        assert _status(result) == 200
        assert node.label == "New Label"
        patched_store.update.assert_called_once_with(graph)

    def test_update_position(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"position_x": 100.5, "position_y": 200.0},
            http,
        )
        assert _status(result) == 200
        assert node.position_x == 100.5
        assert node.position_y == 200.0

    def test_update_status(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"status": "completed"},
            http,
        )
        assert _status(result) == 200
        assert node.status == "completed"

    def test_update_description(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"description": "Updated desc"},
            http,
        )
        assert node.description == "Updated desc"

    def test_update_confidence(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"confidence": 0.95},
            http,
        )
        assert node.confidence == 0.95

    def test_update_data_merges(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        node.data = {"existing": True}
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"data": {"new_key": "val"}},
            http,
        )
        node.data.update.assert_called_once_with({"new_key": "val"})

    def test_update_metadata_merges(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        node.metadata = {"tag": "a"}
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"metadata": {"tag": "b"}},
            http,
        )
        node.metadata.update.assert_called_once_with({"tag": "b"})

    def test_update_multiple_fields(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"label": "New", "status": "done", "position_x": 50},
            http,
        )
        assert _status(result) == 200
        assert node.label == "New"
        assert node.status == "done"
        assert node.position_x == 50.0

    def test_update_graph_not_found(self, patched_store):
        patched_store.get.return_value = None
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-nope/nodes/node-1",
            {"label": "X"},
            http,
        )
        assert _status(result) == 404

    def test_update_node_not_found(self, patched_store):
        graph = _mock_graph()
        graph.nodes = {}
        patched_store.get.return_value = graph
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-missing",
            {"label": "X"},
            http,
        )
        assert _status(result) == 404

    def test_update_invalid_graph_id(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/../../bad/nodes/node-1",
            {"label": "X"},
            http,
        )
        assert _status(result) == 400

    def test_update_invalid_node_id(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/<script>",
            {"label": "X"},
            http,
        )
        assert _status(result) == 400

    def test_update_empty_body_no_changes(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        node.label = "Original"
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {},
            http,
        )
        assert _status(result) == 200
        assert node.label == "Original"
        patched_store.update.assert_called_once()

    def test_update_persists_graph(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"label": "Updated"},
            http,
        )
        patched_store.update.assert_called_once_with(graph)

    def test_patch_wrong_path_returns_none(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/edges/edge-1",
            {"label": "X"},
            http,
        )
        assert result is None

    def test_data_non_dict_ignored(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        node.data = {"existing": True}
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        h.handle_patch(
            "/api/v1/pipeline/graphs/graph-abc123/nodes/node-1",
            {"data": "not-a-dict"},
            http,
        )
        # data.update should NOT have been called with non-dict
        node.data.update.assert_not_called()


# ===========================================================================
# POST /api/v1/pipeline/graphs/:id/execute/:node_id  (execute node)
# ===========================================================================


class TestExecuteNode:
    """Tests for triggering debate execution on a node."""

    def test_execute_graph_not_found(self, patched_store):
        patched_store.get.return_value = None
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graphs/graph-nope/execute/node-1", {}, http
        )
        assert _status(result) == 404

    def test_execute_node_not_found(self, patched_store):
        graph = _mock_graph()
        graph.nodes = {}
        patched_store.get.return_value = graph
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graphs/graph-abc123/execute/node-missing", {}, http
        )
        assert _status(result) == 404

    def test_execute_error_returns_500(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        with patch(
            "aragora.pipeline.dag_operations.DAGOperationsCoordinator",
            side_effect=RuntimeError("execution failed"),
        ):
            result = h.handle_post(
                "/api/v1/pipeline/graphs/graph-abc123/execute/node-1", {}, http
            )
        assert _status(result) == 500

    def test_execute_success(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Debate completed"
        mock_result.metadata = {"rounds": 3}

        mock_coord = MagicMock()

        import asyncio

        async def mock_debate(*a, **kw):
            return mock_result

        mock_coord.debate_node = mock_debate

        h = _make_handler()
        http = _make_http_handler()
        with patch(
            "aragora.pipeline.dag_operations.DAGOperationsCoordinator",
            return_value=mock_coord,
        ):
            result = h.handle_post(
                "/api/v1/pipeline/graphs/graph-abc123/execute/node-1",
                {"rounds": 3},
                http,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["message"] == "Debate completed"

    def test_execute_invalid_graph_id(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graphs/../bad/execute/node-1", {}, http
        )
        assert _status(result) == 400

    def test_execute_invalid_node_id(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graphs/graph-abc123/execute/<script>", {}, http
        )
        assert _status(result) == 400

    def test_execute_passes_agents_and_rounds(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "ok"
        mock_result.metadata = {}

        mock_coord = MagicMock()
        captured_kwargs = {}

        import asyncio

        async def mock_debate(node_id, agents=None, rounds=3):
            captured_kwargs["agents"] = agents
            captured_kwargs["rounds"] = rounds
            return mock_result

        mock_coord.debate_node = mock_debate

        h = _make_handler()
        http = _make_http_handler()
        with patch(
            "aragora.pipeline.dag_operations.DAGOperationsCoordinator",
            return_value=mock_coord,
        ):
            h.handle_post(
                "/api/v1/pipeline/graphs/graph-abc123/execute/node-1",
                {"agents": ["claude", "gpt4"], "rounds": 5},
                http,
            )

        assert captured_kwargs["agents"] == ["claude", "gpt4"]
        assert captured_kwargs["rounds"] == 5


__all__ = []
