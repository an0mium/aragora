"""Tests for the DecompositionHandler.

Covers:
  POST /api/v1/pipeline/:id/decompose/:node_id       Trigger decomposition
  GET  /api/v1/pipeline/:id/decompose/:node_id/tree   Get decomposition tree
  404 for missing graph/node
  can_handle routing
  Rate limiting
  Input validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.pipeline.decomposition import (
    DecompositionHandler,
    _decompose_limiter,
    _parse_decompose_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(ctx: dict[str, Any] | None = None) -> DecompositionHandler:
    return DecompositionHandler(ctx=ctx or {})


def _make_http_handler(client_ip: str = "127.0.0.1") -> MagicMock:
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {"Content-Length": "0"}
    handler.rfile.read.return_value = b"{}"
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


def _mock_graph(graph_id: str = "graph-abc123"):
    """Create a mock graph with a sample node."""
    graph = MagicMock()
    graph.id = graph_id
    graph.nodes = {}
    graph.edges = MagicMock()
    graph.edges.values.return_value = []
    return graph


def _mock_node(node_id: str = "node-abc123", label: str = "Test Node"):
    """Create a mock node."""
    node = MagicMock()
    node.id = node_id
    node.label = label
    node.description = "A test node"
    node.stage = None
    node.status = "active"
    return node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _decompose_limiter._buckets.clear()
    yield
    _decompose_limiter._buckets.clear()


@pytest.fixture(autouse=True)
def _bypass_check_permission(request, monkeypatch):
    """Bypass _check_permission for all tests except those marked no_auto_auth."""
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        return
    monkeypatch.setattr(
        DecompositionHandler,
        "_check_permission",
        lambda self, handler, permission: None,
    )


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the module-level _store between tests."""
    import aragora.server.handlers.pipeline.decomposition as mod

    mod._store = None
    yield
    mod._store = None


@pytest.fixture
def mock_store():
    """Create a mock graph store."""
    store = MagicMock()
    store.get = MagicMock(return_value=None)
    return store


@pytest.fixture
def patched_store(mock_store):
    """Patch _get_store to return the mock store."""
    with patch(
        "aragora.server.handlers.pipeline.decomposition._get_store",
        return_value=mock_store,
    ):
        yield mock_store


# ===========================================================================
# _parse_decompose_path
# ===========================================================================


class TestParseDecomposePath:
    """Tests for the path parser helper."""

    def test_valid_path(self):
        pid, nid, sub = _parse_decompose_path("/api/pipeline/graph-1/decompose/node-1")
        assert pid == "graph-1"
        assert nid == "node-1"
        assert sub is None

    def test_valid_path_with_tree(self):
        pid, nid, sub = _parse_decompose_path(
            "/api/pipeline/graph-1/decompose/node-1/tree"
        )
        assert pid == "graph-1"
        assert nid == "node-1"
        assert sub == "tree"

    def test_too_short(self):
        pid, nid, sub = _parse_decompose_path("/api/pipeline/graph-1")
        assert pid is None
        assert nid is None

    def test_wrong_keyword(self):
        pid, nid, sub = _parse_decompose_path("/api/pipeline/graph-1/other/node-1")
        assert pid is None

    def test_wrong_prefix(self):
        pid, nid, sub = _parse_decompose_path("/other/pipeline/graph-1/decompose/node-1")
        assert pid is None


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_matches_decompose_path(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/pipeline/graph-1/decompose/node-1") is True

    def test_matches_decompose_tree(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/pipeline/graph-1/decompose/node-1/tree") is True

    def test_no_match_without_decompose(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/pipeline/graphs") is False

    def test_no_match_wrong_prefix(self):
        h = _make_handler()
        assert h.can_handle("/api/v1/debates/decompose/node-1") is False


# ===========================================================================
# POST /api/v1/pipeline/:id/decompose/:node_id  (trigger decomposition)
# ===========================================================================


class TestPostDecompose:
    """Tests for triggering decomposition on a node."""

    def test_decompose_graph_not_found(self, patched_store):
        patched_store.get.return_value = None
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graph-1/decompose/node-1", {}, http
        )
        assert _status(result) == 404

    def test_decompose_node_not_found(self, patched_store):
        graph = _mock_graph()
        graph.nodes = {}  # no nodes
        patched_store.get.return_value = graph
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graph-1/decompose/node-missing", {}, http
        )
        assert _status(result) == 404

    def test_decompose_success_via_task_decomposer(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        mock_result = MagicMock()
        mock_result.complexity_score = 0.8
        mock_result.complexity_level = "high"
        mock_result.should_decompose = True
        mock_result.rationale = "Complex task"

        sub1 = MagicMock()
        sub1.id = "sub-1"
        sub1.title = "Subtask 1"
        sub1.description = "First subtask"
        sub1.estimated_complexity = 0.4
        sub1.dependencies = []
        sub1.parent_id = "node-1"
        sub1.depth = 1
        mock_result.subtasks = [sub1]

        h = _make_handler()
        http = _make_http_handler()

        with patch(
            "aragora.server.handlers.pipeline.decomposition.DAGOperationsCoordinator",
            side_effect=ImportError("not available"),
        ), patch(
            "aragora.nomic.task_decomposer.TaskDecomposer"
        ) as MockDecomposer:
            MockDecomposer.return_value.analyze.return_value = mock_result
            result = h.handle_post(
                "/api/v1/pipeline/graph-1/decompose/node-1", {}, http
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["pipeline_id"] == "graph-1"
        assert body["node_id"] == "node-1"
        assert body["success"] is True
        assert body["should_decompose"] is True
        assert len(body["subtasks"]) == 1
        assert body["subtasks"][0]["id"] == "sub-1"

    def test_decompose_unavailable_returns_503(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()

        with patch(
            "aragora.server.handlers.pipeline.decomposition.DAGOperationsCoordinator",
            side_effect=ImportError("not available"),
        ), patch(
            "aragora.nomic.task_decomposer.TaskDecomposer",
            side_effect=ImportError("not available"),
        ):
            result = h.handle_post(
                "/api/v1/pipeline/graph-1/decompose/node-1", {}, http
            )

        assert _status(result) == 503

    def test_decompose_unrecognized_path_returns_none(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post("/api/v1/pipeline/graphs", {}, http)
        assert result is None

    def test_decompose_invalid_pipeline_id(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/../../hack/decompose/node-1", {}, http
        )
        assert _status(result) == 400

    def test_decompose_invalid_node_id(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle_post(
            "/api/v1/pipeline/graph-1/decompose/<script>", {}, http
        )
        assert _status(result) == 400


# ===========================================================================
# GET /api/v1/pipeline/:id/decompose/:node_id/tree  (get tree)
# ===========================================================================


class TestGetDecompositionTree:
    """Tests for getting the decomposition tree in React Flow format."""

    def test_tree_graph_not_found(self, patched_store):
        patched_store.get.return_value = None
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle(
            "/api/v1/pipeline/graph-1/decompose/node-1/tree", {}, http
        )
        assert _status(result) == 404

    def test_tree_node_not_found(self, patched_store):
        graph = _mock_graph()
        graph.nodes = {}
        patched_store.get.return_value = graph
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle(
            "/api/v1/pipeline/graph-1/decompose/node-missing/tree", {}, http
        )
        assert _status(result) == 404

    def test_tree_success_leaf_node(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1", label="Root Task")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle(
            "/api/v1/pipeline/graph-1/decompose/node-1/tree", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["pipeline_id"] == "graph-1"
        assert body["root_node_id"] == "node-1"
        assert "tree" in body
        assert body["tree"]["id"] == "node-1"
        assert "react_flow" in body
        assert "nodes" in body["react_flow"]
        assert "edges" in body["react_flow"]

    def test_tree_react_flow_node_format(self, patched_store):
        graph = _mock_graph()
        node = _mock_node("node-1", label="Root")
        graph.nodes = {"node-1": node}
        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle(
            "/api/v1/pipeline/graph-1/decompose/node-1/tree", {}, http
        )
        body = _body(result)
        rf_nodes = body["react_flow"]["nodes"]
        assert len(rf_nodes) >= 1
        rf_node = rf_nodes[0]
        assert rf_node["id"] == "node-1"
        assert rf_node["type"] == "decomposition"
        assert "position" in rf_node
        assert "data" in rf_node
        assert rf_node["data"]["label"] == "Root"

    def test_tree_with_children(self, patched_store):
        graph = _mock_graph()
        parent = _mock_node("parent", label="Parent")
        child = _mock_node("child", label="Child")
        graph.nodes = {"parent": parent, "child": child}

        # Create an edge that looks like a decomposition
        edge = MagicMock()
        edge.source_id = "parent"
        edge.target_id = "child"
        edge.label = "decomposes_into"
        graph.edges = {"e1": edge}

        patched_store.get.return_value = graph

        h = _make_handler()
        http = _make_http_handler()
        result = h.handle(
            "/api/v1/pipeline/graph-1/decompose/parent/tree", {}, http
        )
        body = _body(result)
        tree = body["tree"]
        assert len(tree["children"]) == 1
        assert tree["children"][0]["id"] == "child"

        rf = body["react_flow"]
        assert len(rf["nodes"]) == 2
        assert len(rf["edges"]) == 1
        assert rf["edges"][0]["source"] == "parent"
        assert rf["edges"][0]["target"] == "child"

    def test_tree_unrecognized_sub_returns_none(self, patched_store):
        h = _make_handler()
        http = _make_http_handler()
        result = h.handle(
            "/api/v1/pipeline/graph-1/decompose/node-1/unknown", {}, http
        )
        assert result is None


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limit enforcement."""

    def test_handle_rate_limited(self, patched_store):
        h = _make_handler()
        http = _make_http_handler(client_ip="10.0.0.1")
        with patch.object(_decompose_limiter, "is_allowed", return_value=False):
            result = h.handle("/api/v1/pipeline/graph-1/decompose/node-1/tree", {}, http)
        assert _status(result) == 429

    def test_handle_post_rate_limited(self, patched_store):
        h = _make_handler()
        http = _make_http_handler(client_ip="10.0.0.2")
        with patch.object(_decompose_limiter, "is_allowed", return_value=False):
            result = h.handle_post(
                "/api/v1/pipeline/graph-1/decompose/node-1", {}, http
            )
        assert _status(result) == 429


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    """Tests for handler initialization."""

    def test_default_ctx(self):
        h = DecompositionHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = DecompositionHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_defaults_to_empty(self):
        h = DecompositionHandler(ctx=None)
        assert h.ctx == {}


__all__ = []
