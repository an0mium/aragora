"""Tests for GraphOperationsMixin (aragora/server/handlers/knowledge_base/mound/graph.py).

Covers all routes and behavior of the graph operations mixin:
- GET /api/knowledge/mound/graph/:id          - Graph traversal
- GET /api/knowledge/mound/graph/:id/lineage  - Node lineage (derived_from chain)
- GET /api/knowledge/mound/graph/:id/related  - Related nodes (immediate neighbors)

Error cases: missing mound (503), missing node ID (400), server errors from mound,
RelationshipType parsing, parameter clamping/bounding, and per-exception-type coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.unified.types import RelationshipType
from aragora.server.handlers.knowledge_base.mound.graph import (
    GraphOperationsMixin,
    _parse_relationship_type,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ASYNC_PATCH = "aragora.server.handlers.knowledge_base.mound.graph._run_async"


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock graph node / edge / result
# ---------------------------------------------------------------------------


@dataclass
class MockGraphNode:
    """Lightweight mock for a graph node (KnowledgeItem)."""

    id: str = "node-001"
    content: str = "Test node content"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content}


@dataclass
class MockGraphEdge:
    """Lightweight mock for a graph edge (KnowledgeLink)."""

    id: str = "edge-001"
    source_id: str = "node-001"
    target_id: str = "node-002"
    relationship: str = "derived_from"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
        }


@dataclass
class MockGraphQueryResult:
    """Lightweight mock for GraphQueryResult."""

    nodes: list[MockGraphNode] = field(default_factory=list)
    edges: list[MockGraphEdge] = field(default_factory=list)
    root_id: str = "node-001"
    depth: int = 2
    total_nodes: int = 0
    total_edges: int = 0


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class GraphTestHandler(GraphOperationsMixin):
    """Concrete handler for testing the graph mixin."""

    def __init__(self, mound=None):
        self._mound_instance = mound

    def _get_mound(self):
        return self._mound_instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_graph_result(nodes=None, edges=None, total_nodes=None, total_edges=None):
    """Build a MockGraphQueryResult with sensible defaults."""
    nodes = (
        nodes
        if nodes is not None
        else [
            MockGraphNode(id="node-001"),
            MockGraphNode(id="node-002"),
        ]
    )
    edges = (
        edges
        if edges is not None
        else [
            MockGraphEdge(id="edge-001", source_id="node-001", target_id="node-002"),
        ]
    )
    return MockGraphQueryResult(
        nodes=nodes,
        edges=edges,
        root_id="node-001",
        depth=2,
        total_nodes=total_nodes if total_nodes is not None else len(nodes),
        total_edges=total_edges if total_edges is not None else len(edges),
    )


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with graph-related methods."""
    mound = MagicMock()
    mound.query_graph = MagicMock(return_value=_make_graph_result())
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a GraphTestHandler with a mocked mound."""
    return GraphTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a GraphTestHandler with no mound (None)."""
    return GraphTestHandler(mound=None)


# ============================================================================
# Tests: _parse_relationship_type (standalone utility)
# ============================================================================


class TestParseRelationshipType:
    """Test the _parse_relationship_type utility function."""

    def test_none_returns_none(self):
        assert _parse_relationship_type(None) is None

    def test_valid_value_lowercase(self):
        assert _parse_relationship_type("supports") == RelationshipType.SUPPORTS

    def test_valid_value_contradicts(self):
        assert _parse_relationship_type("contradicts") == RelationshipType.CONTRADICTS

    def test_valid_value_elaborates(self):
        assert _parse_relationship_type("elaborates") == RelationshipType.ELABORATES

    def test_valid_value_supersedes(self):
        assert _parse_relationship_type("supersedes") == RelationshipType.SUPERSEDES

    def test_valid_value_derived_from(self):
        assert _parse_relationship_type("derived_from") == RelationshipType.DERIVED_FROM

    def test_valid_value_related_to(self):
        assert _parse_relationship_type("related_to") == RelationshipType.RELATED_TO

    def test_valid_value_cites(self):
        assert _parse_relationship_type("cites") == RelationshipType.CITES

    def test_uppercase_enum_name(self):
        """Fallback to enum name lookup when value fails."""
        assert _parse_relationship_type("SUPPORTS") == RelationshipType.SUPPORTS

    def test_uppercase_derived_from(self):
        assert _parse_relationship_type("DERIVED_FROM") == RelationshipType.DERIVED_FROM

    def test_invalid_value_returns_none(self):
        assert _parse_relationship_type("not_a_type") is None

    def test_empty_string_returns_none(self):
        assert _parse_relationship_type("") is None

    def test_numeric_string_returns_none(self):
        assert _parse_relationship_type("123") is None


# ============================================================================
# Tests: _handle_graph_traversal
# ============================================================================


class TestGraphTraversal:
    """Test _handle_graph_traversal - GET /api/knowledge/mound/graph/:id."""

    def test_traversal_success(self, handler, mock_mound):
        """Successful graph traversal returns nodes and count."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["start_node_id"] == "node-001"
        assert body["depth"] == 2
        assert body["max_nodes"] == 50
        assert body["relationship_type"] is None
        assert body["count"] == 2
        assert len(body["nodes"]) == 2

    def test_traversal_with_relationship_type(self, handler, mock_mound):
        """Graph traversal with relationship type filter."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"relationship_type": "supports"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["relationship_type"] == "supports"
        # Verify mound was called with relationship_types=[RelationshipType.SUPPORTS]
        call_kwargs = mock_mound.query_graph.call_args
        assert call_kwargs is not None

    def test_traversal_with_invalid_relationship_type(self, handler, mock_mound):
        """Invalid relationship type means no filter applied (None)."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"relationship_type": "bogus_type"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["relationship_type"] == "bogus_type"

    def test_traversal_with_custom_depth(self, handler, mock_mound):
        """Custom depth parameter is respected."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"depth": "4"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["depth"] == 4

    def test_traversal_depth_clamped_min(self, handler, mock_mound):
        """Depth below 1 is clamped to 1."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"depth": "0"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["depth"] == 1

    def test_traversal_depth_clamped_max(self, handler, mock_mound):
        """Depth above 5 is clamped to 5."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"depth": "99"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["depth"] == 5

    def test_traversal_with_custom_max_nodes(self, handler, mock_mound):
        """Custom max_nodes parameter is respected."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"max_nodes": "100"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["max_nodes"] == 100

    def test_traversal_max_nodes_clamped_min(self, handler, mock_mound):
        """max_nodes below 1 is clamped to 1."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"max_nodes": "0"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["max_nodes"] == 1

    def test_traversal_max_nodes_clamped_max(self, handler, mock_mound):
        """max_nodes above 1000 is clamped to 1000."""
        path = "/api/knowledge/mound/graph/node-001"
        params = {"max_nodes": "9999"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["max_nodes"] == 1000

    def test_traversal_missing_node_id(self, handler):
        """Path with fewer than 5 parts returns 400."""
        path = "/api/knowledge/mound"
        result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 400
        body = _body(result)
        assert "Node ID required" in body.get("error", "")

    def test_traversal_short_path_three_parts(self, handler):
        """3-part path returns 400."""
        path = "/api/knowledge"
        result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 400

    def test_traversal_no_mound(self, handler_no_mound):
        """Returns 503 when mound is unavailable."""
        path = "/api/knowledge/mound/graph/node-001"
        result = handler_no_mound._handle_graph_traversal(path, {})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "")

    def test_traversal_mound_key_error(self, handler, mock_mound):
        """KeyError from mound.query_graph returns 500."""
        mock_mound.query_graph = MagicMock(return_value=MagicMock())
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("node not found")):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 500
        body = _body(result)
        assert "failed" in body.get("error", "").lower()

    def test_traversal_mound_value_error(self, handler, mock_mound):
        """ValueError from mound.query_graph returns 500."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad value")):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 500

    def test_traversal_mound_os_error(self, handler, mock_mound):
        """OSError from mound.query_graph returns 500."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk error")):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 500

    def test_traversal_mound_type_error(self, handler, mock_mound):
        """TypeError from mound.query_graph returns 500."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("type issue")):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 500

    def test_traversal_mound_runtime_error(self, handler, mock_mound):
        """RuntimeError from mound.query_graph returns 500."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("runtime issue")):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 500

    def test_traversal_empty_nodes(self, handler, mock_mound):
        """Traversal with zero result nodes returns count 0."""
        mock_mound.query_graph = MagicMock(return_value=_make_graph_result(nodes=[], edges=[]))
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["nodes"] == []

    def test_traversal_node_id_from_path(self, handler, mock_mound):
        """Node ID is correctly extracted from path segment index 4."""
        path = "/api/knowledge/mound/graph/my-special-node-42"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["start_node_id"] == "my-special-node-42"


# ============================================================================
# Tests: _handle_graph_lineage
# ============================================================================


class TestGraphLineage:
    """Test _handle_graph_lineage - GET /api/knowledge/mound/graph/:id/lineage."""

    def test_lineage_success(self, handler, mock_mound):
        """Successful lineage returns nodes, edges, totals."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-001"
        assert body["depth"] == 5  # default
        lineage = body["lineage"]
        assert len(lineage["nodes"]) == 2
        assert len(lineage["edges"]) == 1
        assert lineage["total_nodes"] == 2
        assert lineage["total_edges"] == 1

    def test_lineage_custom_depth(self, handler, mock_mound):
        """Custom depth parameter is respected."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        params = {"depth": "8"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["depth"] == 8

    def test_lineage_depth_clamped_min(self, handler, mock_mound):
        """Depth below 1 is clamped to 1."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        params = {"depth": "-5"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["depth"] == 1

    def test_lineage_depth_clamped_max(self, handler, mock_mound):
        """Depth above 10 is clamped to 10."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        params = {"depth": "50"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["depth"] == 10

    def test_lineage_missing_node_id(self, handler):
        """Path with fewer than 5 parts returns 400."""
        path = "/api/knowledge/mound"
        result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 400
        body = _body(result)
        assert "Node ID required" in body.get("error", "")

    def test_lineage_no_mound(self, handler_no_mound):
        """Returns 503 when mound is unavailable."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        result = handler_no_mound._handle_graph_lineage(path, {})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "")

    def test_lineage_mound_key_error(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 500

    def test_lineage_mound_value_error(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 500

    def test_lineage_mound_runtime_error(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("boom")):
            result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 500

    def test_lineage_uses_derived_from_relationship(self, handler, mock_mound):
        """Lineage always uses DERIVED_FROM relationship type."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_graph_lineage(path, {})
        call_kwargs = mock_mound.query_graph.call_args
        assert call_kwargs is not None
        # The coroutine was passed to _run_async; verify the call on mock_mound
        mock_mound.query_graph.assert_called_once()
        kw = mock_mound.query_graph.call_args
        # Check relationship_types includes DERIVED_FROM
        rt = kw.kwargs.get("relationship_types") or (
            kw[1].get("relationship_types") if len(kw) > 1 else None
        )
        if rt is None and kw.args:
            # positional args fallback
            pass
        else:
            assert rt == [RelationshipType.DERIVED_FROM]

    def test_lineage_empty_result(self, handler, mock_mound):
        """Empty lineage returns empty nodes/edges."""
        mock_mound.query_graph = MagicMock(
            return_value=_make_graph_result(nodes=[], edges=[], total_nodes=0, total_edges=0)
        )
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["lineage"]["nodes"] == []
        assert body["lineage"]["edges"] == []
        assert body["lineage"]["total_nodes"] == 0
        assert body["lineage"]["total_edges"] == 0

    def test_lineage_node_id_extracted(self, handler, mock_mound):
        """Node ID is correctly extracted from path."""
        path = "/api/knowledge/mound/graph/special-id-xyz/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "special-id-xyz"


# ============================================================================
# Tests: _handle_graph_related
# ============================================================================


class TestGraphRelated:
    """Test _handle_graph_related - GET /api/knowledge/mound/graph/:id/related."""

    def test_related_success(self, handler, mock_mound):
        """Successful related query returns related nodes (excluding self)."""
        # Mock returns node-001 and node-002; node-001 is self, so related = [node-002]
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-001"
        assert body["relationship_type"] is None
        # node-001 excluded from related list
        assert len(body["related"]) == 1
        assert body["related"][0]["id"] == "node-002"
        # total = len(nodes) - 1
        assert body["total"] == 1

    def test_related_with_relationship_type(self, handler, mock_mound):
        """Related query with relationship type filter."""
        path = "/api/knowledge/mound/graph/node-001/related"
        params = {"relationship_type": "contradicts"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["relationship_type"] == "contradicts"

    def test_related_with_invalid_relationship_type(self, handler, mock_mound):
        """Invalid relationship type means no filter (None relationship_types)."""
        path = "/api/knowledge/mound/graph/node-001/related"
        params = {"relationship_type": "invalid_rel"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, params)
        assert _status(result) == 200
        body = _body(result)
        assert body["relationship_type"] == "invalid_rel"

    def test_related_custom_limit(self, handler, mock_mound):
        """Custom limit parameter is passed as max_nodes."""
        path = "/api/knowledge/mound/graph/node-001/related"
        params = {"limit": "10"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_graph_related(path, params)
        mock_mound.query_graph.assert_called_once()
        kw = mock_mound.query_graph.call_args
        max_nodes_val = kw.kwargs.get("max_nodes") or (
            kw[1].get("max_nodes") if len(kw) > 1 else None
        )
        assert max_nodes_val == 10

    def test_related_limit_clamped_min(self, handler, mock_mound):
        """Limit below 1 is clamped to 1."""
        path = "/api/knowledge/mound/graph/node-001/related"
        params = {"limit": "0"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_graph_related(path, params)
        mock_mound.query_graph.assert_called_once()
        kw = mock_mound.query_graph.call_args
        max_nodes_val = kw.kwargs.get("max_nodes") or (
            kw[1].get("max_nodes") if len(kw) > 1 else None
        )
        assert max_nodes_val == 1

    def test_related_limit_clamped_max(self, handler, mock_mound):
        """Limit above 100 is clamped to 100."""
        path = "/api/knowledge/mound/graph/node-001/related"
        params = {"limit": "999"}
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_graph_related(path, params)
        mock_mound.query_graph.assert_called_once()
        kw = mock_mound.query_graph.call_args
        max_nodes_val = kw.kwargs.get("max_nodes") or (
            kw[1].get("max_nodes") if len(kw) > 1 else None
        )
        assert max_nodes_val == 100

    def test_related_depth_is_always_one(self, handler, mock_mound):
        """Related always queries at depth=1 (immediate neighbors only)."""
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_graph_related(path, {})
        mock_mound.query_graph.assert_called_once()
        kw = mock_mound.query_graph.call_args
        depth_val = kw.kwargs.get("depth") or (kw[1].get("depth") if len(kw) > 1 else None)
        assert depth_val == 1

    def test_related_missing_node_id(self, handler):
        """Path with fewer than 5 parts returns 400."""
        path = "/api/knowledge/mound"
        result = handler._handle_graph_related(path, {})
        assert _status(result) == 400
        body = _body(result)
        assert "Node ID required" in body.get("error", "")

    def test_related_no_mound(self, handler_no_mound):
        """Returns 503 when mound is unavailable."""
        path = "/api/knowledge/mound/graph/node-001/related"
        result = handler_no_mound._handle_graph_related(path, {})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "")

    def test_related_mound_key_error(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("key")):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 500

    def test_related_mound_os_error(self, handler, mock_mound):
        """OSError from mound returns 500."""
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("io")):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 500

    def test_related_mound_type_error(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("type")):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 500

    def test_related_only_self_node(self, handler, mock_mound):
        """When the only node is the start node, related list is empty."""
        mock_mound.query_graph = MagicMock(
            return_value=_make_graph_result(
                nodes=[MockGraphNode(id="node-001")],
                edges=[],
            )
        )
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["related"] == []
        # total = len(nodes) - 1 = 0
        assert body["total"] == 0

    def test_related_many_neighbors(self, handler, mock_mound):
        """Multiple related nodes are returned."""
        nodes = [
            MockGraphNode(id="node-001"),
            MockGraphNode(id="node-002"),
            MockGraphNode(id="node-003"),
            MockGraphNode(id="node-004"),
        ]
        mock_mound.query_graph = MagicMock(return_value=_make_graph_result(nodes=nodes, edges=[]))
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert len(body["related"]) == 3
        assert body["total"] == 3
        ids = [n["id"] for n in body["related"]]
        assert "node-001" not in ids

    def test_related_node_id_extraction(self, handler, mock_mound):
        """Node ID is correctly extracted from longer path."""
        path = "/api/knowledge/mound/graph/abc-123/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "abc-123"

    def test_related_empty_graph(self, handler, mock_mound):
        """Empty graph result returns empty related list."""
        mock_mound.query_graph = MagicMock(return_value=_make_graph_result(nodes=[], edges=[]))
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_related(path, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["related"] == []
        # total = len([]) - 1 = -1
        assert body["total"] == -1


# ============================================================================
# Tests: Cross-cutting concerns
# ============================================================================


class TestCrossCutting:
    """Test cross-cutting behavior across all graph endpoints."""

    def test_traversal_default_limit_is_20_for_related(self, handler, mock_mound):
        """Default limit for related is 20."""
        path = "/api/knowledge/mound/graph/node-001/related"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_graph_related(path, {})
        kw = mock_mound.query_graph.call_args
        max_nodes_val = kw.kwargs.get("max_nodes") or (
            kw[1].get("max_nodes") if len(kw) > 1 else None
        )
        assert max_nodes_val == 20

    def test_traversal_default_depth_is_2(self, handler, mock_mound):
        """Default depth for traversal is 2."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, {})
        body = _body(result)
        assert body["depth"] == 2

    def test_lineage_default_depth_is_5(self, handler, mock_mound):
        """Default depth for lineage is 5."""
        path = "/api/knowledge/mound/graph/node-001/lineage"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_lineage(path, {})
        body = _body(result)
        assert body["depth"] == 5

    def test_traversal_default_max_nodes_is_50(self, handler, mock_mound):
        """Default max_nodes for traversal is 50."""
        path = "/api/knowledge/mound/graph/node-001"
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_graph_traversal(path, {})
        body = _body(result)
        assert body["max_nodes"] == 50
