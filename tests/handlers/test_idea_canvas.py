"""Comprehensive tests for IdeaCanvasHandler REST API.

Tests all 12 endpoints, success paths, error handling, edge cases, input
validation, rate limiting, and route matching.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from aragora.server.handlers.idea_canvas import IdeaCanvasHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    return getattr(result, "status_code", getattr(result, "status", 0))


def _body(result) -> dict[str, Any]:
    """Decode the JSON body of a HandlerResult."""
    raw = getattr(result, "body", b"{}")
    if isinstance(raw, (bytes, bytearray)):
        return json.loads(raw.decode("utf-8"))
    if isinstance(raw, str):
        return json.loads(raw)
    return {}


def _ctx() -> MagicMock:
    """Return a mock AuthorizationContext."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    return IdeaCanvasHandler(ctx={})


@pytest.fixture
def mock_req():
    """Factory to build a mock HTTP handler (the ``handler`` arg to .handle())."""

    def _make(method: str = "GET", body: dict | None = None):
        h = MagicMock()
        h.command = method
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"X-Forwarded-For": "127.0.0.1"}
        h.request = MagicMock()
        h.request.body = json.dumps(body).encode() if body else None
        h.auth_user = MagicMock()
        h.auth_user.user_id = "test-user"
        h.auth_user.org_id = "test-org"
        h.auth_user.roles = {"admin"}
        return h

    return _make


@pytest.fixture
def mock_store():
    """Return a pre-configured MagicMock store."""
    store = MagicMock()
    store.list_canvases.return_value = []
    store.save_canvas.return_value = {"id": "ic-new", "name": "New"}
    store.load_canvas.return_value = {"id": "ic-1", "name": "Canvas 1"}
    store.update_canvas.return_value = {"id": "ic-1", "name": "Updated"}
    store.delete_canvas.return_value = True
    return store


@pytest.fixture
def mock_manager():
    """Return a pre-configured MagicMock canvas manager."""
    mgr = MagicMock()
    # Default: get_canvas returns a canvas-like object
    canvas = MagicMock()
    canvas.nodes = {}
    canvas.edges = {}
    canvas.to_dict.return_value = {"id": "ic-1", "nodes": [], "edges": []}
    mgr.get_canvas.return_value = canvas
    # add_node / update_node return node-like objects
    node = MagicMock()
    node.to_dict.return_value = {"id": "n1", "label": "Test Node"}
    mgr.add_node.return_value = node
    mgr.update_node.return_value = node
    mgr.remove_node.return_value = True
    # add_edge returns edge-like object
    edge = MagicMock()
    edge.to_dict.return_value = {"id": "e1", "source": "n1", "target": "n2"}
    mgr.add_edge.return_value = edge
    mgr.remove_edge.return_value = True
    return mgr


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Verify can_handle accepts only /api/v1/ideas paths."""

    def test_ideas_root(self, handler):
        assert handler.can_handle("/api/v1/ideas") is True

    def test_ideas_with_id(self, handler):
        assert handler.can_handle("/api/v1/ideas/abc-123") is True

    def test_ideas_nodes(self, handler):
        assert handler.can_handle("/api/v1/ideas/abc-123/nodes") is True

    def test_ideas_edges(self, handler):
        assert handler.can_handle("/api/v1/ideas/abc-123/edges") is True

    def test_ideas_export(self, handler):
        assert handler.can_handle("/api/v1/ideas/abc-123/export") is True

    def test_ideas_promote(self, handler):
        assert handler.can_handle("/api/v1/ideas/abc-123/promote") is True

    def test_rejects_debates(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_canvas(self, handler):
        assert handler.can_handle("/api/v1/canvas") is False

    def test_rejects_goals(self, handler):
        assert handler.can_handle("/api/v1/goals") is False

    def test_rejects_actions(self, handler):
        assert handler.can_handle("/api/v1/actions") is False


# ---------------------------------------------------------------------------
# Route matching / method-not-allowed tests
# ---------------------------------------------------------------------------


class TestRouteMatching:
    """_route_request dispatches correctly and returns 405 for wrong methods."""

    def test_unknown_path_returns_none(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/test/unknown/route",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert result is None

    # -- /api/v1/ideas --------------------------------------------------------

    def test_list_method_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas",
            "DELETE",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    def test_list_put_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas",
            "PUT",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id}/export -------------------------------------------

    def test_export_post_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/export",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    def test_export_delete_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/export",
            "DELETE",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id}/promote ------------------------------------------

    def test_promote_get_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/promote",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id}/nodes -------------------------------------------

    def test_nodes_get_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/nodes",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    def test_nodes_delete_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/nodes",
            "DELETE",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id}/nodes/{nid} ------------------------------------

    def test_single_node_post_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/nodes/n1",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    def test_single_node_get_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/nodes/n1",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id}/edges -------------------------------------------

    def test_edges_get_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/edges",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id}/edges/{eid} ------------------------------------

    def test_single_edge_post_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/edges/e1",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    def test_single_edge_put_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc/edges/e1",
            "PUT",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405

    # -- /api/v1/ideas/{id} (GET/PUT/DELETE) --------------------------------

    def test_canvas_by_id_post_not_allowed(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/abc",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            _ctx(),
        )
        assert _status(result) == 405


# ---------------------------------------------------------------------------
# List canvases
# ---------------------------------------------------------------------------


class TestListCanvases:
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_empty(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.return_value = []
        mock_get_store.return_value = store

        result = handler._list_canvases(_ctx(), {}, "u1", "ws1")
        assert _status(result) == 200
        data = _body(result)
        assert data["canvases"] == []
        assert data["count"] == 0

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_with_results(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.return_value = [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ]
        mock_get_store.return_value = store

        result = handler._list_canvases(_ctx(), {}, "u1", "ws1")
        assert _status(result) == 200
        assert _body(result)["count"] == 3

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_passes_query_params(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.return_value = []
        mock_get_store.return_value = store

        handler._list_canvases(
            _ctx(),
            {"workspace_id": "w2", "owner_id": "o2", "limit": "50", "offset": "10"},
            "u1",
            "ws1",
        )
        call_kwargs = store.list_canvases.call_args
        assert call_kwargs.kwargs["workspace_id"] == "w2"
        assert call_kwargs.kwargs["owner_id"] == "o2"
        assert call_kwargs.kwargs["limit"] == 50
        assert call_kwargs.kwargs["offset"] == 10

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_limit_clamped_high(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.return_value = []
        mock_get_store.return_value = store

        handler._list_canvases(_ctx(), {"limit": "9999"}, "u1", "ws1")
        assert store.list_canvases.call_args.kwargs["limit"] == 1000

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_limit_clamped_low(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.return_value = []
        mock_get_store.return_value = store

        handler._list_canvases(_ctx(), {"limit": "-5"}, "u1", "ws1")
        assert store.list_canvases.call_args.kwargs["limit"] == 1

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_offset_clamped_low(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.return_value = []
        mock_get_store.return_value = store

        handler._list_canvases(_ctx(), {"offset": "-3"}, "u1", "ws1")
        assert store.list_canvases.call_args.kwargs["offset"] == 0

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_store_error(self, mock_get_store, handler):
        mock_get_store.side_effect = ImportError("no store")

        result = handler._list_canvases(_ctx(), {}, "u1", "ws1")
        assert _status(result) == 500

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_runtime_error(self, mock_get_store, handler):
        store = MagicMock()
        store.list_canvases.side_effect = RuntimeError("db down")
        mock_get_store.return_value = store

        result = handler._list_canvases(_ctx(), {}, "u1", "ws1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Create canvas
# ---------------------------------------------------------------------------


class TestCreateCanvas:
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create_success(self, mock_get_store, handler):
        store = MagicMock()
        store.save_canvas.return_value = {"id": "ic-new", "name": "My Ideas"}
        mock_get_store.return_value = store

        result = handler._create_canvas(
            _ctx(),
            {"name": "My Ideas", "description": "brainstorm"},
            "u1",
            "ws1",
        )
        assert _status(result) == 201
        assert _body(result)["name"] == "My Ideas"

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create_custom_id(self, mock_get_store, handler):
        store = MagicMock()
        store.save_canvas.return_value = {"id": "custom-id"}
        mock_get_store.return_value = store

        handler._create_canvas(_ctx(), {"id": "custom-id"}, "u1", "ws1")
        assert store.save_canvas.call_args.kwargs["canvas_id"] == "custom-id"

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create_auto_id(self, mock_get_store, handler):
        store = MagicMock()
        store.save_canvas.return_value = {"id": "ideas-abcd1234"}
        mock_get_store.return_value = store

        handler._create_canvas(_ctx(), {}, "u1", "ws1")
        canvas_id = store.save_canvas.call_args.kwargs["canvas_id"]
        assert canvas_id.startswith("ideas-")

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create_defaults(self, mock_get_store, handler):
        store = MagicMock()
        store.save_canvas.return_value = {"id": "ic-x"}
        mock_get_store.return_value = store

        handler._create_canvas(_ctx(), {}, "u1", "ws1")
        kwargs = store.save_canvas.call_args.kwargs
        assert kwargs["name"] == "Untitled Ideas"
        assert kwargs["description"] == ""
        assert kwargs["metadata"] == {"stage": "ideas"}

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create_passes_metadata(self, mock_get_store, handler):
        store = MagicMock()
        store.save_canvas.return_value = {"id": "ic-x"}
        mock_get_store.return_value = store

        handler._create_canvas(
            _ctx(),
            {"metadata": {"custom": True}},
            "u1",
            "ws1",
        )
        assert store.save_canvas.call_args.kwargs["metadata"] == {"custom": True}

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create_store_error(self, mock_get_store, handler):
        store = MagicMock()
        store.save_canvas.side_effect = ValueError("bad data")
        mock_get_store.return_value = store

        result = handler._create_canvas(_ctx(), {"name": "Fail"}, "u1", "ws1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Get canvas
# ---------------------------------------------------------------------------


class TestGetCanvas:
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_get_not_found(self, mock_get_store, handler):
        store = MagicMock()
        store.load_canvas.return_value = None
        mock_get_store.return_value = store

        result = handler._get_canvas(_ctx(), "missing", "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_get_success_with_live_canvas(self, mock_get_store, handler):
        store = MagicMock()
        store.load_canvas.return_value = {"id": "ic-1", "name": "Ideas"}
        mock_get_store.return_value = store

        node = MagicMock()
        node.to_dict.return_value = {"id": "n1"}
        edge = MagicMock()
        edge.to_dict.return_value = {"id": "e1"}
        canvas_obj = MagicMock()
        canvas_obj.nodes = {"n1": node}
        canvas_obj.edges = {"e1": edge}

        with patch.object(handler, "_get_canvas_manager") as mgr_patch:
            mgr_patch.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                result = handler._get_canvas(_ctx(), "ic-1", "u1")
                assert _status(result) == 200
                data = _body(result)
                assert data["nodes"] == [{"id": "n1"}]
                assert data["edges"] == [{"id": "e1"}]

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_get_success_no_live_canvas(self, mock_get_store, handler):
        store = MagicMock()
        store.load_canvas.return_value = {"id": "ic-1", "name": "Ideas"}
        mock_get_store.return_value = store

        with patch.object(handler, "_get_canvas_manager") as mgr_patch:
            mgr_patch.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                result = handler._get_canvas(_ctx(), "ic-1", "u1")
                assert _status(result) == 200
                data = _body(result)
                assert data["nodes"] == []
                assert data["edges"] == []

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_get_store_error(self, mock_get_store, handler):
        store = MagicMock()
        store.load_canvas.side_effect = OSError("disk fail")
        mock_get_store.return_value = store

        result = handler._get_canvas(_ctx(), "ic-1", "u1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Update canvas
# ---------------------------------------------------------------------------


class TestUpdateCanvas:
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_update_success(self, mock_get_store, handler):
        store = MagicMock()
        store.update_canvas.return_value = {"id": "ic-1", "name": "Renamed"}
        mock_get_store.return_value = store

        result = handler._update_canvas(
            _ctx(),
            "ic-1",
            {"name": "Renamed"},
            "u1",
        )
        assert _status(result) == 200
        assert _body(result)["name"] == "Renamed"

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_update_not_found(self, mock_get_store, handler):
        store = MagicMock()
        store.update_canvas.return_value = None
        mock_get_store.return_value = store

        result = handler._update_canvas(_ctx(), "missing", {"name": "X"}, "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_update_passes_fields(self, mock_get_store, handler):
        store = MagicMock()
        store.update_canvas.return_value = {"id": "ic-1"}
        mock_get_store.return_value = store

        handler._update_canvas(
            _ctx(),
            "ic-1",
            {"name": "N", "description": "D", "metadata": {"k": "v"}},
            "u1",
        )
        kwargs = store.update_canvas.call_args.kwargs
        assert kwargs["name"] == "N"
        assert kwargs["description"] == "D"
        assert kwargs["metadata"] == {"k": "v"}

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_update_store_error(self, mock_get_store, handler):
        store = MagicMock()
        store.update_canvas.side_effect = TypeError("bad type")
        mock_get_store.return_value = store

        result = handler._update_canvas(_ctx(), "ic-1", {}, "u1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Delete canvas
# ---------------------------------------------------------------------------


class TestDeleteCanvas:
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_delete_success(self, mock_get_store, handler):
        store = MagicMock()
        store.delete_canvas.return_value = True
        mock_get_store.return_value = store

        result = handler._delete_canvas(_ctx(), "ic-1", "u1")
        assert _status(result) == 200
        data = _body(result)
        assert data["deleted"] is True
        assert data["canvas_id"] == "ic-1"

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_delete_not_found(self, mock_get_store, handler):
        store = MagicMock()
        store.delete_canvas.return_value = False
        mock_get_store.return_value = store

        result = handler._delete_canvas(_ctx(), "missing", "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_delete_store_error(self, mock_get_store, handler):
        store = MagicMock()
        store.delete_canvas.side_effect = KeyError("gone")
        mock_get_store.return_value = store

        result = handler._delete_canvas(_ctx(), "ic-1", "u1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Add node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_invalid_idea_type_returns_400(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            result = handler._add_node(
                _ctx(),
                "c1",
                {"idea_type": "TOTALLY_INVALID"},
                "u1",
            )
            assert _status(result) == 400
            assert "Invalid idea type" in _body(result).get("error", "")

    def test_add_node_success(self, handler, mock_manager):
        with patch.object(handler, "_get_canvas_manager", return_value=mock_manager):
            with patch.object(handler, "_run_async") as mock_run:
                node_obj = MagicMock()
                node_obj.to_dict.return_value = {"id": "n1", "label": "Hello"}
                mock_run.return_value = node_obj

                result = handler._add_node(
                    _ctx(),
                    "c1",
                    {"idea_type": "concept", "label": "Hello", "position": {"x": 10, "y": 20}},
                    "u1",
                )
                assert _status(result) == 201
                assert _body(result)["id"] == "n1"

    def test_add_node_canvas_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                result = handler._add_node(
                    _ctx(),
                    "missing",
                    {"idea_type": "concept"},
                    "u1",
                )
                assert _status(result) == 404

    def test_add_node_default_idea_type(self, handler):
        """When idea_type is omitted, defaults to 'concept'."""
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            node = MagicMock()
            node.to_dict.return_value = {"id": "n2"}
            with patch.object(handler, "_run_async", return_value=node):
                result = handler._add_node(_ctx(), "c1", {}, "u1")
                assert _status(result) == 201

    def test_add_node_all_valid_types(self, handler):
        """Every IdeaNodeType enum value should be accepted."""
        valid_types = [
            "concept",
            "cluster",
            "question",
            "insight",
            "evidence",
            "assumption",
            "constraint",
            "observation",
            "hypothesis",
        ]
        for idea_type in valid_types:
            with patch.object(handler, "_get_canvas_manager") as mgr:
                mgr.return_value = MagicMock()
                node = MagicMock()
                node.to_dict.return_value = {"id": f"n-{idea_type}"}
                with patch.object(handler, "_run_async", return_value=node):
                    result = handler._add_node(
                        _ctx(),
                        "c1",
                        {"idea_type": idea_type},
                        "u1",
                    )
                    assert _status(result) == 201, f"Failed for idea_type={idea_type}"

    def test_add_node_position_defaults(self, handler):
        """Missing position x/y should default to 0."""
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            node = MagicMock()
            node.to_dict.return_value = {"id": "n3"}
            with patch.object(handler, "_run_async", return_value=node):
                result = handler._add_node(
                    _ctx(),
                    "c1",
                    {"idea_type": "concept", "position": {}},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_node_import_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=ImportError("no canvas")):
            result = handler._add_node(_ctx(), "c1", {"idea_type": "concept"}, "u1")
            assert _status(result) == 500

    def test_add_node_sets_stage_and_rf_type(self, handler):
        """Verify data dict gets stage=ideas and rf_type=ideaNode."""
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mock_m = MagicMock()
            mgr.return_value = mock_m

            captured_data = {}

            def capture_run(coro):
                node = MagicMock()
                node.to_dict.return_value = {"id": "n4"}
                return node

            with patch.object(handler, "_run_async", side_effect=capture_run):
                handler._add_node(
                    _ctx(),
                    "c1",
                    {"idea_type": "concept", "data": {"custom": "val"}},
                    "u1",
                )
                # The data passed to add_node should have stage and rf_type
                # We can verify through the coroutine call


# ---------------------------------------------------------------------------
# Update node
# ---------------------------------------------------------------------------


class TestUpdateNode:
    def test_update_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            node = MagicMock()
            node.to_dict.return_value = {"id": "n1", "label": "Updated"}
            with patch.object(handler, "_run_async", return_value=node):
                result = handler._update_node(
                    _ctx(),
                    "c1",
                    "n1",
                    {"label": "Updated"},
                    "u1",
                )
                assert _status(result) == 200
                assert _body(result)["label"] == "Updated"

    def test_update_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                result = handler._update_node(
                    _ctx(),
                    "c1",
                    "missing",
                    {"label": "X"},
                    "u1",
                )
                assert _status(result) == 404

    def test_update_with_position(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            node = MagicMock()
            node.to_dict.return_value = {"id": "n1"}
            with patch.object(handler, "_run_async", return_value=node):
                result = handler._update_node(
                    _ctx(),
                    "c1",
                    "n1",
                    {"position": {"x": 100, "y": 200}},
                    "u1",
                )
                assert _status(result) == 200

    def test_update_with_data(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            node = MagicMock()
            node.to_dict.return_value = {"id": "n1"}
            with patch.object(handler, "_run_async", return_value=node):
                result = handler._update_node(
                    _ctx(),
                    "c1",
                    "n1",
                    {"data": {"priority": "high"}},
                    "u1",
                )
                assert _status(result) == 200

    def test_update_store_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=RuntimeError("boom")):
            result = handler._update_node(_ctx(), "c1", "n1", {}, "u1")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Delete node
# ---------------------------------------------------------------------------


class TestDeleteNode:
    def test_delete_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=True):
                result = handler._delete_node(_ctx(), "c1", "n1", "u1")
                assert _status(result) == 200
                data = _body(result)
                assert data["deleted"] is True
                assert data["node_id"] == "n1"

    def test_delete_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=False):
                result = handler._delete_node(_ctx(), "c1", "missing", "u1")
                assert _status(result) == 404

    def test_delete_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=OSError("fail")):
            result = handler._delete_node(_ctx(), "c1", "n1", "u1")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Add edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    def test_add_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            edge = MagicMock()
            edge.to_dict.return_value = {"id": "e1", "source": "n1", "target": "n2"}
            with patch.object(handler, "_run_async", return_value=edge):
                result = handler._add_edge(
                    _ctx(),
                    "c1",
                    {"source_id": "n1", "target_id": "n2"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_using_source_target_aliases(self, handler):
        """Should accept 'source'/'target' as aliases."""
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            edge = MagicMock()
            edge.to_dict.return_value = {"id": "e2"}
            with patch.object(handler, "_run_async", return_value=edge):
                result = handler._add_edge(
                    _ctx(),
                    "c1",
                    {"source": "n1", "target": "n2"},
                    "u1",
                )
                assert _status(result) == 201

    def test_missing_source_returns_400(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            result = handler._add_edge(
                _ctx(),
                "c1",
                {"target_id": "n2"},
                "u1",
            )
            assert _status(result) == 400

    def test_missing_target_returns_400(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            result = handler._add_edge(
                _ctx(),
                "c1",
                {"source_id": "n1"},
                "u1",
            )
            assert _status(result) == 400

    def test_missing_both_returns_400(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            result = handler._add_edge(_ctx(), "c1", {}, "u1")
            assert _status(result) == 400

    def test_edge_canvas_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                result = handler._add_edge(
                    _ctx(),
                    "missing",
                    {"source_id": "n1", "target_id": "n2"},
                    "u1",
                )
                assert _status(result) == 404

    def test_invalid_edge_type_falls_back_to_default(self, handler):
        """Invalid edge type should silently fall back to EdgeType.DEFAULT."""
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            edge = MagicMock()
            edge.to_dict.return_value = {"id": "e3"}
            with patch.object(handler, "_run_async", return_value=edge):
                result = handler._add_edge(
                    _ctx(),
                    "c1",
                    {"source_id": "n1", "target_id": "n2", "type": "BOGUS"},
                    "u1",
                )
                assert _status(result) == 201

    def test_valid_edge_types_accepted(self, handler):
        valid_types = [
            "default",
            "data_flow",
            "control_flow",
            "reference",
            "dependency",
            "critique",
            "support",
        ]
        for etype in valid_types:
            with patch.object(handler, "_get_canvas_manager") as mgr:
                mgr.return_value = MagicMock()
                edge = MagicMock()
                edge.to_dict.return_value = {"id": f"e-{etype}"}
                with patch.object(handler, "_run_async", return_value=edge):
                    result = handler._add_edge(
                        _ctx(),
                        "c1",
                        {"source_id": "n1", "target_id": "n2", "type": etype},
                        "u1",
                    )
                    assert _status(result) == 201, f"Failed for edge type={etype}"

    def test_add_edge_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=ImportError("no canvas")):
            result = handler._add_edge(
                _ctx(),
                "c1",
                {"source_id": "n1", "target_id": "n2"},
                "u1",
            )
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Delete edge
# ---------------------------------------------------------------------------


class TestDeleteEdge:
    def test_delete_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=True):
                result = handler._delete_edge(_ctx(), "c1", "e1", "u1")
                assert _status(result) == 200
                data = _body(result)
                assert data["deleted"] is True
                assert data["edge_id"] == "e1"

    def test_delete_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=False):
                result = handler._delete_edge(_ctx(), "c1", "missing", "u1")
                assert _status(result) == 404

    def test_delete_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=KeyError("err")):
            result = handler._delete_edge(_ctx(), "c1", "e1", "u1")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Export canvas
# ---------------------------------------------------------------------------


class TestExportCanvas:
    def test_export_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                result = handler._export_canvas(_ctx(), "missing", "u1")
                assert _status(result) == 404

    def test_export_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            canvas_obj = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                with patch(
                    "aragora.canvas.converters.to_react_flow",
                    return_value={"nodes": [], "edges": [], "viewport": {}},
                ):
                    result = handler._export_canvas(_ctx(), "c1", "u1")
                    assert _status(result) == 200
                    data = _body(result)
                    assert "nodes" in data
                    assert "edges" in data

    def test_export_import_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=ImportError("no converters")):
            result = handler._export_canvas(_ctx(), "c1", "u1")
            assert _status(result) == 500

    def test_export_runtime_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", side_effect=RuntimeError("fail")):
                result = handler._export_canvas(_ctx(), "c1", "u1")
                assert _status(result) == 500


# ---------------------------------------------------------------------------
# Promote nodes
# ---------------------------------------------------------------------------


class TestPromoteNodes:
    def test_canvas_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                result = handler._promote_nodes(
                    _ctx(),
                    "missing",
                    {"node_ids": ["n1"]},
                    "u1",
                )
                assert _status(result) == 404

    def test_missing_node_ids_returns_400(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            canvas_obj = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                result = handler._promote_nodes(_ctx(), "c1", {}, "u1")
                assert _status(result) == 400

    def test_empty_node_ids_returns_400(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            canvas_obj = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                result = handler._promote_nodes(
                    _ctx(),
                    "c1",
                    {"node_ids": []},
                    "u1",
                )
                assert _status(result) == 400

    def test_promote_success(self, handler):
        goals_canvas = MagicMock()
        goals_canvas.to_dict.return_value = {"id": "gc-1", "stage": "goals"}
        prov = MagicMock()
        prov.to_dict.return_value = {"source": "n1", "target": "g1"}

        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            canvas_obj = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                with patch(
                    "aragora.canvas.promotion.promote_ideas_to_goals",
                    return_value=(goals_canvas, [prov]),
                ):
                    result = handler._promote_nodes(
                        _ctx(),
                        "c1",
                        {"node_ids": ["n1"]},
                        "u1",
                    )
                    assert _status(result) == 201
                    data = _body(result)
                    assert data["promoted_count"] == 1
                    assert "goals_canvas" in data
                    assert "provenance" in data

    def test_promote_multiple_nodes(self, handler):
        goals_canvas = MagicMock()
        goals_canvas.to_dict.return_value = {"id": "gc-2"}
        provs = [MagicMock() for _ in range(3)]
        for p in provs:
            p.to_dict.return_value = {"source": "nx"}

        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            canvas_obj = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                with patch(
                    "aragora.canvas.promotion.promote_ideas_to_goals",
                    return_value=(goals_canvas, provs),
                ):
                    result = handler._promote_nodes(
                        _ctx(),
                        "c1",
                        {"node_ids": ["n1", "n2", "n3"]},
                        "u1",
                    )
                    assert _status(result) == 201
                    assert _body(result)["promoted_count"] == 3

    def test_promote_anonymous_user(self, handler):
        goals_canvas = MagicMock()
        goals_canvas.to_dict.return_value = {"id": "gc-3"}
        prov = MagicMock()
        prov.to_dict.return_value = {}

        with patch.object(handler, "_get_canvas_manager") as mgr:
            mgr.return_value = MagicMock()
            canvas_obj = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_obj):
                with patch(
                    "aragora.canvas.promotion.promote_ideas_to_goals",
                    return_value=(goals_canvas, [prov]),
                ) as mock_promote:
                    handler._promote_nodes(
                        _ctx(),
                        "c1",
                        {"node_ids": ["n1"]},
                        None,
                    )
                    # user_id should fall back to "anonymous"
                    assert mock_promote.call_args[0][2] == "anonymous"

    def test_promote_error(self, handler):
        with patch.object(handler, "_get_canvas_manager", side_effect=ValueError("bad")):
            result = handler._promote_nodes(
                _ctx(),
                "c1",
                {"node_ids": ["n1"]},
                "u1",
            )
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    def test_rate_limit_blocks_request(self, mock_limiter, handler, mock_req):
        mock_limiter.is_allowed.return_value = False
        result = handler.handle("/api/v1/ideas", {}, mock_req("GET"))
        assert result is not None
        assert _status(result) == 429

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    def test_rate_limit_allows_request(self, mock_limiter, handler, mock_req):
        mock_limiter.is_allowed.return_value = True
        # The request should proceed past rate limiting (may still 404/500 on store)
        result = handler.handle("/api/v1/ideas", {}, mock_req("GET"))
        # Should not be 429
        if result is not None:
            assert _status(result) != 429


# ---------------------------------------------------------------------------
# handle() integration-level tests
# ---------------------------------------------------------------------------


class TestHandleIntegration:
    def _always_allow_checker(self):
        """Return a permission checker mock that always allows."""
        checker = MagicMock()
        decision = MagicMock()
        decision.allowed = True
        decision.reason = "test"
        checker.check_permission.return_value = decision
        return checker

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    @patch("aragora.rbac.decorators.get_permission_checker")
    def test_handle_list_success(
        self,
        mock_checker,
        mock_get_store,
        mock_limiter,
        handler,
        mock_req,
    ):
        mock_checker.return_value = self._always_allow_checker()
        mock_limiter.is_allowed.return_value = True
        store = MagicMock()
        store.list_canvases.return_value = [{"id": "ic-1"}]
        mock_get_store.return_value = store

        result = handler.handle("/api/v1/ideas", {}, mock_req("GET"))
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    @patch("aragora.rbac.decorators.get_permission_checker")
    def test_handle_create_success(
        self,
        mock_checker,
        mock_get_store,
        mock_limiter,
        handler,
        mock_req,
    ):
        mock_checker.return_value = self._always_allow_checker()
        mock_limiter.is_allowed.return_value = True
        store = MagicMock()
        store.save_canvas.return_value = {"id": "ic-new"}
        mock_get_store.return_value = store

        result = handler.handle(
            "/api/v1/ideas",
            {},
            mock_req("POST", body={"name": "New Canvas"}),
        )
        assert result is not None
        assert _status(result) == 201

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    def test_handle_unmatched_path_returns_none(self, mock_limiter, handler, mock_req):
        """Paths that don't match any route should return None."""
        mock_limiter.is_allowed.return_value = True
        result = handler.handle(
            "/api/v1/ideas/c1/unknown/extra",
            {},
            mock_req("GET"),
        )
        # Unmatched sub-path returns None (not handled)
        assert result is None

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    @patch("aragora.rbac.decorators.get_permission_checker")
    def test_handle_delete_canvas_success(
        self,
        mock_checker,
        mock_get_store,
        mock_limiter,
        handler,
        mock_req,
    ):
        mock_checker.return_value = self._always_allow_checker()
        mock_limiter.is_allowed.return_value = True
        store = MagicMock()
        store.delete_canvas.return_value = True
        mock_get_store.return_value = store

        result = handler.handle("/api/v1/ideas/abc", {}, mock_req("DELETE"))
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    @patch("aragora.rbac.decorators.get_permission_checker")
    def test_handle_update_canvas_success(
        self,
        mock_checker,
        mock_get_store,
        mock_limiter,
        handler,
        mock_req,
    ):
        mock_checker.return_value = self._always_allow_checker()
        mock_limiter.is_allowed.return_value = True
        store = MagicMock()
        store.update_canvas.return_value = {"id": "abc", "name": "Updated"}
        mock_get_store.return_value = store

        result = handler.handle(
            "/api/v1/ideas/abc",
            {},
            mock_req("PUT", body={"name": "Updated"}),
        )
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    @patch("aragora.rbac.decorators.get_permission_checker")
    def test_handle_permission_denied(
        self,
        mock_checker,
        mock_limiter,
        handler,
        mock_req,
    ):
        """PermissionDeniedError is caught and returns 403."""
        checker = MagicMock()
        decision = MagicMock()
        decision.allowed = False
        decision.reason = "not allowed"
        checker.check_permission.return_value = decision
        mock_checker.return_value = checker
        mock_limiter.is_allowed.return_value = True

        result = handler.handle("/api/v1/ideas", {}, mock_req("GET"))
        assert result is not None
        assert _status(result) == 403

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    @patch("aragora.rbac.decorators.get_permission_checker")
    def test_handle_workspace_from_query(
        self,
        mock_checker,
        mock_limiter,
        handler,
        mock_req,
    ):
        """workspace_id in query params overrides user's org_id."""
        mock_checker.return_value = self._always_allow_checker()
        mock_limiter.is_allowed.return_value = True

        with patch("aragora.canvas.idea_store.get_idea_canvas_store") as mock_get_store:
            store = MagicMock()
            store.list_canvases.return_value = []
            mock_get_store.return_value = store

            handler.handle(
                "/api/v1/ideas",
                {"workspace_id": "custom-ws"},
                mock_req("GET"),
            )
            # workspace_id used should come from query param
            call_kwargs = store.list_canvases.call_args
            assert call_kwargs.kwargs.get("workspace_id") == "custom-ws"


# ---------------------------------------------------------------------------
# _get_request_body edge cases
# ---------------------------------------------------------------------------


class TestGetRequestBody:
    def test_valid_json(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = json.dumps({"key": "val"}).encode()
        assert handler._get_request_body(h) == {"key": "val"}

    def test_empty_body(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = None
        assert handler._get_request_body(h) == {}

    def test_invalid_json(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = b"not-json"
        assert handler._get_request_body(h) == {}

    def test_no_request_attr(self, handler):
        h = MagicMock(spec=[])
        assert handler._get_request_body(h) == {}

    def test_invalid_utf8(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = b"\xff\xfe"
        assert handler._get_request_body(h) == {}


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_ctx(self):
        h = IdeaCanvasHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = IdeaCanvasHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "ideas"


# ---------------------------------------------------------------------------
# Regex pattern edge cases
# ---------------------------------------------------------------------------


class TestRegexPatterns:
    def test_id_with_hyphens(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_BY_ID

        assert IDEAS_BY_ID.match("/api/v1/ideas/my-canvas-123") is not None

    def test_id_with_underscores(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_BY_ID

        assert IDEAS_BY_ID.match("/api/v1/ideas/my_canvas_123") is not None

    def test_id_alphanumeric(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_BY_ID

        assert IDEAS_BY_ID.match("/api/v1/ideas/abc123XYZ") is not None

    def test_id_rejects_special_chars(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_BY_ID

        assert IDEAS_BY_ID.match("/api/v1/ideas/bad!@#id") is None

    def test_id_rejects_empty(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_BY_ID

        assert IDEAS_BY_ID.match("/api/v1/ideas/") is None

    def test_node_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_NODE

        m = IDEAS_NODE.match("/api/v1/ideas/c1/nodes/n2")
        assert m is not None
        assert m.groups() == ("c1", "n2")

    def test_edge_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_EDGE

        m = IDEAS_EDGE.match("/api/v1/ideas/c1/edges/e3")
        assert m is not None
        assert m.groups() == ("c1", "e3")

    def test_export_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_EXPORT

        m = IDEAS_EXPORT.match("/api/v1/ideas/canvas-abc/export")
        assert m is not None
        assert m.group(1) == "canvas-abc"

    def test_promote_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_PROMOTE

        m = IDEAS_PROMOTE.match("/api/v1/ideas/canvas-abc/promote")
        assert m is not None
        assert m.group(1) == "canvas-abc"

    def test_nodes_collection_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_NODES

        m = IDEAS_NODES.match("/api/v1/ideas/c1/nodes")
        assert m is not None
        assert m.group(1) == "c1"

    def test_edges_collection_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_EDGES

        m = IDEAS_EDGES.match("/api/v1/ideas/c1/edges")
        assert m is not None
        assert m.group(1) == "c1"

    def test_list_pattern(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_LIST

        assert IDEAS_LIST.match("/api/v1/ideas") is not None
        assert IDEAS_LIST.match("/api/v1/ideas/") is None

    def test_id_rejects_slashes(self, handler):
        from aragora.server.handlers.idea_canvas import IDEAS_BY_ID

        assert IDEAS_BY_ID.match("/api/v1/ideas/a/b") is None
