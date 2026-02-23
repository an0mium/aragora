"""Tests for GoalCanvasHandler REST API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from aragora.server.handlers.goal_canvas import GoalCanvasHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    return GoalCanvasHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object (the tornado/aiohttp-like handler)."""

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


def _parse_body(result) -> dict:
    """Decode the JSON body from a HandlerResult."""
    if hasattr(result, "body") and isinstance(result.body, bytes):
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result) -> int:
    """Extract the HTTP status code from a HandlerResult."""
    return getattr(result, "status_code", getattr(result, "status", 0))


# ---------------------------------------------------------------------------
# can_handle() tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() path detection."""

    def test_can_handle_goals_root(self, handler):
        assert handler.can_handle("/api/v1/goals") is True

    def test_can_handle_goals_with_id(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123") is True

    def test_can_handle_goals_nodes(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123/nodes") is True

    def test_can_handle_goals_node_id(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123/nodes/n1") is True

    def test_can_handle_goals_edges(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123/edges") is True

    def test_can_handle_goals_edge_id(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123/edges/e1") is True

    def test_can_handle_goals_export(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123/export") is True

    def test_can_handle_goals_advance(self, handler):
        assert handler.can_handle("/api/v1/goals/canvas-123/advance") is True

    def test_cannot_handle_debates(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_actions(self, handler):
        assert handler.can_handle("/api/v1/actions") is False

    def test_cannot_handle_ideas(self, handler):
        assert handler.can_handle("/api/v1/ideas") is False


# ---------------------------------------------------------------------------
# Route matching tests (via _route_request)
# ---------------------------------------------------------------------------


class TestRouting:
    """Tests for _route_request dispatching."""

    def test_unmatched_path_returns_none(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/unknown/extra",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is None

    def test_method_not_allowed_on_list_delete(self, handler):
        result = handler._route_request(
            "/api/v1/goals",
            "DELETE",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_list_put(self, handler):
        result = handler._route_request(
            "/api/v1/goals",
            "PUT",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_export_post(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/export",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_advance_get(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/advance",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_nodes_get(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/nodes",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_single_node_post(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/nodes/n1",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_edges_get(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/edges",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_single_edge_post(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1/edges/e1",
            "POST",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405

    def test_method_not_allowed_on_canvas_by_id_patch(self, handler):
        result = handler._route_request(
            "/api/v1/goals/c1",
            "PATCH",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        assert _status(result) == 405


# ---------------------------------------------------------------------------
# Canvas CRUD tests
# ---------------------------------------------------------------------------


class TestListCanvases:
    """_list_canvases tests."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_empty(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert result is not None
        assert _status(result) == 200
        body = _parse_body(result)
        assert body["canvases"] == []
        assert body["count"] == 0

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_with_results(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = [
            {"id": "gc-1", "name": "Revenue Goals"},
            {"id": "gc-2", "name": "Product Goals"},
        ]
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        body = _parse_body(result)
        assert body["count"] == 2
        assert len(body["canvases"]) == 2

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_passes_workspace_id(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._list_canvases(ctx, {"workspace_id": "ws-custom"}, "u1", "ws1")
        call_kwargs = mock_store.list_canvases.call_args
        assert (
            call_kwargs[1]["workspace_id"] == "ws-custom"
            or call_kwargs.kwargs.get("workspace_id") == "ws-custom"
        )

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_passes_owner_id(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._list_canvases(ctx, {"owner_id": "owner-xyz"}, "u1", "ws1")
        call_kwargs = mock_store.list_canvases.call_args
        assert (
            call_kwargs[1]["owner_id"] == "owner-xyz"
            or call_kwargs.kwargs.get("owner_id") == "owner-xyz"
        )

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_passes_source_canvas_id(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._list_canvases(ctx, {"source_canvas_id": "ideas-abc"}, "u1", "ws1")
        call_kwargs = mock_store.list_canvases.call_args
        assert (
            call_kwargs[1]["source_canvas_id"] == "ideas-abc"
            or call_kwargs.kwargs.get("source_canvas_id") == "ideas-abc"
        )

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_clamps_limit(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._list_canvases(ctx, {"limit": "5000"}, "u1", "ws1")
        call_kwargs = mock_store.list_canvases.call_args
        assert call_kwargs[1]["limit"] <= 1000 or call_kwargs.kwargs.get("limit", 0) <= 1000

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_clamps_limit_minimum(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._list_canvases(ctx, {"limit": "-5"}, "u1", "ws1")
        call_kwargs = mock_store.list_canvases.call_args
        assert call_kwargs[1]["limit"] >= 1 or call_kwargs.kwargs.get("limit", 0) >= 1

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_clamps_offset_minimum(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._list_canvases(ctx, {"offset": "-10"}, "u1", "ws1")
        call_kwargs = mock_store.list_canvases.call_args
        assert call_kwargs[1]["offset"] >= 0 or call_kwargs.kwargs.get("offset", 0) >= 0

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_store_error(self, mock_get_store, handler):
        mock_get_store.side_effect = ImportError("no store")

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert _status(result) == 500

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_list_runtime_error(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.side_effect = RuntimeError("db down")
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert _status(result) == 500


class TestCreateCanvas:
    """_create_canvas tests."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_minimal(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "goals-abc", "name": "Untitled Goals"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._create_canvas(ctx, {}, "u1", "ws1")
        assert _status(result) == 201

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_with_name(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "goals-new", "name": "Q1 Goals"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._create_canvas(
            ctx,
            {"name": "Q1 Goals", "description": "First quarter goals"},
            "u1",
            "ws1",
        )
        assert _status(result) == 201
        mock_store.save_canvas.assert_called_once()
        call_kwargs = mock_store.save_canvas.call_args[1]
        assert call_kwargs["name"] == "Q1 Goals"
        assert call_kwargs["description"] == "First quarter goals"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_with_custom_id(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "my-custom-id"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._create_canvas(ctx, {"id": "my-custom-id"}, "u1", "ws1")
        call_kwargs = mock_store.save_canvas.call_args[1]
        assert call_kwargs["canvas_id"] == "my-custom-id"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_with_source_canvas_id(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "goals-new"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._create_canvas(
            ctx,
            {"source_canvas_id": "ideas-abc"},
            "u1",
            "ws1",
        )
        call_kwargs = mock_store.save_canvas.call_args[1]
        assert call_kwargs["source_canvas_id"] == "ideas-abc"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_with_metadata(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "goals-new"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._create_canvas(
            ctx,
            {"metadata": {"stage": "goals", "priority": "high"}},
            "u1",
            "ws1",
        )
        call_kwargs = mock_store.save_canvas.call_args[1]
        assert call_kwargs["metadata"]["priority"] == "high"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_default_metadata(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "goals-new"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._create_canvas(ctx, {}, "u1", "ws1")
        call_kwargs = mock_store.save_canvas.call_args[1]
        assert call_kwargs["metadata"] == {"stage": "goals"}

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_store_error(self, mock_get_store, handler):
        mock_get_store.side_effect = ImportError("no store")

        ctx = MagicMock()
        result = handler._create_canvas(ctx, {"name": "Test"}, "u1", "ws1")
        assert _status(result) == 500

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_create_save_error(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.side_effect = ValueError("invalid data")
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._create_canvas(ctx, {"name": "Test"}, "u1", "ws1")
        assert _status(result) == 500


class TestGetCanvas:
    """_get_canvas tests."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = None
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._get_canvas(ctx, "missing-id", "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_found_with_live_canvas(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {"id": "gc-1", "name": "My Goals"}
        mock_get_store.return_value = mock_store

        mock_node = MagicMock()
        mock_node.to_dict.return_value = {"id": "n1", "label": "Goal 1"}
        mock_edge = MagicMock()
        mock_edge.to_dict.return_value = {"id": "e1", "source": "n1", "target": "n2"}

        canvas_mock = MagicMock()
        canvas_mock.nodes = {"n1": mock_node}
        canvas_mock.edges = {"e1": mock_edge}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_mock):
                ctx = MagicMock()
                result = handler._get_canvas(ctx, "gc-1", "u1")
                assert _status(result) == 200
                body = _parse_body(result)
                assert body["id"] == "gc-1"
                assert len(body["nodes"]) == 1
                assert len(body["edges"]) == 1

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_found_without_live_canvas(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {"id": "gc-1", "name": "My Goals"}
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._get_canvas(ctx, "gc-1", "u1")
                assert _status(result) == 200
                body = _parse_body(result)
                assert body["nodes"] == []
                assert body["edges"] == []

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_get_canvas_error(self, mock_get_store, handler):
        mock_get_store.side_effect = RuntimeError("db error")

        ctx = MagicMock()
        result = handler._get_canvas(ctx, "gc-1", "u1")
        assert _status(result) == 500


class TestUpdateCanvas:
    """_update_canvas tests."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_update_success(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = {"id": "gc-1", "name": "Updated Name"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._update_canvas(ctx, "gc-1", {"name": "Updated Name"}, "u1")
        assert _status(result) == 200
        body = _parse_body(result)
        assert body["name"] == "Updated Name"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_update_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = None
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._update_canvas(ctx, "missing", {"name": "New"}, "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_update_with_description(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = {"id": "gc-1", "description": "New desc"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._update_canvas(ctx, "gc-1", {"description": "New desc"}, "u1")
        call_kwargs = mock_store.update_canvas.call_args[1]
        assert call_kwargs["description"] == "New desc"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_update_with_metadata(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = {"id": "gc-1"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        handler._update_canvas(ctx, "gc-1", {"metadata": {"stage": "goals"}}, "u1")
        call_kwargs = mock_store.update_canvas.call_args[1]
        assert call_kwargs["metadata"] == {"stage": "goals"}

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_update_store_error(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.side_effect = OSError("disk full")
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._update_canvas(ctx, "gc-1", {"name": "X"}, "u1")
        assert _status(result) == 500


class TestDeleteCanvas:
    """_delete_canvas tests."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_delete_success(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = True
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "gc-1", "u1")
        assert _status(result) == 200
        body = _parse_body(result)
        assert body["deleted"] is True
        assert body["canvas_id"] == "gc-1"

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_delete_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = False
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "missing", "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_delete_store_error(self, mock_get_store, handler):
        mock_get_store.side_effect = ImportError("no store")

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "gc-1", "u1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Node CRUD tests
# ---------------------------------------------------------------------------


class TestAddNode:
    """_add_node tests."""

    def test_add_node_success(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1", "label": "Revenue target"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._add_node(
                    ctx,
                    "gc-1",
                    {"label": "Revenue target", "goal_type": "goal"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_node_canvas_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._add_node(
                    ctx,
                    "missing",
                    {"label": "Goal", "goal_type": "goal"},
                    "u1",
                )
                assert _status(result) == 404

    def test_add_node_invalid_goal_type(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_node(
                ctx,
                "gc-1",
                {"goal_type": "INVALID_NONEXISTENT"},
                "u1",
            )
            assert _status(result) == 400
            body = _parse_body(result)
            assert "Invalid goal type" in body.get("error", "")

    def test_add_node_all_valid_goal_types(self, handler):
        """All GoalNodeType values should be accepted."""
        from aragora.canvas.stages import GoalNodeType

        for goal_type in GoalNodeType:
            node_mock = MagicMock()
            node_mock.to_dict.return_value = {"id": "n1"}

            with patch.object(handler, "_get_canvas_manager") as mock_mgr:
                mock_mgr.return_value = MagicMock()
                with patch.object(handler, "_run_async", return_value=node_mock):
                    ctx = MagicMock()
                    result = handler._add_node(
                        ctx,
                        "gc-1",
                        {"goal_type": goal_type.value, "label": "test"},
                        "u1",
                    )
                    assert _status(result) == 201, f"Failed for goal_type={goal_type.value}"

    def test_add_node_default_goal_type(self, handler):
        """No goal_type in body defaults to 'goal'."""
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._add_node(ctx, "gc-1", {}, "u1")
                assert _status(result) == 201

    def test_add_node_with_position(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._add_node(
                    ctx,
                    "gc-1",
                    {"label": "Goal", "position": {"x": 100, "y": 200}},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_node_with_priority(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._add_node(
                    ctx,
                    "gc-1",
                    {"label": "Goal", "priority": "high"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_node_with_measurable(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._add_node(
                    ctx,
                    "gc-1",
                    {"label": "Goal", "measurable": True},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_node_import_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.side_effect = ImportError("no canvas module")
            ctx = MagicMock()
            result = handler._add_node(ctx, "gc-1", {"label": "G"}, "u1")
            assert _status(result) == 500


class TestUpdateNode:
    """_update_node tests."""

    def test_update_node_success(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1", "label": "Updated Goal"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._update_node(
                    ctx,
                    "gc-1",
                    "n1",
                    {"label": "Updated Goal"},
                    "u1",
                )
                assert _status(result) == 200
                body = _parse_body(result)
                assert body["label"] == "Updated Goal"

    def test_update_node_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._update_node(
                    ctx,
                    "gc-1",
                    "missing",
                    {"label": "X"},
                    "u1",
                )
                assert _status(result) == 404

    def test_update_node_with_position(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._update_node(
                    ctx,
                    "gc-1",
                    "n1",
                    {"position": {"x": 50, "y": 75}},
                    "u1",
                )
                assert _status(result) == 200

    def test_update_node_with_data(self, handler):
        node_mock = MagicMock()
        node_mock.to_dict.return_value = {"id": "n1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=node_mock):
                ctx = MagicMock()
                result = handler._update_node(
                    ctx,
                    "gc-1",
                    "n1",
                    {"data": {"custom": "field"}},
                    "u1",
                )
                assert _status(result) == 200

    def test_update_node_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.side_effect = RuntimeError("boom")
            ctx = MagicMock()
            result = handler._update_node(ctx, "gc-1", "n1", {}, "u1")
            assert _status(result) == 500


class TestDeleteNode:
    """_delete_node tests."""

    def test_delete_node_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=True):
                ctx = MagicMock()
                result = handler._delete_node(ctx, "gc-1", "n1", "u1")
                assert _status(result) == 200
                body = _parse_body(result)
                assert body["deleted"] is True
                assert body["node_id"] == "n1"

    def test_delete_node_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=False):
                ctx = MagicMock()
                result = handler._delete_node(ctx, "gc-1", "n1", "u1")
                assert _status(result) == 404

    def test_delete_node_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.side_effect = OSError("disk error")
            ctx = MagicMock()
            result = handler._delete_node(ctx, "gc-1", "n1", "u1")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Edge CRUD tests
# ---------------------------------------------------------------------------


class TestAddEdge:
    """_add_edge tests."""

    def test_add_edge_success(self, handler):
        edge_mock = MagicMock()
        edge_mock.to_dict.return_value = {"id": "e1", "source": "n1", "target": "n2"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=edge_mock):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source_id": "n1", "target_id": "n2"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_edge_with_source_target_aliases(self, handler):
        edge_mock = MagicMock()
        edge_mock.to_dict.return_value = {"id": "e1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=edge_mock):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source": "n1", "target": "n2"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_edge_missing_source(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_edge(
                ctx,
                "gc-1",
                {"target_id": "n2"},
                "u1",
            )
            assert _status(result) == 400
            body = _parse_body(result)
            assert "source_id" in body.get("error", "")

    def test_add_edge_missing_target(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_edge(
                ctx,
                "gc-1",
                {"source_id": "n1"},
                "u1",
            )
            assert _status(result) == 400
            body = _parse_body(result)
            assert "target_id" in body.get("error", "")

    def test_add_edge_missing_both(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_edge(ctx, "gc-1", {}, "u1")
            assert _status(result) == 400

    def test_add_edge_canvas_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source_id": "n1", "target_id": "n2"},
                    "u1",
                )
                assert _status(result) == 404

    def test_add_edge_with_label(self, handler):
        edge_mock = MagicMock()
        edge_mock.to_dict.return_value = {"id": "e1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=edge_mock):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source_id": "n1", "target_id": "n2", "label": "depends on"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_edge_with_type(self, handler):
        edge_mock = MagicMock()
        edge_mock.to_dict.return_value = {"id": "e1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=edge_mock):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source_id": "n1", "target_id": "n2", "type": "dependency"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_edge_invalid_type_falls_back(self, handler):
        """Invalid edge type should fall back to DEFAULT, not error."""
        edge_mock = MagicMock()
        edge_mock.to_dict.return_value = {"id": "e1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=edge_mock):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source_id": "n1", "target_id": "n2", "type": "INVALID"},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_edge_with_data(self, handler):
        edge_mock = MagicMock()
        edge_mock.to_dict.return_value = {"id": "e1"}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=edge_mock):
                ctx = MagicMock()
                result = handler._add_edge(
                    ctx,
                    "gc-1",
                    {"source_id": "n1", "target_id": "n2", "data": {"weight": 5}},
                    "u1",
                )
                assert _status(result) == 201

    def test_add_edge_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.side_effect = ImportError("no module")
            ctx = MagicMock()
            result = handler._add_edge(
                ctx,
                "gc-1",
                {"source_id": "n1", "target_id": "n2"},
                "u1",
            )
            assert _status(result) == 500


class TestDeleteEdge:
    """_delete_edge tests."""

    def test_delete_edge_success(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=True):
                ctx = MagicMock()
                result = handler._delete_edge(ctx, "gc-1", "e1", "u1")
                assert _status(result) == 200
                body = _parse_body(result)
                assert body["deleted"] is True
                assert body["edge_id"] == "e1"

    def test_delete_edge_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=False):
                ctx = MagicMock()
                result = handler._delete_edge(ctx, "gc-1", "e1", "u1")
                assert _status(result) == 404

    def test_delete_edge_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.side_effect = TypeError("bad arg")
            ctx = MagicMock()
            result = handler._delete_edge(ctx, "gc-1", "e1", "u1")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportCanvas:
    """_export_canvas tests."""

    def test_export_success(self, handler):
        canvas_mock = MagicMock()

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_mock):
                with patch(
                    "aragora.canvas.converters.to_react_flow",
                    return_value={"nodes": [], "edges": []},
                ):
                    ctx = MagicMock()
                    result = handler._export_canvas(ctx, "gc-1", "u1")
                    assert _status(result) == 200
                    body = _parse_body(result)
                    assert "nodes" in body
                    assert "edges" in body

    def test_export_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._export_canvas(ctx, "missing", "u1")
                assert _status(result) == 404

    def test_export_error(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.side_effect = ImportError("no converters")
            ctx = MagicMock()
            result = handler._export_canvas(ctx, "gc-1", "u1")
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Advance to actions tests
# ---------------------------------------------------------------------------


class TestAdvanceToActions:
    """_advance_to_actions tests."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_advance_canvas_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = None
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._advance_to_actions(ctx, "missing", {}, "u1")
        assert _status(result) == 404

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_advance_success_with_nodes(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {
            "id": "gc-1",
            "name": "Goals",
            "metadata": {"stage": "goals"},
        }
        mock_get_store.return_value = mock_store

        mock_node = MagicMock()
        mock_node.to_dict.return_value = {"id": "n1", "label": "Goal 1"}
        mock_edge = MagicMock()
        mock_edge.to_dict.return_value = {"id": "e1"}

        canvas_mock = MagicMock()
        canvas_mock.nodes = {"n1": mock_node}
        canvas_mock.edges = {"e1": mock_edge}

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_mock):
                ctx = MagicMock()
                result = handler._advance_to_actions(ctx, "gc-1", {}, "u1")
                assert _status(result) == 201
                body = _parse_body(result)
                assert body["source_canvas_id"] == "gc-1"
                assert body["source_stage"] == "goals"
                assert body["target_stage"] == "actions"
                assert body["status"] == "ready"
                assert len(body["nodes"]) == 1
                assert len(body["edges"]) == 1

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_advance_success_no_live_canvas(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {
            "id": "gc-1",
            "name": "Goals",
            "metadata": {"stage": "goals"},
        }
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            mock_mgr.return_value = MagicMock()
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._advance_to_actions(ctx, "gc-1", {}, "u1")
                assert _status(result) == 201
                body = _parse_body(result)
                assert body["nodes"] == []
                assert body["edges"] == []
                assert body["metadata"] == {"stage": "goals"}

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_advance_error(self, mock_get_store, handler):
        mock_get_store.side_effect = RuntimeError("fail")

        ctx = MagicMock()
        result = handler._advance_to_actions(ctx, "gc-1", {}, "u1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Rate limit tests
# ---------------------------------------------------------------------------


class TestRateLimit:
    """Rate limiting tests."""

    @patch("aragora.server.handlers.goal_canvas._goals_limiter")
    def test_rate_limit_blocks(self, mock_limiter, handler, mock_http_handler):
        mock_limiter.is_allowed.return_value = False
        result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
        assert result is not None
        assert _status(result) == 429

    @patch("aragora.server.handlers.goal_canvas._goals_limiter")
    def test_rate_limit_allows(self, mock_limiter, handler, mock_http_handler):
        mock_limiter.is_allowed.return_value = True
        with patch.object(handler, "_route_request", return_value=MagicMock(status_code=200)):
            result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
            assert result is not None


# ---------------------------------------------------------------------------
# handle() integration tests
# ---------------------------------------------------------------------------


def _allow_all_checker():
    """Return a mock permission checker that always allows."""
    mock_checker = MagicMock()
    mock_decision = MagicMock()
    mock_decision.allowed = True
    mock_checker.check_permission.return_value = mock_decision
    return mock_checker


class TestHandleIntegration:
    """Tests for the top-level handle() method."""

    @patch("aragora.rbac.decorators.get_permission_checker", _allow_all_checker)
    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_handle_get_list(self, mock_get_store, handler, mock_http_handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.rbac.decorators.get_permission_checker", _allow_all_checker)
    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_handle_post_create(self, mock_get_store, handler, mock_http_handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "gc-new"}
        mock_get_store.return_value = mock_store

        result = handler.handle(
            "/api/v1/goals",
            {},
            mock_http_handler("POST", body={"name": "Test"}),
        )
        assert result is not None
        assert _status(result) == 201

    @patch("aragora.rbac.decorators.get_permission_checker", _allow_all_checker)
    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_handle_get_canvas_by_id(self, mock_get_store, handler, mock_http_handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {"id": "gc-1", "name": "Goals"}
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async", return_value=None):
                result = handler.handle("/api/v1/goals/gc-1", {}, mock_http_handler("GET"))
                assert result is not None
                assert _status(result) == 200

    @patch("aragora.rbac.decorators.get_permission_checker", _allow_all_checker)
    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_handle_put_update(self, mock_get_store, handler, mock_http_handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = {"id": "gc-1", "name": "Updated"}
        mock_get_store.return_value = mock_store

        result = handler.handle(
            "/api/v1/goals/gc-1",
            {},
            mock_http_handler("PUT", body={"name": "Updated"}),
        )
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.rbac.decorators.get_permission_checker", _allow_all_checker)
    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_handle_delete_canvas(self, mock_get_store, handler, mock_http_handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = True
        mock_get_store.return_value = mock_store

        result = handler.handle(
            "/api/v1/goals/gc-1",
            {},
            mock_http_handler("DELETE"),
        )
        assert result is not None
        assert _status(result) == 200

    def test_handle_unmatched_path_returns_none(self, handler, mock_http_handler):
        result = handler.handle(
            "/api/v1/goals/c1/unknown/extra/deep",
            {},
            mock_http_handler("GET"),
        )
        assert result is None

    @patch("aragora.rbac.decorators.get_permission_checker", _allow_all_checker)
    def test_handle_workspace_id_from_query(self, handler, mock_http_handler):
        """workspace_id in query_params should override auth context."""
        with patch.object(handler, "_route_request") as mock_route:
            mock_route.return_value = MagicMock(status_code=200)
            handler.handle(
                "/api/v1/goals",
                {"workspace_id": "ws-override"},
                mock_http_handler("GET"),
            )
            call_args = mock_route.call_args
            # workspace_id is the 6th positional arg (index 5)
            assert call_args[0][5] == "ws-override"


# ---------------------------------------------------------------------------
# _get_request_body tests
# ---------------------------------------------------------------------------


class TestGetRequestBody:
    """Tests for _get_request_body parsing."""

    def test_parse_valid_json(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = json.dumps({"key": "value"}).encode("utf-8")
        result = handler._get_request_body(h)
        assert result == {"key": "value"}

    def test_parse_empty_body(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = None
        result = handler._get_request_body(h)
        assert result == {}

    def test_parse_invalid_json(self, handler):
        h = MagicMock()
        h.request = MagicMock()
        h.request.body = b"not valid json"
        result = handler._get_request_body(h)
        assert result == {}

    def test_no_request_attribute(self, handler):
        h = MagicMock(spec=[])
        result = handler._get_request_body(h)
        assert result == {}


# ---------------------------------------------------------------------------
# Permission denied tests
# ---------------------------------------------------------------------------


class TestPermissionDenied:
    """Tests for PermissionDeniedError handling in handle()."""

    def test_permission_denied_returns_403(self, handler, mock_http_handler):
        from aragora.rbac.decorators import PermissionDeniedError

        with patch.object(handler, "_route_request") as mock_route:
            mock_route.side_effect = PermissionDeniedError("goals:read")
            result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
            assert result is not None
            assert _status(result) == 403


# ---------------------------------------------------------------------------
# Authentication error tests
# ---------------------------------------------------------------------------


class TestAuthErrors:
    """Tests for authentication failure handling."""

    @pytest.mark.no_auto_auth
    def test_auth_value_error(self, handler, mock_http_handler):
        """ValueError during auth returns 401."""
        from aragora.server.handlers.base import BaseHandler

        with patch.object(
            BaseHandler,
            "require_auth_or_error",
            side_effect=ValueError("bad token"),
        ):
            result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
            assert result is not None
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_auth_attribute_error(self, handler, mock_http_handler):
        """AttributeError during auth returns 401."""
        from aragora.server.handlers.base import BaseHandler

        with patch.object(
            BaseHandler,
            "require_auth_or_error",
            side_effect=AttributeError("no user_id"),
        ):
            result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
            assert result is not None
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_auth_key_error(self, handler, mock_http_handler):
        """KeyError during auth returns 401."""
        from aragora.server.handlers.base import BaseHandler

        with patch.object(
            BaseHandler,
            "require_auth_or_error",
            side_effect=KeyError("missing key"),
        ):
            result = handler.handle("/api/v1/goals", {}, mock_http_handler("GET"))
            assert result is not None
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# Regex route pattern tests
# ---------------------------------------------------------------------------


class TestRegexPatterns:
    """Verify regex patterns match expected paths and reject malformed ones."""

    def test_goals_list_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_LIST

        assert GOALS_LIST.match("/api/v1/goals")
        assert not GOALS_LIST.match("/api/v1/goals/")
        assert not GOALS_LIST.match("/api/v1/goals/abc")

    def test_goals_by_id_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_BY_ID

        assert GOALS_BY_ID.match("/api/v1/goals/abc")
        assert GOALS_BY_ID.match("/api/v1/goals/canvas-123")
        assert GOALS_BY_ID.match("/api/v1/goals/a_b-c")
        assert not GOALS_BY_ID.match("/api/v1/goals/abc/nodes")

    def test_goals_nodes_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_NODES

        assert GOALS_NODES.match("/api/v1/goals/c1/nodes")
        assert not GOALS_NODES.match("/api/v1/goals/c1/nodes/n1")

    def test_goals_node_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_NODE

        assert GOALS_NODE.match("/api/v1/goals/c1/nodes/n1")
        m = GOALS_NODE.match("/api/v1/goals/c1/nodes/n1")
        assert m.groups() == ("c1", "n1")

    def test_goals_edges_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_EDGES

        assert GOALS_EDGES.match("/api/v1/goals/c1/edges")
        assert not GOALS_EDGES.match("/api/v1/goals/c1/edges/e1")

    def test_goals_edge_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_EDGE

        assert GOALS_EDGE.match("/api/v1/goals/c1/edges/e1")
        m = GOALS_EDGE.match("/api/v1/goals/c1/edges/e1")
        assert m.groups() == ("c1", "e1")

    def test_goals_export_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_EXPORT

        assert GOALS_EXPORT.match("/api/v1/goals/c1/export")
        assert not GOALS_EXPORT.match("/api/v1/goals/c1/export/extra")

    def test_goals_advance_matches(self):
        from aragora.server.handlers.goal_canvas import GOALS_ADVANCE

        assert GOALS_ADVANCE.match("/api/v1/goals/c1/advance")
        assert not GOALS_ADVANCE.match("/api/v1/goals/c1/advance/extra")

    def test_special_characters_rejected(self):
        from aragora.server.handlers.goal_canvas import GOALS_BY_ID

        assert not GOALS_BY_ID.match("/api/v1/goals/abc def")
        assert not GOALS_BY_ID.match("/api/v1/goals/abc@def")
        assert not GOALS_BY_ID.match("/api/v1/goals/abc/def")


# ---------------------------------------------------------------------------
# Resource type and constructor tests
# ---------------------------------------------------------------------------


class TestHandlerMeta:
    """Tests for handler metadata and configuration."""

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "goals"

    def test_constructor_default_ctx(self):
        h = GoalCanvasHandler()
        assert h.ctx == {}

    def test_constructor_with_ctx(self):
        h = GoalCanvasHandler(ctx={"key": "val"})
        assert h.ctx["key"] == "val"
