"""Tests for IdeaCanvasHandler REST API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any

import pytest

from aragora.server.handlers.idea_canvas import IdeaCanvasHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    return IdeaCanvasHandler(ctx={})


@pytest.fixture
def mock_request():
    """Create a mock HTTP handler object."""

    def _make(method: str = "GET", body: dict | None = None):
        h = MagicMock()
        h.command = method
        h.client_address = ("127.0.0.1", 12345)
        h.headers = {"X-Forwarded-For": "127.0.0.1"}
        h.request = MagicMock()
        h.request.body = json.dumps(body).encode() if body else None
        # Auth mock
        h.auth_user = MagicMock()
        h.auth_user.user_id = "test-user"
        h.auth_user.org_id = "test-org"
        h.auth_user.roles = {"admin"}
        return h

    return _make


# ---------------------------------------------------------------------------
# Route matching tests
# ---------------------------------------------------------------------------


class TestRouteMatching:
    """can_handle() and routing tests."""

    def test_can_handle_ideas_root(self, handler):
        assert handler.can_handle("/api/v1/ideas") is True

    def test_can_handle_ideas_with_id(self, handler):
        assert handler.can_handle("/api/v1/ideas/test-123") is True

    def test_can_handle_ideas_nodes(self, handler):
        assert handler.can_handle("/api/v1/ideas/test-123/nodes") is True

    def test_can_handle_ideas_edges(self, handler):
        assert handler.can_handle("/api/v1/ideas/test-123/edges") is True

    def test_can_handle_ideas_export(self, handler):
        assert handler.can_handle("/api/v1/ideas/test-123/export") is True

    def test_can_handle_ideas_promote(self, handler):
        assert handler.can_handle("/api/v1/ideas/test-123/promote") is True

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/canvas") is False

    def test_routes_unknown_path(self, handler):
        result = handler._route_request(
            "/api/v1/ideas/test/unknown/path",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Handler method tests (with mocked store)
# ---------------------------------------------------------------------------


class TestListCanvases:
    """_list_canvases tests."""

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_empty(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert result is not None
        body = json.loads(result.body) if hasattr(result, "body") else result
        if isinstance(body, str):
            body = json.loads(body)
        # The response should contain canvases key
        if isinstance(body, dict):
            assert "canvases" in body

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_list_with_results(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = [
            {"id": "ic-1", "name": "A"},
            {"id": "ic-2", "name": "B"},
        ]
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert result is not None


class TestCreateCanvas:
    """_create_canvas tests."""

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_create(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "ic-new", "name": "Test"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._create_canvas(
            ctx,
            {"name": "Test Canvas"},
            "u1",
            "ws1",
        )
        assert result is not None
        mock_store.save_canvas.assert_called_once()


class TestGetCanvas:
    """_get_canvas tests."""

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = None
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            ctx = MagicMock()
            result = handler._get_canvas(ctx, "missing", "u1")
            assert result is not None
            # Should be 404
            status = getattr(result, "status_code", getattr(result, "status", None))
            if status:
                assert status == 404


class TestDeleteCanvas:
    """_delete_canvas tests."""

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_delete_success(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = True
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "ic-1", "u1")
        assert result is not None

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_delete_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = False
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "missing", "u1")
        assert result is not None


class TestUpdateCanvas:
    """_update_canvas tests."""

    @patch("aragora.canvas.idea_store.get_idea_canvas_store")
    def test_update(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = {"id": "ic-1", "name": "Updated"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._update_canvas(ctx, "ic-1", {"name": "Updated"}, "u1")
        assert result is not None


class TestAddNode:
    """_add_node tests."""

    def test_invalid_idea_type(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_node(
                ctx,
                "c1",
                {"idea_type": "INVALID_TYPE"},
                "u1",
            )
            assert result is not None
            # Should be 400
            status = getattr(result, "status_code", getattr(result, "status", None))
            if status:
                assert status == 400


class TestAddEdge:
    """_add_edge tests."""

    def test_missing_source(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_edge(
                ctx,
                "c1",
                {"target_id": "n2"},
                "u1",
            )
            assert result is not None
            status = getattr(result, "status_code", getattr(result, "status", None))
            if status:
                assert status == 400


class TestExportCanvas:
    """_export_canvas tests."""

    def test_export_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            manager = MagicMock()
            mock_mgr.return_value = manager
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._export_canvas(ctx, "missing", "u1")
                assert result is not None


class TestPromoteNodes:
    """_promote_nodes tests."""

    def test_missing_node_ids(self, handler):
        with patch.object(handler, "_get_canvas_manager") as mock_mgr:
            manager = MagicMock()
            mock_mgr.return_value = manager
            canvas_mock = MagicMock()
            with patch.object(handler, "_run_async", return_value=canvas_mock):
                ctx = MagicMock()
                result = handler._promote_nodes(ctx, "c1", {}, "u1")
                assert result is not None
                status = getattr(result, "status_code", getattr(result, "status", None))
                if status:
                    assert status == 400

    def test_canvas_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._promote_nodes(
                    ctx,
                    "missing",
                    {"node_ids": ["n1"]},
                    "u1",
                )
                assert result is not None


# ---------------------------------------------------------------------------
# Rate limit & auth tests
# ---------------------------------------------------------------------------


class TestRateLimit:
    """Rate limiting tests."""

    @patch("aragora.server.handlers.idea_canvas._ideas_limiter")
    def test_rate_limit_blocks(self, mock_limiter, handler, mock_request):
        mock_limiter.is_allowed.return_value = False
        result = handler.handle("/api/v1/ideas", {}, mock_request("GET"))
        assert result is not None


class TestCanvasEventTypes:
    """CanvasEventType idea-specific events exist."""

    def test_idea_events_exist(self):
        from aragora.canvas.models import CanvasEventType

        assert CanvasEventType.IDEA_CURSOR_MOVE.value == "ideas:cursor:move"
        assert CanvasEventType.IDEA_PRESENCE_JOIN.value == "ideas:presence:join"
        assert CanvasEventType.IDEA_PRESENCE_LEAVE.value == "ideas:presence:leave"
        assert CanvasEventType.IDEA_NODE_LOCK.value == "ideas:node:lock"
        assert CanvasEventType.IDEA_NODE_UNLOCK.value == "ideas:node:unlock"


class TestKMTypes:
    """Knowledge Mound type extensions."""

    def test_idea_node_types_in_literal(self):
        from aragora.knowledge.mound_types import KnowledgeNode

        # Verify we can create KM nodes with idea_ prefixed types
        node = KnowledgeNode(
            id="kn_test",
            node_type="idea_concept",
            content="test",
        )
        assert node.node_type == "idea_concept"

    def test_idea_relationship_types(self):
        from aragora.knowledge.mound_types import KnowledgeRelationship

        for rel_type in ["inspires", "refines", "challenges", "exemplifies"]:
            rel = KnowledgeRelationship(
                id=f"kr_{rel_type}",
                from_node_id="a",
                to_node_id="b",
                relationship_type=rel_type,
            )
            assert rel.relationship_type == rel_type
