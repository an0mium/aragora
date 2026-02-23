"""Tests for OrchestrationCanvasHandler REST API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from aragora.server.handlers.orchestration_canvas import OrchestrationCanvasHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    return OrchestrationCanvasHandler(ctx={})


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

    def test_can_handle_orchestration_canvas(self, handler):
        assert handler.can_handle("/api/v1/orchestration/canvas") is True

    def test_can_handle_canvas_with_id(self, handler):
        assert handler.can_handle("/api/v1/orchestration/canvas/test-123") is True

    def test_can_handle_canvas_nodes(self, handler):
        assert handler.can_handle("/api/v1/orchestration/canvas/test-123/nodes") is True

    def test_can_handle_canvas_edges(self, handler):
        assert handler.can_handle("/api/v1/orchestration/canvas/test-123/edges") is True

    def test_can_handle_canvas_export(self, handler):
        assert handler.can_handle("/api/v1/orchestration/canvas/test-123/export") is True

    def test_can_handle_canvas_execute(self, handler):
        assert handler.can_handle("/api/v1/orchestration/canvas/test-123/execute") is True

    def test_cannot_handle_deliberation(self, handler):
        # Must NOT handle /api/v1/orchestration (without /canvas)
        assert handler.can_handle("/api/v1/orchestration") is False

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/actions") is False

    def test_routes_unknown_path(self, handler):
        result = handler._route_request(
            "/api/v1/orchestration/canvas/test/unknown/path",
            "GET",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is None

    def test_method_not_allowed_list(self, handler):
        result = handler._route_request(
            "/api/v1/orchestration/canvas",
            "DELETE",
            {},
            {},
            "u1",
            "ws1",
            MagicMock(),
        )
        assert result is not None
        status = getattr(result, "status_code", getattr(result, "status", None))
        if status:
            assert status == 405


# ---------------------------------------------------------------------------
# Canvas CRUD tests
# ---------------------------------------------------------------------------


class TestListCanvases:
    """_list_canvases tests."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_list_empty(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = []
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert result is not None

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_list_with_results(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.list_canvases.return_value = [
            {"id": "orch-1", "name": "Pipeline Alpha"},
            {"id": "orch-2", "name": "Pipeline Beta"},
        ]
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._list_canvases(ctx, {}, "u1", "ws1")
        assert result is not None


class TestCreateCanvas:
    """_create_canvas tests."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_create(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.save_canvas.return_value = {"id": "orch-new", "name": "Test"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._create_canvas(
            ctx,
            {"name": "Pipeline Build"},
            "u1",
            "ws1",
        )
        assert result is not None
        mock_store.save_canvas.assert_called_once()


class TestGetCanvas:
    """_get_canvas tests."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = None
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._get_canvas(ctx, "missing", "u1")
            assert result is not None
            status = getattr(result, "status_code", getattr(result, "status", None))
            if status:
                assert status == 404


class TestDeleteCanvas:
    """_delete_canvas tests."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_delete_success(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = True
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "orch-1", "u1")
        assert result is not None

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_delete_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.delete_canvas.return_value = False
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._delete_canvas(ctx, "missing", "u1")
        assert result is not None


class TestUpdateCanvas:
    """_update_canvas tests."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_update(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.update_canvas.return_value = {"id": "orch-1", "name": "Updated"}
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._update_canvas(ctx, "orch-1", {"name": "Updated"}, "u1")
        assert result is not None


# ---------------------------------------------------------------------------
# Node CRUD tests
# ---------------------------------------------------------------------------


class TestAddNode:
    """_add_node tests."""

    def test_invalid_orchestration_type(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_node(
                ctx,
                "c1",
                {"orchestration_type": "INVALID_TYPE"},
                "u1",
            )
            assert result is not None
            status = getattr(result, "status_code", getattr(result, "status", None))
            if status:
                assert status == 400

    def test_valid_orchestration_types(self, handler):
        """All OrchestrationNodeType values should be accepted."""
        from aragora.canvas.stages import OrchestrationNodeType

        for orch_type in OrchestrationNodeType:
            with patch.object(handler, "_get_canvas_manager") as mock_mgr:
                manager = MagicMock()
                mock_mgr.return_value = manager
                with patch.object(handler, "_run_async") as mock_run:
                    node_mock = MagicMock()
                    node_mock.to_dict.return_value = {"id": "n1"}
                    mock_run.return_value = node_mock
                    ctx = MagicMock()
                    result = handler._add_node(
                        ctx,
                        "c1",
                        {"orchestration_type": orch_type.value, "label": "test"},
                        "u1",
                    )
                    assert result is not None


class TestUpdateNode:
    """_update_node tests."""

    def test_update_node_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async", return_value=None):
                ctx = MagicMock()
                result = handler._update_node(
                    ctx,
                    "c1",
                    "n1",
                    {"label": "Updated"},
                    "u1",
                )
                assert result is not None


class TestDeleteNode:
    """_delete_node tests."""

    def test_delete_node_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async", return_value=False):
                ctx = MagicMock()
                result = handler._delete_node(ctx, "c1", "n1", "u1")
                assert result is not None


# ---------------------------------------------------------------------------
# Edge CRUD tests
# ---------------------------------------------------------------------------


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

    def test_missing_target(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            ctx = MagicMock()
            result = handler._add_edge(
                ctx,
                "c1",
                {"source_id": "n1"},
                "u1",
            )
            assert result is not None
            status = getattr(result, "status_code", getattr(result, "status", None))
            if status:
                assert status == 400


class TestDeleteEdge:
    """_delete_edge tests."""

    def test_delete_edge_not_found(self, handler):
        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async", return_value=False):
                ctx = MagicMock()
                result = handler._delete_edge(ctx, "c1", "e1", "u1")
                assert result is not None


# ---------------------------------------------------------------------------
# Export & Execute tests
# ---------------------------------------------------------------------------


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


class TestExecutePipeline:
    """_execute_pipeline tests."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_canvas_not_found(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = None
        mock_get_store.return_value = mock_store

        ctx = MagicMock()
        result = handler._execute_pipeline(ctx, "missing", {}, "u1")
        assert result is not None
        status = getattr(result, "status_code", getattr(result, "status", None))
        if status:
            assert status == 404

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_execute_success(self, mock_get_store, handler):
        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {
            "id": "orch-1",
            "name": "Pipeline",
            "metadata": {"stage": "orchestration"},
        }
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async") as mock_run:
                canvas_mock = MagicMock()
                canvas_mock.nodes = {}
                canvas_mock.edges = {}
                mock_run.return_value = canvas_mock

                ctx = MagicMock()
                result = handler._execute_pipeline(ctx, "orch-1", {}, "u1")
                assert result is not None


# ---------------------------------------------------------------------------
# Rate limit tests
# ---------------------------------------------------------------------------


class TestRateLimit:
    """Rate limiting tests."""

    @patch("aragora.server.handlers.orchestration_canvas._orchestration_limiter")
    def test_rate_limit_blocks(self, mock_limiter, handler, mock_request):
        mock_limiter.is_allowed.return_value = False
        result = handler.handle("/api/v1/orchestration/canvas", {}, mock_request("GET"))
        assert result is not None
