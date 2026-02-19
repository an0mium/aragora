"""Tests for CanvasPipelineHandler run/status/graph/receipt endpoints."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.canvas_pipeline import (
    CanvasPipelineHandler,
    _get_store,
    _pipeline_objects,
    _pipeline_tasks,
)


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult or raw dict."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _store_put(pipeline_id: str, data: dict) -> None:
    """Helper to insert test data into the pipeline store."""
    _get_store().save(pipeline_id, data)


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear in-memory stores between tests."""
    _pipeline_objects.clear()
    _pipeline_tasks.clear()
    yield
    _pipeline_objects.clear()
    _pipeline_tasks.clear()


@pytest.fixture
def handler():
    return CanvasPipelineHandler()


@pytest.fixture
def mock_store():
    """Provide a mock pipeline store for GET tests."""
    store = MagicMock()
    store.get.return_value = None
    with patch("aragora.server.handlers.canvas_pipeline._get_store", return_value=store):
        yield store


# =========================================================================
# ROUTES list
# =========================================================================


class TestRoutes:
    def test_run_route_registered(self, handler):
        assert "POST /api/v1/canvas/pipeline/run" in handler.ROUTES

    def test_status_route_registered(self, handler):
        assert "GET /api/v1/canvas/pipeline/{id}/status" in handler.ROUTES

    def test_graph_route_registered(self, handler):
        assert "GET /api/v1/canvas/pipeline/{id}/graph" in handler.ROUTES

    def test_receipt_route_registered(self, handler):
        assert "GET /api/v1/canvas/pipeline/{id}/receipt" in handler.ROUTES

    def test_can_handle_canvas_paths(self, handler):
        assert handler.can_handle("/api/v1/canvas/pipeline/run")
        assert handler.can_handle("/api/v1/canvas/pipeline/abc/status")
        assert handler.can_handle("/api/v1/canvas/pipeline/abc/graph")
        assert handler.can_handle("/api/v1/canvas/pipeline/abc/receipt")


# =========================================================================
# handle_run
# =========================================================================


class TestHandleRun:
    @pytest.mark.asyncio
    async def test_run_missing_input(self, handler):
        result = await handler.handle_run({})
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_run_returns_pipeline_id(self, handler):
        result = await handler.handle_run(
            {
                "input_text": "Build a rate limiter",
                "dry_run": True,
            }
        )
        body = _body(result)
        assert "pipeline_id" in body
        assert body["status"] == "running"
        assert "stages" in body
        # Give the background task a moment to complete
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_run_with_custom_stages(self, handler):
        result = await handler.handle_run(
            {
                "input_text": "Test custom stages",
                "stages": ["ideation", "goals"],
                "dry_run": True,
            }
        )
        body = _body(result)
        assert body["stages"] == ["ideation", "goals"]

    @pytest.mark.asyncio
    async def test_run_stores_pipeline(self, handler):
        result = await handler.handle_run(
            {
                "input_text": "Store test",
                "dry_run": True,
            }
        )
        body = _body(result)
        pid = body["pipeline_id"]
        # Pipeline should be tracked (either in objects or tasks)
        assert pid in _pipeline_objects or pid in _pipeline_tasks


# =========================================================================
# handle_status
# =========================================================================


class TestHandleStatus:
    @pytest.mark.asyncio
    async def test_status_not_found(self, handler):
        result = await handler.handle_status("nonexistent")
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_status_for_running_pipeline(self, handler):
        # Start a pipeline
        run_result = await handler.handle_run(
            {
                "input_text": "Status test",
                "dry_run": True,
            }
        )
        run_body = _body(run_result)
        pid = run_body["pipeline_id"]

        status = await handler.handle_status(pid)
        status_body = _body(status)
        assert status_body["pipeline_id"] == pid
        assert status_body["status"] in ("running", "completed")
        assert "stage_status" in status_body

    @pytest.mark.asyncio
    async def test_status_after_completion(self, handler):
        run_result = await handler.handle_run(
            {
                "input_text": "Complete test",
                "stages": ["ideation"],
                "dry_run": True,
            }
        )
        run_body = _body(run_result)
        pid = run_body["pipeline_id"]
        # Wait for task to complete
        if pid in _pipeline_tasks:
            try:
                await asyncio.wait_for(_pipeline_tasks[pid], timeout=5)
            except (asyncio.TimeoutError, Exception):
                pass

        status = await handler.handle_status(pid)
        status_body = _body(status)
        assert status_body["pipeline_id"] == pid


# =========================================================================
# handle_graph
# =========================================================================


class TestHandleGraph:
    @pytest.mark.asyncio
    async def test_graph_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_graph("nonexistent")
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_graph_empty_pipeline(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "stage_status": {},
        }
        result = await handler.handle_graph("pipe-test")
        body = _body(result)
        assert body["pipeline_id"] == "pipe-test"
        assert "graphs" in body

    @pytest.mark.asyncio
    async def test_graph_with_goals(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "goals": {
                "goals": [
                    {"id": "g1", "title": "Goal 1", "dependencies": []},
                    {"id": "g2", "title": "Goal 2", "dependencies": ["g1"]},
                ],
            },
        }
        result = await handler.handle_graph("pipe-test")
        body = _body(result)
        goals_graph = body["graphs"].get("goals", {})
        assert len(goals_graph.get("nodes", [])) == 2
        assert len(goals_graph.get("edges", [])) == 1

    @pytest.mark.asyncio
    async def test_graph_with_workflow(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "final_workflow": {
                "steps": [
                    {"id": "s1", "name": "Step 1"},
                    {"id": "s2", "name": "Step 2"},
                ],
                "transitions": [
                    {"id": "t1", "from_step": "s1", "to_step": "s2"},
                ],
            },
        }
        result = await handler.handle_graph("pipe-test")
        body = _body(result)
        wf_graph = body["graphs"].get("workflow", {})
        assert len(wf_graph.get("nodes", [])) == 2
        assert len(wf_graph.get("edges", [])) == 1

    @pytest.mark.asyncio
    async def test_graph_stage_filter(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "goals": {"goals": [{"id": "g1", "title": "G1", "dependencies": []}]},
            "actions": {"nodes": [], "edges": []},
        }
        result = await handler.handle_graph("pipe-test", {"stage": "goals"})
        body = _body(result)
        assert "goals" in body["graphs"]
        # Actions should not be in result when filtering to goals
        assert "actions" not in body["graphs"]

    @pytest.mark.asyncio
    async def test_graph_goal_nodes_have_rf_format(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "goals": {
                "goals": [{"id": "g1", "title": "Test Goal", "dependencies": []}],
            },
        }
        result = await handler.handle_graph("pipe-test")
        body = _body(result)
        nodes = body["graphs"]["goals"]["nodes"]
        assert nodes[0]["type"] == "goalNode"
        assert "position" in nodes[0]
        assert "data" in nodes[0]


# =========================================================================
# handle_receipt
# =========================================================================


class TestHandleReceipt:
    @pytest.mark.asyncio
    async def test_receipt_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_receipt("nonexistent")
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_receipt_from_result(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "receipt": {"integrity_hash": "def456"},
        }
        result = await handler.handle_receipt("pipe-test")
        body = _body(result)
        assert body["receipt"]["integrity_hash"] == "def456"

    @pytest.mark.asyncio
    async def test_no_receipt_available(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-test",
            "stage_status": {},
        }
        result = await handler.handle_receipt("pipe-test")
        body = _body(result)
        assert "error" in body
