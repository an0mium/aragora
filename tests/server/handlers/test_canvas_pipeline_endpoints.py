"""Tests for CanvasPipelineHandler from-debate/from-ideas/advance/get/stage/convert endpoints."""

from __future__ import annotations

import json

import pytest

from aragora.server.handlers.canvas_pipeline import (
    CanvasPipelineHandler,
    _get_store,
    _pipeline_objects,
)


def _body(result) -> dict:
    """Extract JSON body from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear in-memory stores between tests."""
    _pipeline_objects.clear()
    yield
    _pipeline_objects.clear()


@pytest.fixture
def handler():
    return CanvasPipelineHandler()


@pytest.fixture
def sample_cartographer_data():
    return {
        "nodes": [
            {"id": "n1", "type": "proposal", "summary": "Build rate limiter", "content": "Token bucket"},
            {"id": "n2", "type": "evidence", "summary": "Reduces 429 errors", "content": "Evidence"},
            {"id": "n3", "type": "critique", "summary": "Distributed?", "content": "Question"},
        ],
        "edges": [
            {"source_id": "n2", "target_id": "n1", "relation": "supports"},
            {"source_id": "n3", "target_id": "n1", "relation": "responds_to"},
        ],
    }


# =========================================================================
# Route registration
# =========================================================================


class TestRouteRegistration:
    def test_from_debate_route(self, handler):
        assert "POST /api/v1/canvas/pipeline/from-debate" in handler.ROUTES

    def test_from_ideas_route(self, handler):
        assert "POST /api/v1/canvas/pipeline/from-ideas" in handler.ROUTES

    def test_advance_route(self, handler):
        assert "POST /api/v1/canvas/pipeline/advance" in handler.ROUTES

    def test_get_pipeline_route(self, handler):
        assert "GET /api/v1/canvas/pipeline/{id}" in handler.ROUTES

    def test_get_stage_route(self, handler):
        assert "GET /api/v1/canvas/pipeline/{id}/stage/{stage}" in handler.ROUTES

    def test_convert_debate_route(self, handler):
        assert "POST /api/v1/canvas/convert/debate" in handler.ROUTES

    def test_convert_workflow_route(self, handler):
        assert "POST /api/v1/canvas/convert/workflow" in handler.ROUTES


# =========================================================================
# handle_from_debate
# =========================================================================


class TestHandleFromDebate:
    @pytest.mark.asyncio
    async def test_missing_cartographer_data(self, handler):
        result = await handler.handle_from_debate({})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_empty_cartographer_data(self, handler):
        result = await handler.handle_from_debate({"cartographer_data": {}})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_from_debate_returns_pipeline(self, handler, sample_cartographer_data):
        result = await handler.handle_from_debate({
            "cartographer_data": sample_cartographer_data,
            "auto_advance": True,
        })
        body = _body(result)
        assert "pipeline_id" in body
        assert body["pipeline_id"].startswith("pipe-")
        assert "stage_status" in body
        assert "result" in body

    @pytest.mark.asyncio
    async def test_from_debate_stores_result(self, handler, sample_cartographer_data):
        result = await handler.handle_from_debate({
            "cartographer_data": sample_cartographer_data,
        })
        body = _body(result)
        pid = body["pipeline_id"]
        assert _get_store().get(pid) is not None
        assert pid in _pipeline_objects

    @pytest.mark.asyncio
    async def test_from_debate_no_auto_advance(self, handler, sample_cartographer_data):
        result = await handler.handle_from_debate({
            "cartographer_data": sample_cartographer_data,
            "auto_advance": False,
        })
        status = _body(result)["stage_status"]
        assert status.get("ideas") == "complete"
        assert status.get("goals") == "pending"

    @pytest.mark.asyncio
    async def test_from_debate_total_nodes(self, handler, sample_cartographer_data):
        result = await handler.handle_from_debate({
            "cartographer_data": sample_cartographer_data,
            "auto_advance": True,
        })
        body = _body(result)
        assert "total_nodes" in body
        assert body["total_nodes"] > 0

    @pytest.mark.asyncio
    async def test_from_debate_stages_completed(self, handler, sample_cartographer_data):
        result = await handler.handle_from_debate({
            "cartographer_data": sample_cartographer_data,
            "auto_advance": True,
        })
        assert _body(result)["stages_completed"] == 4


# =========================================================================
# handle_from_ideas
# =========================================================================


class TestHandleFromIdeas:
    @pytest.mark.asyncio
    async def test_missing_ideas(self, handler):
        result = await handler.handle_from_ideas({})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_empty_ideas(self, handler):
        result = await handler.handle_from_ideas({"ideas": []})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_from_ideas_returns_pipeline(self, handler):
        result = await handler.handle_from_ideas({
            "ideas": ["Build a rate limiter", "Add caching"],
        })
        body = _body(result)
        assert "pipeline_id" in body
        assert body["pipeline_id"].startswith("pipe-")
        assert "stage_status" in body
        assert "result" in body

    @pytest.mark.asyncio
    async def test_from_ideas_stores_result(self, handler):
        result = await handler.handle_from_ideas({
            "ideas": ["Idea one", "Idea two"],
        })
        pid = _body(result)["pipeline_id"]
        assert _get_store().get(pid) is not None
        assert pid in _pipeline_objects

    @pytest.mark.asyncio
    async def test_from_ideas_goals_count(self, handler):
        result = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching layer", "Improve docs"],
            "auto_advance": True,
        })
        body = _body(result)
        assert "goals_count" in body
        assert body["goals_count"] > 0

    @pytest.mark.asyncio
    async def test_from_ideas_no_auto_advance(self, handler):
        result = await handler.handle_from_ideas({
            "ideas": ["Some idea"],
            "auto_advance": False,
        })
        status = _body(result)["stage_status"]
        assert status.get("ideas") == "complete"
        assert status.get("goals") == "complete"
        # Actions not generated without auto_advance
        assert status.get("actions") == "pending"


# =========================================================================
# handle_advance
# =========================================================================


class TestHandleAdvance:
    @pytest.mark.asyncio
    async def test_missing_pipeline_id(self, handler):
        result = await handler.handle_advance({})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_missing_target_stage(self, handler):
        result = await handler.handle_advance({"pipeline_id": "pipe-123"})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler):
        result = await handler.handle_advance({
            "pipeline_id": "nonexistent",
            "target_stage": "goals",
        })
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_invalid_stage(self, handler):
        # Create a pipeline first
        ideas_result = await handler.handle_from_ideas({
            "ideas": ["Test idea"],
            "auto_advance": False,
        })
        pid = _body(ideas_result)["pipeline_id"]

        result = await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "invalid_stage",
        })
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_advance_to_actions(self, handler):
        # Create pipeline with goals
        ideas_result = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": False,
        })
        pid = _body(ideas_result)["pipeline_id"]

        result = await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "actions",
        })
        body = _body(result)
        assert body["pipeline_id"] == pid
        assert body["advanced_to"] == "actions"
        assert body["stage_status"]["actions"] == "complete"

    @pytest.mark.asyncio
    async def test_advance_updates_stores(self, handler):
        ideas_result = await handler.handle_from_ideas({
            "ideas": ["Test idea"],
            "auto_advance": False,
        })
        pid = _body(ideas_result)["pipeline_id"]

        await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "actions",
        })
        # Both stores should be updated
        stored = _get_store().get(pid)
        assert stored is not None
        assert stored["stage_status"]["actions"] == "complete"


# =========================================================================
# handle_get_pipeline
# =========================================================================


class TestHandleGetPipeline:
    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        result = await handler.handle_get_pipeline("nonexistent")
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_get_existing_pipeline(self, handler):
        create_result = await handler.handle_from_ideas({
            "ideas": ["Test idea"],
            "auto_advance": True,
        })
        pid = _body(create_result)["pipeline_id"]

        result = await handler.handle_get_pipeline(pid)
        body = _body(result)
        assert "pipeline_id" in body
        assert "ideas" in body
        assert "goals" in body


# =========================================================================
# handle_get_stage
# =========================================================================


class TestHandleGetStage:
    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler):
        result = await handler.handle_get_stage("nonexistent", "ideas")
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_invalid_stage(self, handler):
        create_result = await handler.handle_from_ideas({
            "ideas": ["Test idea"],
            "auto_advance": True,
        })
        pid = _body(create_result)["pipeline_id"]

        result = await handler.handle_get_stage(pid, "nonexistent")
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_get_ideas_stage(self, handler):
        create_result = await handler.handle_from_ideas({
            "ideas": ["Rate limiter", "Caching"],
            "auto_advance": True,
        })
        pid = _body(create_result)["pipeline_id"]

        result = await handler.handle_get_stage(pid, "ideas")
        body = _body(result)
        assert body["stage"] == "ideas"
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_goals_stage(self, handler):
        create_result = await handler.handle_from_ideas({
            "ideas": ["Rate limiter", "Caching"],
            "auto_advance": True,
        })
        pid = _body(create_result)["pipeline_id"]

        result = await handler.handle_get_stage(pid, "goals")
        body = _body(result)
        assert body["stage"] == "goals"
        assert "data" in body

    @pytest.mark.asyncio
    async def test_get_actions_stage(self, handler):
        create_result = await handler.handle_from_ideas({
            "ideas": ["Rate limiter", "Caching"],
            "auto_advance": True,
        })
        pid = _body(create_result)["pipeline_id"]

        result = await handler.handle_get_stage(pid, "actions")
        assert _body(result)["stage"] == "actions"

    @pytest.mark.asyncio
    async def test_get_orchestration_stage(self, handler):
        create_result = await handler.handle_from_ideas({
            "ideas": ["Rate limiter", "Caching"],
            "auto_advance": True,
        })
        pid = _body(create_result)["pipeline_id"]

        result = await handler.handle_get_stage(pid, "orchestration")
        assert _body(result)["stage"] == "orchestration"


# =========================================================================
# handle_convert_debate
# =========================================================================


class TestHandleConvertDebate:
    @pytest.mark.asyncio
    async def test_missing_data(self, handler):
        result = await handler.handle_convert_debate({})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_convert_debate_returns_react_flow(self, handler, sample_cartographer_data):
        result = await handler.handle_convert_debate({
            "cartographer_data": sample_cartographer_data,
        })
        body = _body(result)
        assert "nodes" in body
        assert "edges" in body
        assert len(body["nodes"]) == 3


# =========================================================================
# handle_convert_workflow
# =========================================================================


class TestHandleConvertWorkflow:
    @pytest.mark.asyncio
    async def test_missing_data(self, handler):
        result = await handler.handle_convert_workflow({})
        assert "error" in _body(result)

    @pytest.mark.asyncio
    async def test_convert_workflow_returns_react_flow(self, handler):
        workflow_data = {
            "name": "Test Workflow",
            "steps": [
                {"id": "s1", "name": "Step 1", "type": "task"},
                {"id": "s2", "name": "Step 2", "type": "task"},
            ],
            "transitions": [
                {"from_step": "s1", "to_step": "s2"},
            ],
        }
        result = await handler.handle_convert_workflow({
            "workflow_data": workflow_data,
        })
        body = _body(result)
        assert "nodes" in body
        assert "edges" in body
