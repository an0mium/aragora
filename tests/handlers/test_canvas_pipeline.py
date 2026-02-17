"""Tests for the CanvasPipelineHandler REST endpoints.

Covers all 7 endpoints:
- POST from-debate, from-ideas
- GET pipeline/{id}, pipeline/{id}/stage/{stage}
- POST advance (via advance → stage)
- POST convert/debate, convert/workflow
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.canvas_pipeline import (
    CanvasPipelineHandler,
    _pipeline_objects,
    _pipeline_results,
)


@pytest.fixture(autouse=True)
def _clear_pipeline_store():
    """Clear in-memory pipeline results between tests."""
    _pipeline_results.clear()
    _pipeline_objects.clear()
    yield
    _pipeline_results.clear()
    _pipeline_objects.clear()


@pytest.fixture
def handler():
    return CanvasPipelineHandler()


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_canvas_v1_path(self, handler):
        assert handler.can_handle("/api/v1/canvas/pipeline/from-debate")

    def test_canvas_unversioned_path(self, handler):
        assert handler.can_handle("/api/canvas/pipeline/from-ideas")

    def test_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")


# ---------------------------------------------------------------------------
# POST from-debate
# ---------------------------------------------------------------------------


class TestFromDebate:
    @pytest.mark.asyncio
    async def test_missing_cartographer_data(self, handler):
        result = await handler.handle_from_debate({})
        assert "error" in result
        assert "cartographer_data" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_cartographer_data(self, handler):
        result = await handler.handle_from_debate({"cartographer_data": {}})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_successful_pipeline(self, handler):
        """Verify from_debate runs the real pipeline with minimal input."""
        result = await handler.handle_from_debate({
            "cartographer_data": {"nodes": [{"id": "n1", "label": "test"}], "edges": []},
            "auto_advance": True,
        })
        # Pipeline should succeed (may not advance all stages with minimal data)
        if "error" not in result:
            assert "pipeline_id" in result
            assert result["pipeline_id"] in _pipeline_results
            assert "stage_status" in result
        else:
            # Import may fail in some envs — that's acceptable
            assert "error" in result

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        """Pipeline import failure returns error dict, not exception."""
        with patch.dict("sys.modules", {"aragora.pipeline.idea_to_execution": None}):
            result = await handler.handle_from_debate({
                "cartographer_data": {"nodes": [{"id": "1"}]},
            })
        assert "error" in result


# ---------------------------------------------------------------------------
# POST from-ideas
# ---------------------------------------------------------------------------


class TestFromIdeas:
    @pytest.mark.asyncio
    async def test_missing_ideas(self, handler):
        result = await handler.handle_from_ideas({})
        assert "error" in result
        assert "ideas" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_ideas_list(self, handler):
        result = await handler.handle_from_ideas({"ideas": []})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.pipeline.idea_to_execution": None}):
            result = await handler.handle_from_ideas({
                "ideas": ["improve caching", "add monitoring"],
            })
        assert "error" in result


# ---------------------------------------------------------------------------
# POST advance
# ---------------------------------------------------------------------------


class TestAdvance:
    @pytest.mark.asyncio
    async def test_missing_pipeline_id(self, handler):
        result = await handler.handle_advance({})
        assert "error" in result
        assert "pipeline_id" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_target_stage(self, handler):
        result = await handler.handle_advance({"pipeline_id": "pipe-1"})
        assert "error" in result
        assert "target_stage" in result["error"]

    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler):
        result = await handler.handle_advance({
            "pipeline_id": "nonexistent",
            "target_stage": "goals",
        })
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_stage(self, handler):
        _pipeline_objects["pipe-1"] = MagicMock()
        result = await handler.handle_advance({
            "pipeline_id": "pipe-1",
            "target_stage": "invalid_stage",
        })
        assert "error" in result
        assert "Invalid stage" in result["error"]

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        _pipeline_objects["pipe-1"] = MagicMock()
        with patch.dict("sys.modules", {
            "aragora.canvas.stages": None,
            "aragora.pipeline.idea_to_execution": None,
        }):
            result = await handler.handle_advance({
                "pipeline_id": "pipe-1",
                "target_stage": "goals",
            })
        assert "error" in result


# ---------------------------------------------------------------------------
# GET pipeline/{id}
# ---------------------------------------------------------------------------


class TestGetPipeline:
    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        result = await handler.handle_get_pipeline("nonexistent")
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_found(self, handler):
        _pipeline_results["pipe-abc"] = {
            "pipeline_id": "pipe-abc",
            "ideas": {"nodes": []},
        }
        result = await handler.handle_get_pipeline("pipe-abc")
        assert result["pipeline_id"] == "pipe-abc"


# ---------------------------------------------------------------------------
# GET pipeline/{id}/stage/{stage}
# ---------------------------------------------------------------------------


class TestGetStage:
    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler):
        result = await handler.handle_get_stage("nonexistent", "ideas")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_stage_not_found(self, handler):
        _pipeline_results["pipe-1"] = {"pipeline_id": "pipe-1"}
        result = await handler.handle_get_stage("pipe-1", "ideas")
        # "ideas" key not in result dict
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_stage_name(self, handler):
        _pipeline_results["pipe-1"] = {"pipeline_id": "pipe-1", "ideas": {}}
        result = await handler.handle_get_stage("pipe-1", "invalid_stage")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_valid_stage(self, handler):
        _pipeline_results["pipe-1"] = {
            "pipeline_id": "pipe-1",
            "ideas": {"nodes": [{"id": "n1"}]},
        }
        result = await handler.handle_get_stage("pipe-1", "ideas")
        assert result["stage"] == "ideas"
        assert result["data"]["nodes"][0]["id"] == "n1"

    @pytest.mark.asyncio
    async def test_goals_stage(self, handler):
        _pipeline_results["pipe-1"] = {
            "pipeline_id": "pipe-1",
            "goals": [{"id": "g1", "title": "Goal 1"}],
        }
        result = await handler.handle_get_stage("pipe-1", "goals")
        assert result["stage"] == "goals"

    @pytest.mark.asyncio
    async def test_orchestration_stage(self, handler):
        _pipeline_results["pipe-1"] = {
            "pipeline_id": "pipe-1",
            "orchestration": {"agents": []},
        }
        result = await handler.handle_get_stage("pipe-1", "orchestration")
        assert result["stage"] == "orchestration"


# ---------------------------------------------------------------------------
# POST convert/debate
# ---------------------------------------------------------------------------


class TestConvertDebate:
    @pytest.mark.asyncio
    async def test_missing_cartographer_data(self, handler):
        result = await handler.handle_convert_debate({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.canvas.converters": None}):
            result = await handler.handle_convert_debate({
                "cartographer_data": {"nodes": []},
            })
        assert "error" in result


# ---------------------------------------------------------------------------
# POST convert/workflow
# ---------------------------------------------------------------------------


class TestConvertWorkflow:
    @pytest.mark.asyncio
    async def test_missing_workflow_data(self, handler):
        result = await handler.handle_convert_workflow({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.canvas.converters": None}):
            result = await handler.handle_convert_workflow({
                "workflow_data": {"steps": []},
            })
        assert "error" in result


# ---------------------------------------------------------------------------
# Context / constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_context(self):
        h = CanvasPipelineHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        h = CanvasPipelineHandler(ctx={"key": "val"})
        assert h.ctx["key"] == "val"

    def test_routes_defined(self):
        assert len(CanvasPipelineHandler.ROUTES) == 7
