"""Tests for MCP pipeline tools.

Covers:
- run_pipeline_tool: ideas list, input_text, empty input, dry_run
- extract_goals_tool: valid data, empty data, invalid JSON, threshold
- get_pipeline_status_tool: found, not found, missing ID
- advance_pipeline_stage_tool: valid, invalid stage, missing pipeline
- Import error fallbacks for each tool
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.mcp.tools_module.pipeline import (
    advance_pipeline_stage_tool,
    extract_goals_tool,
    get_pipeline_status_tool,
    run_pipeline_tool,
)


# ---------------------------------------------------------------------------
# run_pipeline_tool
# ---------------------------------------------------------------------------


class TestRunPipelineTool:
    @pytest.mark.asyncio
    async def test_with_ideas_list(self):
        ideas = json.dumps(["Build caching", "Add monitoring"])
        result = await run_pipeline_tool(ideas=ideas)
        assert "error" not in result
        assert "pipeline_id" in result
        assert result["goals_count"] >= 0

    @pytest.mark.asyncio
    async def test_with_input_text(self):
        result = await run_pipeline_tool(input_text="Improve API performance", dry_run=True)
        assert "error" not in result
        assert "pipeline_id" in result

    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await run_pipeline_tool()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_ideas_json(self):
        result = await run_pipeline_tool(ideas="not-json")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ideas_not_array(self):
        result = await run_pipeline_tool(ideas=json.dumps({"key": "val"}))
        assert "error" in result
        assert "array" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_ideas_list(self):
        result = await run_pipeline_tool(ideas=json.dumps([]))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_import_error_fallback(self):
        with patch.dict("sys.modules", {"aragora.pipeline.idea_to_execution": None}):
            result = await run_pipeline_tool(ideas=json.dumps(["test"]))
        assert "error" in result


# ---------------------------------------------------------------------------
# extract_goals_tool
# ---------------------------------------------------------------------------


class TestExtractGoalsTool:
    @pytest.mark.asyncio
    async def test_valid_ideas(self):
        ideas = json.dumps(["Build rate limiter", "Add caching"])
        result = await extract_goals_tool(ideas_json=ideas)
        assert "error" not in result
        assert "goal_graph" in result
        assert "goals_count" in result

    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await extract_goals_tool()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        result = await extract_goals_tool(ideas_json="not valid json")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_not_array(self):
        result = await extract_goals_tool(ideas_json=json.dumps("single string"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_high_confidence_threshold(self):
        ideas = json.dumps(["Build something"])
        result = await extract_goals_tool(ideas_json=ideas, confidence_threshold=0.99)
        assert "error" not in result
        # Most structural goals have lower confidence, so count may be 0
        assert isinstance(result["goals_count"], int)

    @pytest.mark.asyncio
    async def test_import_error_fallback(self):
        with patch.dict("sys.modules", {"aragora.goals.extractor": None}):
            result = await extract_goals_tool(ideas_json=json.dumps(["test"]))
        assert "error" in result


# ---------------------------------------------------------------------------
# get_pipeline_status_tool
# ---------------------------------------------------------------------------


class TestGetPipelineStatusTool:
    @pytest.mark.asyncio
    async def test_missing_id(self):
        result = await get_pipeline_status_tool()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_found(self):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "stage_status": {"ideas": "complete", "goals": "pending"},
            "duration": 1.5,
            "receipt": {"id": "r1"},
        }
        with patch(
            "aragora.storage.pipeline_store.get_pipeline_store",
            return_value=mock_store,
        ):
            result = await get_pipeline_status_tool(pipeline_id="pipe-abc")
        assert "error" not in result
        assert result["pipeline_id"] == "pipe-abc"
        assert result["stage_status"]["ideas"] == "complete"
        assert result["has_receipt"] is True

    @pytest.mark.asyncio
    async def test_not_found(self):
        mock_store = MagicMock()
        mock_store.get.return_value = None
        with patch(
            "aragora.storage.pipeline_store.get_pipeline_store",
            return_value=mock_store,
        ):
            result = await get_pipeline_status_tool(pipeline_id="nonexistent")
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_import_error_fallback(self):
        with patch.dict("sys.modules", {"aragora.storage.pipeline_store": None}):
            result = await get_pipeline_status_tool(pipeline_id="pipe-1")
        assert "error" in result


# ---------------------------------------------------------------------------
# advance_pipeline_stage_tool
# ---------------------------------------------------------------------------


class TestAdvancePipelineStageTool:
    @pytest.mark.asyncio
    async def test_missing_pipeline_id(self):
        result = await advance_pipeline_stage_tool()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_target_stage(self):
        result = await advance_pipeline_stage_tool(pipeline_id="pipe-1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_pipeline_not_found(self):
        result = await advance_pipeline_stage_tool(pipeline_id="nonexistent", target_stage="goals")
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_stage(self):
        from aragora.server.handlers.canvas_pipeline import _pipeline_objects

        _pipeline_objects["pipe-test"] = MagicMock()
        try:
            result = await advance_pipeline_stage_tool(
                pipeline_id="pipe-test", target_stage="invalid"
            )
            assert "error" in result
            assert "Invalid stage" in result["error"]
        finally:
            _pipeline_objects.pop("pipe-test", None)

    @pytest.mark.asyncio
    async def test_import_error_fallback(self):
        with patch.dict(
            "sys.modules",
            {
                "aragora.canvas.stages": None,
                "aragora.pipeline.idea_to_execution": None,
            },
        ):
            result = await advance_pipeline_stage_tool(pipeline_id="pipe-1", target_stage="goals")
        assert "error" in result
