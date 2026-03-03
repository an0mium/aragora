"""Tests for unified-orchestrator wiring in from-braindump handler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler, _pipeline_objects


def _body(result) -> dict:
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


@pytest.fixture(autouse=True)
def _clear_pipeline_objects() -> None:
    _pipeline_objects.clear()
    yield
    _pipeline_objects.clear()


@pytest.fixture
def handler() -> CanvasPipelineHandler:
    return CanvasPipelineHandler()


def _mock_pipeline_result() -> MagicMock:
    result = MagicMock()
    result.pipeline_id = "pipe-brain1234"
    result.stage_status = {"ideas": "complete", "goals": "pending"}
    result.goal_graph = None
    result.to_dict.return_value = {
        "pipeline_id": "pipe-brain1234",
        "stage_status": {"ideas": "complete", "goals": "pending"},
    }
    return result


class TestFromBraindumpUnifiedOrchestrator:
    @pytest.mark.asyncio
    async def test_enabled_orchestrator_adds_summary_and_debate_url(
        self, handler: CanvasPipelineHandler
    ):
        mock_result = _mock_pipeline_result()

        with (
            patch("aragora.pipeline.brain_dump_parser.BrainDumpParser") as MockParser,
            patch("aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline") as MockPipeline,
            patch("aragora.server.handlers.canvas_pipeline._get_store") as mock_store,
            patch.object(
                CanvasPipelineHandler,
                "_run_unified_orchestrator",
                new=AsyncMock(
                    return_value=(
                        {
                            "enabled": True,
                            "run_id": "run-123",
                            "succeeded": True,
                            "stages_completed": ["research", "extend", "debate"],
                            "errors": [],
                            "debate_id": "debate-42",
                            "debate_url": "/debates/debate-42",
                        },
                        "extra context from orchestrator",
                    )
                ),
            ) as mock_orchestrator,
        ):
            MockParser.return_value.parse.return_value = ["idea 1", "idea 2"]
            MockPipeline.return_value.from_ideas.return_value = mock_result
            mock_store.return_value = MagicMock()

            result = await handler.handle_from_braindump(
                {
                    "text": "raw brain dump",
                    "use_unified_orchestrator": True,
                }
            )

            body = _body(result)
            assert body["pipeline_id"] == "pipe-brain1234"
            assert body["ideas_parsed"] == 2
            assert body["unified_orchestrator"]["run_id"] == "run-123"
            assert body["debate_id"] == "debate-42"
            assert body["debate_url"] == "/debates/debate-42"

            mock_orchestrator.assert_awaited_once()
            parse_input = MockParser.return_value.parse.call_args.args[0]
            assert "raw brain dump" in parse_input
            assert "extra context from orchestrator" in parse_input

    @pytest.mark.asyncio
    async def test_orchestrator_failure_degrades_to_pipeline_only(
        self, handler: CanvasPipelineHandler
    ):
        mock_result = _mock_pipeline_result()

        with (
            patch("aragora.pipeline.brain_dump_parser.BrainDumpParser") as MockParser,
            patch("aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline") as MockPipeline,
            patch("aragora.server.handlers.canvas_pipeline._get_store") as mock_store,
            patch.object(
                CanvasPipelineHandler,
                "_run_unified_orchestrator",
                new=AsyncMock(side_effect=RuntimeError("orchestrator unavailable")),
            ),
        ):
            MockParser.return_value.parse.return_value = ["idea 1"]
            MockPipeline.return_value.from_ideas.return_value = mock_result
            mock_store.return_value = MagicMock()

            result = await handler.handle_from_braindump(
                {
                    "text": "raw brain dump",
                    "use_unified_orchestrator": True,
                }
            )

            body = _body(result)
            assert body["pipeline_id"] == "pipe-brain1234"
            assert body["ideas_parsed"] == 1
            assert body["unified_orchestrator"]["succeeded"] is False
            assert "orchestrator unavailable" in body["unified_orchestrator"]["errors"][0]
