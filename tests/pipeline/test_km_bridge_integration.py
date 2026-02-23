"""Tests for Pipeline → KnowledgeMound integration wiring.

Verifies that pipeline handlers and MCP tools call PipelineKMBridge.store_pipeline_result()
after producing a PipelineResult, and that import/runtime errors degrade gracefully.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result(pipeline_id: str = "pipe-test123") -> MagicMock:
    """Create a minimal mock PipelineResult."""
    result = MagicMock()
    result.pipeline_id = pipeline_id
    result.stage_status = {"ideas": "complete"}
    result.ideas_canvas = None
    result.actions_canvas = None
    result.orchestration_canvas = None
    result.universal_graph = None
    result.goal_graph = None
    result.to_dict.return_value = {
        "pipeline_id": pipeline_id,
        "stage_status": {"ideas": "complete"},
    }
    return result


# ---------------------------------------------------------------------------
# Handler integration tests
# ---------------------------------------------------------------------------


class TestPersistCalledFromHandlerFromDebate:
    @pytest.mark.asyncio
    async def test_persist_called_from_handler_from_debate(self):
        """handle_from_debate should call _persist_pipeline_to_km after pipeline runs."""
        mock_result = _make_mock_result()

        with (
            patch(
                "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            ) as MockPipeline,
            patch(
                "aragora.server.handlers.canvas_pipeline._get_store",
            ) as mock_store,
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_pipeline_to_km",
            ) as mock_km_persist,
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_universal_graph",
            ),
        ):
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.from_debate.return_value = mock_result
            MockPipeline.return_value = mock_pipeline_inst
            mock_store.return_value = MagicMock()

            from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler

            handler = CanvasPipelineHandler()
            request_data = {"cartographer_data": {"nodes": [{"id": "n1"}]}}

            result = await handler.handle_from_debate(request_data)

            mock_km_persist.assert_called_once_with(mock_result)


class TestPersistCalledFromHandlerFromIdeas:
    @pytest.mark.asyncio
    async def test_persist_called_from_handler_from_ideas(self):
        """handle_from_ideas should call _persist_pipeline_to_km after pipeline runs."""
        mock_result = _make_mock_result()

        with (
            patch(
                "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            ) as MockPipeline,
            patch(
                "aragora.server.handlers.canvas_pipeline._get_store",
            ) as mock_store,
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_pipeline_to_km",
            ) as mock_km_persist,
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_universal_graph",
            ),
        ):
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.from_ideas.return_value = mock_result
            MockPipeline.return_value = mock_pipeline_inst
            mock_store.return_value = MagicMock()

            from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler

            handler = CanvasPipelineHandler()
            request_data = {"ideas": ["idea 1", "idea 2"]}

            result = await handler.handle_from_ideas(request_data)

            mock_km_persist.assert_called_once_with(mock_result)


class TestPersistCalledFromHandlerFromBraindump:
    @pytest.mark.asyncio
    async def test_persist_called_from_handler_from_braindump(self):
        """handle_from_braindump should call _persist_pipeline_to_km after pipeline runs."""
        mock_result = _make_mock_result()

        with (
            patch(
                "aragora.pipeline.brain_dump_parser.BrainDumpParser",
            ) as MockParser,
            patch(
                "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            ) as MockPipeline,
            patch(
                "aragora.server.handlers.canvas_pipeline._get_store",
            ) as mock_store,
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_pipeline_to_km",
            ) as mock_km_persist,
        ):
            mock_parser_inst = MagicMock()
            mock_parser_inst.parse.return_value = ["parsed idea 1"]
            MockParser.return_value = mock_parser_inst

            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.from_ideas.return_value = mock_result
            MockPipeline.return_value = mock_pipeline_inst
            mock_store.return_value = MagicMock()

            from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler

            handler = CanvasPipelineHandler()
            request_data = {"text": "some braindump text with ideas"}

            result = await handler.handle_from_braindump(request_data)

            mock_km_persist.assert_called_once_with(mock_result)


class TestPersistGracefulDegradationImportError:
    @pytest.mark.asyncio
    async def test_persist_graceful_degradation_import_error(self):
        """Handler should complete normally when km_bridge import fails."""
        mock_result = _make_mock_result()

        with (
            patch(
                "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            ) as MockPipeline,
            patch(
                "aragora.server.handlers.canvas_pipeline._get_store",
            ) as mock_store,
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_universal_graph",
            ),
            patch(
                "aragora.server.handlers.canvas_pipeline._persist_pipeline_to_km",
                side_effect=ImportError("no km_bridge"),
            ),
        ):
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.from_ideas.return_value = mock_result
            MockPipeline.return_value = mock_pipeline_inst
            mock_store.return_value = MagicMock()

            from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler

            handler = CanvasPipelineHandler()
            request_data = {"ideas": ["idea 1"]}

            # The _persist_pipeline_to_km function itself handles exceptions
            # internally. Here we test that even if it somehow raises, the
            # handler's own try/except catches ImportError gracefully.
            result = await handler.handle_from_ideas(request_data)
            # Should still produce a valid error or success response (ImportError caught)
            assert result is not None


class TestPersistGracefulDegradationRuntimeError:
    @pytest.mark.asyncio
    async def test_persist_graceful_degradation_runtime_error(self):
        """Handler should complete when store_pipeline_result raises RuntimeError."""
        from aragora.server.handlers.canvas_pipeline import _persist_pipeline_to_km

        mock_result = _make_mock_result()

        # Test the helper function directly with a RuntimeError from the bridge
        with patch(
            "aragora.pipeline.km_bridge.PipelineKMBridge",
        ) as MockBridge:
            mock_bridge_inst = MagicMock()
            mock_bridge_inst.store_pipeline_result.side_effect = RuntimeError("KM down")
            MockBridge.return_value = mock_bridge_inst

            # Should not raise — exceptions are caught inside _persist_pipeline_to_km
            _persist_pipeline_to_km(mock_result)


# ---------------------------------------------------------------------------
# MCP tool integration tests
# ---------------------------------------------------------------------------


class TestPersistCalledFromMCPSyncPath:
    @pytest.mark.asyncio
    async def test_persist_called_from_mcp_sync_path(self):
        """run_pipeline_tool sync path (ideas) should call PipelineKMBridge."""
        mock_result = _make_mock_result()

        with (
            patch(
                "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            ) as MockPipeline,
            patch(
                "aragora.pipeline.km_bridge.PipelineKMBridge",
            ) as MockBridge,
        ):
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.from_ideas.return_value = mock_result
            MockPipeline.return_value = mock_pipeline_inst

            mock_bridge_inst = MagicMock()
            MockBridge.return_value = mock_bridge_inst

            from aragora.mcp.tools_module.pipeline import run_pipeline_tool

            result = await run_pipeline_tool(ideas='["idea 1", "idea 2"]')

            mock_bridge_inst.store_pipeline_result.assert_called_once_with(mock_result)
            assert result.get("pipeline_id") == "pipe-test123"


class TestPersistCalledFromMCPAsyncPath:
    @pytest.mark.asyncio
    async def test_persist_called_from_mcp_async_path(self):
        """run_pipeline_tool async path (input_text) should call PipelineKMBridge."""
        mock_result = _make_mock_result()

        with (
            patch(
                "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            ) as MockPipeline,
            patch(
                "aragora.pipeline.km_bridge.PipelineKMBridge",
            ) as MockBridge,
            patch(
                "aragora.storage.pipeline_store.get_pipeline_store",
            ) as mock_store_fn,
        ):
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.run = AsyncMock(return_value=mock_result)
            MockPipeline.return_value = mock_pipeline_inst

            mock_bridge_inst = MagicMock()
            MockBridge.return_value = mock_bridge_inst

            mock_store_fn.return_value = MagicMock()

            from aragora.mcp.tools_module.pipeline import run_pipeline_tool

            result = await run_pipeline_tool(input_text="Build a rate limiter")

            mock_bridge_inst.store_pipeline_result.assert_called_once_with(mock_result)
            assert result.get("pipeline_id") == "pipe-test123"


# ---------------------------------------------------------------------------
# Bridge unit test
# ---------------------------------------------------------------------------


class TestBridgeStorePipelineResultDelegatesToAdapter:
    def test_bridge_store_pipeline_result_delegates_to_adapter(self):
        """PipelineKMBridge.store_pipeline_result should delegate to DecisionPlanAdapter."""
        mock_km = MagicMock()
        mock_result = _make_mock_result()

        with patch(
            "aragora.knowledge.mound.adapters.decision_plan_adapter.DecisionPlanAdapter",
        ) as MockAdapter:
            mock_adapter_inst = MagicMock()
            MockAdapter.return_value = mock_adapter_inst

            from aragora.pipeline.km_bridge import PipelineKMBridge

            bridge = PipelineKMBridge(knowledge_mound=mock_km)
            success = bridge.store_pipeline_result(mock_result)

            MockAdapter.assert_called_once_with(mock_km)
            mock_adapter_inst.store.assert_called_once_with(mock_result.to_dict())
            assert success is True
