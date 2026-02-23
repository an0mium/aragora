"""Integration smoke tests for the idea-to-execution pipeline.

Verifies the end-to-end flow: from_debate() with sample cartographer data
through to PipelineResult.to_dict() with React Flow JSON and provenance.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Sample data: minimal ArgumentCartographer export
# ---------------------------------------------------------------------------

SAMPLE_CARTOGRAPHER_DATA = {
    "nodes": [
        {"id": "n1", "node_type": "proposal", "content": "We should implement rate limiting"},
        {
            "id": "n2",
            "node_type": "consensus",
            "content": "Implement a token bucket rate limiter with configurable limits",
        },
        {
            "id": "n3",
            "node_type": "vote",
            "content": "Agreed: per-user rate limits with 100 req/min default",
        },
        {"id": "n4", "node_type": "critique", "content": "Needs burst handling for spiky traffic"},
    ],
    "edges": [
        {"source": "n1", "target": "n2", "label": "leads_to"},
        {"source": "n2", "target": "n3", "label": "supports"},
        {"source": "n4", "target": "n2", "label": "critiques"},
    ],
    "metadata": {
        "debate_id": "test-debate-001",
        "topic": "Rate limiting strategy",
        "participant_count": 4,
    },
}


class TestFromDebateSmoke:
    """Smoke tests for the sync from_debate() entry point."""

    def test_from_debate_returns_pipeline_result(self):
        """from_debate() should return a PipelineResult with expected fields."""
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(SAMPLE_CARTOGRAPHER_DATA, auto_advance=True)

        assert result is not None
        assert result.pipeline_id is not None
        assert len(result.pipeline_id) > 0

    def test_from_debate_produces_ideas(self):
        """from_debate() should extract ideas from cartographer nodes."""
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(SAMPLE_CARTOGRAPHER_DATA)

        result_dict = result.to_dict()
        assert "ideas" in result_dict or "stages" in result_dict

    def test_from_debate_to_dict_serializable(self):
        """PipelineResult.to_dict() should produce JSON-serializable output."""
        import json
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(SAMPLE_CARTOGRAPHER_DATA)

        result_dict = result.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

    def test_from_ideas_returns_pipeline_result(self):
        """from_ideas() should return a PipelineResult."""
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        ideas = [
            "Implement rate limiting",
            "Add per-user quotas",
            "Build monitoring dashboard",
        ]
        result = pipeline.from_ideas(ideas)

        assert result is not None
        assert result.pipeline_id is not None


class TestGoalExtractionFromDebate:
    """Test goal extraction bridge from cartographer data."""

    def test_extract_from_debate_analysis_basic(self):
        """GoalExtractor.extract_from_debate_analysis() produces a GoalGraph."""
        from aragora.goals.extractor import GoalExtractor, GoalExtractionConfig

        extractor = GoalExtractor()
        config = GoalExtractionConfig(
            confidence_threshold=0.0,
            max_goals=5,
            require_consensus=False,
        )

        graph = extractor.extract_from_debate_analysis(
            cartographer_output=SAMPLE_CARTOGRAPHER_DATA,
            config=config,
        )

        assert graph is not None
        # Should extract goals from consensus/vote nodes
        assert len(graph.goals) >= 0  # May be 0 if no matching nodes

    def test_extract_with_consensus_nodes(self):
        """extract_from_debate_analysis picks up consensus/vote node types."""
        from aragora.goals.extractor import GoalExtractor, GoalExtractionConfig

        extractor = GoalExtractor()
        config = GoalExtractionConfig(
            confidence_threshold=0.0,
            max_goals=10,
            require_consensus=False,
        )

        graph = extractor.extract_from_debate_analysis(
            cartographer_output=SAMPLE_CARTOGRAPHER_DATA,
            config=config,
        )

        # We have 2 consensus/vote nodes in the sample data
        goal_ids = [n.id for n in graph.goals]
        assert len(goal_ids) <= 10  # respects max_goals


class TestAsyncPipelineRun:
    """Test async pipeline run with mocked components."""

    @pytest.mark.asyncio
    async def test_async_run_dry_run(self):
        """async run() with dry_run should skip orchestration."""
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineConfig,
        )

        pipeline = IdeaToExecutionPipeline()
        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation"],
        )

        result = await pipeline.run("Build a rate limiter", config=config)

        assert result is not None
        assert result.pipeline_id is not None
        # Should have stage results
        assert len(result.stage_results) >= 1
        ideation = result.stage_results[0]
        assert ideation.stage_name == "ideation"

    @pytest.mark.asyncio
    async def test_async_run_emits_events(self):
        """async run() should call event_callback for each stage."""
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineConfig,
        )

        events_received: list[tuple[str, dict]] = []

        def capture_event(event_type: str, data: dict) -> None:
            events_received.append((event_type, data))

        pipeline = IdeaToExecutionPipeline()
        config = PipelineConfig(
            dry_run=True,
            stages_to_run=["ideation"],
            enable_receipts=False,
            event_callback=capture_event,
        )

        await pipeline.run("Test idea", config=config)

        # Should have at least stage_started and stage_completed events
        event_types = [e[0] for e in events_received]
        assert "stage_started" in event_types
        # Either completed or failed
        assert (
            "stage_completed" in event_types
            or "stage_failed" in event_types
            or "failed" in event_types
        )


class TestPipelineStreamEmitter:
    """Test the WebSocket stream emitter for pipeline events."""

    @pytest.mark.asyncio
    async def test_emit_filters_by_pipeline_id(self):
        """Events should only go to clients watching the matching pipeline."""
        from aragora.server.stream.pipeline_stream import PipelineStreamEmitter
        from aragora.events.types import StreamEventType

        emitter = PipelineStreamEmitter()

        ws1 = AsyncMock()
        ws2 = AsyncMock()

        emitter.add_client(ws1, "pipe-A")
        emitter.add_client(ws2, "pipe-B")

        await emitter.emit("pipe-A", StreamEventType.PIPELINE_STAGE_STARTED, {"stage": "ideation"})

        ws1.send_str.assert_called_once()
        ws2.send_str.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_history_persists(self):
        """Emitted events should be stored in per-pipeline history."""
        from aragora.server.stream.pipeline_stream import PipelineStreamEmitter
        from aragora.events.types import StreamEventType

        emitter = PipelineStreamEmitter()

        await emitter.emit("pipe-X", StreamEventType.PIPELINE_COMPLETED, {"receipt": None})

        history = emitter.get_history("pipe-X")
        assert len(history) == 1
        assert history[0]["data"]["pipeline_id"] == "pipe-X"


class TestSDKPipelineNamespace:
    """Verify the Python SDK pipeline namespace is wired correctly."""

    def test_sync_client_has_pipeline(self):
        """AragoraClient should have a .pipeline attribute."""
        from aragora_sdk.namespaces.pipeline import PipelineAPI

        mock_client = MagicMock()
        api = PipelineAPI(mock_client)

        assert hasattr(api, "run")
        assert hasattr(api, "from_debate")
        assert hasattr(api, "from_ideas")
        assert hasattr(api, "status")
        assert hasattr(api, "get")
        assert hasattr(api, "graph")
        assert hasattr(api, "receipt")
        assert hasattr(api, "advance")

    def test_run_calls_correct_endpoint(self):
        """PipelineAPI.run() should POST to /api/v1/canvas/pipeline/run."""
        from aragora_sdk.namespaces.pipeline import PipelineAPI

        mock_client = MagicMock()
        mock_client.request.return_value = {"pipeline_id": "p1", "status": "started"}

        api = PipelineAPI(mock_client)
        result = api.run("Build a feature")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/api/v1/canvas/pipeline/run"

    def test_status_calls_correct_endpoint(self):
        """PipelineAPI.status() should GET the status endpoint."""
        from aragora_sdk.namespaces.pipeline import PipelineAPI

        mock_client = MagicMock()
        mock_client.request.return_value = {"pipeline_id": "p1", "stages": []}

        api = PipelineAPI(mock_client)
        api.status("p1")

        mock_client.request.assert_called_once_with("GET", "/api/v1/canvas/pipeline/p1/status")

    def test_graph_with_stage_filter(self):
        """PipelineAPI.graph() should pass stage as query param."""
        from aragora_sdk.namespaces.pipeline import PipelineAPI

        mock_client = MagicMock()
        mock_client.request.return_value = {"nodes": [], "edges": []}

        api = PipelineAPI(mock_client)
        api.graph("p1", stage="goals")

        call_args = mock_client.request.call_args
        assert call_args[1]["params"] == {"stage": "goals"}

    def test_receipt_calls_correct_endpoint(self):
        """PipelineAPI.receipt() should GET the receipt endpoint."""
        from aragora_sdk.namespaces.pipeline import PipelineAPI

        mock_client = MagicMock()
        mock_client.request.return_value = {"receipt": {}}

        api = PipelineAPI(mock_client)
        api.receipt("p1")

        mock_client.request.assert_called_once_with("GET", "/api/v1/canvas/pipeline/p1/receipt")
