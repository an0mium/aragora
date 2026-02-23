"""Tests for SpectatorStream integration in the Idea-to-Execution Pipeline.

Verifies that SpectatorStream.emit() is called at each pipeline stage
transition:
  - Pipeline started/completed/failed
  - Stage started/completed for ideation, goals, actions, orchestration
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    _spectate,
)
from aragora.spectate.events import SpectatorEvents


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def pipeline():
    """Default pipeline with no AI agent."""
    return IdeaToExecutionPipeline()


@pytest.fixture
def sample_ideas():
    """Sample idea strings for testing."""
    return [
        "Build a rate limiter for API endpoints",
        "Add Redis-backed caching for frequently accessed data",
        "Improve API docs with OpenAPI interactive playground",
    ]


@pytest.fixture
def sample_cartographer_data():
    """Sample ArgumentCartographer output."""
    return {
        "nodes": [
            {
                "id": "n1",
                "type": "proposal",
                "summary": "Build a rate limiter",
                "content": "Token bucket",
            },
            {"id": "n2", "type": "evidence", "summary": "Reduces errors", "content": "Evidence"},
        ],
        "edges": [
            {"source_id": "n2", "target_id": "n1", "relation": "supports"},
        ],
    }


@pytest.fixture
def mock_spectator():
    """Patch the _spectate function to capture calls."""
    with patch("aragora.pipeline.idea_to_execution._spectate") as mock:
        yield mock


@pytest.fixture
def mock_spectator_stream():
    """Patch SpectatorStream at the module level to capture emit calls."""
    mock_stream = MagicMock()
    mock_stream.enabled = True
    with patch(
        "aragora.pipeline.idea_to_execution._get_spectator_stream",
        return_value=mock_stream,
    ):
        yield mock_stream


# =========================================================================
# Event constant tests
# =========================================================================


class TestPipelineSpectatorEvents:
    """Verify pipeline event constants are registered."""

    def test_pipeline_started_constant(self):
        assert SpectatorEvents.PIPELINE_STARTED == "pipeline.started"

    def test_pipeline_stage_started_constant(self):
        assert SpectatorEvents.PIPELINE_STAGE_STARTED == "pipeline.stage_started"

    def test_pipeline_stage_completed_constant(self):
        assert SpectatorEvents.PIPELINE_STAGE_COMPLETED == "pipeline.stage_completed"

    def test_pipeline_completed_constant(self):
        assert SpectatorEvents.PIPELINE_COMPLETED == "pipeline.completed"

    def test_pipeline_failed_constant(self):
        assert SpectatorEvents.PIPELINE_FAILED == "pipeline.failed"

    def test_pipeline_events_in_valid_set(self):
        from aragora.spectate.stream import VALID_EVENT_TYPES

        assert SpectatorEvents.PIPELINE_STARTED in VALID_EVENT_TYPES
        assert SpectatorEvents.PIPELINE_STAGE_STARTED in VALID_EVENT_TYPES
        assert SpectatorEvents.PIPELINE_STAGE_COMPLETED in VALID_EVENT_TYPES
        assert SpectatorEvents.PIPELINE_COMPLETED in VALID_EVENT_TYPES
        assert SpectatorEvents.PIPELINE_FAILED in VALID_EVENT_TYPES


# =========================================================================
# _spectate helper tests
# =========================================================================


class TestSpectateHelper:
    """Test the _spectate graceful degradation helper."""

    def test_spectate_calls_emit(self, mock_spectator_stream):
        """Verify _spectate forwards to SpectatorStream.emit."""
        _spectate("pipeline.started", "pipeline_id=test123")
        mock_spectator_stream.emit.assert_called_once_with(
            event_type="pipeline.started",
            details="pipeline_id=test123",
        )

    def test_spectate_survives_import_error(self):
        """Verify _spectate silently handles ImportError."""
        with patch(
            "aragora.pipeline.idea_to_execution._get_spectator_stream",
            side_effect=ImportError("no spectate"),
        ):
            # Should not raise
            _spectate("pipeline.started", "test")

    def test_spectate_survives_type_error(self):
        """Verify _spectate silently handles TypeError."""
        with patch(
            "aragora.pipeline.idea_to_execution._get_spectator_stream",
            side_effect=TypeError("bad args"),
        ):
            # Should not raise
            _spectate("pipeline.started", "test")


# =========================================================================
# from_ideas spectator integration
# =========================================================================


class TestFromIdeasSpectator:
    """Test spectator emissions during from_ideas() pipeline."""

    def test_from_ideas_emits_pipeline_started(self, pipeline, sample_ideas, mock_spectator):
        pipeline.from_ideas(sample_ideas, auto_advance=True)
        started_calls = [
            c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.started"
        ]
        assert len(started_calls) == 1
        assert "source=ideas" in started_calls[0].args[1]

    def test_from_ideas_emits_pipeline_completed(self, pipeline, sample_ideas, mock_spectator):
        pipeline.from_ideas(sample_ideas, auto_advance=True)
        completed_calls = [
            c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.completed"
        ]
        assert len(completed_calls) == 1

    def test_from_ideas_emits_ideation_stage(self, pipeline, sample_ideas, mock_spectator):
        pipeline.from_ideas(sample_ideas, auto_advance=True)
        ideation_started = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_started" and "ideation" in c.args[1]
        ]
        ideation_completed = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_completed" and "ideation" in c.args[1]
        ]
        assert len(ideation_started) == 1
        assert len(ideation_completed) == 1

    def test_from_ideas_emits_goals_stage(self, pipeline, sample_ideas, mock_spectator):
        pipeline.from_ideas(sample_ideas, auto_advance=True)
        goals_completed = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_completed" and "goals" in c.args[1]
        ]
        assert len(goals_completed) == 1

    def test_from_ideas_emits_actions_stage(self, pipeline, sample_ideas, mock_spectator):
        pipeline.from_ideas(sample_ideas, auto_advance=True)
        actions_started = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_started" and "actions" in c.args[1]
        ]
        actions_completed = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_completed" and "actions" in c.args[1]
        ]
        assert len(actions_started) == 1
        assert len(actions_completed) == 1

    def test_from_ideas_emits_orchestration_stage(self, pipeline, sample_ideas, mock_spectator):
        pipeline.from_ideas(sample_ideas, auto_advance=True)
        orch_started = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_started" and "orchestration" in c.args[1]
        ]
        orch_completed = [
            c
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_completed" and "orchestration" in c.args[1]
        ]
        assert len(orch_started) == 1
        assert len(orch_completed) == 1

    def test_from_ideas_all_stages_emitted_in_order(self, pipeline, sample_ideas, mock_spectator):
        """Verify the full sequence of spectator events."""
        pipeline.from_ideas(sample_ideas, auto_advance=True)

        event_types = [c.args[0] for c in mock_spectator.call_args_list]

        # Must start with pipeline.started
        assert event_types[0] == "pipeline.started"
        # Must end with pipeline.completed
        assert event_types[-1] == "pipeline.completed"

        # Verify stage ordering: ideation before goals before actions before orchestration
        stage_events = [
            (c.args[0], c.args[1])
            for c in mock_spectator.call_args_list
            if c.args[0] in ("pipeline.stage_started", "pipeline.stage_completed")
        ]
        stage_names = [details.split("=")[1] for _, details in stage_events]
        expected_order = [
            "ideation",
            "ideation",  # started, completed
            "goals",  # completed (goals extracted inline)
            "actions",
            "actions",  # started, completed
            "orchestration",
            "orchestration",  # started, completed
        ]
        assert stage_names == expected_order


# =========================================================================
# from_debate spectator integration
# =========================================================================


class TestFromDebateSpectator:
    """Test spectator emissions during from_debate() pipeline."""

    def test_from_debate_emits_pipeline_started(
        self,
        pipeline,
        sample_cartographer_data,
        mock_spectator,
    ):
        pipeline.from_debate(sample_cartographer_data, auto_advance=True)
        started_calls = [
            c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.started"
        ]
        assert len(started_calls) == 1
        assert "source=debate" in started_calls[0].args[1]

    def test_from_debate_emits_pipeline_completed(
        self,
        pipeline,
        sample_cartographer_data,
        mock_spectator,
    ):
        pipeline.from_debate(sample_cartographer_data, auto_advance=True)
        completed_calls = [
            c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.completed"
        ]
        assert len(completed_calls) == 1

    def test_from_debate_emits_all_four_stages(
        self,
        pipeline,
        sample_cartographer_data,
        mock_spectator,
    ):
        pipeline.from_debate(sample_cartographer_data, auto_advance=True)
        stage_starts = [
            c.args[1]
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_started"
        ]
        stage_completes = [
            c.args[1]
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_completed"
        ]
        assert "stage=ideation" in stage_starts
        assert "stage=goals" in stage_starts
        assert "stage=actions" in stage_starts
        assert "stage=orchestration" in stage_starts
        assert "stage=ideation" in stage_completes
        assert "stage=goals" in stage_completes
        assert "stage=actions" in stage_completes
        assert "stage=orchestration" in stage_completes


# =========================================================================
# Async run() spectator integration
# =========================================================================


class TestAsyncRunSpectator:
    """Test spectator emissions during async run() pipeline."""

    @pytest.mark.asyncio
    async def test_async_run_emits_pipeline_started(self, pipeline, mock_spectator):
        cfg = PipelineConfig(dry_run=True)
        await pipeline.run("Build a rate limiter", config=cfg)
        started_calls = [
            c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.started"
        ]
        assert len(started_calls) == 1
        assert "source=async" in started_calls[0].args[1]

    @pytest.mark.asyncio
    async def test_async_run_emits_pipeline_completed(self, pipeline, mock_spectator):
        cfg = PipelineConfig(dry_run=True)
        await pipeline.run("Build a rate limiter", config=cfg)
        completed_calls = [
            c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.completed"
        ]
        assert len(completed_calls) == 1

    @pytest.mark.asyncio
    async def test_async_run_emits_stage_events(self, pipeline, mock_spectator):
        cfg = PipelineConfig(dry_run=True)
        await pipeline.run("Build a rate limiter", config=cfg)
        stage_starts = [
            c.args[1]
            for c in mock_spectator.call_args_list
            if c.args[0] == "pipeline.stage_started"
        ]
        # At minimum, ideation and goals stages should have started
        assert any("ideation" in s for s in stage_starts)
        assert any("goals" in s for s in stage_starts)

    @pytest.mark.asyncio
    async def test_async_run_failure_emits_failed(self, pipeline, mock_spectator):
        """When the pipeline raises, pipeline.failed should be emitted."""
        cfg = PipelineConfig(
            stages_to_run=["ideation"],
            dry_run=False,
        )
        # Patch _run_ideation to raise
        with patch.object(
            pipeline,
            "_run_ideation",
            side_effect=RuntimeError("boom"),
        ):
            await pipeline.run("Build a rate limiter", config=cfg)

        failed_calls = [c for c in mock_spectator.call_args_list if c.args[0] == "pipeline.failed"]
        assert len(failed_calls) == 1
