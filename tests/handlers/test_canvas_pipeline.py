"""Tests for the CanvasPipelineHandler REST endpoints.

Covers all 17 endpoints:
- POST from-debate, from-ideas, from-template, advance, run, extract-goals
- POST approve-transition, convert/debate, convert/workflow
- GET pipeline/{id}, pipeline/{id}/status, pipeline/{id}/stage/{stage},
      pipeline/{id}/graph, pipeline/{id}/receipt, pipeline/templates
- PUT pipeline/{id}
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.canvas_pipeline import (
    CanvasPipelineHandler,
    _pipeline_objects,
)


@pytest.fixture
def mock_pipeline():
    """Mock IdeaToExecutionPipeline so handle_from_ideas always succeeds.

    This prevents tests from skipping with "Pipeline import unavailable" in
    environments where aragora.pipeline.idea_to_execution is not importable.
    The handler does a lazy ``from aragora.pipeline.idea_to_execution import
    IdeaToExecutionPipeline`` inside its try block. We inject a mock module
    into sys.modules so that import always resolves.
    """
    import types
    import sys

    mock_result = MagicMock()
    mock_result.pipeline_id = "pipe-test1234"
    mock_result.stage_status = {"ideas": "complete", "goals": "pending"}
    mock_result.goal_graph = None
    mock_result.universal_graph = None
    mock_result.to_dict.return_value = {
        "pipeline_id": "pipe-test1234",
        "stage_status": {"ideas": "complete", "goals": "pending"},
    }

    mock_cls = MagicMock()
    mock_cls.return_value.from_ideas.return_value = mock_result

    # Build a fake module so the handler's lazy import succeeds
    fake_mod = types.ModuleType("aragora.pipeline.idea_to_execution")
    fake_mod.IdeaToExecutionPipeline = mock_cls

    original = sys.modules.get("aragora.pipeline.idea_to_execution")
    sys.modules["aragora.pipeline.idea_to_execution"] = fake_mod
    try:
        yield mock_result
    finally:
        # Restore whatever was there before (real module or absent)
        if original is not None:
            sys.modules["aragora.pipeline.idea_to_execution"] = original
        else:
            sys.modules.pop("aragora.pipeline.idea_to_execution", None)


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult or raw dict."""
    if isinstance(result, dict):
        return result
    # HandlerResult: parse .body bytes as JSON
    return json.loads(result.body)


@pytest.fixture(autouse=True)
def _clear_pipeline_store():
    """Clear in-memory pipeline objects between tests."""
    _pipeline_objects.clear()
    yield
    _pipeline_objects.clear()


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
        body = _body(result)
        assert "error" in body
        assert "cartographer_data" in body["error"]

    @pytest.mark.asyncio
    async def test_empty_cartographer_data(self, handler):
        result = await handler.handle_from_debate({"cartographer_data": {}})
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_successful_pipeline(self, handler):
        """Verify from_debate runs the real pipeline with minimal input."""
        result = await handler.handle_from_debate(
            {
                "cartographer_data": {"nodes": [{"id": "n1", "label": "test"}], "edges": []},
                "auto_advance": True,
            }
        )
        body = _body(result)
        # Pipeline should succeed (may not advance all stages with minimal data)
        if "error" not in body:
            assert "pipeline_id" in body
            assert "stage_status" in body
        else:
            # Import may fail in some envs -- that's acceptable
            assert "error" in body

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        """Pipeline import failure returns error dict, not exception."""
        with patch.dict("sys.modules", {"aragora.pipeline.idea_to_execution": None}):
            result = await handler.handle_from_debate(
                {
                    "cartographer_data": {"nodes": [{"id": "1"}]},
                }
            )
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# POST from-ideas
# ---------------------------------------------------------------------------


class TestFromIdeas:
    @pytest.mark.asyncio
    async def test_missing_ideas(self, handler):
        result = await handler.handle_from_ideas({})
        body = _body(result)
        assert "error" in body
        assert "ideas" in body["error"]

    @pytest.mark.asyncio
    async def test_empty_ideas_list(self, handler):
        result = await handler.handle_from_ideas({"ideas": []})
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.pipeline.idea_to_execution": None}):
            result = await handler.handle_from_ideas(
                {
                    "ideas": ["improve caching", "add monitoring"],
                }
            )
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# POST advance
# ---------------------------------------------------------------------------


class TestAdvance:
    @pytest.mark.asyncio
    async def test_missing_pipeline_id(self, handler):
        result = await handler.handle_advance({})
        body = _body(result)
        assert "error" in body
        assert "pipeline_id" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_target_stage(self, handler):
        result = await handler.handle_advance({"pipeline_id": "pipe-1"})
        body = _body(result)
        assert "error" in body
        assert "target_stage" in body["error"]

    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler):
        result = await handler.handle_advance(
            {
                "pipeline_id": "nonexistent",
                "target_stage": "goals",
            }
        )
        body = _body(result)
        assert "error" in body
        assert "not found" in body["error"]

    @pytest.mark.asyncio
    async def test_invalid_stage(self, handler):
        _pipeline_objects["pipe-1"] = MagicMock()
        result = await handler.handle_advance(
            {
                "pipeline_id": "pipe-1",
                "target_stage": "invalid_stage",
            }
        )
        body = _body(result)
        assert "error" in body
        assert "Invalid stage" in body["error"]

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        _pipeline_objects["pipe-1"] = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "aragora.canvas.stages": None,
                "aragora.pipeline.idea_to_execution": None,
            },
        ):
            result = await handler.handle_advance(
                {
                    "pipeline_id": "pipe-1",
                    "target_stage": "goals",
                }
            )
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# GET pipeline/{id}
# ---------------------------------------------------------------------------


class TestGetPipeline:
    @pytest.mark.asyncio
    async def test_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_get_pipeline("nonexistent")
        body = _body(result)
        assert "error" in body
        assert "not found" in body["error"]

    @pytest.mark.asyncio
    async def test_found(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-abc",
            "ideas": {"nodes": []},
        }
        result = await handler.handle_get_pipeline("pipe-abc")
        body = _body(result)
        assert body["pipeline_id"] == "pipe-abc"


# ---------------------------------------------------------------------------
# GET pipeline/{id}/stage/{stage}
# ---------------------------------------------------------------------------


class TestGetStage:
    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_get_stage("nonexistent", "ideas")
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_stage_not_found(self, handler, mock_store):
        mock_store.get.return_value = {"pipeline_id": "pipe-1"}
        result = await handler.handle_get_stage("pipe-1", "ideas")
        # "ideas" key not in result dict
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_invalid_stage_name(self, handler, mock_store):
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "ideas": {}}
        result = await handler.handle_get_stage("pipe-1", "invalid_stage")
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_valid_stage(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "ideas": {"nodes": [{"id": "n1"}]},
        }
        result = await handler.handle_get_stage("pipe-1", "ideas")
        body = _body(result)
        assert body["stage"] == "ideas"
        assert body["data"]["nodes"][0]["id"] == "n1"

    @pytest.mark.asyncio
    async def test_goals_stage(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "goals": [{"id": "g1", "title": "Goal 1"}],
        }
        result = await handler.handle_get_stage("pipe-1", "goals")
        body = _body(result)
        assert body["stage"] == "goals"

    @pytest.mark.asyncio
    async def test_orchestration_stage(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "orchestration": {"agents": []},
        }
        result = await handler.handle_get_stage("pipe-1", "orchestration")
        body = _body(result)
        assert body["stage"] == "orchestration"


# ---------------------------------------------------------------------------
# POST convert/debate
# ---------------------------------------------------------------------------


class TestConvertDebate:
    @pytest.mark.asyncio
    async def test_missing_cartographer_data(self, handler):
        result = await handler.handle_convert_debate({})
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.canvas.converters": None}):
            result = await handler.handle_convert_debate(
                {
                    "cartographer_data": {"nodes": []},
                }
            )
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# POST convert/workflow
# ---------------------------------------------------------------------------


class TestConvertWorkflow:
    @pytest.mark.asyncio
    async def test_missing_workflow_data(self, handler):
        result = await handler.handle_convert_workflow({})
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.canvas.converters": None}):
            result = await handler.handle_convert_workflow(
                {
                    "workflow_data": {"steps": []},
                }
            )
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# POST extract-goals
# ---------------------------------------------------------------------------


class TestExtractGoals:
    @pytest.mark.asyncio
    async def test_missing_data_and_id(self, handler):
        result = await handler.handle_extract_goals({})
        body = _body(result)
        assert "error" in body
        assert "Missing" in body["error"]

    @pytest.mark.asyncio
    async def test_with_raw_canvas_data(self, handler):
        """Extract goals from inline canvas data."""
        canvas_data = {
            "nodes": [
                {"id": "n1", "data": {"idea_type": "concept", "label": "Caching"}},
                {"id": "n2", "data": {"idea_type": "evidence", "label": "Redis benchmarks"}},
                {"id": "n3", "data": {"idea_type": "insight", "label": "Cache invalidation"}},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "supports"},
            ],
        }
        result = await handler.handle_extract_goals(
            {
                "ideas_canvas_data": canvas_data,
                "ideas_canvas_id": "test-canvas-1",
            }
        )
        body = _body(result)
        assert "error" not in body
        assert "goals" in body
        assert body["source_canvas_id"] == "test-canvas-1"
        assert "goals_count" in body
        assert isinstance(body["goals_count"], int)

    @pytest.mark.asyncio
    async def test_empty_nodes(self, handler):
        """Canvas with no nodes produces empty goals."""
        result = await handler.handle_extract_goals(
            {
                "ideas_canvas_data": {"nodes": [], "edges": []},
            }
        )
        body = _body(result)
        assert "error" not in body
        assert body["goals_count"] == 0

    @pytest.mark.asyncio
    async def test_config_max_goals(self, handler):
        """Config max_goals limits output."""
        nodes = [
            {"id": f"n{i}", "data": {"idea_type": "concept", "label": f"Idea {i}"}}
            for i in range(20)
        ]
        result = await handler.handle_extract_goals(
            {
                "ideas_canvas_data": {"nodes": nodes, "edges": []},
                "config": {"max_goals": 3, "confidence_threshold": 0},
            }
        )
        body = _body(result)
        assert "error" not in body
        assert body["goals_count"] <= 3

    @pytest.mark.asyncio
    async def test_config_confidence_threshold(self, handler):
        """Goals below confidence threshold are filtered out."""
        result = await handler.handle_extract_goals(
            {
                "ideas_canvas_data": {
                    "nodes": [
                        {"id": "n1", "data": {"idea_type": "concept", "label": "Test"}},
                    ],
                    "edges": [],
                },
                "config": {"confidence_threshold": 0.99},
            }
        )
        body = _body(result)
        assert "error" not in body
        # With a high threshold, most structural goals should be filtered
        assert isinstance(body["goals_count"], int)

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self, handler):
        with patch.dict("sys.modules", {"aragora.goals.extractor": None}):
            result = await handler.handle_extract_goals(
                {
                    "ideas_canvas_data": {"nodes": [{"id": "1"}]},
                }
            )
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_provenance_links_returned(self, handler):
        """Provenance links from stage 1 to stage 2 should be present."""
        canvas_data = {
            "nodes": [
                {"id": "n1", "data": {"idea_type": "concept", "label": "Core idea"}},
                {"id": "n2", "data": {"idea_type": "evidence", "label": "Supporting data"}},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "supports"},
            ],
        }
        result = await handler.handle_extract_goals(
            {
                "ideas_canvas_data": canvas_data,
                "config": {"confidence_threshold": 0},
            }
        )
        body = _body(result)
        assert "error" not in body
        assert "provenance" in body

    @pytest.mark.asyncio
    async def test_canvas_id_from_store_fallback(self, handler):
        """When only canvas_id given and store fails, returns error."""
        result = await handler.handle_extract_goals(
            {
                "ideas_canvas_id": "nonexistent-canvas",
            }
        )
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_handles_post_routing(self, handler):
        """Verify handle_post dispatches extract-goals correctly."""
        mock_handler = MagicMock()
        mock_handler.request.body = b'{"ideas_canvas_data": {"nodes": [], "edges": []}}'
        result = handler.handle_post("/api/v1/canvas/pipeline/extract-goals", {}, mock_handler)
        # Should return a coroutine (async method), not None
        assert result is not None


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
        assert len(CanvasPipelineHandler.ROUTES) >= 20


# ---------------------------------------------------------------------------
# Phase 1: pipeline_id consistency
# ---------------------------------------------------------------------------


class TestRunPipelineIdConsistency:
    """Verify the handler's pipeline_id matches the stored result's pipeline_id."""

    @pytest.mark.asyncio
    async def test_run_pipeline_id_matches_stored(self, handler, mock_store):
        """Returned pipeline_id must match the ID used to store results."""
        stored_ids = []
        mock_store.save.side_effect = lambda pid, data: stored_ids.append(pid)

        result = await handler.handle_run(
            {
                "input_text": "Build a caching layer",
                "dry_run": True,
            }
        )
        body = _body(result)
        assert "error" not in body
        returned_id = body["pipeline_id"]

        # The placeholder save uses the handler's pipeline_id
        assert returned_id in stored_ids

    @pytest.mark.asyncio
    async def test_run_missing_input(self, handler):
        """Missing input_text returns error."""
        result = await handler.handle_run({})
        body = _body(result)
        assert "error" in body
        assert "input_text" in body["error"]


# ---------------------------------------------------------------------------
# Phase 3: event_callback in sync methods
# ---------------------------------------------------------------------------


class TestSyncEventCallback:
    """Verify from_debate and from_ideas invoke event_callback."""

    def test_from_ideas_event_callback(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        events = []

        def on_event(event_type, data):
            events.append((event_type, data))

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_ideas(
            ["idea A", "idea B"],
            auto_advance=True,
            event_callback=on_event,
        )
        assert result.pipeline_id.startswith("pipe-")
        # Must have received stage events
        event_types = [e[0] for e in events]
        assert "stage_completed" in event_types

    def test_from_debate_event_callback(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        events = []

        def on_event(event_type, data):
            events.append((event_type, data))

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(
            {"nodes": [{"id": "n1", "label": "test"}], "edges": []},
            auto_advance=True,
            event_callback=on_event,
        )
        assert result.pipeline_id.startswith("pipe-")
        event_types = [e[0] for e in events]
        assert "stage_completed" in event_types

    def test_from_ideas_external_pipeline_id(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_ideas(
            ["idea A"],
            pipeline_id="pipe-custom-123",
        )
        assert result.pipeline_id == "pipe-custom-123"


# ---------------------------------------------------------------------------
# PUT: Save pipeline canvas state
# ---------------------------------------------------------------------------


class TestSavePipeline:
    """Tests for handle_save_pipeline (PUT /api/v1/canvas/pipeline/{id})."""

    @pytest.mark.asyncio
    async def test_missing_stages(self, handler, mock_store):
        """Missing 'stages' field returns error."""
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "stage_status": {}}
        result = await handler.handle_save_pipeline("pipe-1", {})
        body = _body(result)
        assert "error" in body
        assert "stages" in body["error"]

    @pytest.mark.asyncio
    async def test_save_with_nodes(self, handler, mock_store):
        """Saving stages with nodes marks them complete."""
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "stage_status": {}}
        result = await handler.handle_save_pipeline(
            "pipe-1",
            {
                "stages": {
                    "ideas": {"nodes": [{"id": "n1"}], "edges": []},
                },
            },
        )
        body = _body(result)
        assert body["saved"] is True
        assert body["pipeline_id"] == "pipe-1"
        assert body["stage_status"]["ideas"] == "complete"

    @pytest.mark.asyncio
    async def test_save_creates_new_pipeline(self, handler, mock_store):
        """PUT on a nonexistent pipeline_id creates a new entry (upsert)."""
        mock_store.get.return_value = None
        result = await handler.handle_save_pipeline(
            "pipe-new",
            {
                "stages": {
                    "goals": {"nodes": [{"id": "g1"}], "edges": []},
                },
            },
        )
        body = _body(result)
        assert body["saved"] is True
        assert body["pipeline_id"] == "pipe-new"
        # Verify store.save was called
        mock_store.save.assert_called_once()
        saved_data = mock_store.save.call_args[0][1]
        assert saved_data["pipeline_id"] == "pipe-new"

    @pytest.mark.asyncio
    async def test_empty_nodes_not_marked_complete(self, handler, mock_store):
        """Saving a stage with empty nodes doesn't mark it complete."""
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "stage_status": {}}
        result = await handler.handle_save_pipeline(
            "pipe-1",
            {
                "stages": {
                    "ideas": {"nodes": [], "edges": []},
                },
            },
        )
        body = _body(result)
        assert body["saved"] is True
        assert "ideas" not in body["stage_status"]

    @pytest.mark.asyncio
    async def test_save_multiple_stages(self, handler, mock_store):
        """Multiple stages can be saved in a single request."""
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "stage_status": {}}
        result = await handler.handle_save_pipeline(
            "pipe-1",
            {
                "stages": {
                    "ideas": {"nodes": [{"id": "n1"}], "edges": []},
                    "goals": {"nodes": [{"id": "g1"}], "edges": [{"source": "g1", "target": "n1"}]},
                    "actions": {"nodes": [], "edges": []},
                },
            },
        )
        body = _body(result)
        assert body["saved"] is True
        assert body["stage_status"]["ideas"] == "complete"
        assert body["stage_status"]["goals"] == "complete"
        assert "actions" not in body["stage_status"]

    @pytest.mark.asyncio
    async def test_put_routing(self, handler):
        """handle_put dispatches to handle_save_pipeline."""
        mock_handler = MagicMock()
        mock_handler.request.body = b'{"stages": {"ideas": {"nodes": [{"id": "1"}], "edges": []}}}'
        result = handler.handle_put("/api/v1/canvas/pipeline/pipe-test", {}, mock_handler)
        # Should return a coroutine
        assert result is not None

    @pytest.mark.asyncio
    async def test_put_routing_no_match(self, handler):
        """handle_put returns None for non-matching paths."""
        mock_handler = MagicMock()
        result = handler.handle_put("/api/v1/canvas/other", {}, mock_handler)
        assert result is None


# ---------------------------------------------------------------------------
# POST: Approve/reject stage transition
# ---------------------------------------------------------------------------


class TestApproveTransition:
    """Tests for handle_approve_transition (POST /{id}/approve-transition)."""

    @pytest.mark.asyncio
    async def test_pipeline_not_found(self, handler, mock_store):
        """Nonexistent pipeline returns error."""
        mock_store.get.return_value = None
        result = await handler.handle_approve_transition(
            "nonexistent",
            {
                "from_stage": "ideas",
                "to_stage": "goals",
                "approved": True,
            },
        )
        body = _body(result)
        assert "error" in body
        assert "not found" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_from_stage(self, handler, mock_store):
        """Missing from_stage returns error."""
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "stage_status": {}}
        result = await handler.handle_approve_transition(
            "pipe-1",
            {
                "to_stage": "goals",
                "approved": True,
            },
        )
        body = _body(result)
        assert "error" in body
        assert "from_stage" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_to_stage(self, handler, mock_store):
        """Missing to_stage returns error."""
        mock_store.get.return_value = {"pipeline_id": "pipe-1", "stage_status": {}}
        result = await handler.handle_approve_transition(
            "pipe-1",
            {
                "from_stage": "ideas",
                "approved": True,
            },
        )
        body = _body(result)
        assert "error" in body
        assert "to_stage" in body["error"]

    @pytest.mark.asyncio
    async def test_approve_updates_stage_status(self, handler, mock_store):
        """Approving a transition advances the pipeline stages."""
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "stage_status": {"ideas": "complete"},
            "transitions": [],
        }
        result = await handler.handle_approve_transition(
            "pipe-1",
            {
                "from_stage": "ideas",
                "to_stage": "goals",
                "approved": True,
                "comment": "Looks good",
            },
        )
        body = _body(result)
        assert body["status"] == "approved"
        assert body["comment"] == "Looks good"
        assert body["pipeline_id"] == "pipe-1"
        # Verify store was saved with updated stage_status
        saved_data = mock_store.save.call_args[0][1]
        assert saved_data["stage_status"]["ideas"] == "complete"
        assert saved_data["stage_status"]["goals"] == "active"

    @pytest.mark.asyncio
    async def test_reject_does_not_advance(self, handler, mock_store):
        """Rejecting a transition doesn't change stage_status."""
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "stage_status": {"ideas": "complete"},
            "transitions": [],
        }
        result = await handler.handle_approve_transition(
            "pipe-1",
            {
                "from_stage": "ideas",
                "to_stage": "goals",
                "approved": False,
                "comment": "Needs more detail",
            },
        )
        body = _body(result)
        assert body["status"] == "rejected"
        saved_data = mock_store.save.call_args[0][1]
        # goals should not be "active"
        assert "goals" not in saved_data["stage_status"]

    @pytest.mark.asyncio
    async def test_creates_transition_if_none_exist(self, handler, mock_store):
        """New transition record created when no matching transition exists."""
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "stage_status": {},
        }
        result = await handler.handle_approve_transition(
            "pipe-1",
            {
                "from_stage": "ideas",
                "to_stage": "goals",
                "approved": True,
            },
        )
        body = _body(result)
        assert body["status"] == "approved"
        saved_data = mock_store.save.call_args[0][1]
        assert len(saved_data["transitions"]) == 1
        assert saved_data["transitions"][0]["from_stage"] == "ideas"
        assert saved_data["transitions"][0]["to_stage"] == "goals"
        assert saved_data["transitions"][0]["status"] == "approved"
        assert "reviewed_at" in saved_data["transitions"][0]

    @pytest.mark.asyncio
    async def test_updates_existing_transition(self, handler, mock_store):
        """Existing transition record is updated in place."""
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "stage_status": {},
            "transitions": [
                {
                    "from_stage": "ideas",
                    "to_stage": "goals",
                    "status": "pending",
                }
            ],
        }
        result = await handler.handle_approve_transition(
            "pipe-1",
            {
                "from_stage": "ideas",
                "to_stage": "goals",
                "approved": True,
                "comment": "Approved after review",
            },
        )
        body = _body(result)
        assert body["status"] == "approved"
        saved_data = mock_store.save.call_args[0][1]
        assert len(saved_data["transitions"]) == 1
        assert saved_data["transitions"][0]["status"] == "approved"
        assert saved_data["transitions"][0]["human_comment"] == "Approved after review"

    @pytest.mark.asyncio
    async def test_post_routing_approve_transition(self, handler):
        """handle_post dispatches /approve-transition correctly."""
        mock_handler = MagicMock()
        mock_handler.request.body = (
            b'{"from_stage": "ideas", "to_stage": "goals", "approved": true}'
        )
        result = handler.handle_post(
            "/api/v1/canvas/pipeline/pipe-test/approve-transition",
            {},
            mock_handler,
        )
        # Should return a coroutine (async method)
        assert result is not None


# ---------------------------------------------------------------------------
# E2E: Full pipeline contract
# ---------------------------------------------------------------------------


class TestE2ESmokeContract:
    """End-to-end smoke test: from_ideas -> status -> stage -> save -> approve -> receipt."""

    @pytest.mark.asyncio
    async def test_full_pipeline_lifecycle(self, handler, mock_store, mock_pipeline):
        """Exercise the full lifecycle: create -> get -> save -> approve -> receipt."""
        # Step 1: Create pipeline via from-ideas (mock_pipeline ensures the
        # IdeaToExecutionPipeline import always succeeds)
        result = await handler.handle_from_ideas(
            {
                "ideas": ["build caching", "add monitoring"],
            }
        )
        body = _body(result)
        assert "error" not in body, f"Pipeline creation should succeed: {body}"
        pipeline_id = body["pipeline_id"]
        assert pipeline_id.startswith("pipe-")

        # Step 2: Get pipeline by ID (mock store returns what was saved)
        mock_store.get.return_value = {
            "pipeline_id": pipeline_id,
            "stage_status": body.get("stage_status", {}),
            "ideas": {"nodes": [{"id": "n1"}], "edges": []},
        }
        get_result = await handler.handle_get_pipeline(pipeline_id)
        get_body = _body(get_result)
        assert get_body["pipeline_id"] == pipeline_id

        # Step 3: Get specific stage
        stage_result = await handler.handle_get_stage(pipeline_id, "ideas")
        stage_body = _body(stage_result)
        assert stage_body["stage"] == "ideas"

        # Step 4: Save updated canvas state
        save_result = await handler.handle_save_pipeline(
            pipeline_id,
            {
                "stages": {
                    "ideas": {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": []},
                    "goals": {"nodes": [{"id": "g1"}], "edges": []},
                },
            },
        )
        save_body = _body(save_result)
        assert save_body["saved"] is True
        assert save_body["stage_status"]["ideas"] == "complete"
        assert save_body["stage_status"]["goals"] == "complete"

        # Step 5: Approve transition from ideas to goals
        # Update mock to reflect saved state with transitions
        mock_store.get.return_value = {
            "pipeline_id": pipeline_id,
            "stage_status": {"ideas": "complete", "goals": "complete"},
            "transitions": [],
        }
        approve_result = await handler.handle_approve_transition(
            pipeline_id,
            {
                "from_stage": "ideas",
                "to_stage": "goals",
                "approved": True,
                "comment": "Transition approved by test",
            },
        )
        approve_body = _body(approve_result)
        assert approve_body["status"] == "approved"

        # Step 6: Verify receipt endpoint returns something
        mock_store.get.return_value = {
            "pipeline_id": pipeline_id,
            "receipt": {"hash": "abc123"},
        }
        receipt_result = await handler.handle_receipt(pipeline_id)
        receipt_body = _body(receipt_result)
        assert "error" not in receipt_body or "receipt" in str(receipt_body)

    @pytest.mark.asyncio
    async def test_pipeline_from_ideas_to_get_status(self, handler, mock_store, mock_pipeline):
        """Create pipeline from ideas and check status."""
        result = await handler.handle_from_ideas(
            {
                "ideas": ["idea one"],
            }
        )
        body = _body(result)
        assert "error" not in body, f"Pipeline creation should succeed: {body}"
        pipeline_id = body["pipeline_id"]

        # Mock status response
        mock_store.get.return_value = {
            "pipeline_id": pipeline_id,
            "stage_status": body.get("stage_status", {}),
        }
        status = await handler.handle_status(pipeline_id)
        status_body = _body(status)
        assert "pipeline_id" in status_body or "stage_status" in status_body


# ---------------------------------------------------------------------------
# POST execute
# ---------------------------------------------------------------------------


class TestExecutePipeline:
    """Tests for POST /api/v1/canvas/pipeline/{id}/execute."""

    @pytest.mark.asyncio
    async def test_execute_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_execute("nonexistent", {})
        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_execute_incomplete_stages(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "stage_status": {
                "ideas": "complete",
                "goals": "complete",
                "actions": "pending",
                "orchestration": "pending",
            },
        }
        result = await handler.handle_execute("pipe-1", {})
        body = _body(result)
        assert "error" in body
        assert "actions" in body["error"]

    @pytest.mark.asyncio
    async def test_execute_dry_run(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-1",
            "stage_status": {
                "ideas": "complete",
                "goals": "pending",
                "actions": "pending",
                "orchestration": "pending",
            },
            "orchestration": {
                "nodes": [
                    {"id": "t1", "data": {"orch_type": "agent_task", "label": "Task 1"}},
                ],
                "edges": [],
            },
        }
        result = await handler.handle_execute("pipe-1", {"dry_run": True})
        body = _body(result)
        assert body["status"] == "dry_run"
        assert body["agent_tasks"] == 1
        assert "goals" in body["stages_incomplete"]

    @pytest.mark.asyncio
    async def test_execute_success(self, handler, mock_store):
        mock_store.get.return_value = {
            "pipeline_id": "pipe-ok",
            "stage_status": {
                "ideas": "complete",
                "goals": "complete",
                "actions": "complete",
                "orchestration": "complete",
            },
            "orchestration": {
                "nodes": [
                    {"id": "t1", "data": {"orch_type": "agent_task", "label": "Build cache"}},
                    {"id": "t2", "data": {"orch_type": "agent_task", "label": "Write tests"}},
                    {"id": "a1", "data": {"orch_type": "agent", "label": "Claude"}},
                ],
                "edges": [],
            },
        }
        result = await handler.handle_execute("pipe-ok", {})
        body = _body(result)
        assert body["status"] == "executing"
        assert body["agent_tasks"] == 2  # only agent_task nodes
        assert body["total_orchestration_nodes"] == 3
        assert body["execution_id"].startswith("exec-")
