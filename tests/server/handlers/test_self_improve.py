"""Tests for the self-improvement REST API handler.

Tests cover:
- Route matching (can_handle)
- Run listing (GET /api/self-improve/runs)
- Run details (GET /api/self-improve/runs/:id)
- Run creation (POST /api/self-improve/start)
- Run cancellation (POST /api/self-improve/runs/:id/cancel)
- History alias (GET /api/self-improve/history)
- Dry run mode
- Store unavailability
- Path extraction
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.stores.run_store import RunStatus, SelfImproveRun, SelfImproveRunStore
from aragora.server.handlers.self_improve import SelfImproveHandler, _extract_run_id

from .conftest import parse_handler_response


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def handler(mock_server_context):
    """Create a SelfImproveHandler instance."""
    return SelfImproveHandler(server_context=mock_server_context)


@pytest.fixture
def mock_store():
    """Create a mock SelfImproveRunStore."""
    store = MagicMock(spec=SelfImproveRunStore)
    return store


@pytest.fixture
def sample_run():
    """Create a sample SelfImproveRun."""
    return SelfImproveRun(
        run_id="abc12345",
        goal="Improve test coverage",
        status=RunStatus.RUNNING,
        tracks=["qa", "developer"],
        mode="flat",
        max_cycles=5,
        created_at="2026-02-15T10:00:00+00:00",
        started_at="2026-02-15T10:00:01+00:00",
    )


@pytest.fixture
def completed_run():
    """Create a completed SelfImproveRun."""
    return SelfImproveRun(
        run_id="def67890",
        goal="Fix bugs",
        status=RunStatus.COMPLETED,
        tracks=["developer"],
        mode="flat",
        max_cycles=3,
        created_at="2026-02-15T09:00:00+00:00",
        started_at="2026-02-15T09:00:01+00:00",
        completed_at="2026-02-15T09:10:00+00:00",
        total_subtasks=5,
        completed_subtasks=5,
        failed_subtasks=0,
        summary="All subtasks completed successfully",
    )


@pytest.fixture
def handler_with_store(handler, mock_store):
    """Handler with pre-injected mock store."""
    handler._store = mock_store
    return handler


@pytest.fixture
def mock_http_post_body(mock_http_handler):
    """Create a POST handler with JSON body."""

    def _create(body: dict[str, Any]):
        return mock_http_handler(method="POST", body=body)

    return _create


@pytest.fixture(autouse=True)
def _clear_active_tasks():
    """Clear active tasks between tests."""
    from aragora.server.handlers.self_improve import _active_tasks

    _active_tasks.clear()
    yield
    _active_tasks.clear()


# ============================================================================
# Path Extraction Tests
# ============================================================================


class TestExtractRunId:
    """Tests for _extract_run_id helper."""

    def test_extracts_from_runs_path(self):
        assert _extract_run_id("/api/self-improve/runs/abc123") == "abc123"

    def test_extracts_from_cancel_path(self):
        assert _extract_run_id("/api/self-improve/runs/abc123/cancel") == "abc123"

    def test_returns_none_for_runs_list(self):
        assert _extract_run_id("/api/self-improve/runs") is None

    def test_returns_none_for_start(self):
        assert _extract_run_id("/api/self-improve/start") is None

    def test_returns_none_for_short_path(self):
        assert _extract_run_id("/api/self-improve") is None

    def test_returns_none_for_wrong_segment(self):
        assert _extract_run_id("/api/self-improve/history") is None

    def test_handles_leading_slash(self):
        assert _extract_run_id("/api/self-improve/runs/xyz") == "xyz"

    def test_handles_uuid_style_id(self):
        assert _extract_run_id("/api/self-improve/runs/a1b2c3d4") == "a1b2c3d4"


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_handles_start(self, handler):
        assert handler.can_handle("/api/self-improve/start") is True

    def test_handles_runs(self, handler):
        assert handler.can_handle("/api/self-improve/runs") is True

    def test_handles_history(self, handler):
        assert handler.can_handle("/api/self-improve/history") is True

    def test_handles_run_by_id(self, handler):
        assert handler.can_handle("/api/self-improve/runs/abc123") is True

    def test_handles_cancel(self, handler):
        assert handler.can_handle("/api/self-improve/runs/abc123/cancel") is True

    def test_rejects_unrelated(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/self-improvement") is False

    def test_handles_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/self-improve/runs") is True

    def test_handles_versioned_start(self, handler):
        assert handler.can_handle("/api/v1/self-improve/start") is True


# ============================================================================
# GET /api/self-improve/runs Tests
# ============================================================================


class TestListRuns:
    """Tests for listing self-improvement runs."""

    @pytest.mark.asyncio
    async def test_list_runs_success(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        mock_store.list_runs.return_value = [sample_run]
        result = await handler_with_store.handle(
            "/api/self-improve/runs", {}, mock_http_handler()
        )
        body = parse_handler_response(result)
        assert result.status_code == 200
        assert len(body["runs"]) == 1
        assert body["runs"][0]["run_id"] == "abc12345"
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.list_runs.return_value = []
        result = await handler_with_store.handle(
            "/api/self-improve/runs", {}, mock_http_handler()
        )
        body = parse_handler_response(result)
        assert body["runs"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_runs_with_pagination(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.list_runs.return_value = []
        await handler_with_store.handle(
            "/api/self-improve/runs", {"limit": "10", "offset": "5"}, mock_http_handler()
        )
        mock_store.list_runs.assert_called_once_with(limit=10, offset=5, status=None)

    @pytest.mark.asyncio
    async def test_list_runs_with_status_filter(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.list_runs.return_value = []
        await handler_with_store.handle(
            "/api/self-improve/runs", {"status": "running"}, mock_http_handler()
        )
        mock_store.list_runs.assert_called_once_with(limit=50, offset=0, status="running")

    @pytest.mark.asyncio
    async def test_list_runs_store_unavailable(self, handler, mock_http_handler):
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle(
                "/api/self-improve/runs", {}, mock_http_handler()
            )
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_history_alias(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.list_runs.return_value = []
        result = await handler_with_store.handle(
            "/api/self-improve/history", {}, mock_http_handler()
        )
        body = parse_handler_response(result)
        assert body["runs"] == []
        mock_store.list_runs.assert_called_once()


# ============================================================================
# GET /api/self-improve/runs/:id Tests
# ============================================================================


class TestGetRun:
    """Tests for getting a specific run."""

    @pytest.mark.asyncio
    async def test_get_run_success(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        mock_store.get_run.return_value = sample_run
        result = await handler_with_store.handle(
            "/api/self-improve/runs/abc12345", {}, mock_http_handler()
        )
        body = parse_handler_response(result)
        assert result.status_code == 200
        assert body["run_id"] == "abc12345"
        assert body["goal"] == "Improve test coverage"
        assert body["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.get_run.return_value = None
        result = await handler_with_store.handle(
            "/api/self-improve/runs/nonexistent", {}, mock_http_handler()
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_run_store_unavailable(self, handler, mock_http_handler):
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle(
                "/api/self-improve/runs/abc123", {}, mock_http_handler()
            )
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_completed_run(self, handler_with_store, mock_store, completed_run, mock_http_handler):
        mock_store.get_run.return_value = completed_run
        result = await handler_with_store.handle(
            "/api/self-improve/runs/def67890", {}, mock_http_handler()
        )
        body = parse_handler_response(result)
        assert body["status"] == "completed"
        assert body["total_subtasks"] == 5
        assert body["completed_subtasks"] == 5
        assert body["summary"] == "All subtasks completed successfully"


# ============================================================================
# POST /api/self-improve/start Tests
# ============================================================================


class TestStartRun:
    """Tests for starting a self-improvement run."""

    @pytest.mark.asyncio
    async def test_start_run_success(self, handler_with_store, mock_store, sample_run, mock_http_post_body):
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        http = mock_http_post_body({"goal": "Improve test coverage"})

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "Improve test coverage"}):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/start", {}, http
                )

        body = parse_handler_response(result)
        assert result.status_code == 202
        assert body["status"] == "started"
        assert body["run_id"] == "abc12345"

    @pytest.mark.asyncio
    async def test_start_run_missing_goal(self, handler_with_store, mock_http_post_body):
        http = mock_http_post_body({})

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/start", {}, http
            )

        assert result.status_code == 400
        body = parse_handler_response(result)
        assert "goal" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_start_run_empty_goal(self, handler_with_store, mock_http_post_body):
        http = mock_http_post_body({"goal": "   "})

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "   "}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/start", {}, http
            )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_start_run_invalid_mode(self, handler_with_store, mock_http_post_body):
        http = mock_http_post_body({"goal": "Test", "mode": "invalid"})

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "Test", "mode": "invalid"}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/start", {}, http
            )

        assert result.status_code == 400
        body = parse_handler_response(result)
        assert "mode" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_start_run_store_unavailable(self, handler, mock_http_post_body):
        http = mock_http_post_body({"goal": "Test"})

        with patch.object(handler, "_get_store", return_value=None):
            with patch.object(handler, "read_json_body", return_value={"goal": "Test"}):
                result = await handler.handle_post(
                    "/api/self-improve/start", {}, http
                )

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_start_run_with_tracks(self, handler_with_store, mock_store, sample_run, mock_http_post_body):
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        body_data = {"goal": "Improve", "tracks": ["qa", "developer"]}
        http = mock_http_post_body(body_data)

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/start", {}, http
                )

        assert result.status_code == 202
        mock_store.create_run.assert_called_once()
        call_kwargs = mock_store.create_run.call_args[1]
        assert call_kwargs["tracks"] == ["qa", "developer"]

    @pytest.mark.asyncio
    async def test_start_run_hierarchical_mode(self, handler_with_store, mock_store, sample_run, mock_http_post_body):
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        body_data = {"goal": "Improve", "mode": "hierarchical"}
        http = mock_http_post_body(body_data)

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/start", {}, http
                )

        assert result.status_code == 202
        call_kwargs = mock_store.create_run.call_args[1]
        assert call_kwargs["mode"] == "hierarchical"


# ============================================================================
# Dry Run Tests
# ============================================================================


class TestDryRun:
    """Tests for dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_plan(self, handler_with_store, mock_store, mock_http_post_body):
        dry_run = SelfImproveRun(
            run_id="dry12345",
            goal="Refactor module",
            dry_run=True,
        )
        mock_store.create_run.return_value = dry_run
        mock_store.update_run.return_value = dry_run

        body_data = {"goal": "Refactor module", "dry_run": True}
        http = mock_http_post_body(body_data)

        mock_plan = {"goal": "Refactor module", "subtasks": [], "tracks": []}

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_generate_plan", new_callable=AsyncMock, return_value=mock_plan):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/start", {}, http
                )

        body = parse_handler_response(result)
        assert result.status_code == 200
        assert body["status"] == "preview"
        assert body["plan"] == mock_plan
        assert body["run_id"] == "dry12345"

    @pytest.mark.asyncio
    async def test_dry_run_updates_store_completed(self, handler_with_store, mock_store, mock_http_post_body):
        dry_run = SelfImproveRun(
            run_id="dry12345",
            goal="Refactor module",
            dry_run=True,
        )
        mock_store.create_run.return_value = dry_run
        mock_store.update_run.return_value = dry_run

        body_data = {"goal": "Refactor module", "dry_run": True}
        http = mock_http_post_body(body_data)

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_generate_plan", new_callable=AsyncMock, return_value={}):
                await handler_with_store.handle_post(
                    "/api/self-improve/start", {}, http
                )

        mock_store.update_run.assert_called_once()
        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "completed"
        assert "plan" in call_kwargs


# ============================================================================
# POST /api/self-improve/runs/:id/cancel Tests
# ============================================================================


class TestCancelRun:
    """Tests for cancelling a run."""

    @pytest.mark.asyncio
    async def test_cancel_running_run(self, handler_with_store, mock_store, sample_run, mock_http_post_body):
        sample_run.status = RunStatus.CANCELLED
        mock_store.cancel_run.return_value = sample_run

        http = mock_http_post_body({})

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/runs/abc12345/cancel", {}, http
            )

        body = parse_handler_response(result)
        assert result.status_code == 200
        assert body["status"] == "cancelled"
        assert body["run_id"] == "abc12345"

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, handler_with_store, mock_store, mock_http_post_body):
        mock_store.cancel_run.return_value = None

        http = mock_http_post_body({})

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/runs/nonexistent/cancel", {}, http
            )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_stops_active_task(self, handler_with_store, mock_store, mock_http_post_body):
        from aragora.server.handlers.self_improve import _active_tasks

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        _active_tasks["abc12345"] = mock_task

        cancelled_run = SelfImproveRun(
            run_id="abc12345",
            goal="Test",
            status=RunStatus.CANCELLED,
        )
        mock_store.cancel_run.return_value = cancelled_run

        http = mock_http_post_body({})

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/runs/abc12345/cancel", {}, http
            )

        mock_task.cancel.assert_called_once()
        assert "abc12345" not in _active_tasks

    @pytest.mark.asyncio
    async def test_cancel_already_done_task(self, handler_with_store, mock_store, mock_http_post_body):
        from aragora.server.handlers.self_improve import _active_tasks

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = True
        _active_tasks["abc12345"] = mock_task

        cancelled_run = SelfImproveRun(
            run_id="abc12345",
            goal="Test",
            status=RunStatus.CANCELLED,
        )
        mock_store.cancel_run.return_value = cancelled_run

        http = mock_http_post_body({})

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            await handler_with_store.handle_post(
                "/api/self-improve/runs/abc12345/cancel", {}, http
            )

        mock_task.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_store_unavailable(self, handler, mock_http_post_body):
        http = mock_http_post_body({})

        with patch.object(handler, "_get_store", return_value=None):
            with patch.object(handler, "read_json_body", return_value={}):
                result = await handler.handle_post(
                    "/api/self-improve/runs/abc123/cancel", {}, http
                )

        assert result.status_code == 503


# ============================================================================
# Generate Plan Tests
# ============================================================================


class TestGeneratePlan:
    """Tests for plan generation."""

    @pytest.mark.asyncio
    async def test_generate_plan_success(self, handler):
        mock_result = MagicMock()
        mock_subtask = MagicMock()
        mock_subtask.description = "Subtask 1"
        mock_subtask.track = "qa"
        mock_subtask.priority = 1
        mock_result.subtasks = [mock_subtask]
        mock_result.complexity_score = 0.7

        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.task_decomposer.TaskDecomposer"
        ) as MockDecomposer:
            MockDecomposer.return_value.analyze.return_value = mock_result
            plan = await handler._generate_plan("Improve tests", ["qa"])

        assert plan["goal"] == "Improve tests"
        assert plan["tracks"] == ["qa"]
        assert len(plan["subtasks"]) == 1
        assert plan["subtasks"][0]["description"] == "Subtask 1"
        assert plan["complexity"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_plan_import_error(self, handler):
        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.task_decomposer.TaskDecomposer",
            side_effect=ImportError("No decomposer"),
        ):
            plan = await handler._generate_plan("Test", None)

        assert plan["goal"] == "Test"
        assert plan["tracks"] == []
        assert plan["subtasks"] == []
        assert "error" in plan

    @pytest.mark.asyncio
    async def test_generate_plan_no_subtasks_attr(self, handler):
        mock_result = MagicMock(spec=[])

        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.task_decomposer.TaskDecomposer"
        ) as MockDecomposer:
            MockDecomposer.return_value.analyze.return_value = mock_result
            plan = await handler._generate_plan("Test", ["dev"])

        assert plan["subtasks"] == []

    @pytest.mark.asyncio
    async def test_generate_plan_none_tracks(self, handler):
        mock_result = MagicMock()
        mock_result.subtasks = []
        mock_result.complexity_score = 0.1

        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.task_decomposer.TaskDecomposer"
        ) as MockDecomposer:
            MockDecomposer.return_value.analyze.return_value = mock_result
            plan = await handler._generate_plan("Test", None)

        assert plan["tracks"] == []


# ============================================================================
# Execute Run Tests
# ============================================================================


class TestExecuteRun:
    """Tests for background run execution."""

    @pytest.mark.asyncio
    async def test_execute_run_success(self, handler_with_store, mock_store):
        from aragora.server.handlers.self_improve import _active_tasks

        _active_tasks["run123"] = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_subtasks = 3
        mock_result.completed_subtasks = 3
        mock_result.failed_subtasks = 0
        mock_result.summary = "Done"
        mock_result.error = None

        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.hardened_orchestrator.HardenedOrchestrator"
        ) as MockOrch:
            mock_orch_instance = MockOrch.return_value
            mock_orch_instance.execute_goal_coordinated = AsyncMock(return_value=mock_result)
            await handler_with_store._execute_run("run123", "Test goal", None, "flat", None, 5)

        mock_store.update_run.assert_called()
        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "completed"
        assert last_call[1]["total_subtasks"] == 3
        assert "run123" not in _active_tasks

    @pytest.mark.asyncio
    async def test_execute_run_failure(self, handler_with_store, mock_store):
        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.hardened_orchestrator.HardenedOrchestrator"
        ) as MockOrch:
            mock_orch_instance = MockOrch.return_value
            mock_orch_instance.execute_goal_coordinated = AsyncMock(
                side_effect=RuntimeError("Orchestration failed")
            )
            await handler_with_store._execute_run("run123", "Test", None, "flat", None, 5)

        mock_store.update_run.assert_called()
        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "failed"
        assert "failed" in last_call[1]["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_run_cancelled(self, handler_with_store, mock_store):
        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.hardened_orchestrator.HardenedOrchestrator"
        ) as MockOrch:
            mock_orch_instance = MockOrch.return_value
            mock_orch_instance.execute_goal_coordinated = AsyncMock(
                side_effect=asyncio.CancelledError()
            )
            await handler_with_store._execute_run("run123", "Test", None, "flat", None, 5)

        mock_store.update_run.assert_called()
        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_execute_run_store_unavailable(self, handler):
        with patch.object(handler, "_get_store", return_value=None):
            await handler._execute_run("run123", "Test", None, "flat", None, 5)
        # Should return early without error

    @pytest.mark.asyncio
    async def test_execute_run_cleans_active_tasks(self, handler_with_store, mock_store):
        from aragora.server.handlers.self_improve import _active_tasks

        _active_tasks["run123"] = MagicMock()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.total_subtasks = 1
        mock_result.completed_subtasks = 0
        mock_result.failed_subtasks = 1
        mock_result.summary = "Failed"
        mock_result.error = "Test error"

        with patch(
            "aragora.nomic.self_improve.SelfImprovePipeline",
            side_effect=ImportError("skip pipeline"),
        ), patch(
            "aragora.nomic.hardened_orchestrator.HardenedOrchestrator"
        ) as MockOrch:
            mock_orch_instance = MockOrch.return_value
            mock_orch_instance.execute_goal_coordinated = AsyncMock(return_value=mock_result)
            await handler_with_store._execute_run("run123", "Test", None, "flat", None, 5)

        assert "run123" not in _active_tasks


# ============================================================================
# Store Initialization Tests
# ============================================================================


class TestStoreInit:
    """Tests for lazy store initialization."""

    def test_get_store_lazy_init(self, handler):
        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore"
        ) as MockStore:
            store_instance = MagicMock()
            MockStore.return_value = store_instance

            result = handler._get_store()
            assert result is store_instance

            # Second call returns cached
            result2 = handler._get_store()
            assert result2 is store_instance
            MockStore.assert_called_once()

    def test_get_store_import_error(self, handler):
        import sys

        # Temporarily remove the module to force ImportError on local import
        saved = sys.modules.pop("aragora.nomic.stores.run_store", None)
        sys.modules["aragora.nomic.stores.run_store"] = None  # type: ignore[assignment]
        try:
            result = handler._get_store()
            assert result is None
        finally:
            if saved is not None:
                sys.modules["aragora.nomic.stores.run_store"] = saved
            else:
                sys.modules.pop("aragora.nomic.stores.run_store", None)

    def test_get_store_os_error(self, handler):
        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            side_effect=OSError("Permission denied"),
        ):
            result = handler._get_store()
            assert result is None


# ============================================================================
# Unhandled Path Tests
# ============================================================================


class TestUnhandledPaths:
    """Tests for paths that return None (not handled)."""

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_cancel_path(self, handler_with_store, mock_http_handler):
        """GET to /cancel should return None (cancel is POST only)."""
        result = await handler_with_store.handle(
            "/api/self-improve/runs/abc123/cancel", {}, mock_http_handler()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown(self, handler_with_store, mock_http_post_body):
        http = mock_http_post_body({})
        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/unknown", {}, http
            )
        assert result is None


# ============================================================================
# Run Store Unit Tests
# ============================================================================


class TestSelfImproveRunStore:
    """Tests for the SelfImproveRunStore."""

    def test_create_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test goal", tracks=["qa"])
        assert run.goal == "Test goal"
        assert run.tracks == ["qa"]
        assert run.status == RunStatus.PENDING
        assert run.run_id is not None

    def test_get_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        created = store.create_run(goal="Test")
        fetched = store.get_run(created.run_id)
        assert fetched is not None
        assert fetched.run_id == created.run_id

    def test_get_run_not_found(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        assert store.get_run("nonexistent") is None

    def test_update_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test")
        updated = store.update_run(run.run_id, status="running")
        assert updated is not None
        assert updated.status == RunStatus.RUNNING

    def test_update_run_not_found(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        assert store.update_run("nonexistent", status="running") is None

    def test_list_runs(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        store.create_run(goal="Run 1")
        store.create_run(goal="Run 2")
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_with_status_filter(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        store.create_run(goal="Run 1")
        run2 = store.create_run(goal="Run 2")
        store.update_run(run2.run_id, status="running")

        pending = store.list_runs(status="pending")
        assert len(pending) == 1

        running = store.list_runs(status="running")
        assert len(running) == 1

    def test_list_runs_pagination(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        for i in range(5):
            store.create_run(goal=f"Run {i}")

        page1 = store.list_runs(limit=2, offset=0)
        assert len(page1) == 2

        page2 = store.list_runs(limit=2, offset=2)
        assert len(page2) == 2

        page3 = store.list_runs(limit=2, offset=4)
        assert len(page3) == 1

    def test_cancel_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test")
        cancelled = store.cancel_run(run.run_id)
        assert cancelled is not None
        assert cancelled.status == RunStatus.CANCELLED
        assert cancelled.completed_at is not None

    def test_cancel_completed_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test")
        store.update_run(run.run_id, status="completed")
        assert store.cancel_run(run.run_id) is None

    def test_cancel_nonexistent_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        assert store.cancel_run("nonexistent") is None

    def test_persistence(self, tmp_path):
        store1 = SelfImproveRunStore(data_dir=tmp_path)
        store1.create_run(goal="Persistent")

        store2 = SelfImproveRunStore(data_dir=tmp_path)
        runs = store2.list_runs()
        assert len(runs) == 1
        assert runs[0].goal == "Persistent"

    def test_run_to_dict(self):
        run = SelfImproveRun(run_id="test123", goal="Test", status=RunStatus.RUNNING)
        d = run.to_dict()
        assert d["run_id"] == "test123"
        assert d["status"] == "running"
        assert d["goal"] == "Test"

    def test_run_from_dict(self):
        d = {"run_id": "test123", "goal": "Test", "status": "completed"}
        run = SelfImproveRun.from_dict(d)
        assert run.run_id == "test123"
        assert run.status == RunStatus.COMPLETED

    def test_run_from_dict_unknown_fields(self):
        d = {"run_id": "test", "goal": "Test", "status": "pending", "extra_field": "ignored"}
        run = SelfImproveRun.from_dict(d)
        assert run.run_id == "test"
        assert not hasattr(run, "extra_field")


# ============================================================================
# Worktree Endpoint Tests
# ============================================================================


class TestListWorktrees:
    """Tests for GET /api/self-improve/worktrees."""

    def test_can_handle_worktrees(self, handler):
        assert handler.can_handle("/api/self-improve/worktrees") is True

    def test_can_handle_worktrees_versioned(self, handler):
        assert handler.can_handle("/api/v1/self-improve/worktrees") is True

    @pytest.mark.asyncio
    async def test_list_worktrees_success(self, handler, mock_http_handler):
        mock_wt = MagicMock()
        mock_wt.branch_name = "dev/qa-coverage-001"
        mock_wt.worktree_path = "/tmp/worktrees/qa-coverage"
        mock_wt.track = "qa"
        mock_wt.created_at = datetime(2026, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
        mock_wt.assignment_id = "assign-001"

        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator"
        ) as MockCoord:
            MockCoord.return_value.list_worktrees.return_value = [mock_wt]
            result = await handler.handle(
                "/api/self-improve/worktrees", {}, mock_http_handler()
            )

        body = parse_handler_response(result)
        assert result.status_code == 200
        assert body["total"] == 1
        assert body["worktrees"][0]["branch_name"] == "dev/qa-coverage-001"
        assert body["worktrees"][0]["track"] == "qa"

    @pytest.mark.asyncio
    async def test_list_worktrees_empty(self, handler, mock_http_handler):
        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator"
        ) as MockCoord:
            MockCoord.return_value.list_worktrees.return_value = []
            result = await handler.handle(
                "/api/self-improve/worktrees", {}, mock_http_handler()
            )

        body = parse_handler_response(result)
        assert body["worktrees"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_worktrees_import_error(self, handler, mock_http_handler):
        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator",
            side_effect=ImportError("No coordinator"),
        ):
            result = await handler.handle(
                "/api/self-improve/worktrees", {}, mock_http_handler()
            )

        body = parse_handler_response(result)
        assert body["worktrees"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_worktrees_no_created_at(self, handler, mock_http_handler):
        mock_wt = MagicMock()
        mock_wt.branch_name = "dev/test-branch"
        mock_wt.worktree_path = "/tmp/worktrees/test"
        mock_wt.track = None
        mock_wt.created_at = None
        mock_wt.assignment_id = None

        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator"
        ) as MockCoord:
            MockCoord.return_value.list_worktrees.return_value = [mock_wt]
            result = await handler.handle(
                "/api/self-improve/worktrees", {}, mock_http_handler()
            )

        body = parse_handler_response(result)
        assert body["worktrees"][0]["created_at"] is None
        assert body["worktrees"][0]["track"] is None


class TestCleanupWorktrees:
    """Tests for POST /api/self-improve/worktrees/cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self, handler, mock_http_post_body):
        http = mock_http_post_body({})

        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator"
        ) as MockCoord:
            MockCoord.return_value.cleanup_all_worktrees.return_value = 3
            with patch.object(handler, "read_json_body", return_value={}):
                result = await handler.handle_post(
                    "/api/self-improve/worktrees/cleanup", {}, http
                )

        body = parse_handler_response(result)
        assert result.status_code == 200
        assert body["removed"] == 3
        assert body["status"] == "cleaned"

    @pytest.mark.asyncio
    async def test_cleanup_nothing(self, handler, mock_http_post_body):
        http = mock_http_post_body({})

        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator"
        ) as MockCoord:
            MockCoord.return_value.cleanup_all_worktrees.return_value = 0
            with patch.object(handler, "read_json_body", return_value={}):
                result = await handler.handle_post(
                    "/api/self-improve/worktrees/cleanup", {}, http
                )

        body = parse_handler_response(result)
        assert body["removed"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_import_error(self, handler, mock_http_post_body):
        http = mock_http_post_body({})

        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator",
            side_effect=ImportError("No coordinator"),
        ):
            with patch.object(handler, "read_json_body", return_value={}):
                result = await handler.handle_post(
                    "/api/self-improve/worktrees/cleanup", {}, http
                )

        assert result.status_code == 503
