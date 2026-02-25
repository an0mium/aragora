"""Tests for self-improvement handler (aragora/server/handlers/self_improve.py).

Covers all routes and behavior of the SelfImproveHandler class:
- can_handle() routing for all ROUTES (versioned and unversioned)
- GET  /api/self-improve/status           - Get current cycle status
- GET  /api/self-improve/runs             - List all runs
- GET  /api/self-improve/history          - List runs (alias)
- GET  /api/self-improve/runs/:id         - Get specific run
- GET  /api/self-improve/worktrees        - List worktrees
- POST /api/self-improve/run              - Start a run
- POST /api/self-improve/start            - Start a run (legacy alias)
- POST /api/self-improve/coordinate       - Start hierarchical coordination
- POST /api/self-improve/runs/:id/cancel  - Cancel a running run
- POST /api/self-improve/worktrees/cleanup- Clean up worktrees
- Rate limiting behavior (handle, handle_post)
- Error handling (store unavailable, missing fields, invalid modes)
- Dry run mode (plan generation fallback chain)
- Background execution (pipeline, orchestrator, coordination)
- Worktree management (list, cleanup, import failures)
- RBAC permission checks
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.self_improve import (
    SelfImproveHandler,
    _active_tasks,
    _extract_run_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for SelfImproveHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        client_address: tuple[str, int] = ("127.0.0.1", 12345),
    ):
        self.command = method
        self.client_address = client_address
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.path = ""

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


class MockRun:
    """Mock run object returned by the store."""

    def __init__(
        self,
        run_id: str = "run-001",
        goal: str = "Improve test coverage",
        status: str = "running",
        mode: str = "flat",
        tracks: list[str] | None = None,
    ):
        self.run_id = run_id
        self.goal = goal
        self.status = status
        self.mode = mode
        self.tracks = tracks or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "mode": self.mode,
            "tracks": self.tracks,
        }


class MockWorktree:
    """Mock worktree object returned by BranchCoordinator."""

    def __init__(
        self,
        branch_name: str = "nomic/qa-1",
        worktree_path: str = "/tmp/worktree-1",
        track: str = "qa",
        created_at: str = "2026-02-23T00:00:00Z",
        assignment_id: str = "assign-001",
    ):
        self.branch_name = branch_name
        self.worktree_path = worktree_path
        self.track = track
        self.created_at = created_at
        self.assignment_id = assignment_id


class MockPipelineResult:
    """Mock result from SelfImprovePipeline.run()."""

    def __init__(
        self,
        subtasks_completed: int = 3,
        subtasks_total: int = 5,
        subtasks_failed: int = 2,
    ):
        self.subtasks_completed = subtasks_completed
        self.subtasks_total = subtasks_total
        self.subtasks_failed = subtasks_failed


class MockOrchestratorResult:
    """Mock result from HardenedOrchestrator.execute_goal_coordinated()."""

    def __init__(
        self,
        success: bool = True,
        total_subtasks: int = 4,
        completed_subtasks: int = 3,
        failed_subtasks: int = 1,
        summary: str = "Completed 3/4 subtasks",
        error: str | None = None,
    ):
        self.success = success
        self.total_subtasks = total_subtasks
        self.completed_subtasks = completed_subtasks
        self.failed_subtasks = failed_subtasks
        self.summary = summary
        self.error = error


class MockCoordinationResult:
    """Mock result from HierarchicalCoordinator.coordinate()."""

    def __init__(
        self,
        success: bool = True,
        cycles_used: int = 2,
        worker_reports: list | None = None,
    ):
        self.success = success
        self.cycles_used = cycles_used
        self.worker_reports = worker_reports or []


class MockWorkerReport:
    """Mock worker report."""

    def __init__(self, success: bool = True):
        self.success = success


class MockDecomposerResult:
    """Mock result from TaskDecomposer.analyze()."""

    def __init__(self, subtasks: list | None = None, complexity_score: float = 0.5):
        self.subtasks = subtasks or []
        self.complexity_score = complexity_score


class MockSubtask:
    """Mock subtask from decomposer."""

    def __init__(
        self,
        description: str = "Write tests",
        track: str = "qa",
        priority: int = 1,
    ):
        self.description = description
        self.track = track
        self.priority = priority


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SelfImproveHandler with minimal server context."""
    return SelfImproveHandler(server_context={})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler for GET requests."""
    return MockHTTPHandler()


@pytest.fixture
def mock_store():
    """Create a mock run store."""
    store = MagicMock()
    store.list_runs.return_value = [
        MockRun("run-001", "Improve tests", "completed"),
        MockRun("run-002", "Fix bugs", "running"),
    ]
    store.get_run.return_value = MockRun("run-001")
    store.create_run.return_value = MockRun("run-new", "New goal", "pending")
    store.cancel_run.return_value = MockRun("run-001", status="cancelled")
    store.update_run.return_value = None
    return store


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters between tests to prevent cross-test pollution."""
    yield
    try:
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def clear_active_tasks():
    """Clear the module-level _active_tasks dict between tests."""
    _active_tasks.clear()
    yield
    # Also cancel any lingering asyncio tasks
    for task in _active_tasks.values():
        if hasattr(task, "cancel"):
            task.cancel()
    _active_tasks.clear()


# ===========================================================================
# _extract_run_id tests
# ===========================================================================


class TestExtractRunId:
    """Tests for the _extract_run_id helper function."""

    def test_extract_from_runs_path(self):
        assert _extract_run_id("/api/self-improve/runs/run-123") == "run-123"

    def test_extract_from_cancel_path(self):
        assert _extract_run_id("/api/self-improve/runs/run-456/cancel") == "run-456"

    def test_returns_none_for_short_path(self):
        assert _extract_run_id("/api/self-improve") is None

    def test_returns_none_for_non_runs_path(self):
        assert _extract_run_id("/api/self-improve/status") is None

    def test_returns_none_for_just_runs(self):
        # parts = ["api", "self-improve", "runs"] -> len 3 < 4
        assert _extract_run_id("/api/self-improve/runs") is None

    def test_extract_uuid_style_id(self):
        assert _extract_run_id("/api/self-improve/runs/abc-def-123") == "abc-def-123"

    def test_extract_with_trailing_slash(self):
        result = _extract_run_id("/api/self-improve/runs/run-001/")
        assert result is not None

    def test_empty_path(self):
        assert _extract_run_id("") is None

    def test_root_path(self):
        assert _extract_run_id("/") is None


# ===========================================================================
# can_handle tests
# ===========================================================================


class TestCanHandle:
    """Tests for SelfImproveHandler.can_handle() routing."""

    def test_handles_status(self, handler):
        assert handler.can_handle("/api/self-improve/status") is True

    def test_handles_runs(self, handler):
        assert handler.can_handle("/api/self-improve/runs") is True

    def test_handles_history(self, handler):
        assert handler.can_handle("/api/self-improve/history") is True

    def test_handles_run(self, handler):
        assert handler.can_handle("/api/self-improve/run") is True

    def test_handles_start(self, handler):
        assert handler.can_handle("/api/self-improve/start") is True

    def test_handles_coordinate(self, handler):
        assert handler.can_handle("/api/self-improve/coordinate") is True

    def test_handles_worktrees(self, handler):
        assert handler.can_handle("/api/self-improve/worktrees") is True

    def test_handles_worktrees_cleanup(self, handler):
        assert handler.can_handle("/api/self-improve/worktrees/cleanup") is True

    def test_handles_worktrees_autopilot_status(self, handler):
        assert handler.can_handle("/api/self-improve/worktrees/autopilot/status") is True

    def test_handles_worktrees_autopilot_ensure(self, handler):
        assert handler.can_handle("/api/self-improve/worktrees/autopilot/ensure") is True

    def test_handles_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/self-improve/status") is True

    def test_handles_v2_versioned_path(self, handler):
        assert handler.can_handle("/api/v2/self-improve/runs") is True

    def test_handles_run_by_id(self, handler):
        assert handler.can_handle("/api/self-improve/runs/run-001") is True

    def test_handles_cancel(self, handler):
        assert handler.can_handle("/api/self-improve/runs/run-001/cancel") is True

    def test_does_not_handle_other_paths(self, handler):
        assert handler.can_handle("/api/debates/list") is False

    def test_does_not_handle_random_api(self, handler):
        assert handler.can_handle("/api/v1/billing/plans") is False

    def test_handles_any_self_improve_subpath(self, handler):
        assert handler.can_handle("/api/self-improve/custom-endpoint") is True

    def test_method_parameter_accepted(self, handler):
        assert handler.can_handle("/api/self-improve/status", method="POST") is True

    def test_does_not_handle_partial_match(self, handler):
        assert handler.can_handle("/api/self-improveX/status") is False


# ===========================================================================
# GET /api/self-improve/status
# ===========================================================================


class TestGetStatus:
    """Tests for GET /api/self-improve/status endpoint."""

    @pytest.mark.asyncio
    async def test_idle_when_no_active_tasks(self, handler, http_handler):
        result = await handler.handle("/api/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "idle"
        assert body["active_runs"] == 0
        assert body["runs"] == []

    @pytest.mark.asyncio
    async def test_running_with_active_task(self, handler, http_handler, mock_store):
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _active_tasks["run-001"] = mock_task

        handler._store = mock_store
        mock_store.get_run.return_value = MockRun("run-001", status="running")

        result = await handler.handle("/api/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "running"
        assert body["active_runs"] == 1
        assert len(body["runs"]) == 1
        assert body["runs"][0]["run_id"] == "run-001"

    @pytest.mark.asyncio
    async def test_idle_when_task_is_done(self, handler, http_handler):
        mock_task = MagicMock()
        mock_task.done.return_value = True
        _active_tasks["run-001"] = mock_task

        result = await handler.handle("/api/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "idle"
        assert body["active_runs"] == 0

    @pytest.mark.asyncio
    async def test_running_no_store(self, handler, http_handler):
        """Active tasks but store unavailable => running state, empty runs list."""
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _active_tasks["run-001"] = mock_task

        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle("/api/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "running"
        assert body["active_runs"] == 1
        assert body["runs"] == []

    @pytest.mark.asyncio
    async def test_running_store_returns_none_for_run(self, handler, http_handler, mock_store):
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _active_tasks["run-001"] = mock_task

        handler._store = mock_store
        mock_store.get_run.return_value = None

        result = await handler.handle("/api/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "running"
        assert body["active_runs"] == 1
        assert body["runs"] == []

    @pytest.mark.asyncio
    async def test_multiple_active_tasks(self, handler, http_handler, mock_store):
        for i in range(3):
            t = MagicMock()
            t.done.return_value = False
            _active_tasks[f"run-{i:03d}"] = t

        handler._store = mock_store
        mock_store.get_run.side_effect = lambda rid: MockRun(rid, status="running")

        result = await handler.handle("/api/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "running"
        assert body["active_runs"] == 3
        assert len(body["runs"]) == 3

    @pytest.mark.asyncio
    async def test_status_via_versioned_path(self, handler, http_handler):
        result = await handler.handle("/api/v1/self-improve/status", {}, http_handler)
        body = _body(result)
        assert body["state"] == "idle"

    @pytest.mark.asyncio
    async def test_status_200_code(self, handler, http_handler):
        result = await handler.handle("/api/self-improve/status", {}, http_handler)
        assert _status(result) == 200


# ===========================================================================
# GET /api/self-improve/runs and /history
# ===========================================================================


class TestListRuns:
    """Tests for GET /api/self-improve/runs and /history endpoints."""

    @pytest.mark.asyncio
    async def test_list_runs(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle("/api/self-improve/runs", {}, http_handler)
        body = _body(result)
        assert body["total"] == 2
        assert len(body["runs"]) == 2
        assert body["limit"] == 50
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_runs_with_pagination(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle(
            "/api/self-improve/runs",
            {"limit": "10", "offset": "5"},
            http_handler,
        )
        mock_store.list_runs.assert_called_with(limit=10, offset=5, status=None)

    @pytest.mark.asyncio
    async def test_list_runs_with_status_filter(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle(
            "/api/self-improve/runs",
            {"status": "completed"},
            http_handler,
        )
        mock_store.list_runs.assert_called_with(limit=50, offset=0, status="completed")

    @pytest.mark.asyncio
    async def test_history_alias(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle("/api/self-improve/history", {}, http_handler)
        body = _body(result)
        assert "runs" in body
        mock_store.list_runs.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_runs_store_unavailable(self, handler, http_handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle("/api/self-improve/runs", {}, http_handler)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_list_runs_versioned_path(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle("/api/v2/self-improve/runs", {}, http_handler)
        body = _body(result)
        assert "runs" in body

    @pytest.mark.asyncio
    async def test_list_runs_empty_result(self, handler, http_handler, mock_store):
        handler._store = mock_store
        mock_store.list_runs.return_value = []
        result = await handler.handle("/api/self-improve/runs", {}, http_handler)
        body = _body(result)
        assert body["total"] == 0
        assert body["runs"] == []

    @pytest.mark.asyncio
    async def test_history_store_unavailable(self, handler, http_handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle("/api/self-improve/history", {}, http_handler)
        assert _status(result) == 503


# ===========================================================================
# GET /api/self-improve/runs/:id
# ===========================================================================


class TestGetRun:
    """Tests for GET /api/self-improve/runs/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_by_id(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle("/api/self-improve/runs/run-001", {}, http_handler)
        body = _body(result)
        assert body["run_id"] == "run-001"

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, handler, http_handler, mock_store):
        handler._store = mock_store
        mock_store.get_run.return_value = None
        result = await handler.handle("/api/self-improve/runs/nonexistent", {}, http_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_get_run_store_unavailable(self, handler, http_handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle("/api/self-improve/runs/run-001", {}, http_handler)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_run_does_not_match_cancel_path(self, handler, http_handler):
        """GET to /runs/:id/cancel should not be treated as get_run."""
        handler._store = MagicMock()
        result = await handler.handle("/api/self-improve/runs/run-001/cancel", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_run_200_status(self, handler, http_handler, mock_store):
        handler._store = mock_store
        result = await handler.handle("/api/self-improve/runs/run-001", {}, http_handler)
        assert _status(result) == 200


# ===========================================================================
# POST /api/self-improve/run and /start
# (Mock _execute_run to prevent background execution)
# ===========================================================================


class TestStartRun:
    """Tests for POST /api/self-improve/run and /start endpoints."""

    @pytest.mark.asyncio
    async def test_start_run(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Improve coverage"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 202
        body = _body(result)
        assert body["status"] == "started"
        assert "run_id" in body

    @pytest.mark.asyncio
    async def test_start_run_legacy_alias(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Fix bugs"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/start", {}, http)
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_start_run_missing_goal(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 400
        assert "goal" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_start_run_empty_goal(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "   "})
        result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_start_run_invalid_mode(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test", "mode": "invalid"})
        result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 400
        assert "mode" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_start_run_flat_mode(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test", "mode": "flat"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_start_run_hierarchical_mode(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test", "mode": "hierarchical"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_start_run_store_unavailable(self, handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            http = MockHTTPHandler(body={"goal": "Test"})
            result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_start_run_with_all_options(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(
            body={
                "goal": "Full options test",
                "mode": "flat",
                "tracks": ["qa", "developer"],
                "budget_limit_usd": 5.0,
                "max_cycles": 3,
                "dry_run": False,
                "scan_mode": False,
                "quick_mode": True,
                "require_approval": True,
            }
        )
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_start_run_creates_task(self, handler, mock_store):
        handler._store = mock_store
        mock_store.create_run.return_value = MockRun("run-task-test", "Test", "pending")
        http = MockHTTPHandler(body={"goal": "Test"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert "run-task-test" in _active_tasks

    @pytest.mark.asyncio
    async def test_start_run_versioned_path(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test versioned"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            result = await handler.handle_post("/api/v1/self-improve/run", {}, http)
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_start_run_updates_store_status(self, handler, mock_store):
        """Starting a run should update the store with 'running' status."""
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test update"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/run", {}, http)
        # update_run should be called with status="running"
        mock_store.update_run.assert_called()
        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "running"


# ===========================================================================
# POST /api/self-improve/run (dry_run mode)
# ===========================================================================


class TestDryRun:
    """Tests for dry-run mode via POST /api/self-improve/run."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self, handler, mock_store):
        handler._store = mock_store
        with patch.object(
            handler,
            "_generate_plan",
            new_callable=AsyncMock,
            return_value={"goal": "Test", "subtasks": [{"desc": "do stuff"}]},
        ):
            http = MockHTTPHandler(body={"goal": "Test plan", "dry_run": True})
            result = await handler.handle_post("/api/self-improve/run", {}, http)

        body = _body(result)
        assert body["status"] == "preview"
        assert "plan" in body
        assert "run_id" in body

    @pytest.mark.asyncio
    async def test_dry_run_updates_store_completed(self, handler, mock_store):
        handler._store = mock_store
        with patch.object(
            handler,
            "_generate_plan",
            new_callable=AsyncMock,
            return_value={"goal": "Test"},
        ):
            http = MockHTTPHandler(body={"goal": "Test", "dry_run": True})
            await handler.handle_post("/api/self-improve/run", {}, http)

        mock_store.update_run.assert_called()
        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "completed"

    @pytest.mark.asyncio
    async def test_dry_run_emits_websocket_event(self, handler, mock_store):
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_phase_completed = AsyncMock()

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.object(
                handler,
                "_generate_plan",
                new_callable=AsyncMock,
                return_value={"goal": "Test"},
            ):
                http = MockHTTPHandler(body={"goal": "Test", "dry_run": True})
                await handler.handle_post("/api/self-improve/run", {}, http)

        mock_stream.emit_phase_completed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dry_run_stream_emit_failure(self, handler, mock_store):
        """WebSocket emit failure should not break dry run."""
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_phase_completed = AsyncMock(side_effect=RuntimeError("ws fail"))

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.object(
                handler,
                "_generate_plan",
                new_callable=AsyncMock,
                return_value={"goal": "Test"},
            ):
                http = MockHTTPHandler(body={"goal": "Test", "dry_run": True})
                result = await handler.handle_post("/api/self-improve/run", {}, http)

        assert _body(result)["status"] == "preview"

    @pytest.mark.asyncio
    async def test_dry_run_os_error_in_stream(self, handler, mock_store):
        """OSError in stream emit should also be handled."""
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_phase_completed = AsyncMock(side_effect=OSError("ws fail"))

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.object(
                handler,
                "_generate_plan",
                new_callable=AsyncMock,
                return_value={"goal": "Test"},
            ):
                http = MockHTTPHandler(body={"goal": "Test", "dry_run": True})
                result = await handler.handle_post("/api/self-improve/run", {}, http)

        assert _body(result)["status"] == "preview"

    @pytest.mark.asyncio
    async def test_dry_run_no_active_task(self, handler, mock_store):
        """Dry run should NOT create an active task."""
        handler._store = mock_store
        mock_store.create_run.return_value = MockRun("dry-run-001", "Test", "pending")

        with patch.object(
            handler,
            "_generate_plan",
            new_callable=AsyncMock,
            return_value={"goal": "Test"},
        ):
            http = MockHTTPHandler(body={"goal": "Test", "dry_run": True})
            await handler.handle_post("/api/self-improve/run", {}, http)

        assert "dry-run-001" not in _active_tasks

    @pytest.mark.asyncio
    async def test_dry_run_no_stream_server(self, handler, mock_store):
        """No stream server => event emit is skipped without error."""
        handler._store = mock_store

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.object(
                handler,
                "_generate_plan",
                new_callable=AsyncMock,
                return_value={"goal": "Test"},
            ):
                http = MockHTTPHandler(body={"goal": "Test", "dry_run": True})
                result = await handler.handle_post("/api/self-improve/run", {}, http)

        assert _body(result)["status"] == "preview"


# ===========================================================================
# _generate_plan tests
# ===========================================================================


class TestGeneratePlan:
    """Tests for the _generate_plan method (plan generation fallback chain)."""

    @pytest.mark.asyncio
    async def test_generate_plan_pipeline_path(self, handler):
        """Test plan generation via SelfImprovePipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.dry_run = AsyncMock(return_value={"steps": ["a"]})

        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline
        mock_module.SelfImproveConfig.return_value = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
            plan = await handler._generate_plan("Improve tests", ["qa"])

        assert plan["tracks"] == ["qa"]

    @pytest.mark.asyncio
    async def test_generate_plan_decomposer_fallback(self, handler):
        """When pipeline is unavailable, fall back to TaskDecomposer."""
        mock_subtask = MockSubtask("Write tests", "qa", 1)
        mock_result = MockDecomposerResult(subtasks=[mock_subtask], complexity_score=0.7)

        mock_decomposer = MagicMock()
        mock_decomposer.analyze.return_value = mock_result
        mock_decomposer_module = MagicMock()
        mock_decomposer_module.TaskDecomposer.return_value = mock_decomposer

        with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
            with patch.dict(
                "sys.modules",
                {"aragora.nomic.task_decomposer": mock_decomposer_module},
            ):
                plan = await handler._generate_plan("Test", ["qa"])

        assert plan["goal"] == "Test"
        assert plan["tracks"] == ["qa"]
        assert len(plan["subtasks"]) == 1
        assert plan["subtasks"][0]["description"] == "Write tests"
        assert plan["complexity"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_plan_full_fallback(self, handler):
        """When both pipeline and decomposer fail, return minimal plan."""
        with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
            with patch.dict("sys.modules", {"aragora.nomic.task_decomposer": None}):
                plan = await handler._generate_plan("Test", None)

        assert plan["goal"] == "Test"
        assert plan["tracks"] == []
        assert plan["subtasks"] == []
        assert "error" in plan

    @pytest.mark.asyncio
    async def test_generate_plan_pipeline_runtime_error(self, handler):
        """RuntimeError in pipeline dry_run falls back to decomposer."""
        mock_pipeline = MagicMock()
        mock_pipeline.dry_run = AsyncMock(side_effect=RuntimeError("fail"))

        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline
        mock_module.SelfImproveConfig.return_value = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
            with patch.dict("sys.modules", {"aragora.nomic.task_decomposer": None}):
                plan = await handler._generate_plan("Test", [])

        # Falls through to full fallback
        assert plan["goal"] == "Test"
        assert "error" in plan

    @pytest.mark.asyncio
    async def test_generate_plan_no_tracks(self, handler):
        """Tracks=None should produce empty tracks list."""
        with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
            with patch.dict("sys.modules", {"aragora.nomic.task_decomposer": None}):
                plan = await handler._generate_plan("Test", None)
        assert plan["tracks"] == []

    @pytest.mark.asyncio
    async def test_generate_plan_scan_mode_params(self, handler):
        """Verify scan_mode and quick_mode are passed to config."""
        mock_pipeline = MagicMock()
        mock_pipeline.dry_run = AsyncMock(return_value={"steps": []})

        mock_config_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline
        mock_module.SelfImproveConfig = mock_config_cls

        with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
            await handler._generate_plan("Test", [], scan_mode=True, quick_mode=True)

        mock_config_cls.assert_called_once_with(scan_mode=True, quick_mode=True)


# ===========================================================================
# POST /api/self-improve/runs/:id/cancel
# ===========================================================================


class TestCancelRun:
    """Tests for POST /api/self-improve/runs/:id/cancel endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_run(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/runs/run-001/cancel", {}, http)
        body = _body(result)
        assert body["status"] == "cancelled"
        assert body["run_id"] == "run-001"

    @pytest.mark.asyncio
    async def test_cancel_run_not_found(self, handler, mock_store):
        handler._store = mock_store
        mock_store.cancel_run.return_value = None
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/runs/nonexistent/cancel", {}, http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_cancel_cancels_active_task(self, handler, mock_store):
        handler._store = mock_store
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _active_tasks["run-001"] = mock_task

        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/runs/run-001/cancel", {}, http)
        mock_task.cancel.assert_called_once()
        assert "run-001" not in _active_tasks

    @pytest.mark.asyncio
    async def test_cancel_done_task_not_cancelled(self, handler, mock_store):
        handler._store = mock_store
        mock_task = MagicMock()
        mock_task.done.return_value = True
        _active_tasks["run-001"] = mock_task

        http = MockHTTPHandler(body={})
        await handler.handle_post("/api/self-improve/runs/run-001/cancel", {}, http)
        mock_task.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_store_unavailable(self, handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            http = MockHTTPHandler(body={})
            result = await handler.handle_post("/api/self-improve/runs/run-001/cancel", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cancel_no_active_task(self, handler, mock_store):
        """Cancel a run that has no active task (already cleaned up)."""
        handler._store = mock_store
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/runs/run-001/cancel", {}, http)
        body = _body(result)
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_200_status(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/runs/run-001/cancel", {}, http)
        assert _status(result) == 200


# ===========================================================================
# POST /api/self-improve/coordinate
# ===========================================================================


class TestCoordinate:
    """Tests for POST /api/self-improve/coordinate endpoint."""

    @pytest.mark.asyncio
    async def test_coordinate_start(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Coordinate test"})
        with patch.object(handler, "_execute_coordination", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 202
        body = _body(result)
        assert body["status"] == "coordinating"
        assert body["mode"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_coordinate_missing_goal(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_coordinate_empty_goal(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": ""})
        result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_coordinate_whitespace_goal(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "   \n\t  "})
        result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_coordinate_store_unavailable(self, handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            http = MockHTTPHandler(body={"goal": "Test"})
            result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_coordinate_with_options(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(
            body={
                "goal": "Big project",
                "tracks": ["qa"],
                "max_cycles": 5,
                "quality_threshold": 0.8,
                "max_parallel_workers": 8,
            }
        )
        with patch.object(handler, "_execute_coordination", new_callable=AsyncMock):
            result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 202

    @pytest.mark.asyncio
    async def test_coordinate_creates_task(self, handler, mock_store):
        handler._store = mock_store
        mock_store.create_run.return_value = MockRun("coord-001", "Test", "pending")
        http = MockHTTPHandler(body={"goal": "Coordinate"})
        with patch.object(handler, "_execute_coordination", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert "coord-001" in _active_tasks

    @pytest.mark.asyncio
    async def test_coordinate_updates_store(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test"})
        with patch.object(handler, "_execute_coordination", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/coordinate", {}, http)
        mock_store.update_run.assert_called()
        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "running"

    @pytest.mark.asyncio
    async def test_coordinate_default_max_cycles(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test coord defaults"})
        with patch.object(handler, "_execute_coordination", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/coordinate", {}, http)
        mock_store.create_run.assert_called_once()
        call_kwargs = mock_store.create_run.call_args[1]
        assert call_kwargs["mode"] == "hierarchical"
        assert call_kwargs["max_cycles"] == 3


# ===========================================================================
# _execute_coordination tests
# ===========================================================================


class TestExecuteCoordination:
    """Tests for the _execute_coordination background method."""

    @pytest.mark.asyncio
    async def test_coordination_success(self, handler, mock_store):
        handler._store = mock_store

        mock_result = MockCoordinationResult(
            success=True,
            cycles_used=2,
            worker_reports=[MockWorkerReport(True), MockWorkerReport(False)],
        )

        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.HierarchicalCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": mock_module}):
            await handler._execute_coordination("run-c01", "Improve", None)

        mock_store.update_run.assert_called()
        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "completed"
        assert call_kwargs["completed_subtasks"] == 1
        assert call_kwargs["failed_subtasks"] == 1

    @pytest.mark.asyncio
    async def test_coordination_failure(self, handler, mock_store):
        handler._store = mock_store

        mock_result = MockCoordinationResult(success=False, cycles_used=3, worker_reports=[])

        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.HierarchicalCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": mock_module}):
            await handler._execute_coordination("run-c02", "Bad goal", None)

        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_coordination_import_error(self, handler, mock_store):
        handler._store = mock_store

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": None}):
            await handler._execute_coordination("run-c03", "Test", None)

        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_coordination_runtime_error(self, handler, mock_store):
        handler._store = mock_store

        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(side_effect=RuntimeError("boom"))
        mock_module = MagicMock()
        mock_module.HierarchicalCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": mock_module}):
            await handler._execute_coordination("run-c04", "Test", None)

        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "failed"
        assert call_kwargs["error"] == "Coordination failed"

    @pytest.mark.asyncio
    async def test_coordination_clears_active_tasks(self, handler, mock_store):
        handler._store = mock_store
        _active_tasks["run-c05"] = MagicMock()

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": None}):
            await handler._execute_coordination("run-c05", "Test", None)

        assert "run-c05" not in _active_tasks

    @pytest.mark.asyncio
    async def test_coordination_no_store(self, handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            await handler._execute_coordination("run-c06", "Test", None)

    @pytest.mark.asyncio
    async def test_coordination_value_error(self, handler, mock_store):
        handler._store = mock_store

        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(side_effect=ValueError("bad"))
        mock_module = MagicMock()
        mock_module.HierarchicalCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": mock_module}):
            await handler._execute_coordination("run-c07", "Test", None)

        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["status"] == "failed"

    @pytest.mark.asyncio
    async def test_coordination_summary_on_success(self, handler, mock_store):
        handler._store = mock_store

        mock_result = MockCoordinationResult(success=True, cycles_used=2, worker_reports=[])
        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.HierarchicalCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": mock_module}):
            await handler._execute_coordination("run-c08", "Test", None)

        call_kwargs = mock_store.update_run.call_args[1]
        assert "2 cycles" in call_kwargs["summary"]

    @pytest.mark.asyncio
    async def test_coordination_summary_on_failure(self, handler, mock_store):
        handler._store = mock_store

        mock_result = MockCoordinationResult(success=False, cycles_used=3, worker_reports=[])
        mock_coordinator = MagicMock()
        mock_coordinator.coordinate = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.HierarchicalCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.hierarchical_coordinator": mock_module}):
            await handler._execute_coordination("run-c09", "Test", None)

        call_kwargs = mock_store.update_run.call_args[1]
        assert call_kwargs["summary"] == "Coordination failed"


# ===========================================================================
# _execute_run tests
# ===========================================================================


class TestExecuteRun:
    """Tests for the _execute_run background method."""

    @pytest.mark.asyncio
    async def test_execute_run_hierarchical_delegates(self, handler, mock_store):
        handler._store = mock_store

        with patch.object(handler, "_execute_coordination", new_callable=AsyncMock) as mock_coord:
            await handler._execute_run("run-e01", "Test", None, "hierarchical", None, 3)
        mock_coord.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_run_no_store(self, handler):
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            await handler._execute_run("run-e02", "Test", None, "flat", None, 5)

    @pytest.mark.asyncio
    async def test_execute_run_pipeline_success(self, handler, mock_store):
        handler._store = mock_store
        mock_result = MockPipelineResult(3, 5, 2)

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
                await handler._execute_run("run-e03", "Improve", None, "flat", None, 5)

        update_calls = [c for c in mock_store.update_run.call_args_list if c[0][0] == "run-e03"]
        assert any(c[1].get("status") == "completed" for c in update_calls)

    @pytest.mark.asyncio
    async def test_execute_run_pipeline_import_error_falls_back(self, handler, mock_store):
        """Pipeline ImportError should fall back to HardenedOrchestrator."""
        handler._store = mock_store

        mock_orch_result = MockOrchestratorResult()
        mock_orch = MagicMock()
        mock_orch.execute_goal_coordinated = AsyncMock(return_value=mock_orch_result)
        mock_orch_module = MagicMock()
        mock_orch_module.HardenedOrchestrator.return_value = mock_orch

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict(
                    "sys.modules", {"aragora.nomic.hardened_orchestrator": mock_orch_module}
                ):
                    await handler._execute_run("run-e04", "Test", None, "flat", None, 5)

        update_calls = mock_store.update_run.call_args_list
        assert any(c[1].get("status") == "completed" for c in update_calls if c[0][0] == "run-e04")

    @pytest.mark.asyncio
    async def test_execute_run_both_fail(self, handler, mock_store):
        """Both pipeline and orchestrator fail => status=failed."""
        handler._store = mock_store

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict("sys.modules", {"aragora.nomic.hardened_orchestrator": None}):
                    await handler._execute_run("run-e05", "Test", None, "flat", None, 5)

        update_calls = mock_store.update_run.call_args_list
        assert any(c[1].get("status") == "failed" for c in update_calls if c[0][0] == "run-e05")

    @pytest.mark.asyncio
    async def test_execute_run_clears_active_tasks(self, handler, mock_store):
        handler._store = mock_store
        _active_tasks["run-e06"] = MagicMock()

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict("sys.modules", {"aragora.nomic.hardened_orchestrator": None}):
                    await handler._execute_run("run-e06", "Test", None, "flat", None, 5)

        assert "run-e06" not in _active_tasks

    @pytest.mark.asyncio
    async def test_execute_run_emits_websocket_events(self, handler, mock_store):
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_loop_started = AsyncMock()
        mock_stream.emit_phase_started = AsyncMock()
        mock_stream.emit_loop_stopped = AsyncMock()

        mock_result = MockPipelineResult(1, 1, 0)
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
                await handler._execute_run("run-e07", "Test", None, "flat", None, 5)

        mock_stream.emit_loop_started.assert_awaited_once()
        mock_stream.emit_loop_stopped.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_run_ws_loop_started_failure(self, handler, mock_store):
        """WebSocket emit_loop_started failure should not break execution."""
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_loop_started = AsyncMock(side_effect=RuntimeError("ws fail"))
        mock_stream.emit_phase_started = AsyncMock()
        mock_stream.emit_loop_stopped = AsyncMock()

        mock_result = MockPipelineResult(1, 1, 0)
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
                await handler._execute_run("run-e08", "Test", None, "flat", None, 5)

        # Should still complete
        update_calls = mock_store.update_run.call_args_list
        assert any(c[1].get("status") == "completed" for c in update_calls if c[0][0] == "run-e08")

    @pytest.mark.asyncio
    async def test_execute_run_pipeline_runtime_error_falls_back(self, handler, mock_store):
        """RuntimeError from pipeline should fall back to orchestrator."""
        handler._store = mock_store

        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=RuntimeError("pipeline boom"))

        mock_pipeline_module = MagicMock()
        mock_pipeline_module.SelfImprovePipeline.return_value = mock_pipeline

        mock_orch_result = MockOrchestratorResult()
        mock_orch = MagicMock()
        mock_orch.execute_goal_coordinated = AsyncMock(return_value=mock_orch_result)
        mock_orch_module = MagicMock()
        mock_orch_module.HardenedOrchestrator.return_value = mock_orch

        mock_stream = MagicMock()
        mock_stream.emit_loop_started = AsyncMock()
        mock_stream.emit_phase_started = AsyncMock()
        mock_stream.emit_error = AsyncMock()
        mock_stream.emit_loop_stopped = AsyncMock()

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_pipeline_module}):
                with patch.dict(
                    "sys.modules", {"aragora.nomic.hardened_orchestrator": mock_orch_module}
                ):
                    await handler._execute_run("run-e09", "Test", None, "flat", None, 5)

        mock_stream.emit_error.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_run_orchestrator_error_emits_stream(self, handler, mock_store):
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_loop_started = AsyncMock()
        mock_stream.emit_error = AsyncMock()

        mock_orch = MagicMock()
        mock_orch.execute_goal_coordinated = AsyncMock(side_effect=ValueError("bad"))
        mock_orch_module = MagicMock()
        mock_orch_module.HardenedOrchestrator.return_value = mock_orch

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict(
                    "sys.modules", {"aragora.nomic.hardened_orchestrator": mock_orch_module}
                ):
                    await handler._execute_run("run-e10", "Test", None, "flat", None, 5)

        mock_stream.emit_error.assert_awaited()

    @pytest.mark.asyncio
    async def test_execute_run_stream_error_emit_failure_swallowed(self, handler, mock_store):
        """Stream emit_error failure should be swallowed silently."""
        handler._store = mock_store

        mock_stream = MagicMock()
        mock_stream.emit_loop_started = AsyncMock()
        mock_stream.emit_error = AsyncMock(side_effect=OSError("ws dead"))

        mock_orch = MagicMock()
        mock_orch.execute_goal_coordinated = AsyncMock(side_effect=TypeError("bad type"))
        mock_orch_module = MagicMock()
        mock_orch_module.HardenedOrchestrator.return_value = mock_orch

        with patch.object(handler, "_get_stream_server", return_value=mock_stream):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict(
                    "sys.modules", {"aragora.nomic.hardened_orchestrator": mock_orch_module}
                ):
                    await handler._execute_run("run-e11", "Test", None, "flat", None, 5)

        # Should not raise
        update_calls = mock_store.update_run.call_args_list
        assert any(c[1].get("status") == "failed" for c in update_calls if c[0][0] == "run-e11")


# ===========================================================================
# Pipeline zero-completed subtasks path
# ===========================================================================


class TestPipelineZeroCompleted:
    """Test pipeline result with zero completed subtasks."""

    @pytest.mark.asyncio
    async def test_zero_completed_sets_failed(self, handler, mock_store):
        handler._store = mock_store

        mock_result = MockPipelineResult(subtasks_completed=0, subtasks_total=5, subtasks_failed=5)
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.SelfImprovePipeline.return_value = mock_pipeline

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": mock_module}):
                await handler._execute_run("run-zero", "Test", None, "flat", None, 5)

        update_calls = mock_store.update_run.call_args_list
        assert any(c[1].get("status") == "failed" for c in update_calls if c[0][0] == "run-zero")


# ===========================================================================
# Orchestrator success/failure status mapping
# ===========================================================================


class TestOrchestratorStatusMapping:
    """Test that orchestrator result.success maps correctly."""

    @pytest.mark.asyncio
    async def test_orchestrator_success_true(self, handler, mock_store):
        handler._store = mock_store
        mock_result = MockOrchestratorResult(success=True)
        mock_orch = MagicMock()
        mock_orch.execute_goal_coordinated = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.HardenedOrchestrator.return_value = mock_orch

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict(
                    "sys.modules", {"aragora.nomic.hardened_orchestrator": mock_module}
                ):
                    await handler._execute_run("run-orch-ok", "Test", None, "flat", None, 5)

        update_calls = mock_store.update_run.call_args_list
        assert any(
            c[1].get("status") == "completed" for c in update_calls if c[0][0] == "run-orch-ok"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_success_false(self, handler, mock_store):
        handler._store = mock_store
        mock_result = MockOrchestratorResult(success=False)
        mock_orch = MagicMock()
        mock_orch.execute_goal_coordinated = AsyncMock(return_value=mock_result)
        mock_module = MagicMock()
        mock_module.HardenedOrchestrator.return_value = mock_orch

        with patch.object(handler, "_get_stream_server", return_value=None):
            with patch.dict("sys.modules", {"aragora.nomic.self_improve": None}):
                with patch.dict(
                    "sys.modules", {"aragora.nomic.hardened_orchestrator": mock_module}
                ):
                    await handler._execute_run("run-orch-fail", "Test", None, "flat", None, 5)

        update_calls = mock_store.update_run.call_args_list
        assert any(
            c[1].get("status") == "failed" for c in update_calls if c[0][0] == "run-orch-fail"
        )


# ===========================================================================
# Worktree endpoints
# ===========================================================================


class TestWorktrees:
    """Tests for worktree management endpoints."""

    @pytest.mark.asyncio
    async def test_list_worktrees(self, handler, http_handler):
        mock_wt = MockWorktree()
        mock_coordinator = MagicMock()
        mock_coordinator.list_worktrees.return_value = [mock_wt]
        mock_module = MagicMock()
        mock_module.BranchCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            result = await handler.handle("/api/self-improve/worktrees", {}, http_handler)

        body = _body(result)
        assert body["total"] == 1
        assert body["worktrees"][0]["branch_name"] == "nomic/qa-1"
        assert body["worktrees"][0]["track"] == "qa"

    @pytest.mark.asyncio
    async def test_list_worktrees_empty(self, handler, http_handler):
        mock_coordinator = MagicMock()
        mock_coordinator.list_worktrees.return_value = []
        mock_module = MagicMock()
        mock_module.BranchCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            result = await handler.handle("/api/self-improve/worktrees", {}, http_handler)

        body = _body(result)
        assert body["total"] == 0
        assert body["worktrees"] == []

    @pytest.mark.asyncio
    async def test_list_worktrees_import_error(self, handler, http_handler):
        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": None}):
            result = await handler.handle("/api/self-improve/worktrees", {}, http_handler)

        body = _body(result)
        assert body["total"] == 0
        assert "error" in body

    @pytest.mark.asyncio
    async def test_list_worktrees_value_error(self, handler, http_handler):
        mock_module = MagicMock()
        mock_module.BranchCoordinator.side_effect = ValueError("bad config")

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            result = await handler.handle("/api/self-improve/worktrees", {}, http_handler)

        body = _body(result)
        assert body["total"] == 0
        assert "error" in body

    @pytest.mark.asyncio
    async def test_list_worktrees_os_error(self, handler, http_handler):
        mock_module = MagicMock()
        mock_module.BranchCoordinator.side_effect = OSError("disk")

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            result = await handler.handle("/api/self-improve/worktrees", {}, http_handler)

        body = _body(result)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_worktrees(self, handler):
        mock_coordinator = MagicMock()
        mock_coordinator.cleanup_all_worktrees.return_value = 3
        mock_module = MagicMock()
        mock_module.BranchCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            http = MockHTTPHandler(body={})
            result = await handler.handle_post("/api/self-improve/worktrees/cleanup", {}, http)

        body = _body(result)
        assert body["status"] == "cleaned"
        assert body["removed"] == 3

    @pytest.mark.asyncio
    async def test_cleanup_worktrees_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": None}):
            http = MockHTTPHandler(body={})
            result = await handler.handle_post("/api/self-improve/worktrees/cleanup", {}, http)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cleanup_worktrees_os_error(self, handler):
        mock_module = MagicMock()
        mock_module.BranchCoordinator.return_value.cleanup_all_worktrees.side_effect = OSError(
            "disk"
        )

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            http = MockHTTPHandler(body={})
            result = await handler.handle_post("/api/self-improve/worktrees/cleanup", {}, http)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cleanup_worktrees_value_error(self, handler):
        mock_module = MagicMock()
        mock_module.BranchCoordinator.side_effect = ValueError("bad")

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            http = MockHTTPHandler(body={})
            result = await handler.handle_post("/api/self-improve/worktrees/cleanup", {}, http)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_multiple_worktrees(self, handler, http_handler):
        worktrees = [
            MockWorktree("branch-1", "/tmp/wt1", "qa", "2026-01-01", "a1"),
            MockWorktree("branch-2", "/tmp/wt2", "dev", "2026-01-02", "a2"),
        ]
        mock_coordinator = MagicMock()
        mock_coordinator.list_worktrees.return_value = worktrees
        mock_module = MagicMock()
        mock_module.BranchCoordinator.return_value = mock_coordinator

        with patch.dict("sys.modules", {"aragora.nomic.branch_coordinator": mock_module}):
            result = await handler.handle("/api/self-improve/worktrees", {}, http_handler)

        body = _body(result)
        assert body["total"] == 2
        assert body["worktrees"][0]["branch_name"] == "branch-1"
        assert body["worktrees"][1]["track"] == "dev"
        assert body["worktrees"][1]["assignment_id"] == "a2"
        assert body["worktrees"][0]["worktree_path"] == "/tmp/wt1"

    @pytest.mark.asyncio
    async def test_autopilot_status(self, handler, http_handler):
        mock_proc = MagicMock(returncode=0, stdout='{"sessions": []}', stderr="")
        mock_service = MagicMock()
        mock_service.run_autopilot_action.return_value = mock_proc

        with patch("aragora.worktree.WorktreeLifecycleService", return_value=mock_service):
            result = await handler.handle(
                "/api/self-improve/worktrees/autopilot/status", {}, http_handler
            )

        body = _body(result)
        assert body["action"] == "status"
        assert body["ok"] is True
        assert body["result"]["sessions"] == []

    @pytest.mark.asyncio
    async def test_autopilot_ensure(self, handler):
        mock_proc = MagicMock(
            returncode=0,
            stdout='{"ok": true, "session": {"branch": "codex/test"}}',
            stderr="",
        )
        mock_service = MagicMock()
        mock_service.run_autopilot_action.return_value = mock_proc

        with patch("aragora.worktree.WorktreeLifecycleService", return_value=mock_service):
            http = MockHTTPHandler(body={"agent": "codex-ci", "managed_dir": ".worktrees/codex-auto"})
            result = await handler.handle_post("/api/self-improve/worktrees/autopilot/ensure", {}, http)

        body = _body(result)
        assert body["action"] == "ensure"
        assert body["ok"] is True
        assert body["result"]["session"]["branch"] == "codex/test"

    @pytest.mark.asyncio
    async def test_autopilot_failure_returns_503(self, handler):
        mock_proc = MagicMock(returncode=2, stdout='{"ok": false}', stderr="merge conflict")
        mock_service = MagicMock()
        mock_service.run_autopilot_action.return_value = mock_proc

        with patch("aragora.worktree.WorktreeLifecycleService", return_value=mock_service):
            http = MockHTTPHandler(body={"managed_dir": ".worktrees/codex-auto"}, method="POST")
            result = await handler.handle_post(
                "/api/self-improve/worktrees/autopilot/reconcile",
                {},
                http,
            )

        assert _status(result) == 503
        body = _body(result)
        assert body["ok"] is False
        assert "stderr" in body


# ===========================================================================
# handle returns None for unrecognized paths
# ===========================================================================


class TestUnrecognizedPaths:
    """Tests for unrecognized paths returning None."""

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_get(self, handler, http_handler):
        result = await handler.handle("/api/self-improve/unknown", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown_post(self, handler):
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/unknown", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_cancel_without_run_id(self, handler):
        """POST to a /cancel path that doesn't have a run id."""
        http = MockHTTPHandler(body={})
        result = await handler.handle_post("/api/self-improve/cancel", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_get_for_post_only_route(self, handler, http_handler):
        """GET /api/self-improve/run should return None (it's a POST-only route)."""
        result = await handler.handle("/api/self-improve/run", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_get_coordinate(self, handler, http_handler):
        """GET /api/self-improve/coordinate should return None."""
        result = await handler.handle("/api/self-improve/coordinate", {}, http_handler)
        assert result is None


# ===========================================================================
# Store initialization
# ===========================================================================


class TestStoreInitialization:
    """Tests for the _get_store lazy loading."""

    def test_store_initially_none(self, handler):
        assert handler._store is None

    def test_get_store_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.nomic.stores.run_store": None}):
            result = handler._get_store()
        assert result is None

    def test_get_store_caches(self, handler):
        mock_store = MagicMock()
        handler._store = mock_store
        assert handler._get_store() is mock_store

    def test_get_store_os_error(self, handler):
        mock_module = MagicMock()
        mock_module.SelfImproveRunStore.side_effect = OSError("disk failure")
        with patch.dict("sys.modules", {"aragora.nomic.stores.run_store": mock_module}):
            result = handler._get_store()
        assert result is None

    def test_get_stream_server_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.server.stream.nomic_loop_stream": None}):
            result = handler._get_stream_server()
        assert result is None

    def test_get_stream_server_caches(self, handler):
        mock_stream = MagicMock()
        handler._stream_server = mock_stream
        mock_module = MagicMock()
        mock_module.NomicLoopStreamServer = MagicMock
        with patch.dict("sys.modules", {"aragora.server.stream.nomic_loop_stream": mock_module}):
            result = handler._get_stream_server()
        assert result is mock_stream


# ===========================================================================
# Handler ROUTES and class attributes
# ===========================================================================


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_contains_expected_paths(self, handler):
        expected = [
            "/api/self-improve/run",
            "/api/self-improve/start",
            "/api/self-improve/status",
            "/api/self-improve/runs",
            "/api/self-improve/runs/*",
            "/api/self-improve/runs/*/cancel",
            "/api/self-improve/history",
            "/api/self-improve/coordinate",
            "/api/self-improve/worktrees",
            "/api/self-improve/worktrees/cleanup",
            "/api/self-improve/worktrees/autopilot/status",
            "/api/self-improve/worktrees/autopilot/ensure",
            "/api/self-improve/worktrees/autopilot/reconcile",
            "/api/self-improve/worktrees/autopilot/cleanup",
            "/api/self-improve/worktrees/autopilot/maintain",
        ]
        for route in expected:
            assert route in handler.ROUTES

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "self_improve"

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) >= 20


# ===========================================================================
# Edge case: body parsing
# ===========================================================================


class TestBodyParsing:
    """Tests for body parsing edge cases."""

    @pytest.mark.asyncio
    async def test_start_run_with_null_body(self, handler, mock_store):
        """read_json_body returns None => body defaults to {}."""
        handler._store = mock_store
        http = MockHTTPHandler()
        result = await handler.handle_post("/api/self-improve/run", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_coordinate_with_null_body(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler()
        result = await handler.handle_post("/api/self-improve/coordinate", {}, http)
        assert _status(result) == 400


# ===========================================================================
# Default parameter values
# ===========================================================================


class TestDefaultParams:
    """Tests for default parameter handling in start_run and coordinate."""

    @pytest.mark.asyncio
    async def test_start_run_defaults(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test defaults"})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/run", {}, http)

        mock_store.create_run.assert_called_once()
        call_kwargs = mock_store.create_run.call_args[1]
        assert call_kwargs["mode"] == "flat"
        assert call_kwargs["max_cycles"] == 5
        assert call_kwargs["dry_run"] is False
        assert call_kwargs["tracks"] == []

    @pytest.mark.asyncio
    async def test_start_run_custom_budget(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test", "budget_limit_usd": 25.0})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/run", {}, http)

        mock_store.create_run.assert_called_once()
        call_kwargs = mock_store.create_run.call_args[1]
        assert call_kwargs["budget_limit_usd"] == 25.0

    @pytest.mark.asyncio
    async def test_start_run_with_tracks(self, handler, mock_store):
        handler._store = mock_store
        http = MockHTTPHandler(body={"goal": "Test", "tracks": ["qa", "dev"]})
        with patch.object(handler, "_execute_run", new_callable=AsyncMock):
            await handler.handle_post("/api/self-improve/run", {}, http)

        call_kwargs = mock_store.create_run.call_args[1]
        assert call_kwargs["tracks"] == ["qa", "dev"]
