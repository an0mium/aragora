"""Tests for autonomous self-improvement REST API handler.

Tests cover:
- Route matching (can_handle)
- POST /api/v1/autonomous/improve - create a run and return run_id
- GET  /api/v1/autonomous/improve - list all runs
- GET  /api/v1/autonomous/improve/{run_id} - retrieve run status
- Permission enforcement via RBAC
- Budget limit validation (min/max bounds, invalid types)
- Invalid goal handling (missing, empty, non-string)
- Track validation
- require_approval validation
- Store unavailability (503)
- Run not found (404)
- Response envelope format ({"data": ...})
- Background execution fallback (pipeline -> orchestrator)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.stores.run_store import RunStatus, SelfImproveRun, SelfImproveRunStore
from aragora.server.handlers.autonomous.improve import (
    AutonomousImproveHandler,
    MAX_BUDGET_USD,
    MIN_BUDGET_USD,
    _active_improve_tasks,
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


def _parse_data(result) -> dict:
    """Extract the 'data' envelope from a HandlerResult."""
    body = _body(result)
    return body.get("data", body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_server_context():
    """Minimal server context for handler tests."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "nomic_dir": None,
    }


@pytest.fixture
def handler(mock_server_context):
    """Create an AutonomousImproveHandler instance."""
    return AutonomousImproveHandler(server_context=mock_server_context)


@pytest.fixture
def mock_store():
    """Create a mock SelfImproveRunStore."""
    store = MagicMock(spec=SelfImproveRunStore)
    return store


@pytest.fixture
def sample_run():
    """Create a sample running SelfImproveRun."""
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
def mock_http_handler():
    """Create a mock HTTP handler factory."""

    def _create(
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> MagicMock:
        mock = MagicMock()
        mock.command = method
        body_bytes = json.dumps(body or {}).encode()
        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body_bytes)
        mock.headers = {"Content-Length": str(len(body_bytes))}
        mock.client_address = ("127.0.0.1", 12345)
        return mock

    return _create


# ---------------------------------------------------------------------------
# can_handle routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test route matching."""

    def test_matches_improve_endpoint(self, handler):
        assert handler.can_handle("/api/v1/autonomous/improve") is True

    def test_matches_improve_with_run_id(self, handler):
        assert handler.can_handle("/api/v1/autonomous/improve/abc123") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_too_deep_path(self, handler):
        assert handler.can_handle("/api/v1/autonomous/improve/abc/extra") is False

    def test_rejects_other_autonomous_paths(self, handler):
        assert handler.can_handle("/api/v1/autonomous/triggers") is False

    def test_matches_without_version_prefix(self, handler):
        assert handler.can_handle("/api/autonomous/improve") is True


# ---------------------------------------------------------------------------
# _extract_run_id helper
# ---------------------------------------------------------------------------


class TestExtractRunId:
    """Test run ID extraction from path."""

    def test_extracts_run_id(self):
        assert _extract_run_id("/api/autonomous/improve/abc123") == "abc123"

    def test_returns_none_for_list_path(self):
        assert _extract_run_id("/api/autonomous/improve") is None

    def test_returns_none_for_short_path(self):
        assert _extract_run_id("/api/autonomous") is None


# ---------------------------------------------------------------------------
# POST /api/v1/autonomous/improve - create run
# ---------------------------------------------------------------------------


class TestCreateRun:
    """Test POST endpoint for creating improvement runs."""

    @pytest.mark.asyncio
    async def test_creates_run_and_returns_run_id(
        self, handler_with_store, mock_store, mock_http_handler
    ):
        """POST creates a run and returns run_id with queued status."""
        mock_run = SelfImproveRun(
            run_id="new-run-1",
            goal="Improve performance",
            status=RunStatus.PENDING,
        )
        mock_store.create_run.return_value = mock_run
        mock_store.update_run.return_value = mock_run

        http = mock_http_handler(
            method="POST",
            body={"goal": "Improve performance"},
        )

        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)

        assert _status(result) == 202
        data = _parse_data(result)
        assert data["run_id"] == "new-run-1"
        assert data["status"] == "queued"
        mock_store.create_run.assert_called_once()

        # Clean up background task
        task = _active_improve_tasks.pop("new-run-1", None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_missing_goal_returns_400(self, handler_with_store, mock_http_handler):
        """POST without goal returns 400."""
        http = mock_http_handler(method="POST", body={})
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400
        assert "goal" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_goal_returns_400(self, handler_with_store, mock_http_handler):
        """POST with empty string goal returns 400."""
        http = mock_http_handler(method="POST", body={"goal": "   "})
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_non_string_goal_returns_400(self, handler_with_store, mock_http_handler):
        """POST with non-string goal returns 400."""
        http = mock_http_handler(method="POST", body={"goal": 42})
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_budget_limit_accepted(self, handler_with_store, mock_store, mock_http_handler):
        """POST with valid budget_limit succeeds."""
        mock_run = SelfImproveRun(run_id="budg-1", goal="test", budget_limit_usd=5.0)
        mock_store.create_run.return_value = mock_run
        mock_store.update_run.return_value = mock_run

        http = mock_http_handler(
            method="POST",
            body={"goal": "Improve docs", "budget_limit": 5.0},
        )
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 202

        task = _active_improve_tasks.pop("budg-1", None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_budget_too_high_returns_400(self, handler_with_store, mock_http_handler):
        """POST with budget above MAX returns 400."""
        http = mock_http_handler(
            method="POST",
            body={"goal": "test", "budget_limit": MAX_BUDGET_USD + 1},
        )
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400
        assert "budget_limit" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_budget_too_low_returns_400(self, handler_with_store, mock_http_handler):
        """POST with budget below MIN returns 400."""
        http = mock_http_handler(
            method="POST",
            body={"goal": "test", "budget_limit": 0.001},
        )
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_budget_type_returns_400(self, handler_with_store, mock_http_handler):
        """POST with non-numeric budget returns 400."""
        http = mock_http_handler(
            method="POST",
            body={"goal": "test", "budget_limit": "expensive"},
        )
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_tracks_returns_400(self, handler_with_store, mock_http_handler):
        """POST with non-list tracks returns 400."""
        http = mock_http_handler(
            method="POST",
            body={"goal": "test", "tracks": "qa"},
        )
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_require_approval_returns_400(
        self, handler_with_store, mock_http_handler
    ):
        """POST with non-boolean require_approval returns 400."""
        http = mock_http_handler(
            method="POST",
            body={"goal": "test", "require_approval": "yes"},
        )
        result = await handler_with_store.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_store_unavailable_returns_503(self, handler, mock_http_handler):
        """POST when store is unavailable returns 503."""
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            http = mock_http_handler(
                method="POST",
                body={"goal": "test"},
            )
            result = await handler.handle_post("/api/v1/autonomous/improve", {}, http)
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/autonomous/improve/{run_id} - get run status
# ---------------------------------------------------------------------------


class TestGetRun:
    """Test GET endpoint for retrieving a specific run."""

    @pytest.mark.asyncio
    async def test_returns_run_with_data_envelope(self, handler_with_store, mock_store, sample_run):
        """GET returns run in {"data": {...}} envelope."""
        mock_store.get_run.return_value = sample_run

        result = await handler_with_store.handle(
            "/api/v1/autonomous/improve/abc12345", {}, MagicMock()
        )

        assert _status(result) == 200
        data = _parse_data(result)
        assert data["run_id"] == "abc12345"
        assert data["status"] == "running"
        assert data["goal"] == "Improve test coverage"
        assert "progress" in data
        assert data["result"] is None  # not completed yet

    @pytest.mark.asyncio
    async def test_completed_run_includes_result(
        self, handler_with_store, mock_store, completed_run
    ):
        """GET on completed run includes result object."""
        mock_store.get_run.return_value = completed_run

        result = await handler_with_store.handle(
            "/api/v1/autonomous/improve/def67890", {}, MagicMock()
        )

        data = _parse_data(result)
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["summary"] == "All subtasks completed successfully"

    @pytest.mark.asyncio
    async def test_progress_has_percent_complete(
        self, handler_with_store, mock_store, completed_run
    ):
        """GET returns progress with percent_complete."""
        mock_store.get_run.return_value = completed_run

        result = await handler_with_store.handle(
            "/api/v1/autonomous/improve/def67890", {}, MagicMock()
        )

        data = _parse_data(result)
        assert data["progress"]["percent_complete"] == 100.0

    @pytest.mark.asyncio
    async def test_run_not_found_returns_404(self, handler_with_store, mock_store):
        """GET for non-existent run returns 404."""
        mock_store.get_run.return_value = None

        result = await handler_with_store.handle(
            "/api/v1/autonomous/improve/nonexistent", {}, MagicMock()
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET /api/v1/autonomous/improve - list runs
# ---------------------------------------------------------------------------


class TestListRuns:
    """Test GET endpoint for listing all runs."""

    @pytest.mark.asyncio
    async def test_returns_runs_in_data_envelope(
        self, handler_with_store, mock_store, sample_run, completed_run
    ):
        """GET list returns runs in {"data": {"runs": [...]}} envelope."""
        mock_store.list_runs.return_value = [sample_run, completed_run]

        result = await handler_with_store.handle("/api/v1/autonomous/improve", {}, MagicMock())

        assert _status(result) == 200
        data = _parse_data(result)
        assert "runs" in data
        assert len(data["runs"]) == 2
        assert "total" in data

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_array(self, handler_with_store, mock_store):
        """GET list with no runs returns empty array."""
        mock_store.list_runs.return_value = []

        result = await handler_with_store.handle("/api/v1/autonomous/improve", {}, MagicMock())

        data = _parse_data(result)
        assert data["runs"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_pagination_params_forwarded(self, handler_with_store, mock_store):
        """GET list forwards limit and offset to store."""
        mock_store.list_runs.return_value = []

        await handler_with_store.handle(
            "/api/v1/autonomous/improve",
            {"limit": "10", "offset": "5"},
            MagicMock(),
        )

        mock_store.list_runs.assert_called_once_with(limit=10, offset=5, status=None)

    @pytest.mark.asyncio
    async def test_store_unavailable_returns_503(self, handler):
        """GET list when store unavailable returns 503."""
        handler._store = None
        with patch.object(handler, "_get_store", return_value=None):
            result = await handler.handle("/api/v1/autonomous/improve", {}, MagicMock())
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# RBAC permission checks
# ---------------------------------------------------------------------------


class TestPermissions:
    """Test RBAC permission enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_requires_autonomous_read(self, handler, mock_http_handler):
        """GET endpoints require autonomous:read permission."""
        # Without mocked auth, get_auth_context should fail -> 401
        http = mock_http_handler()
        result = await handler.handle("/api/v1/autonomous/improve", {}, http)
        # Should be 401 or 403 (not 200/503)
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_post_requires_autonomous_improve(self, handler, mock_http_handler):
        """POST endpoint requires autonomous:improve permission."""
        http = mock_http_handler(
            method="POST",
            body={"goal": "test"},
        )
        result = await handler.handle_post("/api/v1/autonomous/improve", {}, http)
        assert _status(result) in (401, 403)


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


class TestBackgroundExecution:
    """Test background execution logic."""

    @pytest.mark.asyncio
    async def test_execute_run_pipeline_success(self, handler_with_store, mock_store):
        """Background execution via SelfImprovePipeline updates store on success."""
        mock_result = MagicMock()
        mock_result.subtasks_completed = 3
        mock_result.subtasks_total = 5
        mock_result.subtasks_failed = 2

        mock_pipeline = AsyncMock()
        mock_pipeline.run = AsyncMock(return_value=mock_result)

        with patch.dict(
            "sys.modules",
            {
                "aragora.nomic.self_improve": MagicMock(
                    SelfImproveConfig=MagicMock(return_value=MagicMock()),
                    SelfImprovePipeline=MagicMock(return_value=mock_pipeline),
                ),
            },
        ):
            await handler_with_store._execute_run(
                run_id="exec-1",
                goal="Improve tests",
                tracks=["qa"],
                budget_limit=5.0,
                require_approval=False,
            )

        # Verify store was updated with completed status
        mock_store.update_run.assert_called()
        update_calls = [c for c in mock_store.update_run.call_args_list if c.args[0] == "exec-1"]
        assert len(update_calls) >= 1
        last_call = update_calls[-1]
        assert last_call.kwargs.get("status") in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_execute_run_falls_back_to_orchestrator(self, handler_with_store, mock_store):
        """Background execution falls back to HardenedOrchestrator when pipeline unavailable."""
        mock_orch_result = MagicMock()
        mock_orch_result.success = True
        mock_orch_result.total_subtasks = 4
        mock_orch_result.completed_subtasks = 4
        mock_orch_result.failed_subtasks = 0
        mock_orch_result.summary = "All done"
        mock_orch_result.error = None

        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_goal_coordinated = AsyncMock(return_value=mock_orch_result)

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.nomic.self_improve": None,  # Make import fail
                },
            ),
            patch(
                "aragora.nomic.hardened_orchestrator.HardenedOrchestrator",
                return_value=mock_orchestrator,
            ),
        ):
            await handler_with_store._execute_run(
                run_id="fallback-1",
                goal="Fix bugs",
                tracks=None,
                budget_limit=None,
            )

        mock_store.update_run.assert_called()

    @pytest.mark.asyncio
    async def test_execute_run_cleans_up_task_on_failure(self, handler_with_store, mock_store):
        """Background task is removed from _active_improve_tasks even on double failure."""
        _active_improve_tasks["cleanup-test"] = MagicMock()

        # Both pipeline and orchestrator fail
        with (
            patch.dict(
                "sys.modules",
                {"aragora.nomic.self_improve": None},
            ),
            patch.dict(
                "sys.modules",
                {"aragora.nomic.hardened_orchestrator": None},
            ),
        ):
            await handler_with_store._execute_run(
                run_id="cleanup-test",
                goal="test",
                tracks=None,
                budget_limit=None,
            )

        # Task should be cleaned up
        assert "cleanup-test" not in _active_improve_tasks

    @pytest.mark.asyncio
    async def test_execute_run_handles_cancellation(self, handler_with_store, mock_store):
        """Background execution handles CancelledError gracefully."""
        mock_pipeline = AsyncMock()
        mock_pipeline.run = AsyncMock(side_effect=asyncio.CancelledError)

        with patch.dict(
            "sys.modules",
            {
                "aragora.nomic.self_improve": MagicMock(
                    SelfImproveConfig=MagicMock(return_value=MagicMock()),
                    SelfImprovePipeline=MagicMock(return_value=mock_pipeline),
                ),
            },
        ):
            await handler_with_store._execute_run(
                run_id="cancel-1",
                goal="test",
                tracks=None,
                budget_limit=None,
            )

        # Should be updated to cancelled
        mock_store.update_run.assert_called()
        last_call = mock_store.update_run.call_args_list[-1]
        assert last_call.kwargs.get("status") == "cancelled"
