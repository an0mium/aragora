"""Tests for the self-improvement REST API handler.

Tests cover:
- Route matching (can_handle)
- Status endpoint (GET /api/self-improve/status)
- Run listing (GET /api/self-improve/runs)
- Run details (GET /api/self-improve/runs/:id)
- Run creation (POST /api/self-improve/run and /start)
- Run cancellation (POST /api/self-improve/runs/:id/cancel)
- History alias (GET /api/self-improve/history)
- Dry run mode with SelfImprovePipeline
- Config overrides (scan_mode, quick_mode, budget_limit_usd)
- Hierarchical coordination (POST /api/self-improve/coordinate)
- Worktree management
- Store unavailability
- Path extraction
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.stores.run_store import RunStatus, SelfImproveRun, SelfImproveRunStore
from aragora.server.handlers.self_improve import SelfImproveHandler, _extract_run_id


# ============================================================================
# Fixtures
# ============================================================================


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
def mock_http_handler():
    """Create a mock HTTP handler factory."""
    import json as _json

    def _create(
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> MagicMock:
        mock = MagicMock()
        mock.command = method
        body_bytes = _json.dumps(body or {}).encode()
        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body_bytes)
        mock.headers = {"Content-Length": str(len(body_bytes))}
        mock.client_address = ("127.0.0.1", 12345)
        return mock

    return _create


@pytest.fixture(autouse=True)
def _clear_active_tasks():
    """Clear active tasks between tests."""
    from aragora.server.handlers.self_improve import _active_tasks

    _active_tasks.clear()
    yield
    _active_tasks.clear()


def _parse_response(result) -> dict[str, Any]:
    """Parse JSON from a HandlerResult."""
    import json as _json

    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return _json.loads(body) if body else {}


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


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_handles_run(self, handler):
        assert handler.can_handle("/api/self-improve/run") is True

    def test_handles_start(self, handler):
        assert handler.can_handle("/api/self-improve/start") is True

    def test_handles_status(self, handler):
        assert handler.can_handle("/api/self-improve/status") is True

    def test_handles_runs(self, handler):
        assert handler.can_handle("/api/self-improve/runs") is True

    def test_handles_history(self, handler):
        assert handler.can_handle("/api/self-improve/history") is True

    def test_handles_coordinate(self, handler):
        assert handler.can_handle("/api/self-improve/coordinate") is True

    def test_handles_run_by_id(self, handler):
        assert handler.can_handle("/api/self-improve/runs/abc123") is True

    def test_handles_cancel(self, handler):
        assert handler.can_handle("/api/self-improve/runs/abc123/cancel") is True

    def test_rejects_unrelated(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_handles_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/self-improve/runs") is True

    def test_handles_versioned_status(self, handler):
        assert handler.can_handle("/api/v1/self-improve/status") is True

    def test_handles_versioned_coordinate(self, handler):
        assert handler.can_handle("/api/v1/self-improve/coordinate") is True


# ============================================================================
# GET /api/self-improve/status Tests
# ============================================================================


class TestGetStatus:
    """Tests for the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_idle(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/self-improve/status", {}, mock_http_handler()
        )
        body = _parse_response(result)
        assert result.status_code == 200
        assert body["state"] == "idle"
        assert body["active_runs"] == 0
        assert body["runs"] == []

    @pytest.mark.asyncio
    async def test_status_running(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        from aragora.server.handlers.self_improve import _active_tasks

        # Simulate an active task
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        _active_tasks["abc12345"] = mock_task

        mock_store.get_run.return_value = sample_run

        result = await handler_with_store.handle(
            "/api/self-improve/status", {}, mock_http_handler()
        )
        body = _parse_response(result)
        assert result.status_code == 200
        assert body["state"] == "running"
        assert body["active_runs"] == 1
        assert len(body["runs"]) == 1
        assert body["runs"][0]["run_id"] == "abc12345"

    @pytest.mark.asyncio
    async def test_status_ignores_done_tasks(self, handler, mock_http_handler):
        from aragora.server.handlers.self_improve import _active_tasks

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = True
        _active_tasks["done123"] = mock_task

        result = await handler.handle(
            "/api/self-improve/status", {}, mock_http_handler()
        )
        body = _parse_response(result)
        assert body["state"] == "idle"

    @pytest.mark.asyncio
    async def test_status_versioned_path(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/self-improve/status", {}, mock_http_handler()
        )
        body = _parse_response(result)
        assert body["state"] == "idle"


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
        body = _parse_response(result)
        assert result.status_code == 200
        assert len(body["runs"]) == 1
        assert body["runs"][0]["run_id"] == "abc12345"

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.list_runs.return_value = []
        result = await handler_with_store.handle(
            "/api/self-improve/runs", {}, mock_http_handler()
        )
        body = _parse_response(result)
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
        body = _parse_response(result)
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
        body = _parse_response(result)
        assert result.status_code == 200
        assert body["run_id"] == "abc12345"
        assert body["goal"] == "Improve test coverage"

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.get_run.return_value = None
        result = await handler_with_store.handle(
            "/api/self-improve/runs/nonexistent", {}, mock_http_handler()
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_completed_run(self, handler_with_store, mock_store, completed_run, mock_http_handler):
        mock_store.get_run.return_value = completed_run
        result = await handler_with_store.handle(
            "/api/self-improve/runs/def67890", {}, mock_http_handler()
        )
        body = _parse_response(result)
        assert body["status"] == "completed"
        assert body["total_subtasks"] == 5


# ============================================================================
# POST /api/self-improve/run Tests
# ============================================================================


class TestStartRun:
    """Tests for starting a self-improvement run."""

    @pytest.mark.asyncio
    async def test_start_run_via_run_endpoint(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        """POST /api/self-improve/run starts a cycle."""
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        http = mock_http_handler(method="POST", body={"goal": "Improve tests"})

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "Improve tests"}):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/run", {}, http
                )

        body = _parse_response(result)
        assert result.status_code == 202
        assert body["status"] == "started"

    @pytest.mark.asyncio
    async def test_start_run_via_start_endpoint(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        """POST /api/self-improve/start also works (legacy alias)."""
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        http = mock_http_handler(method="POST", body={"goal": "Fix bugs"})

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "Fix bugs"}):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/start", {}, http
                )

        assert result.status_code == 202

    @pytest.mark.asyncio
    async def test_start_run_missing_goal(self, handler_with_store, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/run", {}, http
            )

        assert result.status_code == 400
        body = _parse_response(result)
        assert "goal" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_start_run_empty_goal(self, handler_with_store, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "   "}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/run", {}, http
            )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_start_run_invalid_mode(self, handler_with_store, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "Test", "mode": "invalid"}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/run", {}, http
            )

        assert result.status_code == 400
        body = _parse_response(result)
        assert "mode" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_start_run_with_config_overrides(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        """Config overrides (scan_mode, quick_mode, budget_limit) are passed through."""
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        body_data = {
            "goal": "Improve coverage",
            "scan_mode": False,
            "quick_mode": True,
            "budget_limit_usd": 25.0,
            "require_approval": True,
        }
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock) as mock_exec:
                result = await handler_with_store.handle_post(
                    "/api/self-improve/run", {}, http
                )

        assert result.status_code == 202
        # Verify config overrides were passed to _execute_run
        mock_exec.assert_called_once()
        call_kwargs = mock_exec.call_args
        # Positional args: run_id, goal, tracks, mode, budget_limit, max_cycles
        # Keyword args: scan_mode, quick_mode, require_approval
        assert call_kwargs.kwargs["scan_mode"] is False
        assert call_kwargs.kwargs["quick_mode"] is True
        assert call_kwargs.kwargs["require_approval"] is True

    @pytest.mark.asyncio
    async def test_start_run_store_unavailable(self, handler, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch.object(handler, "_get_store", return_value=None):
            with patch.object(handler, "read_json_body", return_value={"goal": "Test"}):
                result = await handler.handle_post(
                    "/api/self-improve/run", {}, http
                )

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_start_run_versioned_path(self, handler_with_store, mock_store, sample_run, mock_http_handler):
        mock_store.create_run.return_value = sample_run
        mock_store.update_run.return_value = sample_run

        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={"goal": "Test"}):
            with patch.object(handler_with_store, "_execute_run", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/v1/self-improve/run", {}, http
                )

        assert result.status_code == 202


# ============================================================================
# Dry Run Tests
# ============================================================================


class TestDryRun:
    """Tests for dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_plan(self, handler_with_store, mock_store, mock_http_handler):
        dry_run = SelfImproveRun(run_id="dry12345", goal="Refactor module", dry_run=True)
        mock_store.create_run.return_value = dry_run
        mock_store.update_run.return_value = dry_run

        body_data = {"goal": "Refactor module", "dry_run": True}
        http = mock_http_handler(method="POST")

        mock_plan = {"goal": "Refactor module", "subtasks": [], "tracks": []}

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_generate_plan", new_callable=AsyncMock, return_value=mock_plan):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/run", {}, http
                )

        body = _parse_response(result)
        assert result.status_code == 200
        assert body["status"] == "preview"
        assert body["plan"] == mock_plan

    @pytest.mark.asyncio
    async def test_dry_run_passes_config_to_generate_plan(self, handler_with_store, mock_store, mock_http_handler):
        dry_run = SelfImproveRun(run_id="dry123", goal="Test", dry_run=True)
        mock_store.create_run.return_value = dry_run
        mock_store.update_run.return_value = dry_run

        body_data = {"goal": "Test", "dry_run": True, "scan_mode": False, "quick_mode": True}
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_generate_plan", new_callable=AsyncMock, return_value={}) as mock_gen:
                await handler_with_store.handle_post(
                    "/api/self-improve/run", {}, http
                )

        mock_gen.assert_called_once_with("Test", None, False, True)


# ============================================================================
# POST /api/self-improve/coordinate Tests
# ============================================================================


class TestCoordinate:
    """Tests for the hierarchical coordination endpoint."""

    @pytest.mark.asyncio
    async def test_coordinate_success(self, handler_with_store, mock_store, mock_http_handler):
        coord_run = SelfImproveRun(run_id="coord123", goal="Refactor", mode="hierarchical")
        mock_store.create_run.return_value = coord_run
        mock_store.update_run.return_value = coord_run

        body_data = {"goal": "Refactor modules"}
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_execute_coordination", new_callable=AsyncMock):
                result = await handler_with_store.handle_post(
                    "/api/self-improve/coordinate", {}, http
                )

        body = _parse_response(result)
        assert result.status_code == 202
        assert body["status"] == "coordinating"
        assert body["mode"] == "hierarchical"
        assert body["run_id"] == "coord123"

    @pytest.mark.asyncio
    async def test_coordinate_missing_goal(self, handler_with_store, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/coordinate", {}, http
            )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_coordinate_store_unavailable(self, handler, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch.object(handler, "_get_store", return_value=None):
            with patch.object(handler, "read_json_body", return_value={"goal": "Test"}):
                result = await handler.handle_post(
                    "/api/self-improve/coordinate", {}, http
                )

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_coordinate_with_config(self, handler_with_store, mock_store, mock_http_handler):
        coord_run = SelfImproveRun(run_id="coord456", goal="Test", mode="hierarchical")
        mock_store.create_run.return_value = coord_run
        mock_store.update_run.return_value = coord_run

        body_data = {
            "goal": "Test goal",
            "max_cycles": 5,
            "quality_threshold": 0.8,
            "max_parallel_workers": 2,
        }
        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value=body_data):
            with patch.object(handler_with_store, "_execute_coordination", new_callable=AsyncMock) as mock_exec:
                result = await handler_with_store.handle_post(
                    "/api/self-improve/coordinate", {}, http
                )

        assert result.status_code == 202
        mock_exec.assert_called_once()
        call_kwargs = mock_exec.call_args.kwargs
        assert call_kwargs["max_cycles"] == 5
        assert call_kwargs["quality_threshold"] == 0.8
        assert call_kwargs["max_parallel"] == 2


# ============================================================================
# Execute Coordination Tests
# ============================================================================


class TestExecuteCoordination:
    """Tests for background coordination execution."""

    @pytest.mark.asyncio
    async def test_execute_coordination_success(self, handler_with_store, mock_store):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.cycles_used = 2
        mock_report = MagicMock()
        mock_report.success = True
        mock_result.worker_reports = [mock_report]

        with patch(
            "aragora.nomic.hierarchical_coordinator.HierarchicalCoordinator"
        ) as MockCoord:
            mock_instance = MockCoord.return_value
            mock_instance.coordinate = AsyncMock(return_value=mock_result)
            await handler_with_store._execute_coordination("run123", "Test goal", None)

        mock_store.update_run.assert_called()
        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "completed"
        assert last_call[1]["total_subtasks"] == 1
        assert last_call[1]["completed_subtasks"] == 1

    @pytest.mark.asyncio
    async def test_execute_coordination_failure(self, handler_with_store, mock_store):
        with patch(
            "aragora.nomic.hierarchical_coordinator.HierarchicalCoordinator"
        ) as MockCoord:
            mock_instance = MockCoord.return_value
            mock_instance.coordinate = AsyncMock(side_effect=RuntimeError("Boom"))
            await handler_with_store._execute_coordination("run123", "Test", None)

        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "failed"
        assert last_call[1]["error"] == "Coordination failed"

    @pytest.mark.asyncio
    async def test_execute_coordination_cancelled(self, handler_with_store, mock_store):
        with patch(
            "aragora.nomic.hierarchical_coordinator.HierarchicalCoordinator"
        ) as MockCoord:
            mock_instance = MockCoord.return_value
            mock_instance.coordinate = AsyncMock(side_effect=asyncio.CancelledError())
            await handler_with_store._execute_coordination("run123", "Test", None)

        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_execute_coordination_import_error(self, handler_with_store, mock_store):
        with patch(
            "aragora.nomic.hierarchical_coordinator.HierarchicalCoordinator",
            side_effect=ImportError("No coordinator"),
        ):
            await handler_with_store._execute_coordination("run123", "Test", None)

        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_coordination_cleans_active_tasks(self, handler_with_store, mock_store):
        from aragora.server.handlers.self_improve import _active_tasks

        _active_tasks["run123"] = MagicMock()

        with patch(
            "aragora.nomic.hierarchical_coordinator.HierarchicalCoordinator"
        ) as MockCoord:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.cycles_used = 1
            mock_result.worker_reports = []
            mock_instance = MockCoord.return_value
            mock_instance.coordinate = AsyncMock(return_value=mock_result)
            await handler_with_store._execute_coordination("run123", "Test", None)

        assert "run123" not in _active_tasks


# ============================================================================
# POST /api/self-improve/runs/:id/cancel Tests
# ============================================================================


class TestCancelRun:
    """Tests for cancelling a run."""

    @pytest.mark.asyncio
    async def test_cancel_running_run(self, handler_with_store, mock_store, mock_http_handler):
        cancelled_run = SelfImproveRun(run_id="abc12345", goal="Test", status=RunStatus.CANCELLED)
        mock_store.cancel_run.return_value = cancelled_run

        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/runs/abc12345/cancel", {}, http
            )

        body = _parse_response(result)
        assert result.status_code == 200
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, handler_with_store, mock_store, mock_http_handler):
        mock_store.cancel_run.return_value = None

        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/runs/nonexistent/cancel", {}, http
            )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_stops_active_task(self, handler_with_store, mock_store, mock_http_handler):
        from aragora.server.handlers.self_improve import _active_tasks

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        _active_tasks["abc12345"] = mock_task

        cancelled_run = SelfImproveRun(run_id="abc12345", goal="Test", status=RunStatus.CANCELLED)
        mock_store.cancel_run.return_value = cancelled_run

        http = mock_http_handler(method="POST")

        with patch.object(handler_with_store, "read_json_body", return_value={}):
            result = await handler_with_store.handle_post(
                "/api/self-improve/runs/abc12345/cancel", {}, http
            )

        mock_task.cancel.assert_called_once()
        assert "abc12345" not in _active_tasks


# ============================================================================
# Generate Plan Tests
# ============================================================================


class TestGeneratePlan:
    """Tests for plan generation with SelfImprovePipeline integration."""

    @pytest.mark.asyncio
    async def test_generate_plan_via_pipeline(self, handler):
        """Uses SelfImprovePipeline.dry_run() when available."""
        mock_plan = {
            "objective": "Test",
            "goals": [],
            "subtasks": [],
            "config": {"use_worktrees": True},
        }

        with patch("aragora.nomic.self_improve.SelfImprovePipeline") as MockPipeline:
            mock_instance = MockPipeline.return_value
            mock_instance.dry_run = AsyncMock(return_value=mock_plan)
            plan = await handler._generate_plan("Test", ["qa"], scan_mode=True, quick_mode=False)

        assert plan["objective"] == "Test"
        assert plan["tracks"] == ["qa"]

    @pytest.mark.asyncio
    async def test_generate_plan_pipeline_config(self, handler):
        """SelfImproveConfig receives scan_mode and quick_mode."""
        mock_plan = {"objective": "X", "goals": [], "subtasks": [], "config": {}}

        with patch("aragora.nomic.self_improve.SelfImprovePipeline") as MockPipeline:
            with patch("aragora.nomic.self_improve.SelfImproveConfig") as MockConfig:
                mock_instance = MockPipeline.return_value
                mock_instance.dry_run = AsyncMock(return_value=mock_plan)
                await handler._generate_plan("X", None, scan_mode=False, quick_mode=True)

        MockConfig.assert_called_once_with(scan_mode=False, quick_mode=True)

    @pytest.mark.asyncio
    async def test_generate_plan_fallback_to_decomposer(self, handler):
        """Falls back to TaskDecomposer if SelfImprovePipeline unavailable."""
        mock_result = MagicMock()
        mock_subtask = MagicMock()
        mock_subtask.description = "Subtask 1"
        mock_subtask.track = "qa"
        mock_subtask.priority = 1
        mock_result.subtasks = [mock_subtask]
        mock_result.complexity_score = 0.7

        with patch("aragora.nomic.self_improve.SelfImprovePipeline", side_effect=ImportError):
            with patch("aragora.nomic.task_decomposer.TaskDecomposer") as MockDecomposer:
                MockDecomposer.return_value.analyze.return_value = mock_result
                plan = await handler._generate_plan("Test", ["dev"])

        assert plan["goal"] == "Test"
        assert len(plan["subtasks"]) == 1
        assert plan["subtasks"][0]["description"] == "Subtask 1"

    @pytest.mark.asyncio
    async def test_generate_plan_all_unavailable(self, handler):
        """Returns error dict when both pipeline and decomposer unavailable."""
        with patch("aragora.nomic.self_improve.SelfImprovePipeline", side_effect=ImportError):
            with patch("aragora.nomic.task_decomposer.TaskDecomposer", side_effect=ImportError):
                plan = await handler._generate_plan("Test", None)

        assert plan["goal"] == "Test"
        assert plan["subtasks"] == []
        assert "error" in plan


# ============================================================================
# Execute Run Tests
# ============================================================================


class TestExecuteRun:
    """Tests for background run execution."""

    @pytest.mark.asyncio
    async def test_execute_run_via_pipeline(self, handler_with_store, mock_store):
        """Uses SelfImprovePipeline for flat mode."""
        mock_result = MagicMock()
        mock_result.subtasks_total = 3
        mock_result.subtasks_completed = 3
        mock_result.subtasks_failed = 0

        with patch("aragora.nomic.self_improve.SelfImprovePipeline") as MockPipeline:
            with patch("aragora.nomic.self_improve.SelfImproveConfig"):
                mock_instance = MockPipeline.return_value
                mock_instance.run = AsyncMock(return_value=mock_result)
                await handler_with_store._execute_run(
                    "run123", "Test goal", None, "flat", 10.0, 5,
                    scan_mode=True, quick_mode=False,
                )

        mock_store.update_run.assert_called()
        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "completed"
        assert last_call[1]["total_subtasks"] == 3

    @pytest.mark.asyncio
    async def test_execute_run_pipeline_fallback(self, handler_with_store, mock_store):
        """Falls back to HardenedOrchestrator when pipeline not available."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_subtasks = 2
        mock_result.completed_subtasks = 2
        mock_result.failed_subtasks = 0
        mock_result.summary = "Done"
        mock_result.error = None

        with patch("aragora.nomic.self_improve.SelfImprovePipeline", side_effect=ImportError):
            with patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as MockOrch:
                mock_instance = MockOrch.return_value
                mock_instance.execute_goal_coordinated = AsyncMock(return_value=mock_result)
                await handler_with_store._execute_run(
                    "run123", "Test", None, "flat", None, 5,
                )

        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_run_hierarchical_delegates(self, handler_with_store, mock_store):
        """Hierarchical mode delegates to _execute_coordination."""
        with patch.object(
            handler_with_store, "_execute_coordination", new_callable=AsyncMock
        ) as mock_coord:
            await handler_with_store._execute_run(
                "run123", "Test", None, "hierarchical", None, 3,
            )

        mock_coord.assert_called_once_with("run123", "Test", None, max_cycles=3)

    @pytest.mark.asyncio
    async def test_execute_run_cancelled(self, handler_with_store, mock_store):
        with patch("aragora.nomic.self_improve.SelfImprovePipeline") as MockPipeline:
            with patch("aragora.nomic.self_improve.SelfImproveConfig"):
                mock_instance = MockPipeline.return_value
                mock_instance.run = AsyncMock(side_effect=asyncio.CancelledError())
                await handler_with_store._execute_run("run123", "Test", None, "flat", None, 5)

        last_call = mock_store.update_run.call_args
        assert last_call[1]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_execute_run_store_unavailable(self, handler):
        with patch.object(handler, "_get_store", return_value=None):
            await handler._execute_run("run123", "Test", None, "flat", None, 5)
        # Should return early without error


# ============================================================================
# Store Init Tests
# ============================================================================


class TestStoreInit:
    """Tests for lazy store initialization."""

    def test_get_store_lazy_init(self, handler):
        with patch("aragora.nomic.stores.run_store.SelfImproveRunStore") as MockStore:
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


# ============================================================================
# Worktree Endpoint Tests
# ============================================================================


class TestListWorktrees:
    """Tests for GET /api/self-improve/worktrees."""

    @pytest.mark.asyncio
    async def test_list_worktrees_success(self, handler, mock_http_handler):
        mock_wt = MagicMock()
        mock_wt.branch_name = "dev/qa-coverage-001"
        mock_wt.worktree_path = "/tmp/worktrees/qa-coverage"
        mock_wt.track = "qa"
        mock_wt.created_at = datetime(2026, 2, 15, 10, 0, 0, tzinfo=timezone.utc)
        mock_wt.assignment_id = "assign-001"

        with patch("aragora.nomic.branch_coordinator.BranchCoordinator") as MockCoord:
            MockCoord.return_value.list_worktrees.return_value = [mock_wt]
            result = await handler.handle(
                "/api/self-improve/worktrees", {}, mock_http_handler()
            )

        body = _parse_response(result)
        assert result.status_code == 200
        assert body["total"] == 1
        assert body["worktrees"][0]["branch_name"] == "dev/qa-coverage-001"

    @pytest.mark.asyncio
    async def test_list_worktrees_import_error(self, handler, mock_http_handler):
        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator",
            side_effect=ImportError("No coordinator"),
        ):
            result = await handler.handle(
                "/api/self-improve/worktrees", {}, mock_http_handler()
            )

        body = _parse_response(result)
        assert body["worktrees"] == []
        assert body["total"] == 0


class TestCleanupWorktrees:
    """Tests for POST /api/self-improve/worktrees/cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_success(self, handler, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch("aragora.nomic.branch_coordinator.BranchCoordinator") as MockCoord:
            MockCoord.return_value.cleanup_all_worktrees.return_value = 3
            with patch.object(handler, "read_json_body", return_value={}):
                result = await handler.handle_post(
                    "/api/self-improve/worktrees/cleanup", {}, http
                )

        body = _parse_response(result)
        assert result.status_code == 200
        assert body["removed"] == 3

    @pytest.mark.asyncio
    async def test_cleanup_import_error(self, handler, mock_http_handler):
        http = mock_http_handler(method="POST")

        with patch(
            "aragora.nomic.branch_coordinator.BranchCoordinator",
            side_effect=ImportError("No coordinator"),
        ):
            with patch.object(handler, "read_json_body", return_value={}):
                result = await handler.handle_post(
                    "/api/self-improve/worktrees/cleanup", {}, http
                )

        assert result.status_code == 503


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
    async def test_handle_post_returns_none_for_unknown(self, handler_with_store, mock_http_handler):
        http = mock_http_handler(method="POST")
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

    def test_get_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        created = store.create_run(goal="Test")
        fetched = store.get_run(created.run_id)
        assert fetched is not None
        assert fetched.run_id == created.run_id

    def test_update_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test")
        updated = store.update_run(run.run_id, status="running")
        assert updated is not None
        assert updated.status == RunStatus.RUNNING

    def test_cancel_run(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test")
        cancelled = store.cancel_run(run.run_id)
        assert cancelled is not None
        assert cancelled.status == RunStatus.CANCELLED

    def test_cancel_completed_run_returns_none(self, tmp_path):
        store = SelfImproveRunStore(data_dir=tmp_path)
        run = store.create_run(goal="Test")
        store.update_run(run.run_id, status="completed")
        assert store.cancel_run(run.run_id) is None

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

    def test_run_from_dict(self):
        d = {"run_id": "test123", "goal": "Test", "status": "completed"}
        run = SelfImproveRun.from_dict(d)
        assert run.run_id == "test123"
        assert run.status == RunStatus.COMPLETED
