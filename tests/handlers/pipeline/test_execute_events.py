"""Tests for WebSocket event emission during pipeline execution.

Covers:
  - PipelineStreamEmitter is wired during execution
  - pipeline_started event emitted on execution start
  - pipeline_completed / pipeline_failed events on finish
  - step_progress events forwarded from progress_callback
  - Dry-run does NOT emit events
  - Events include correct pipeline_id
  - In-memory execution state updated from progress events
  - Emitter unavailability is handled gracefully
  - emit_execution_progress convenience method
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.pipeline.execute import (
    PipelineExecuteHandler,
    _executions,
    _execution_tasks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_executions():
    """Reset module-level state between tests."""
    _executions.clear()
    _execution_tasks.clear()
    yield
    _executions.clear()
    _execution_tasks.clear()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from aragora.server.handlers.pipeline.execute import _execute_limiter

    _execute_limiter._buckets.clear()
    yield
    _execute_limiter._buckets.clear()


def _make_handler() -> PipelineExecuteHandler:
    return PipelineExecuteHandler(ctx={})


def _mock_outcome(
    *,
    success: bool = True,
    tasks_completed: int = 2,
    tasks_total: int = 3,
    error: str | None = None,
) -> MagicMock:
    outcome = MagicMock()
    outcome.success = success
    outcome.tasks_completed = tasks_completed
    outcome.tasks_total = tasks_total
    outcome.error = error
    outcome.to_dict.return_value = {
        "success": success,
        "tasks_completed": tasks_completed,
        "tasks_total": tasks_total,
        "error": error,
    }
    return outcome


def _make_emitter() -> MagicMock:
    emitter = MagicMock()
    emitter.emit_started = AsyncMock()
    emitter.emit_completed = AsyncMock()
    emitter.emit_failed = AsyncMock()
    return emitter


@contextmanager
def _patch_decision_plan_runtime(
    *,
    outcome: Any | None = None,
    record: dict[str, Any] | None = None,
    decision_receipt: dict[str, Any] | None = None,
    execute_side_effect: BaseException | None = None,
    tasks_count: int = 1,
    plan: Any | None = None,
):
    plan = plan or SimpleNamespace(id="dp-test")
    tasks = [MagicMock(name=f"task-{idx}") for idx in range(tasks_count)]
    store = MagicMock()
    store.get.return_value = plan
    launch = {
        "plan_id": plan.id,
        "execution_id": "exec-test",
        "correlation_id": "corr-test",
        "status": "queued",
        "run_id": "run-test",
    }

    with (
        patch(
            "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
            return_value=(plan, tasks),
        ) as build,
        patch(
            "aragora.pipeline.canonical_execution.queue_plan_execution",
            return_value=launch,
        ) as queue,
        patch(
            "aragora.pipeline.canonical_execution.execute_queued_plan",
            new_callable=AsyncMock,
        ) as execute,
        patch("aragora.pipeline.plan_store.get_plan_store", return_value=store) as get_store,
        patch(
            "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
            new_callable=AsyncMock,
            return_value={"receipt_id": "pipe-receipt"},
        ) as receipt_gen,
    ):
        if execute_side_effect is not None:
            execute.side_effect = execute_side_effect
        else:
            execute.return_value = (
                outcome or _mock_outcome(),
                record or {"summary": "ok"},
                decision_receipt or {"decision": "receipt"},
            )
        yield SimpleNamespace(
            build=build,
            queue=queue,
            execute=execute,
            get_store=get_store,
            receipt_gen=receipt_gen,
            plan=plan,
            tasks=tasks,
            store=store,
            launch=launch,
        )


def _make_http_handler(body: dict[str, Any] | None = None) -> MagicMock:
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    if body is not None:
        raw = json.dumps(body).encode()
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile.read.return_value = raw
    else:
        handler.headers = {"Content-Length": "2"}
        handler.rfile.read.return_value = b"{}"
    return handler


def _mock_orch_nodes(count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "id": f"orch-node-{i}",
            "stage": "orchestration",
            "label": f"Task {i + 1}",
            "orch_type": "agent_task",
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Emitter Wiring During Execution
# ---------------------------------------------------------------------------


class TestEmitterWiring:
    @pytest.mark.asyncio
    async def test_emitter_started_event_emitted(self):
        """pipeline_started event is emitted at execution start."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(
                outcome=_mock_outcome(success=True, tasks_completed=2, tasks_total=2),
                tasks_count=2,
            ),
        ):
            await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_started.assert_awaited_once()
        call_args = mock_emitter.emit_started.call_args
        assert call_args[0][0] == "pipe-ws"
        assert call_args[0][1]["cycle_id"] == "cycle-1"
        assert call_args[0][1]["goal_count"] == 1
        assert call_args[0][1]["plan_id"] == "dp-test"
        assert call_args[0][1]["execution_id"] == "exec-test"

    @pytest.mark.asyncio
    async def test_emitter_completed_event_on_success(self):
        """pipeline_completed event is emitted on successful execution."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(
                outcome=_mock_outcome(success=True, tasks_completed=3, tasks_total=3),
                record={"status": "completed"},
            ),
        ):
            await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_completed.assert_awaited_once()
        assert mock_emitter.emit_completed.call_args[0][0] == "pipe-ws"
        assert _executions["pipe-ws"]["status"] == "completed"
        assert _executions["pipe-ws"]["completed_subtasks"] == 3

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_unsuccessful_outcome(self):
        """pipeline_failed event is emitted when the execution outcome is unsuccessful."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(
                outcome=_mock_outcome(
                    success=False,
                    tasks_completed=0,
                    tasks_total=3,
                    error="Pipeline execution failed",
                ),
                record={"status": "failed"},
                tasks_count=3,
            ),
        ):
            await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert mock_emitter.emit_failed.call_args[0] == (
            "pipe-ws",
            "Pipeline execution failed",
        )
        assert _executions["pipe-ws"]["status"] == "failed"
        assert _executions["pipe-ws"]["failed_subtasks"] == 3

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_runtime_error(self):
        """pipeline_failed event emitted on RuntimeError."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(execute_side_effect=RuntimeError("Boom")),
        ):
            await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert mock_emitter.emit_failed.call_args[0] == ("pipe-ws", "Boom")
        assert _executions["pipe-ws"]["error"] == "Boom"

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_cancel(self):
        """pipeline_failed event emitted on CancelledError."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(execute_side_effect=asyncio.CancelledError()),
        ):
            await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert mock_emitter.emit_failed.call_args[0] == ("pipe-ws", "Pipeline cancelled")
        assert _executions["pipe-ws"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_emitter_failed_event_when_plan_lookup_misses(self):
        """pipeline_failed event is emitted when the queued plan cannot be loaded."""
        h = _make_handler()
        _executions["pipe-ws"] = {
            "pipeline_id": "pipe-ws",
            "status": "started",
            "plan_id": "dp-missing",
            "execution_id": "exec-missing",
            "correlation_id": "corr-missing",
        }
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(plan=SimpleNamespace(id="dp-missing")) as runtime,
        ):
            runtime.store.get.return_value = None
            await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert "Plan not found" in mock_emitter.emit_failed.call_args[0][1]


# ---------------------------------------------------------------------------
# Execution State Wiring
# ---------------------------------------------------------------------------


class TestExecutionState:
    @pytest.mark.asyncio
    async def test_execution_state_records_runtime_metadata(self):
        """Successful execution stores plan/runtime metadata on the execution state."""
        h = _make_handler()
        _executions["pipe-cb"] = {"pipeline_id": "pipe-cb", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=_make_emitter(),
            ),
            _patch_decision_plan_runtime(
                outcome=_mock_outcome(success=True, tasks_completed=1, tasks_total=1),
                record={"status": "completed", "summary": "done"},
                decision_receipt={"decision": "ok"},
            ),
        ):
            await h._execute_pipeline("pipe-cb", "cycle-1", goals, None, False)

        state = _executions["pipe-cb"]
        assert state["status"] == "completed"
        assert state["runtime"] == "decision_plan"
        assert state["plan_id"] == "dp-test"
        assert state["execution_id"] == "exec-test"
        assert state["correlation_id"] == "corr-test"
        assert state["run_id"] == "run-test"
        assert state["record"]["summary"] == "done"
        assert state["receipt"]["decision"] == "ok"
        assert state["receipt"]["pipeline_receipt"]["receipt_id"] == "pipe-receipt"

    @pytest.mark.asyncio
    async def test_existing_execution_metadata_is_reused(self):
        """Pre-seeded plan/execution identifiers skip the synthetic rebuild path."""
        h = _make_handler()
        _executions["pipe-cb"] = {
            "pipeline_id": "pipe-cb",
            "status": "started",
            "plan_id": "dp-existing",
            "execution_id": "exec-existing",
            "correlation_id": "corr-existing",
        }
        goals = [MagicMock(description="Goal 1")]

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=_make_emitter(),
            ),
            _patch_decision_plan_runtime(
                plan=SimpleNamespace(id="dp-existing"),
                outcome=_mock_outcome(success=True, tasks_completed=1, tasks_total=1),
            ) as runtime,
        ):
            await h._execute_pipeline("pipe-cb", "cycle-1", goals, None, False)

        runtime.build.assert_not_called()
        runtime.queue.assert_not_called()
        runtime.execute.assert_awaited_once()
        runtime.execute.assert_awaited_with(
            runtime.plan,
            execution_id="exec-existing",
            correlation_id="corr-existing",
            execution_mode="workflow",
        )


# ---------------------------------------------------------------------------
# Dry Run Does NOT Emit Events
# ---------------------------------------------------------------------------


class TestDryRunNoEvents:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_emitter(self):
        """Dry run returns preview without emitting WS events."""
        h = _make_handler()
        http = _make_http_handler(body={"dry_run": True})
        orch_nodes = _mock_orch_nodes(2)

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with patch.object(h, "_load_orchestration_nodes", return_value=orch_nodes):
                result = await h.handle_post("/api/v1/pipeline/pipe-dry/execute", {}, http)

        # Dry run exits before _execute_pipeline, so no emitter calls
        mock_emitter.emit_started.assert_not_awaited()
        mock_emitter.emit_completed.assert_not_awaited()
        assert result is not None


# ---------------------------------------------------------------------------
# Emitter Unavailability
# ---------------------------------------------------------------------------


class TestEmitterUnavailable:
    @pytest.mark.asyncio
    async def test_execution_succeeds_without_emitter(self):
        """Execution completes normally when emitter returns None."""
        h = _make_handler()
        _executions["pipe-no-ws"] = {"pipeline_id": "pipe-no-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=None,
            ),
            _patch_decision_plan_runtime(
                outcome=_mock_outcome(success=True, tasks_completed=1, tasks_total=1),
            ),
        ):
            await h._execute_pipeline("pipe-no-ws", "cycle-1", goals, None, False)

        assert _executions["pipe-no-ws"]["status"] == "completed"
        assert (
            _executions["pipe-no-ws"]["receipt"]["pipeline_receipt"]["receipt_id"] == "pipe-receipt"
        )

    @pytest.mark.asyncio
    async def test_execution_failure_without_emitter_still_records_error(self):
        """Execution failures still update in-memory state when no emitter is available."""
        h = _make_handler()
        _executions["pipe-no-ws"] = {"pipeline_id": "pipe-no-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=None,
            ),
            _patch_decision_plan_runtime(execute_side_effect=RuntimeError("no emitter boom")),
        ):
            await h._execute_pipeline("pipe-no-ws", "cycle-1", goals, None, False)

        assert _executions["pipe-no-ws"]["status"] == "failed"
        assert _executions["pipe-no-ws"]["error"] == "no emitter boom"


# ---------------------------------------------------------------------------
# emit_execution_progress Convenience Method
# ---------------------------------------------------------------------------


class TestEmitExecutionProgress:
    @pytest.mark.asyncio
    async def test_emit_execution_progress_calculates_progress(self):
        """emit_execution_progress computes progress from completed/total."""
        from aragora.server.stream.pipeline_stream import PipelineStreamEmitter

        emitter = PipelineStreamEmitter()
        emitter.emit = AsyncMock()

        await emitter.emit_execution_progress("pipe-1", 3, 8, "Testing subtask")

        emitter.emit.assert_awaited_once()
        call_args = emitter.emit.call_args
        assert call_args[0][0] == "pipe-1"
        data = call_args[0][2]
        assert data["completed"] == 3
        assert data["total"] == 8
        assert data["current_task"] == "Testing subtask"
        assert data["step"] == "Testing subtask"
        assert abs(data["progress"] - 3 / 8) < 0.001

    @pytest.mark.asyncio
    async def test_emit_execution_progress_zero_total(self):
        """emit_execution_progress handles zero total gracefully."""
        from aragora.server.stream.pipeline_stream import PipelineStreamEmitter

        emitter = PipelineStreamEmitter()
        emitter.emit = AsyncMock()

        await emitter.emit_execution_progress("pipe-1", 0, 0, "Init")

        data = emitter.emit.call_args[0][2]
        assert data["progress"] == 0.0

    @pytest.mark.asyncio
    async def test_emit_execution_progress_full_completion(self):
        """emit_execution_progress reports 1.0 for fully complete."""
        from aragora.server.stream.pipeline_stream import PipelineStreamEmitter

        emitter = PipelineStreamEmitter()
        emitter.emit = AsyncMock()

        await emitter.emit_execution_progress("pipe-1", 5, 5, "Done")

        data = emitter.emit.call_args[0][2]
        assert data["progress"] == 1.0


# ---------------------------------------------------------------------------
# Event Pipeline ID Correctness
# ---------------------------------------------------------------------------


class TestEventPipelineId:
    @pytest.mark.asyncio
    async def test_all_events_include_correct_pipeline_id(self):
        """All emitted events contain the correct pipeline_id."""
        h = _make_handler()
        _executions["pipe-id-check"] = {"pipeline_id": "pipe-id-check", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(
                outcome=_mock_outcome(success=True, tasks_completed=1, tasks_total=1),
            ),
        ):
            await h._execute_pipeline("pipe-id-check", "cycle-1", goals, None, False)

        # Check started event
        assert mock_emitter.emit_started.call_args[0][0] == "pipe-id-check"
        # Check completed event
        assert mock_emitter.emit_completed.call_args[0][0] == "pipe-id-check"

    @pytest.mark.asyncio
    async def test_failed_event_has_correct_pipeline_id(self):
        """Failed events contain the correct pipeline_id."""
        h = _make_handler()
        _executions["pipe-fail-id"] = {"pipeline_id": "pipe-fail-id", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = _make_emitter()

        with (
            patch(
                "aragora.server.handlers.pipeline.execute._get_emitter",
                return_value=mock_emitter,
            ),
            _patch_decision_plan_runtime(execute_side_effect=RuntimeError("Boom")),
        ):
            await h._execute_pipeline("pipe-fail-id", "cycle-1", goals, None, False)

        assert mock_emitter.emit_failed.call_args[0][0] == "pipe-fail-id"


# ---------------------------------------------------------------------------
# _get_emitter Helper
# ---------------------------------------------------------------------------


class TestGetEmitterHelper:
    def test_get_emitter_returns_emitter(self):
        """_get_emitter returns the global emitter."""
        from aragora.server.handlers.pipeline.execute import _get_emitter

        emitter = _get_emitter()
        assert emitter is not None

    def test_get_emitter_returns_none_on_import_error(self):
        """_get_emitter returns None when import fails."""
        from aragora.server.handlers.pipeline.execute import _get_emitter

        with patch.dict("sys.modules", {"aragora.server.stream.pipeline_stream": None}):
            result = _get_emitter()
        assert result is None
