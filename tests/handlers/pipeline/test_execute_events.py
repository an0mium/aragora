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


def _mock_result(completed: int = 2, total: int = 3, failed: int = 1) -> MagicMock:
    result = MagicMock()
    result.subtasks_completed = completed
    result.subtasks_total = total
    result.subtasks_failed = failed
    return result


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


def _mock_plan(plan_id: str = "plan-123") -> MagicMock:
    plan = MagicMock()
    plan.id = plan_id
    return plan


def _mock_launch(
    plan_id: str = "plan-123",
    execution_id: str = "exec-123",
    correlation_id: str = "corr-123",
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "execution_id": execution_id,
        "correlation_id": correlation_id,
        "execution_mode": "workflow",
        "status": "queued",
    }


def _mock_outcome(
    *,
    success: bool = True,
    tasks_total: int = 1,
    tasks_completed: int | None = None,
    receipt_id: str = "rcpt-123",
    error: str | None = None,
) -> MagicMock:
    completed = tasks_completed if tasks_completed is not None else tasks_total
    outcome = MagicMock()
    outcome.success = success
    outcome.tasks_total = tasks_total
    outcome.tasks_completed = completed
    outcome.error = error
    outcome.receipt_id = receipt_id
    outcome.to_dict.return_value = {
        "success": success,
        "tasks_total": tasks_total,
        "tasks_completed": completed,
        "error": error,
        "receipt_id": receipt_id,
    }
    return outcome


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

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=2, tasks_completed=2),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_started.assert_awaited_once()
        call_args = mock_emitter.emit_started.call_args
        assert call_args[0][0] == "pipe-ws"
        assert call_args[0][1]["cycle_id"] == "cycle-1"
        assert call_args[0][1]["goal_count"] == 1

    @pytest.mark.asyncio
    async def test_emitter_completed_event_on_success(self):
        """pipeline_completed event is emitted on successful execution."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=3, tasks_completed=3),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_completed.assert_awaited_once()
        assert mock_emitter.emit_completed.call_args[0][0] == "pipe-ws"

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_zero_subtasks(self):
        """pipeline_failed event emitted when no subtasks complete."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_failed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=False, tasks_total=3, tasks_completed=0),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_runtime_error(self):
        """pipeline_failed event emitted on RuntimeError."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_failed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(side_effect=RuntimeError("Boom")),
                ),
            ):
                await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert "boom" in mock_emitter.emit_failed.call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_cancel(self):
        """pipeline_failed event emitted on CancelledError."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_failed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(side_effect=asyncio.CancelledError()),
                ),
            ):
                await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert "cancel" in mock_emitter.emit_failed.call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_emitter_failed_event_on_import_error(self):
        """pipeline_failed event emitted when canonical execution is unavailable."""
        h = _make_handler()
        _executions["pipe-ws"] = {"pipeline_id": "pipe-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_failed = AsyncMock()

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with patch(
                "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                side_effect=ImportError("canonical execution not available"),
            ):
                await h._execute_pipeline("pipe-ws", "cycle-1", goals, None, False)

        mock_emitter.emit_failed.assert_awaited_once()
        assert "not available" in mock_emitter.emit_failed.call_args[0][1].lower()


# ---------------------------------------------------------------------------
# Execution State Wiring
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_execution_records_launch_metadata(self):
        """Launch metadata is stored on the in-memory execution record."""
        h = _make_handler()
        _executions["pipe-cb"] = {"pipeline_id": "pipe-cb", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=1, tasks_completed=1),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-cb", "cycle-1", goals, None, False)

        execution = _executions["pipe-cb"]
        assert execution["plan_id"] == "plan-123"
        assert execution["execution_id"] == "exec-123"
        assert execution["correlation_id"] == "corr-123"

    @pytest.mark.asyncio
    async def test_execution_updates_status_on_success(self):
        """Successful execution updates the in-memory execution status."""
        h = _make_handler()
        _executions["pipe-cb"] = {"pipeline_id": "pipe-cb", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=1, tasks_completed=1),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-cb", "cycle-1", goals, None, False)

        assert _executions["pipe-cb"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execution_updates_subtask_counts(self):
        """Successful execution stores subtask counts from the outcome."""
        h = _make_handler()
        _executions["pipe-cb"] = {"pipeline_id": "pipe-cb", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock(), MagicMock(), MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=3, tasks_completed=2),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-cb", "cycle-1", goals, None, False)

        execution = _executions["pipe-cb"]
        assert execution["total_subtasks"] == 3
        assert execution["completed_subtasks"] == 2
        assert execution["failed_subtasks"] == 1

    @pytest.mark.asyncio
    async def test_execution_forwards_lifecycle_events_to_emitter(self):
        """Execution start and completion are forwarded to the WebSocket emitter."""
        h = _make_handler()
        _executions["pipe-cb"] = {"pipeline_id": "pipe-cb", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=1, tasks_completed=1),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-cb", "cycle-1", goals, None, False)

        mock_emitter.emit_started.assert_awaited_once()
        mock_emitter.emit_completed.assert_awaited_once()


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

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=None,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=1, tasks_completed=1),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-no-ws", "cycle-1", goals, None, False)

        assert _executions["pipe-no-ws"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execution_records_metadata_without_emitter(self):
        """Plan launch metadata is still recorded when no emitter is available."""
        h = _make_handler()
        _executions["pipe-no-ws"] = {"pipeline_id": "pipe-no-ws", "status": "started"}
        goals = [MagicMock(description="Goal 1")]

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=None,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=1, tasks_completed=1),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
                ),
            ):
                await h._execute_pipeline("pipe-no-ws", "cycle-1", goals, None, False)

        execution = _executions["pipe-no-ws"]
        assert execution["plan_id"] == "plan-123"
        assert execution["execution_id"] == "exec-123"


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

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_completed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(
                        return_value=(
                            _mock_outcome(success=True, tasks_total=1, tasks_completed=1),
                            {"execution_id": "exec-123"},
                            {"receipt_id": "rcpt-123"},
                        )
                    ),
                ),
                patch(
                    "aragora.pipeline.receipt_generator.generate_pipeline_receipt",
                    new_callable=AsyncMock,
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

        mock_emitter = MagicMock()
        mock_emitter.emit_started = AsyncMock()
        mock_emitter.emit_failed = AsyncMock()
        mock_emitter.as_event_callback.return_value = lambda e, d: None

        with patch(
            "aragora.server.handlers.pipeline.execute._get_emitter",
            return_value=mock_emitter,
        ):
            with (
                patch(
                    "aragora.pipeline.canonical_execution.build_decision_plan_from_orchestration",
                    return_value=(_mock_plan(), [MagicMock()]),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.queue_plan_execution",
                    return_value=_mock_launch(),
                ),
                patch(
                    "aragora.pipeline.canonical_execution.execute_queued_plan",
                    new=AsyncMock(side_effect=RuntimeError("Boom")),
                ),
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
