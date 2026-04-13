"""Tests for the TestFixer task worker loop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.control_plane.workers.testfixer_task_worker as worker_module


def make_task(
    *,
    task_id: str = "task-123",
    task_type: str = worker_module.TESTFIXER_TASK_TYPE,
    payload: dict[str, object] | None = None,
) -> SimpleNamespace:
    """Create a lightweight scheduler task stub."""

    return SimpleNamespace(id=task_id, task_type=task_type, payload=payload or {"repo": "demo"})


@pytest.fixture
def scheduler_bridge() -> MagicMock:
    """Build a mocked scheduler bridge."""

    bridge = MagicMock()
    bridge.claim_task = AsyncMock()
    bridge.complete_task = AsyncMock()
    bridge.fail_task = AsyncMock()
    return bridge


@pytest.fixture
def integration(scheduler_bridge: MagicMock) -> MagicMock:
    """Build a mocked integration object with the expected nesting."""

    integration = MagicMock()
    integration._coordinator = MagicMock()
    integration._coordinator._scheduler_bridge = scheduler_bridge
    return integration


@pytest.fixture
def worker(integration: MagicMock) -> worker_module.TestFixerTaskWorker:
    """Create a worker with a stable test id."""

    return worker_module.TestFixerTaskWorker(integration, worker_id="worker-123")


def test_scheduler_bridge_property_returns_nested_bridge(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """The worker exposes the coordinator bridge via its helper property."""

    assert worker._scheduler_bridge is scheduler_bridge


def test_create_handler_uses_integration(
    worker: worker_module.TestFixerTaskWorker,
    integration: MagicMock,
) -> None:
    """Handler construction should be delegated to TestFixerControlPlane."""

    with patch.object(worker_module, "TestFixerControlPlane", autospec=True) as handler_cls:
        handler = worker._create_handler()

    handler_cls.assert_called_once_with(integration)
    assert handler is handler_cls.return_value


@pytest.mark.asyncio
async def test_stop_sets_worker_state_false(worker: worker_module.TestFixerTaskWorker) -> None:
    """Stopping the worker should flip the run flag immediately."""

    worker._running = True

    await worker.stop()

    assert worker._running is False


@pytest.mark.asyncio
async def test_start_run_once_sleeps_when_no_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """An empty poll should sleep briefly, then stop in run-once mode."""

    scheduler_bridge.claim_task.return_value = None

    with patch.object(worker_module.asyncio, "sleep", new=AsyncMock()) as sleep_mock:
        await worker.start(run_once=True)

    scheduler_bridge.claim_task.assert_awaited_once_with(
        agent_id="worker-123",
        capabilities=["testfixer"],
        block_ms=2000,
    )
    sleep_mock.assert_awaited_once_with(0.5)
    assert worker._running is False


@pytest.mark.asyncio
async def test_start_run_once_handles_supported_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """A supported task should be handed to the task handler once."""

    task = make_task(payload={"repo_path": "/tmp/repo"})
    scheduler_bridge.claim_task.return_value = task
    worker._handle_task = AsyncMock()

    await worker.start(run_once=True)

    worker._handle_task.assert_awaited_once_with(task)
    assert worker._running is False


@pytest.mark.asyncio
async def test_start_requeues_unsupported_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """Unsupported task types should be requeued instead of executed."""

    task = make_task(task_type="debate")
    scheduler_bridge.claim_task.return_value = task

    await worker.start(run_once=True)

    scheduler_bridge.fail_task.assert_awaited_once_with(
        task.id,
        "Unsupported task type",
        agent_id="worker-123",
        requeue=True,
    )
    assert worker._running is False


@pytest.mark.asyncio
async def test_start_exits_when_stopped_during_multi_run_loop(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """The worker should honor stop() even when not running in run-once mode."""

    scheduler_bridge.claim_task.return_value = make_task()

    async def stop_after_handling(task: SimpleNamespace) -> None:
        await worker.stop()

    worker._handle_task = AsyncMock(side_effect=stop_after_handling)

    await worker.start(run_once=False)

    scheduler_bridge.claim_task.assert_awaited_once()
    worker._handle_task.assert_awaited_once()
    assert worker._running is False


@pytest.mark.asyncio
async def test_handle_task_completes_task_on_success(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """Successful handler results should complete the scheduler task."""

    task = make_task(payload={"repo_path": "/tmp/repo"})
    handler = MagicMock()
    handler.execute = AsyncMock(return_value={"status": "ok"})
    worker._create_handler = MagicMock(return_value=handler)

    await worker._handle_task(task)

    handler.execute.assert_awaited_once_with(task.payload)
    scheduler_bridge.complete_task.assert_awaited_once_with(
        task.id,
        result={"status": "ok"},
        agent_id="worker-123",
    )


@pytest.mark.asyncio
async def test_handle_task_fails_task_on_runtime_error(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Runtime errors from the handler should fail the scheduler task."""

    task = make_task()
    handler = MagicMock()
    handler.execute = AsyncMock(side_effect=RuntimeError("boom"))
    worker._create_handler = MagicMock(return_value=handler)

    with caplog.at_level("ERROR"):
        await worker._handle_task(task)

    scheduler_bridge.fail_task.assert_awaited_once_with(
        task.id,
        error="boom",
        agent_id="worker-123",
    )
    assert "TestFixer task task-123 failed: boom" in caplog.text


@pytest.mark.asyncio
async def test_handle_task_fails_task_on_timeout_error(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """Timeout errors should be reported as task failures."""

    task = make_task()
    handler = MagicMock()
    handler.execute = AsyncMock(side_effect=TimeoutError("timed out"))
    worker._create_handler = MagicMock(return_value=handler)

    await worker._handle_task(task)

    scheduler_bridge.fail_task.assert_awaited_once_with(
        task.id,
        error="timed out",
        agent_id="worker-123",
    )
