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


# ---------------------------------------------------------------------------
# Additional coverage (B0-cohort #5183)
# ---------------------------------------------------------------------------


def test_init_stores_integration_and_defaults(integration: MagicMock) -> None:
    """Construction wires the integration and applies safe defaults."""

    fresh = worker_module.TestFixerTaskWorker(integration)

    assert fresh._integration is integration
    assert fresh._worker_id == "testfixer-worker"
    assert fresh._running is False


@pytest.mark.asyncio
async def test_default_worker_id_used_when_claiming(
    integration: MagicMock,
    scheduler_bridge: MagicMock,
) -> None:
    """The default worker id is passed through to claim_task."""

    fresh = worker_module.TestFixerTaskWorker(integration)
    scheduler_bridge.claim_task.return_value = None

    with patch.object(worker_module.asyncio, "sleep", new=AsyncMock()):
        await fresh.start(run_once=True)

    scheduler_bridge.claim_task.assert_awaited_once_with(
        agent_id="testfixer-worker",
        capabilities=["testfixer"],
        block_ms=2000,
    )


@pytest.mark.asyncio
async def test_start_logs_startup_message(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Starting the worker emits the startup log line."""

    scheduler_bridge.claim_task.return_value = None

    with patch.object(worker_module.asyncio, "sleep", new=AsyncMock()):
        with caplog.at_level("INFO", logger=worker_module.__name__):
            await worker.start(run_once=True)

    assert "TestFixer task worker started" in caplog.text


@pytest.mark.asyncio
async def test_worker_is_running_while_handling_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """start() flips the run flag on before the first task is processed."""

    scheduler_bridge.claim_task.return_value = make_task()
    observed: list[bool] = []

    async def record_running(task: SimpleNamespace) -> None:
        observed.append(worker._running)

    worker._handle_task = AsyncMock(side_effect=record_running)

    await worker.start(run_once=True)

    assert observed == [True]


@pytest.mark.asyncio
async def test_start_continues_polling_after_empty_claim(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """In continuous mode, an empty poll sleeps and the loop claims again."""

    task = make_task()
    scheduler_bridge.claim_task.side_effect = [None, task]

    async def stop_after_handling(t: SimpleNamespace) -> None:
        await worker.stop()

    worker._handle_task = AsyncMock(side_effect=stop_after_handling)

    with patch.object(worker_module.asyncio, "sleep", new=AsyncMock()) as sleep_mock:
        await worker.start(run_once=False)

    assert scheduler_bridge.claim_task.await_count == 2
    sleep_mock.assert_awaited_once_with(0.5)
    worker._handle_task.assert_awaited_once_with(task)


@pytest.mark.asyncio
async def test_start_continues_after_requeueing_unsupported_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """In continuous mode, a requeued task does not stop the loop."""

    bad_task = make_task(task_id="task-bad", task_type="debate")
    good_task = make_task(task_id="task-good")
    scheduler_bridge.claim_task.side_effect = [bad_task, good_task]

    async def stop_after_handling(t: SimpleNamespace) -> None:
        await worker.stop()

    worker._handle_task = AsyncMock(side_effect=stop_after_handling)

    await worker.start(run_once=False)

    scheduler_bridge.fail_task.assert_awaited_once_with(
        "task-bad",
        "Unsupported task type",
        agent_id="worker-123",
        requeue=True,
    )
    assert scheduler_bridge.claim_task.await_count == 2
    worker._handle_task.assert_awaited_once_with(good_task)


@pytest.mark.asyncio
async def test_start_run_once_with_task_does_not_sleep(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """The idle backoff sleep only happens on empty polls."""

    scheduler_bridge.claim_task.return_value = make_task()
    worker._handle_task = AsyncMock()

    with patch.object(worker_module.asyncio, "sleep", new=AsyncMock()) as sleep_mock:
        await worker.start(run_once=True)

    sleep_mock.assert_not_awaited()
    scheduler_bridge.fail_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_is_idempotent(worker: worker_module.TestFixerTaskWorker) -> None:
    """stop() can be called repeatedly, including before start()."""

    await worker.stop()
    await worker.stop()

    assert worker._running is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad value"),
        OSError("disk gone"),
        ConnectionError("link down"),
    ],
    ids=["value-error", "os-error", "connection-error"],
)
async def test_handle_task_fails_task_on_each_caught_exception(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
    exc: Exception,
) -> None:
    """Every declared-recoverable exception type fails the scheduler task."""

    task = make_task()
    handler = MagicMock()
    handler.execute = AsyncMock(side_effect=exc)
    worker._create_handler = MagicMock(return_value=handler)

    await worker._handle_task(task)

    scheduler_bridge.fail_task.assert_awaited_once_with(
        task.id,
        error=str(exc),
        agent_id="worker-123",
    )
    scheduler_bridge.complete_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_task_propagates_unexpected_exception(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """Exception types outside the caught tuple bubble up to the caller."""

    task = make_task()
    handler = MagicMock()
    handler.execute = AsyncMock(side_effect=KeyError("missing"))
    worker._create_handler = MagicMock(return_value=handler)

    with pytest.raises(KeyError):
        await worker._handle_task(task)

    scheduler_bridge.complete_task.assert_not_awaited()
    scheduler_bridge.fail_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_task_success_does_not_fail_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """A successful execution never reports failure to the scheduler."""

    task = make_task()
    handler = MagicMock()
    handler.execute = AsyncMock(return_value={"fixed": 3})
    worker._create_handler = MagicMock(return_value=handler)

    await worker._handle_task(task)

    scheduler_bridge.fail_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_task_creates_fresh_handler_per_task(
    worker: worker_module.TestFixerTaskWorker,
    scheduler_bridge: MagicMock,
) -> None:
    """Each task gets its own handler instance rather than a cached one."""

    handler = MagicMock()
    handler.execute = AsyncMock(return_value={})
    worker._create_handler = MagicMock(return_value=handler)

    await worker._handle_task(make_task(task_id="t1"))
    await worker._handle_task(make_task(task_id="t2"))

    assert worker._create_handler.call_count == 2
    assert scheduler_bridge.complete_task.await_count == 2
