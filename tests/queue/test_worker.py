"""
Tests for debate worker module.

Tests cover:
- DebateWorker construction and properties (worker_id, is_running, active_jobs)
- get_stats (uptime, counters)
- _process_job (success path, retry path, permanent failure path)
- stop (graceful shutdown, timeout)
- _handle_signal
- Default implementations of JobQueue.complete() and JobQueue.fail()
- Job.mark_retrying lifecycle method
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.base import Job, JobQueue, JobStatus
from aragora.queue.config import QueueConfig, reset_queue_config, set_queue_config
from aragora.queue.retry import RetryPolicy
from aragora.queue.worker import DebateWorker


@pytest.fixture(autouse=True)
def _reset_config():
    set_queue_config(QueueConfig())
    yield
    reset_queue_config()


def _make_worker(
    executor: AsyncMock | None = None,
    max_concurrent: int = 3,
) -> tuple[DebateWorker, AsyncMock, AsyncMock]:
    """Create a DebateWorker with mock queue and executor."""
    mock_queue = AsyncMock(spec=JobQueue)
    mock_executor = executor or AsyncMock(return_value={"answer": "test"})
    worker = DebateWorker(
        queue=mock_queue,
        worker_id="test-worker-1",
        executor=mock_executor,
        max_concurrent=max_concurrent,
        retry_policy=RetryPolicy(max_attempts=3, jitter=False),
    )
    return worker, mock_queue, mock_executor


class TestWorkerConstruction:
    """Tests for DebateWorker construction and properties."""

    def test_worker_id(self):
        worker, _, _ = _make_worker()
        assert worker.worker_id == "test-worker-1"

    def test_is_running_initially_false(self):
        worker, _, _ = _make_worker()
        assert worker.is_running is False

    def test_active_jobs_initially_zero(self):
        worker, _, _ = _make_worker()
        assert worker.active_jobs == 0

    def test_active_jobs_with_acquired_semaphore(self):
        worker, _, _ = _make_worker(max_concurrent=5)
        worker._semaphore._value = 3  # 2 slots used
        assert worker.active_jobs == 2


class TestWorkerStats:
    """Tests for get_stats."""

    def test_stats_not_running(self):
        worker, _, _ = _make_worker()
        stats = worker.get_stats()

        assert stats["worker_id"] == "test-worker-1"
        assert stats["running"] is False
        assert stats["active_jobs"] == 0
        assert stats["max_concurrent"] == 3
        assert stats["jobs_processed"] == 0
        assert stats["jobs_failed"] == 0
        assert stats["uptime_seconds"] == 0

    def test_stats_with_activity(self):
        worker, _, _ = _make_worker()
        worker._running = True
        worker._start_time = time.time() - 100
        worker._jobs_processed = 50
        worker._jobs_failed = 5

        stats = worker.get_stats()

        assert stats["running"] is True
        assert stats["jobs_processed"] == 50
        assert stats["jobs_failed"] == 5
        assert stats["uptime_seconds"] >= 99


class TestProcessJob:
    """Tests for _process_job."""

    @pytest.mark.asyncio
    async def test_successful_job(self):
        mock_executor = AsyncMock(return_value={"result": "success"})
        worker, mock_queue, _ = _make_worker(executor=mock_executor)

        job = Job(id="j-ok", payload={"q": "test"}, max_attempts=3)
        job.mark_processing("test-worker-1")

        # Acquire semaphore (normally done by start loop)
        await worker._semaphore.acquire()

        await worker._process_job(job)

        mock_executor.assert_called_once_with(job)
        mock_queue.ack.assert_called_once_with("j-ok")
        assert worker._jobs_processed == 1
        assert worker._jobs_failed == 0

    @pytest.mark.asyncio
    async def test_retryable_failure(self):
        mock_executor = AsyncMock(side_effect=ConnectionError("lost connection"))
        worker, mock_queue, _ = _make_worker(executor=mock_executor)

        job = Job(id="j-retry", payload={}, max_attempts=3)
        job.mark_processing("test-worker-1")  # attempts = 1

        await worker._semaphore.acquire()
        await worker._process_job(job)

        assert worker._jobs_failed == 1
        mock_queue.nack.assert_called_once_with("j-retry", requeue=True)

    @pytest.mark.asyncio
    async def test_non_retryable_failure(self):
        mock_executor = AsyncMock(side_effect=ValueError("invalid input"))
        worker, mock_queue, _ = _make_worker(executor=mock_executor)

        job = Job(id="j-perm-fail", payload={}, max_attempts=3)
        job.mark_processing("test-worker-1")

        await worker._semaphore.acquire()
        await worker._process_job(job)

        assert worker._jobs_failed == 1
        mock_queue.nack.assert_called_once_with("j-perm-fail", requeue=False)

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        mock_executor = AsyncMock(side_effect=RuntimeError("still failing"))
        worker, mock_queue, _ = _make_worker(executor=mock_executor)

        job = Job(id="j-max", payload={}, max_attempts=3, attempts=2)
        job.mark_processing("test-worker-1")  # attempts now 3

        await worker._semaphore.acquire()
        await worker._process_job(job)

        assert worker._jobs_failed == 1
        mock_queue.nack.assert_called_once_with("j-max", requeue=False)

    @pytest.mark.asyncio
    async def test_releases_semaphore(self):
        mock_executor = AsyncMock(return_value={"ok": True})
        worker, _, _ = _make_worker(executor=mock_executor, max_concurrent=1)

        job = Job(id="j-sem", payload={})
        job.mark_processing("test-worker-1")

        await worker._semaphore.acquire()
        assert worker._semaphore._value == 0

        await worker._process_job(job)
        assert worker._semaphore._value == 1  # released

    @pytest.mark.asyncio
    async def test_releases_semaphore_on_failure(self):
        mock_executor = AsyncMock(side_effect=RuntimeError("boom"))
        worker, _, _ = _make_worker(executor=mock_executor, max_concurrent=1)

        job = Job(id="j-sem-fail", payload={}, max_attempts=1)
        job.mark_processing("test-worker-1")

        await worker._semaphore.acquire()
        await worker._process_job(job)
        assert worker._semaphore._value == 1  # released even on failure


class TestWorkerStop:
    """Tests for stop."""

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        worker, mock_queue, _ = _make_worker()
        await worker.stop()
        mock_queue.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_running_no_tasks(self):
        worker, mock_queue, _ = _make_worker()
        worker._running = True

        await worker.stop(timeout=1.0)

        assert worker._running is False
        assert worker._shutdown_event.is_set()
        mock_queue.close.assert_called_once()


class TestHandleSignal:
    """Tests for _handle_signal."""

    def test_sets_shutdown_state(self):
        worker, _, _ = _make_worker()
        worker._running = True

        worker._handle_signal()

        assert worker._running is False
        assert worker._shutdown_event.is_set()


class TestJobMarkRetrying:
    """Tests for Job.mark_retrying lifecycle method (gap in existing tests)."""

    def test_mark_retrying(self):
        job = Job(payload={})
        job.mark_retrying("temporary error")

        assert job.status == JobStatus.RETRYING
        assert job.error == "temporary error"
        # completed_at should NOT be set
        assert job.completed_at is None

    def test_mark_retrying_preserves_attempts(self):
        job = Job(payload={}, attempts=2)
        job.mark_retrying("retry error")
        assert job.attempts == 2  # mark_retrying doesn't increment


class TestJobQueueDefaultMethods:
    """Tests for JobQueue default complete() and fail() methods."""

    @pytest.mark.asyncio
    async def test_complete_delegates_to_ack(self):
        mock_queue = AsyncMock(spec=JobQueue)
        mock_queue.ack.return_value = True
        # Call the default implementation directly
        result = await JobQueue.complete(mock_queue, "j-1", result={"ok": True})
        mock_queue.ack.assert_called_once_with("j-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_fail_delegates_to_nack(self):
        mock_queue = AsyncMock(spec=JobQueue)
        mock_queue.nack.return_value = True
        result = await JobQueue.fail(mock_queue, "j-1", error="oops", requeue=True)
        mock_queue.nack.assert_called_once_with("j-1", requeue=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_fail_default_no_requeue(self):
        mock_queue = AsyncMock(spec=JobQueue)
        mock_queue.nack.return_value = True
        result = await JobQueue.fail(mock_queue, "j-1", error="permanent")
        mock_queue.nack.assert_called_once_with("j-1", requeue=False)


class TestRetryPolicyFromConfig:
    """Tests for RetryPolicy.from_config integration."""

    def test_from_config_uses_queue_config(self):
        set_queue_config(
            QueueConfig(
                retry_max_attempts=7,
                retry_base_delay=2.0,
                retry_max_delay=600.0,
            )
        )

        policy = RetryPolicy.from_config()

        assert policy.max_attempts == 7
        assert policy.base_delay_seconds == 2.0
        assert policy.max_delay_seconds == 600.0
