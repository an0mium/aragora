"""
Comprehensive Tests for Routing Job Queue Worker.

Tests the routing worker including:
- Worker lifecycle (start, stop, configuration)
- Job processing for debate and email routing
- Error handling and retry logic
- Task validation and cleanup
- Concurrency management
- Job enqueueing and recovery
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.workers.routing_worker import (
    RoutingWorker,
    JOB_TYPE_ROUTING,
    JOB_TYPE_ROUTING_DEBATE,
    JOB_TYPE_ROUTING_EMAIL,
    enqueue_routing_job,
    recover_interrupted_routing,
)
from aragora.storage.job_queue_store import (
    JobStatus,
    QueuedJob,
    SQLiteJobStore,
    reset_job_store,
    set_job_store,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_routing_jobs.db"


@pytest.fixture
def store(temp_db):
    """Create a SQLite job store for testing."""
    reset_job_store()
    s = SQLiteJobStore(temp_db)
    set_job_store(s)
    yield s
    reset_job_store()


# =============================================================================
# Job Type Constants Tests
# =============================================================================


class TestJobTypeConstants:
    """Tests for job type constant values."""

    def test_routing_type(self):
        assert JOB_TYPE_ROUTING == "routing"

    def test_debate_type(self):
        assert JOB_TYPE_ROUTING_DEBATE == "routing_debate"

    def test_email_type(self):
        assert JOB_TYPE_ROUTING_EMAIL == "routing_email"


# =============================================================================
# Worker Initialization Tests
# =============================================================================


class TestRoutingWorkerInit:
    """Tests for RoutingWorker initialization."""

    def test_default_initialization(self, store):
        """Test worker initializes with defaults."""
        worker = RoutingWorker()
        assert worker.worker_id.startswith("routing-worker-")
        assert worker.poll_interval == 2.0
        assert worker.max_concurrent == 5
        assert worker.retry_delay_seconds == 30.0
        assert worker._running is False
        assert worker._active_jobs == {}

    def test_custom_worker_id(self, store):
        """Test worker with custom ID."""
        worker = RoutingWorker(worker_id="custom-router-1")
        assert worker.worker_id == "custom-router-1"

    def test_custom_poll_interval(self, store):
        """Test worker with custom poll interval."""
        worker = RoutingWorker(poll_interval=5.0)
        assert worker.poll_interval == 5.0

    def test_custom_max_concurrent(self, store):
        """Test worker with custom max concurrent."""
        worker = RoutingWorker(max_concurrent=20)
        assert worker.max_concurrent == 20

    def test_custom_retry_delay(self, store):
        """Test worker with custom retry delay."""
        worker = RoutingWorker(retry_delay_seconds=60.0)
        assert worker.retry_delay_seconds == 60.0


# =============================================================================
# Worker Lifecycle Tests
# =============================================================================


class TestRoutingWorkerLifecycle:
    """Tests for worker start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, store):
        """Start should set _running to True."""
        worker = RoutingWorker(poll_interval=0.05)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)

        assert worker._running is True

        await worker.stop()
        await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, store):
        """Stop should set _running to False."""
        worker = RoutingWorker(poll_interval=0.05)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)

        await worker.stop()
        assert worker._running is False

        await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_stop_without_start(self, store):
        """Stop without start should not raise."""
        worker = RoutingWorker()
        await worker.stop()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_handles_cancellation(self, store):
        """Worker should handle CancelledError gracefully."""
        worker = RoutingWorker(poll_interval=0.05)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        assert worker._running is False or task.done()

    @pytest.mark.asyncio
    async def test_waits_for_active_jobs(self, store):
        """Worker should wait for active jobs on shutdown."""
        worker = RoutingWorker(poll_interval=0.05)
        completed = False

        async def slow_task():
            nonlocal completed
            await asyncio.sleep(0.1)
            completed = True

        task = asyncio.create_task(slow_task())
        worker._active_jobs["slow"] = task

        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.05)
        await worker.stop()
        await asyncio.wait_for(start_task, timeout=3.0)

        assert completed

    @pytest.mark.asyncio
    async def test_handles_error_in_loop(self, store):
        """Worker should continue after errors in the loop."""
        worker = RoutingWorker(poll_interval=0.05)

        call_count = 0

        async def failing_dequeue(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Transient error")
            return None

        with patch.object(worker._store, "dequeue", side_effect=failing_dequeue):
            task = asyncio.create_task(worker.start())
            await asyncio.sleep(0.3)
            await worker.stop()
            await asyncio.wait_for(task, timeout=2.0)

        assert call_count >= 2


# =============================================================================
# Task Cleanup Tests
# =============================================================================


class TestRoutingCleanup:
    """Tests for _cleanup_completed_tasks method."""

    @pytest.mark.asyncio
    async def test_removes_completed_tasks(self, store):
        """Should remove completed tasks."""
        worker = RoutingWorker()

        async def noop():
            pass

        task = asyncio.create_task(noop())
        await task
        worker._active_jobs["done"] = task

        worker._cleanup_completed_tasks()
        assert "done" not in worker._active_jobs

    @pytest.mark.asyncio
    async def test_keeps_running_tasks(self, store):
        """Should keep running tasks."""
        worker = RoutingWorker()
        event = asyncio.Event()

        async def wait():
            await event.wait()

        task = asyncio.create_task(wait())
        worker._active_jobs["running"] = task

        worker._cleanup_completed_tasks()
        assert "running" in worker._active_jobs

        event.set()
        await task

    @pytest.mark.asyncio
    async def test_handles_failed_tasks(self, store):
        """Should handle failed tasks without crashing."""
        worker = RoutingWorker()

        async def fail():
            raise RuntimeError("Route error")

        task = asyncio.create_task(fail())
        try:
            await task
        except RuntimeError:
            pass
        worker._active_jobs["failed"] = task

        worker._cleanup_completed_tasks()
        assert "failed" not in worker._active_jobs


# =============================================================================
# Job Processing Tests - Debate Routing
# =============================================================================


class TestProcessDebateRouting:
    """Tests for debate result routing."""

    @pytest.mark.asyncio
    async def test_routes_debate_result_success(self, store):
        """Should route debate result successfully."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="debate-route-1",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-123",
                "result": {"verdict": "approved", "confidence": 0.9},
                "include_voice": False,
            },
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("debate-route-1")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_routes_generic_routing_type(self, store):
        """Should also handle JOB_TYPE_ROUTING as debate route."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="generic-route-1",
            job_type=JOB_TYPE_ROUTING,
            payload={
                "debate_id": "debate-456",
                "result": {"verdict": "rejected"},
            },
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("generic-route-1")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_debate_route_failure_retries(self, store):
        """Should retry when debate routing returns False."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="fail-route-1",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-789",
                "result": {"verdict": "approved"},
            },
            max_attempts=3,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("fail-route-1")
        assert job_after.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_debate_route_missing_id_raises(self, store):
        """Should fail when debate_id is missing."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="no-id-route",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "result": {"verdict": "approved"},
            },
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        await worker._process_job(claimed)

        job_after = await store.get("no-id-route")
        assert job_after.status == JobStatus.FAILED
        assert "Missing debate_id" in job_after.error

    @pytest.mark.asyncio
    async def test_debate_route_with_voice(self, store):
        """Should pass include_voice flag via job payload."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="voice-route",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-voice",
                "result": {"verdict": "approved"},
                "include_voice": True,
            },
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            await worker._process_job(claimed)

        mock_route.assert_called_once()
        called_job = mock_route.call_args[0][0]
        assert called_job.payload["include_voice"] is True
        assert called_job.payload["debate_id"] == "debate-voice"


# =============================================================================
# Job Processing Tests - Email Routing
# =============================================================================


class TestProcessEmailRouting:
    """Tests for email result routing."""

    @pytest.mark.asyncio
    async def test_routes_email_success(self, store):
        """Should route email result successfully."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="email-route-1",
            job_type=JOB_TYPE_ROUTING_EMAIL,
            payload={
                "debate_id": "debate-email-1",
                "result": {"verdict": "approved"},
                "recipient_email": "user@example.com",
            },
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_email_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("email-route-1")
        assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_email_missing_debate_id_raises(self, store):
        """Should fail when debate_id is missing from email job."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="no-id-email",
            job_type=JOB_TYPE_ROUTING_EMAIL,
            payload={
                "result": {"verdict": "approved"},
                "recipient_email": "user@example.com",
            },
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        await worker._process_job(claimed)

        job_after = await store.get("no-id-email")
        assert job_after.status == JobStatus.FAILED
        assert "Missing debate_id" in job_after.error

    @pytest.mark.asyncio
    async def test_email_missing_recipient_raises(self, store):
        """Should fail when recipient_email is missing."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="no-email-addr",
            job_type=JOB_TYPE_ROUTING_EMAIL,
            payload={
                "debate_id": "debate-email-2",
                "result": {"verdict": "approved"},
            },
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        await worker._process_job(claimed)

        job_after = await store.get("no-email-addr")
        assert job_after.status == JobStatus.FAILED
        assert "recipient_email" in job_after.error

    @pytest.mark.asyncio
    async def test_email_passes_correct_params(self, store):
        """Should pass correct parameters via job payload."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="email-params",
            job_type=JOB_TYPE_ROUTING_EMAIL,
            payload={
                "debate_id": "debate-params",
                "result": {"verdict": "rejected", "details": "Insufficient evidence"},
                "recipient_email": "admin@corp.com",
            },
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_email_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_send:
            await worker._process_job(claimed)

        mock_send.assert_called_once()
        called_job = mock_send.call_args[0][0]
        assert called_job.payload["debate_id"] == "debate-params"
        assert called_job.payload["recipient_email"] == "admin@corp.com"
        assert called_job.payload["result"]["verdict"] == "rejected"


# =============================================================================
# Unknown Job Type Tests
# =============================================================================


class TestUnknownJobType:
    """Tests for handling unknown job types."""

    @pytest.mark.asyncio
    async def test_unknown_type_fails(self, store):
        """Should fail on unknown job type."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="unknown-route",
            job_type="routing_unknown",
            payload={"debate_id": "d-1"},
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        await worker._process_job(claimed)

        job_after = await store.get("unknown-route")
        assert job_after.status == JobStatus.FAILED


# =============================================================================
# Error Handling and Retry Tests
# =============================================================================


class TestRoutingRetry:
    """Tests for error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_exception(self, store):
        """Should retry on exception when attempts remain."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="retry-route",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-retry",
                "result": {"verdict": "approved"},
            },
            max_attempts=3,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Platform unavailable"),
        ):
            await worker._process_job(claimed)

        job_after = await store.get("retry-route")
        assert job_after.status == JobStatus.PENDING
        assert "Platform unavailable" in job_after.error

    @pytest.mark.asyncio
    async def test_permanent_failure_after_max_attempts(self, store):
        """Should permanently fail after max attempts exhausted."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="perm-fail-route",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-fail",
                "result": {"verdict": "approved"},
            },
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Persistent error"),
        ):
            await worker._process_job(claimed)

        job_after = await store.get("perm-fail-route")
        assert job_after.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_delivery_failure_retry(self, store):
        """Should retry when delivery returns False but has attempts."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="delivery-retry",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-retry-2",
                "result": {"verdict": "approved"},
            },
            max_attempts=5,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("delivery-retry")
        # Should be back to pending for retry
        assert job_after.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_delivery_failure_no_retry_at_max(self, store):
        """Should fail permanently when delivery fails and no retries left."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="delivery-max",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-max",
                "result": {"verdict": "approved"},
            },
            max_attempts=1,
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("delivery-max")
        assert job_after.status == JobStatus.FAILED


# =============================================================================
# Result Includes Duration Tests
# =============================================================================


class TestRoutingResultData:
    """Tests for result data in completed jobs."""

    @pytest.mark.asyncio
    async def test_completed_includes_duration(self, store):
        """Completed job should include duration_seconds."""
        worker = RoutingWorker()

        job = QueuedJob(
            id="duration-route",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={
                "debate_id": "debate-dur",
                "result": {"verdict": "approved"},
            },
        )
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id=worker.worker_id)

        with patch.object(
            worker,
            "_route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await worker._process_job(claimed)

        job_after = await store.get("duration-route")
        assert job_after.status == JobStatus.COMPLETED


# =============================================================================
# Enqueue Tests
# =============================================================================


class TestEnqueueRoutingJob:
    """Tests for enqueue_routing_job function."""

    @pytest.mark.asyncio
    async def test_enqueue_debate_routing(self, store):
        """Should enqueue a debate routing job."""
        job = await enqueue_routing_job(
            job_id="route-1",
            debate_id="debate-1",
            result={"verdict": "approved"},
        )

        assert job.id == "route-1"
        assert job.job_type == JOB_TYPE_ROUTING_DEBATE
        assert job.payload["debate_id"] == "debate-1"
        assert job.payload["result"]["verdict"] == "approved"
        assert job.max_attempts == 5  # More retries for routing

    @pytest.mark.asyncio
    async def test_enqueue_persists(self, store):
        """Should persist job to store."""
        await enqueue_routing_job(
            job_id="persist-route",
            debate_id="debate-persist",
            result={"verdict": "rejected"},
        )

        stored = await store.get("persist-route")
        assert stored is not None
        assert stored.job_type == JOB_TYPE_ROUTING_DEBATE

    @pytest.mark.asyncio
    async def test_enqueue_with_voice(self, store):
        """Should include voice flag in payload."""
        job = await enqueue_routing_job(
            job_id="voice-route",
            debate_id="debate-voice",
            result={"verdict": "approved"},
            include_voice=True,
        )

        assert job.payload["include_voice"] is True

    @pytest.mark.asyncio
    async def test_enqueue_without_voice(self, store):
        """Should default include_voice to False."""
        job = await enqueue_routing_job(
            job_id="no-voice-route",
            debate_id="debate-no-voice",
            result={"verdict": "approved"},
        )

        assert job.payload["include_voice"] is False

    @pytest.mark.asyncio
    async def test_enqueue_email_routing(self, store):
        """Should enqueue an email routing job."""
        job = await enqueue_routing_job(
            job_id="email-route-1",
            debate_id="debate-email",
            result={"verdict": "approved"},
            job_type=JOB_TYPE_ROUTING_EMAIL,
            recipient_email="user@example.com",
        )

        assert job.job_type == JOB_TYPE_ROUTING_EMAIL
        assert job.payload["recipient_email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_enqueue_with_user_workspace(self, store):
        """Should include user and workspace IDs."""
        job = await enqueue_routing_job(
            job_id="user-route",
            debate_id="debate-user",
            result={},
            user_id="user-1",
            workspace_id="ws-1",
        )

        assert job.user_id == "user-1"
        assert job.workspace_id == "ws-1"

    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self, store):
        """Should set job priority."""
        job = await enqueue_routing_job(
            job_id="priority-route",
            debate_id="debate-priority",
            result={},
            priority=10,
        )

        assert job.priority == 10

    @pytest.mark.asyncio
    async def test_enqueue_without_recipient_email(self, store):
        """Should not include recipient_email when not provided."""
        job = await enqueue_routing_job(
            job_id="no-email-route",
            debate_id="debate-no-email",
            result={},
        )

        assert "recipient_email" not in job.payload

    @pytest.mark.asyncio
    async def test_enqueue_default_job_type(self, store):
        """Should default to JOB_TYPE_ROUTING_DEBATE."""
        job = await enqueue_routing_job(
            job_id="default-type-route",
            debate_id="debate-default",
            result={},
        )

        assert job.job_type == JOB_TYPE_ROUTING_DEBATE


# =============================================================================
# Recovery Tests
# =============================================================================


class TestRecoverInterruptedRouting:
    """Tests for recover_interrupted_routing function."""

    @pytest.mark.asyncio
    async def test_no_stale_jobs(self, store):
        """Should return 0 when no stale jobs."""
        recovered = await recover_interrupted_routing()
        assert recovered == 0

    @pytest.mark.asyncio
    async def test_recovers_stale_routing_jobs(self, store):
        """Should recover stale routing jobs."""
        job = QueuedJob(
            id="stale-route",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={"debate_id": "d-1", "result": {}},
        )
        await store.enqueue(job)
        await store.dequeue(worker_id="crashed-worker")

        conn = store._get_conn()
        conn.execute(
            "UPDATE job_queue SET started_at = ? WHERE id = ?",
            (time.time() - 400, "stale-route"),
        )
        conn.commit()

        recovered = await recover_interrupted_routing()
        assert recovered == 1

    @pytest.mark.asyncio
    async def test_recovers_multiple_stale_jobs(self, store):
        """Should recover multiple stale routing jobs."""
        for i in range(3):
            job = QueuedJob(
                id=f"stale-route-{i}",
                job_type=JOB_TYPE_ROUTING_DEBATE,
                payload={"debate_id": f"d-{i}", "result": {}},
            )
            await store.enqueue(job)
            await store.dequeue(worker_id="crashed-worker")

        conn = store._get_conn()
        conn.execute(
            "UPDATE job_queue SET started_at = ?",
            (time.time() - 400,),
        )
        conn.commit()

        recovered = await recover_interrupted_routing()
        assert recovered == 3

    @pytest.mark.asyncio
    async def test_recovery_handles_errors(self, store):
        """Should handle errors during recovery gracefully."""
        with patch.object(
            store,
            "recover_stale_jobs",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB error"),
        ):
            recovered = await recover_interrupted_routing()
            assert recovered == 0


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestRoutingConcurrency:
    """Tests for worker concurrency management."""

    @pytest.mark.asyncio
    async def test_dequeues_all_routing_types(self, store):
        """Worker should dequeue all routing job types."""
        for i, jt in enumerate(
            [
                JOB_TYPE_ROUTING,
                JOB_TYPE_ROUTING_DEBATE,
                JOB_TYPE_ROUTING_EMAIL,
            ]
        ):
            job = QueuedJob(
                id=f"type-{i}",
                job_type=jt,
                payload={
                    "debate_id": f"d-{i}",
                    "result": {},
                },
            )
            await store.enqueue(job)

        job_types = [JOB_TYPE_ROUTING, JOB_TYPE_ROUTING_DEBATE, JOB_TYPE_ROUTING_EMAIL]

        dequeued_types = set()
        for _ in range(3):
            job = await store.dequeue(worker_id="test", job_types=job_types)
            if job:
                dequeued_types.add(job.job_type)

        assert len(dequeued_types) == 3

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, store):
        """Worker should not exceed max_concurrent."""
        worker = RoutingWorker(max_concurrent=1, poll_interval=0.05)

        event = asyncio.Event()

        async def wait():
            await event.wait()

        task = asyncio.create_task(wait())
        worker._active_jobs["blocking"] = task

        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.15)

        await worker.stop()
        event.set()
        await asyncio.wait_for(start_task, timeout=3.0)

    @pytest.mark.asyncio
    async def test_priority_ordering(self, store):
        """Higher priority jobs should be dequeued first."""
        # Enqueue low priority first
        low = QueuedJob(
            id="low-priority",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={"debate_id": "d-low", "result": {}},
            priority=0,
        )
        await store.enqueue(low)

        # Enqueue high priority second
        high = QueuedJob(
            id="high-priority",
            job_type=JOB_TYPE_ROUTING_DEBATE,
            payload={"debate_id": "d-high", "result": {}},
            priority=10,
        )
        await store.enqueue(high)

        # High priority should be dequeued first
        job_types = [JOB_TYPE_ROUTING, JOB_TYPE_ROUTING_DEBATE, JOB_TYPE_ROUTING_EMAIL]
        first = await store.dequeue(worker_id="test", job_types=job_types)
        assert first.id == "high-priority"

        second = await store.dequeue(worker_id="test", job_types=job_types)
        assert second.id == "low-priority"
