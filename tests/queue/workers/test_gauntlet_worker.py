"""
Tests for Gauntlet Job Queue Worker.

Tests durable job processing for gauntlet stress-testing.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.workers.gauntlet_worker import (
    GauntletWorker,
    JOB_TYPE_GAUNTLET,
    enqueue_gauntlet_job,
    recover_interrupted_gauntlets,
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
        yield Path(tmpdir) / "test_gauntlet_jobs.db"


@pytest.fixture
def store(temp_db):
    """Create a SQLite job store for testing."""
    reset_job_store()
    store = SQLiteJobStore(temp_db)
    set_job_store(store)
    yield store
    reset_job_store()


class TestEnqueueGauntletJob:
    """Tests for enqueue_gauntlet_job function."""

    @pytest.mark.asyncio
    async def test_enqueues_job(self, store):
        """Should enqueue a gauntlet job."""
        job = await enqueue_gauntlet_job(
            gauntlet_id="gauntlet-test-123",
            input_content="Test content",
            input_type="spec",
            persona="sec-auditor",
            agents=["anthropic-api"],
            profile="default",
        )

        assert job.id == "gauntlet-test-123"
        assert job.job_type == JOB_TYPE_GAUNTLET
        assert job.payload["input_content"] == "Test content"
        assert job.payload["persona"] == "sec-auditor"

    @pytest.mark.asyncio
    async def test_persists_to_store(self, store):
        """Should persist job to store."""
        await enqueue_gauntlet_job(
            gauntlet_id="gauntlet-persist-test",
            input_content="Persist test",
            input_type="policy",
            persona=None,
            agents=["openai-api"],
            profile="strict",
        )

        # Verify in store
        stored = await store.get("gauntlet-persist-test")
        assert stored is not None
        assert stored.job_type == JOB_TYPE_GAUNTLET
        assert stored.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_includes_user_workspace(self, store):
        """Should include user and workspace IDs."""
        job = await enqueue_gauntlet_job(
            gauntlet_id="gauntlet-user-test",
            input_content="User test",
            input_type="spec",
            persona=None,
            agents=["anthropic-api"],
            profile="default",
            user_id="user-123",
            workspace_id="ws-456",
        )

        assert job.user_id == "user-123"
        assert job.workspace_id == "ws-456"

    @pytest.mark.asyncio
    async def test_priority(self, store):
        """Should set job priority."""
        job = await enqueue_gauntlet_job(
            gauntlet_id="gauntlet-priority-test",
            input_content="Priority test",
            input_type="spec",
            persona=None,
            agents=["anthropic-api"],
            profile="default",
            priority=10,
        )

        assert job.priority == 10


class TestGauntletWorker:
    """Tests for GauntletWorker class."""

    @pytest.mark.asyncio
    async def test_initializes(self, store):
        """Should initialize with defaults."""
        worker = GauntletWorker()
        assert worker.worker_id.startswith("gauntlet-worker-")
        assert worker.poll_interval == 2.0
        assert worker.max_concurrent == 3

    @pytest.mark.asyncio
    async def test_custom_worker_id(self, store):
        """Should accept custom worker ID."""
        worker = GauntletWorker(worker_id="custom-worker-1")
        assert worker.worker_id == "custom-worker-1"

    @pytest.mark.asyncio
    async def test_dequeues_gauntlet_jobs(self, store):
        """Should only dequeue gauntlet job types."""
        # Enqueue a gauntlet job
        await enqueue_gauntlet_job(
            gauntlet_id="gauntlet-dequeue-test",
            input_content="Dequeue test",
            input_type="spec",
            persona=None,
            agents=["anthropic-api"],
            profile="default",
        )

        # Enqueue a non-gauntlet job
        other_job = QueuedJob(
            id="other-job",
            job_type="workflow",
            payload={},
        )
        await store.enqueue(other_job)

        # Dequeue with gauntlet filter
        claimed = await store.dequeue(
            worker_id="test-worker",
            job_types=[JOB_TYPE_GAUNTLET],
        )

        assert claimed is not None
        assert claimed.id == "gauntlet-dequeue-test"

    @pytest.mark.asyncio
    async def test_stop_gracefully(self, store):
        """Should stop gracefully."""
        worker = GauntletWorker()

        # Start and immediately stop
        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)
        await worker.stop()

        # Wait for task to complete
        await asyncio.wait_for(start_task, timeout=2.0)

        assert not worker._running


class TestRecoverInterruptedGauntlets:
    """Tests for recover_interrupted_gauntlets function."""

    @pytest.mark.asyncio
    async def test_recovers_stale_jobs(self, store):
        """Should recover stale jobs from queue."""
        # Enqueue a job and manually make it stale
        job = QueuedJob(
            id="stale-gauntlet",
            job_type=JOB_TYPE_GAUNTLET,
            payload={
                "gauntlet_id": "stale-gauntlet",
                "input_content": "Stale test",
                "input_type": "spec",
                "persona": None,
                "agents": ["anthropic-api"],
                "profile": "default",
            },
        )
        await store.enqueue(job)

        # Claim it (simulating a worker that crashed)
        await store.dequeue(worker_id="crashed-worker")

        # Manually backdate it
        import time

        conn = store._get_conn()
        conn.execute(
            "UPDATE job_queue SET started_at = ? WHERE id = ?",
            (time.time() - 400, "stale-gauntlet"),
        )
        conn.commit()

        # Recover stale jobs
        recovered = await store.recover_stale_jobs(stale_threshold_seconds=300.0)
        assert recovered == 1

        # Job should be pending again
        job_after = await store.get("stale-gauntlet")
        assert job_after is not None
        assert job_after.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_no_stale_jobs(self, store):
        """Should handle case with no stale jobs."""
        recovered = await recover_interrupted_gauntlets()
        assert recovered == 0


class TestWorkerJobProcessing:
    """Tests for worker job processing logic."""

    @pytest.mark.asyncio
    async def test_marks_completed_on_success(self, store):
        """Should mark job completed on successful execution."""
        # Mock the gauntlet execution
        mock_result = MagicMock()
        mock_result.verdict.value = "pass"
        mock_result.confidence = 0.95
        mock_result.risk_score = 0.1
        mock_result.robustness_score = 0.9
        mock_result.total_findings = 2
        mock_result.critical_findings = []
        mock_result.high_findings = []
        mock_result.medium_findings = []
        mock_result.low_findings = []

        with patch(
            "aragora.queue.workers.gauntlet_worker.GauntletWorker._execute_gauntlet",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.return_value = {
                "gauntlet_id": "test-gauntlet",
                "verdict": "pass",
                "confidence": 0.95,
            }

            worker = GauntletWorker()

            # Create a job
            job = QueuedJob(
                id="test-job",
                job_type=JOB_TYPE_GAUNTLET,
                payload={
                    "gauntlet_id": "test-gauntlet",
                    "input_content": "Test",
                    "input_type": "spec",
                    "persona": None,
                    "agents": ["anthropic-api"],
                    "profile": "default",
                },
            )
            await store.enqueue(job)
            claimed = await store.dequeue(worker_id=worker.worker_id)

            # Process job
            await worker._process_job(claimed)

            # Job should be completed
            job_after = await store.get("test-job")
            assert job_after is not None
            assert job_after.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_marks_failed_on_error(self, store):
        """Should mark job failed and schedule retry on error."""
        with patch(
            "aragora.queue.workers.gauntlet_worker.GauntletWorker._execute_gauntlet",
            new_callable=AsyncMock,
        ) as mock_execute:
            mock_execute.side_effect = RuntimeError("Test error")

            worker = GauntletWorker()

            # Create a job with 3 max attempts
            job = QueuedJob(
                id="fail-job",
                job_type=JOB_TYPE_GAUNTLET,
                payload={
                    "gauntlet_id": "fail-gauntlet",
                    "input_content": "Test",
                    "input_type": "spec",
                    "persona": None,
                    "agents": ["anthropic-api"],
                    "profile": "default",
                },
                max_attempts=3,
            )
            await store.enqueue(job)
            claimed = await store.dequeue(worker_id=worker.worker_id)

            # Process job (should fail)
            await worker._process_job(claimed)

            # Job should be back to pending for retry
            job_after = await store.get("fail-job")
            assert job_after is not None
            assert job_after.status == JobStatus.PENDING
            assert job_after.error == "Test error"
