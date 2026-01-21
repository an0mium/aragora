"""
Tests for Job Queue Store.

Tests durable job queue storage with SQLite backend.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pytest

from aragora.storage.job_queue_store import (
    JobStatus,
    QueuedJob,
    SQLiteJobStore,
    get_job_store,
    reset_job_store,
    set_job_store,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_jobs.db"


@pytest.fixture
def store(temp_db):
    """Create a SQLite job store for testing."""
    return SQLiteJobStore(temp_db)


def create_test_job(
    job_type: str = "gauntlet",
    priority: int = 0,
    **kwargs,
) -> QueuedJob:
    """Create a test job."""
    return QueuedJob(
        id=str(uuid4()),
        job_type=job_type,
        payload=kwargs.get("payload", {"test": True}),
        priority=priority,
        user_id=kwargs.get("user_id"),
        workspace_id=kwargs.get("workspace_id"),
    )


class TestQueuedJob:
    """Tests for QueuedJob dataclass."""

    def test_creates_with_defaults(self):
        """Should create job with sensible defaults."""
        job = QueuedJob(id="test-1", job_type="gauntlet", payload={})
        assert job.status == JobStatus.PENDING
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.priority == 0

    def test_to_dict(self):
        """Should serialize to dict."""
        job = QueuedJob(
            id="test-1",
            job_type="gauntlet",
            payload={"question": "test?"},
            priority=5,
        )
        data = job.to_dict()
        assert data["id"] == "test-1"
        assert data["job_type"] == "gauntlet"
        assert data["payload"]["question"] == "test?"
        assert data["status"] == "pending"
        assert data["priority"] == 5

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "id": "test-2",
            "job_type": "workflow",
            "payload": {"step": 1},
            "status": "processing",
            "priority": 10,
            "attempts": 1,
        }
        job = QueuedJob.from_dict(data)
        assert job.id == "test-2"
        assert job.job_type == "workflow"
        assert job.status == JobStatus.PROCESSING
        assert job.priority == 10
        assert job.attempts == 1


class TestSQLiteJobStore:
    """Tests for SQLite job store."""

    @pytest.mark.asyncio
    async def test_enqueue_and_get(self, store):
        """Should enqueue and retrieve a job."""
        job = create_test_job()
        await store.enqueue(job)

        retrieved = await store.get(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.job_type == job.job_type
        assert retrieved.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_dequeue_claims_job(self, store):
        """Should claim job when dequeuing."""
        job = create_test_job()
        await store.enqueue(job)

        claimed = await store.dequeue(worker_id="worker-1")
        assert claimed is not None
        assert claimed.id == job.id
        assert claimed.status == JobStatus.PROCESSING
        assert claimed.worker_id == "worker-1"
        assert claimed.attempts == 1

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_when_empty(self, store):
        """Should return None when no jobs available."""
        result = await store.dequeue(worker_id="worker-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_dequeue_filters_by_type(self, store):
        """Should filter by job type when dequeuing."""
        job1 = create_test_job(job_type="gauntlet")
        job2 = create_test_job(job_type="workflow")
        await store.enqueue(job1)
        await store.enqueue(job2)

        # Should only get workflow job
        claimed = await store.dequeue(worker_id="w1", job_types=["workflow"])
        assert claimed is not None
        assert claimed.job_type == "workflow"

    @pytest.mark.asyncio
    async def test_dequeue_respects_priority(self, store):
        """Should dequeue higher priority jobs first."""
        low_priority = create_test_job(priority=1)
        high_priority = create_test_job(priority=10)

        # Enqueue low first
        await store.enqueue(low_priority)
        await store.enqueue(high_priority)

        # Should get high priority first
        claimed = await store.dequeue(worker_id="w1")
        assert claimed is not None
        assert claimed.id == high_priority.id

    @pytest.mark.asyncio
    async def test_complete_marks_done(self, store):
        """Should mark job as completed."""
        job = create_test_job()
        await store.enqueue(job)
        await store.dequeue(worker_id="w1")

        await store.complete(job.id, result={"success": True})

        completed = await store.get(job.id)
        assert completed is not None
        assert completed.status == JobStatus.COMPLETED
        assert completed.result == {"success": True}
        assert completed.completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_with_retry(self, store):
        """Should schedule retry on failure."""
        job = create_test_job()
        await store.enqueue(job)
        await store.dequeue(worker_id="w1")

        await store.fail(job.id, error="Test error", should_retry=True)

        failed = await store.get(job.id)
        assert failed is not None
        assert failed.status == JobStatus.PENDING  # Back to pending for retry
        assert failed.error == "Test error"

    @pytest.mark.asyncio
    async def test_fail_permanent(self, store):
        """Should mark as permanently failed after max attempts."""
        job = QueuedJob(
            id=str(uuid4()),
            job_type="gauntlet",
            payload={},
            attempts=3,
            max_attempts=3,
        )
        await store.enqueue(job)

        await store.fail(job.id, error="Final error")

        failed = await store.get(job.id)
        assert failed is not None
        assert failed.status == JobStatus.FAILED
        assert failed.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, store):
        """Should cancel a pending job."""
        job = create_test_job()
        await store.enqueue(job)

        result = await store.cancel(job.id)
        assert result is True

        cancelled = await store.get(job.id)
        assert cancelled is not None
        assert cancelled.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_processing_job_fails(self, store):
        """Should not cancel a processing job."""
        job = create_test_job()
        await store.enqueue(job)
        await store.dequeue(worker_id="w1")

        result = await store.cancel(job.id)
        assert result is False

    @pytest.mark.asyncio
    async def test_recover_stale_jobs(self, store):
        """Should recover jobs stuck in processing."""
        job = create_test_job()
        await store.enqueue(job)
        claimed = await store.dequeue(worker_id="w1")

        # Manually backdate the started_at time
        conn = store._get_conn()
        conn.execute(
            "UPDATE job_queue SET started_at = ? WHERE id = ?",
            (time.time() - 400, job.id),  # 400 seconds ago
        )
        conn.commit()

        # Recover with 300s threshold
        recovered = await store.recover_stale_jobs(stale_threshold_seconds=300.0)
        assert recovered == 1

        # Job should be pending again
        job_after = await store.get(job.id)
        assert job_after is not None
        assert job_after.status == JobStatus.PENDING
        assert job_after.worker_id is None

    @pytest.mark.asyncio
    async def test_list_jobs(self, store):
        """Should list jobs with filtering."""
        job1 = create_test_job(job_type="gauntlet")
        job2 = create_test_job(job_type="workflow")
        await store.enqueue(job1)
        await store.enqueue(job2)

        # List all
        all_jobs = await store.list_jobs()
        assert len(all_jobs) == 2

        # Filter by type
        gauntlet_jobs = await store.list_jobs(job_type="gauntlet")
        assert len(gauntlet_jobs) == 1
        assert gauntlet_jobs[0].job_type == "gauntlet"

    @pytest.mark.asyncio
    async def test_get_stats(self, store):
        """Should return queue statistics."""
        job1 = create_test_job()
        job2 = create_test_job()
        await store.enqueue(job1)
        await store.enqueue(job2)
        await store.dequeue(worker_id="w1")

        stats = await store.get_stats()
        assert stats["pending"] == 1
        assert stats["processing"] == 1
        assert stats["total"] == 2


class TestJobStoreFactory:
    """Tests for job store factory functions."""

    def test_get_job_store_returns_singleton(self, temp_db, monkeypatch):
        """Should return same instance on multiple calls."""
        reset_job_store()
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(temp_db.parent))

        store1 = get_job_store()
        store2 = get_job_store()

        assert store1 is store2
        reset_job_store()

    def test_set_job_store_overrides(self, store):
        """Should allow setting custom store."""
        reset_job_store()
        set_job_store(store)

        retrieved = get_job_store()
        assert retrieved is store
        reset_job_store()
