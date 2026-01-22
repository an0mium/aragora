"""
Tests for Queue Handler - Job Queue Management API.

Tests cover:
- Job submission
- Job listing and filtering
- Job status retrieval
- Job retry functionality
- Job cancellation
- Queue statistics
- Worker status
- Path validation (security)
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum

import pytest


class MockJobStatus(Enum):
    """Mock job status enum."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class MockJob:
    """Mock job for testing."""

    def __init__(
        self,
        id: str = "job-123",
        status: MockJobStatus = MockJobStatus.PENDING,
        job_type: str = "debate",
        created_at: float = 1704067200.0,  # 2024-01-01 00:00:00
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        failed_at: Optional[float] = None,
        attempts: int = 0,
        max_attempts: int = 3,
        priority: int = 0,
        error: Optional[str] = None,
        worker_id: Optional[str] = None,
        payload: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.id = id
        self.status = status
        self.job_type = job_type
        self.created_at = created_at
        self.failed_at = failed_at
        self.started_at = started_at
        self.completed_at = completed_at
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.priority = priority
        self.error = error
        self.worker_id = worker_id
        self.payload = payload or {}
        self.metadata = metadata or {}


class MockQueue:
    """Mock queue for testing."""

    def __init__(self):
        self.jobs: Dict[str, MockJob] = {}
        self.stream_key = "aragora:jobs"
        self._redis = AsyncMock()
        self._status_tracker = MagicMock()

    async def enqueue(self, job, priority: int = 0) -> str:
        job_id = getattr(job, "id", f"job-{len(self.jobs)}")
        self.jobs[job_id] = job
        return job_id

    async def get_status(self, job_id: str) -> Optional[MockJob]:
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status in (MockJobStatus.PENDING, MockJobStatus.RETRYING):
            job.status = MockJobStatus.CANCELLED
            return True
        return False

    async def get_queue_stats(self) -> Dict[str, int]:
        return {
            "pending": 5,
            "processing": 2,
            "completed": 100,
            "failed": 3,
            "cancelled": 1,
            "retrying": 0,
            "stream_length": 111,
            "pending_in_group": 5,
        }

    async def list_jobs(self, status=None, limit: int = 100, offset: int = 0) -> List[MockJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs[offset : offset + limit]


def create_mock_handler(
    method: str = "GET", body: Optional[Dict] = None, path: str = "/api/v1/queue/jobs"
) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {"Content-Type": "application/json"}

    if body:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = BytesIO(body_bytes)
        handler.headers["Content-Length"] = str(len(body_bytes))
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> Dict:
    """Extract body from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
    else:
        body = result[0]

    if isinstance(body, dict):
        return body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable rate limits for all tests."""

    def _always_allowed(key: str) -> bool:
        return True

    import sys

    if "aragora.server.handlers.utils.rate_limit" in sys.modules:
        rl_module = sys.modules["aragora.server.handlers.utils.rate_limit"]
        original = {}
        for name, limiter in rl_module._limiters.items():
            original[name] = limiter.is_allowed
            limiter.is_allowed = _always_allowed
        yield
        for name, orig in original.items():
            if name in rl_module._limiters:
                rl_module._limiters[name].is_allowed = orig
    else:
        yield


@pytest.fixture
def mock_queue():
    """Create a mock queue."""
    return MockQueue()


@pytest.fixture
def queue_handler():
    """Create queue handler instance."""
    from aragora.server.handlers.queue import QueueHandler

    return QueueHandler({})


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_jobs_endpoint(self, queue_handler):
        assert queue_handler.can_handle("/api/v1/queue/jobs") is True

    def test_handles_specific_job_endpoint(self, queue_handler):
        assert queue_handler.can_handle("/api/v1/queue/jobs/job-123") is True

    def test_handles_retry_endpoint(self, queue_handler):
        assert queue_handler.can_handle("/api/v1/queue/jobs/job-123/retry") is True

    def test_handles_stats_endpoint(self, queue_handler):
        assert queue_handler.can_handle("/api/v1/queue/stats") is True

    def test_handles_workers_endpoint(self, queue_handler):
        assert queue_handler.can_handle("/api/v1/queue/workers") is True

    def test_does_not_handle_other_routes(self, queue_handler):
        assert queue_handler.can_handle("/api/v1/debates") is False
        assert queue_handler.can_handle("/api/v1/users") is False


class TestQueueStats:
    """Tests for queue statistics endpoint."""

    @pytest.mark.asyncio
    async def test_stats_returns_queue_unavailable(self, queue_handler):
        """Should return 503 when queue is not available."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await queue_handler.handle("/api/v1/queue/stats", "GET")

            assert get_status(result) == 503
            body = get_body(result)
            assert "error" in body

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, queue_handler, mock_queue):
        """Should return queue statistics."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/stats", "GET")

            assert get_status(result) == 200
            body = get_body(result)
            assert "stats" in body
            assert body["stats"]["pending"] == 5
            assert body["stats"]["completed"] == 100


class TestJobSubmission:
    """Tests for job submission endpoint."""

    @pytest.mark.asyncio
    async def test_submit_requires_question(self, queue_handler, mock_queue):
        """Should require question field."""
        mock_handler = create_mock_handler("POST", {"agents": ["claude"]})

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/jobs", "POST", mock_handler)

            assert get_status(result) == 400
            body = get_body(result)
            assert "question" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_submit_creates_job(self, queue_handler, mock_queue):
        """Should create and enqueue a job."""
        mock_handler = create_mock_handler(
            "POST", {"question": "What is the meaning of life?", "rounds": 3}
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.create_debate_job") as mock_create:
                mock_job = MockJob()
                mock_create.return_value = mock_job

                result = await queue_handler.handle("/api/v1/queue/jobs", "POST", mock_handler)

                assert get_status(result) == 202
                body = get_body(result)
                assert "job_id" in body
                assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_submit_queue_unavailable(self, queue_handler):
        """Should return 503 when queue unavailable."""
        mock_handler = create_mock_handler("POST", {"question": "test?"})

        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await queue_handler.handle("/api/v1/queue/jobs", "POST", mock_handler)

            assert get_status(result) == 503


class TestJobListing:
    """Tests for job listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_jobs_returns_jobs(self, queue_handler, mock_queue):
        """Should return list of jobs."""
        mock_queue.jobs["job-1"] = MockJob(id="job-1")
        mock_queue.jobs["job-2"] = MockJob(id="job-2")
        mock_queue._status_tracker.list_jobs = AsyncMock(
            return_value=[
                MockJob(id="job-1"),
                MockJob(id="job-2"),
            ]
        )
        mock_queue._status_tracker.get_counts_by_status = AsyncMock(
            return_value={"pending": 2, "completed": 0}
        )

        mock_handler = create_mock_handler("GET")

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/jobs", "GET", mock_handler)

                assert get_status(result) == 200
                body = get_body(result)
                assert "jobs" in body
                assert len(body["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_queue_unavailable(self, queue_handler):
        """Should return 503 when queue unavailable."""
        mock_handler = create_mock_handler("GET")

        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await queue_handler.handle("/api/v1/queue/jobs", "GET", mock_handler)

            assert get_status(result) == 503


class TestJobRetrieval:
    """Tests for getting specific job status."""

    @pytest.mark.asyncio
    async def test_get_job_returns_status(self, queue_handler, mock_queue):
        """Should return job details."""
        mock_queue.jobs["job-123"] = MockJob(
            id="job-123",
            status=MockJobStatus.COMPLETED,
            completed_at=1704153600.0,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/jobs/job-123", "GET")

            assert get_status(result) == 200
            body = get_body(result)
            assert body["job_id"] == "job-123"
            assert body["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, queue_handler, mock_queue):
        """Should return 404 for missing job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/jobs/nonexistent", "GET")

            assert get_status(result) == 404


class TestJobRetry:
    """Tests for job retry endpoint."""

    @pytest.mark.asyncio
    async def test_retry_failed_job(self, queue_handler, mock_queue):
        """Should retry failed jobs."""
        mock_queue.jobs["job-123"] = MockJob(
            id="job-123",
            status=MockJobStatus.FAILED,
            error="Network timeout",
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/jobs/job-123/retry", "POST")

                assert get_status(result) == 200
                body = get_body(result)
                assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_retry_non_failed_job_rejected(self, queue_handler, mock_queue):
        """Should reject retry of non-failed jobs."""
        mock_queue.jobs["job-123"] = MockJob(
            id="job-123",
            status=MockJobStatus.PROCESSING,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/jobs/job-123/retry", "POST")

                assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_retry_not_found(self, queue_handler, mock_queue):
        """Should return 404 for missing job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/jobs/nonexistent/retry", "POST")

                assert get_status(result) == 404


class TestJobCancellation:
    """Tests for job cancellation endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, queue_handler, mock_queue):
        """Should cancel pending jobs."""
        mock_queue.jobs["job-123"] = MockJob(
            id="job-123",
            status=MockJobStatus.PENDING,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/jobs/job-123", "DELETE")

            assert get_status(result) == 200
            body = get_body(result)
            assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_processing_job_rejected(self, queue_handler, mock_queue):
        """Should reject cancellation of processing jobs."""
        mock_queue.jobs["job-123"] = MockJob(
            id="job-123",
            status=MockJobStatus.PROCESSING,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/jobs/job-123", "DELETE")

            assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, queue_handler, mock_queue):
        """Should return 404 for missing job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/jobs/nonexistent", "DELETE")

            assert get_status(result) == 404


class TestPathValidation:
    """Tests for path segment validation (security)."""

    @pytest.mark.asyncio
    async def test_rejects_path_traversal_in_job_id(self, queue_handler, mock_queue):
        """Should reject path traversal attempts.

        Note: Paths with too many segments don't match routes (return None).
        This tests a job_id that contains '..' but matches the route pattern.
        """
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            # Use a job_id with path traversal characters that matches route length
            result = await queue_handler.handle("/api/v1/queue/jobs/..passwd", "GET")

            assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_rejects_special_chars_in_job_id(self, queue_handler, mock_queue):
        """Should reject special characters in job ID."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            # Use a job_id with invalid characters that matches route length
            result = await queue_handler.handle("/api/v1/queue/jobs/job@invalid", "GET")

            assert get_status(result) == 400


class TestWorkerStatus:
    """Tests for worker status endpoint."""

    @pytest.mark.asyncio
    async def test_workers_queue_unavailable(self, queue_handler):
        """Should return 503 when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await queue_handler.handle("/api/v1/queue/workers", "GET")

            assert get_status(result) == 503
            body = get_body(result)
            assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_workers_returns_list(self, queue_handler, mock_queue):
        """Should return worker list."""
        mock_queue._redis.xinfo_groups = AsyncMock(
            return_value=[{"name": "workers", "consumers": 2}]
        )
        mock_queue._redis.xinfo_consumers = AsyncMock(
            return_value=[
                {"name": "worker-1", "pending": 1, "idle": 1000},
                {"name": "worker-2", "pending": 0, "idle": 5000},
            ]
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await queue_handler.handle("/api/v1/queue/workers", "GET")

            assert get_status(result) == 200
            body = get_body(result)
            assert "workers" in body
            assert body["total"] == 2


class TestDeadLetterQueue:
    """Tests for dead-letter queue (DLQ) endpoints."""

    @pytest.mark.asyncio
    async def test_list_dlq(self, queue_handler, mock_queue):
        """Should list failed jobs in DLQ."""
        # Add a failed job with max retries exceeded
        mock_queue.jobs["dlq-job-1"] = MockJob(
            id="dlq-job-1",
            status=MockJobStatus.FAILED,
            attempts=3,
            max_attempts=3,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/dlq", "GET")

                assert get_status(result) == 200
                body = get_body(result)
                assert "jobs" in body
                assert "total" in body

    @pytest.mark.asyncio
    async def test_list_dlq_queue_unavailable(self, queue_handler):
        """Should return 503 when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await queue_handler.handle("/api/v1/queue/dlq", "GET")

            assert get_status(result) == 503

    @pytest.mark.asyncio
    async def test_requeue_dlq_job(self, queue_handler, mock_queue):
        """Should requeue a specific DLQ job."""
        mock_queue.jobs["dlq-job-1"] = MockJob(
            id="dlq-job-1",
            status=MockJobStatus.FAILED,
            attempts=3,
            max_attempts=3,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/dlq/dlq-job-1/requeue", "POST")

                assert get_status(result) == 200
                body = get_body(result)
                assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_requeue_dlq_job_not_found(self, queue_handler, mock_queue):
        """Should return 404 for missing DLQ job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/dlq/nonexistent/requeue", "POST")

                assert get_status(result) == 404

    @pytest.mark.asyncio
    async def test_requeue_dlq_job_not_in_dlq(self, queue_handler, mock_queue):
        """Should reject requeue of non-DLQ job."""
        mock_queue.jobs["job-123"] = MockJob(
            id="job-123",
            status=MockJobStatus.PENDING,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/dlq/job-123/requeue", "POST")

                assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_requeue_all_dlq(self, queue_handler, mock_queue):
        """Should requeue all DLQ jobs."""
        mock_queue.jobs["dlq-job-1"] = MockJob(
            id="dlq-job-1",
            status=MockJobStatus.FAILED,
            attempts=3,
            max_attempts=3,
        )
        mock_queue.jobs["dlq-job-2"] = MockJob(
            id="dlq-job-2",
            status=MockJobStatus.FAILED,
            attempts=3,
            max_attempts=3,
        )

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await queue_handler.handle("/api/v1/queue/dlq/requeue", "POST")

                assert get_status(result) == 200
                body = get_body(result)
                assert "requeued" in body

    @pytest.mark.asyncio
    async def test_requeue_all_dlq_queue_unavailable(self, queue_handler):
        """Should return 503 when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await queue_handler.handle("/api/v1/queue/dlq/requeue", "POST")

            assert get_status(result) == 503

    def test_handles_dlq_endpoints(self, queue_handler):
        """Should recognize DLQ endpoints."""
        assert queue_handler.can_handle("/api/v1/queue/dlq") is True
        assert queue_handler.can_handle("/api/v1/queue/dlq/requeue") is True
        assert queue_handler.can_handle("/api/v1/queue/dlq/job-123/requeue") is True
