"""
Tests for QueueHandler.

Tests the REST API endpoints for job queue management:
- Job submission
- Job listing and status
- Job retry and cancellation
- Queue statistics
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum

import pytest

from aragora.server.handlers.queue import QueueHandler, _get_queue


class MockJobStatus(Enum):
    """Mock job status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MockJob:
    """Mock job for testing."""

    def __init__(
        self,
        id: str,
        status: MockJobStatus = MockJobStatus.PENDING,
        payload: Dict[str, Any] = None,
    ):
        self.id = id
        self.status = status
        self.payload = payload or {"question": "Test question"}
        self.created_at = datetime.now().timestamp()
        self.started_at = None
        self.completed_at = None
        self.attempts = 0
        self.max_attempts = 3
        self.priority = 0
        self.error = None
        self.worker_id = None
        self.metadata = {}


class MockQueue:
    """Mock queue for testing."""

    def __init__(self):
        self._jobs: Dict[str, MockJob] = {}
        self._status_tracker = MagicMock()
        self._redis = MagicMock()
        self.stream_key = "aragora:jobs"

    async def get_queue_stats(self) -> Dict[str, int]:
        pending = sum(1 for j in self._jobs.values() if j.status == MockJobStatus.PENDING)
        processing = sum(1 for j in self._jobs.values() if j.status == MockJobStatus.PROCESSING)
        completed = sum(1 for j in self._jobs.values() if j.status == MockJobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == MockJobStatus.FAILED)
        cancelled = sum(1 for j in self._jobs.values() if j.status == MockJobStatus.CANCELLED)
        return {
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "retrying": 0,
            "stream_length": len(self._jobs),
            "pending_in_group": pending,
        }

    async def enqueue(self, job: MockJob, priority: int = 0) -> str:
        self._jobs[job.id] = job
        job.priority = priority
        return job.id

    async def get_status(self, job_id: str) -> Optional[MockJob]:
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in (MockJobStatus.PENDING, MockJobStatus.PROCESSING):
            job.status = MockJobStatus.CANCELLED
            return True
        return False


@pytest.fixture
def mock_queue():
    """Create a mock queue."""
    return MockQueue()


@pytest.fixture
def handler():
    """Create a handler."""
    return QueueHandler({})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.path = "/api/queue/jobs"
    return handler


class TestQueueHandlerRouting:
    """Test handler routing."""

    def test_can_handle_queue_paths(self, handler):
        """Test handler recognizes queue paths."""
        assert handler.can_handle("/api/queue/jobs") is True
        assert handler.can_handle("/api/queue/stats") is True
        assert handler.can_handle("/api/queue/workers") is True

    def test_cannot_handle_other_paths(self, handler):
        """Test handler rejects non-queue paths."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False


class TestGetStats:
    """Test GET /api/queue/stats."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_queue, mock_http_handler):
        """Test getting queue statistics."""
        # Add some jobs
        mock_queue._jobs["job-1"] = MockJob("job-1", MockJobStatus.PENDING)
        mock_queue._jobs["job-2"] = MockJob("job-2", MockJobStatus.COMPLETED)
        mock_queue._jobs["job-3"] = MockJob("job-3", MockJobStatus.FAILED)

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._get_stats()

        assert result is not None
        data = json.loads(result.body)
        assert "stats" in data
        assert data["stats"]["pending"] == 1
        assert data["stats"]["completed"] == 1
        assert data["stats"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_queue_unavailable(self, handler, mock_http_handler):
        """Test stats when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._get_stats()

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data["error"] or "not available" in data.get("message", "")


class TestGetWorkers:
    """Test GET /api/queue/workers."""

    @pytest.mark.asyncio
    async def test_get_workers_queue_unavailable(self, handler, mock_http_handler):
        """Test workers when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._get_workers()

        assert result.status_code == 503
        data = json.loads(result.body)
        assert data["workers"] == []


class TestSubmitJob:
    """Test POST /api/queue/jobs."""

    @pytest.mark.asyncio
    async def test_submit_job_queue_unavailable(self, handler, mock_http_handler):
        """Test job submission when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._submit_job(mock_http_handler)

        assert result.status_code == 503


class TestListJobs:
    """Test GET /api/queue/jobs."""

    @pytest.mark.asyncio
    async def test_list_jobs_queue_unavailable(self, handler, mock_http_handler):
        """Test listing jobs when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._list_jobs({})

        assert result.status_code == 503
        data = json.loads(result.body)
        assert data["jobs"] == []


class TestGetJob:
    """Test GET /api/queue/jobs/:id."""

    @pytest.mark.asyncio
    async def test_get_job_success(self, handler, mock_queue, mock_http_handler):
        """Test getting job status."""
        job = MockJob("job-1", MockJobStatus.PENDING)
        mock_queue._jobs["job-1"] = job

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._get_job("job-1")

        assert result is not None
        data = json.loads(result.body)
        assert data["job_id"] == "job-1"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, handler, mock_queue, mock_http_handler):
        """Test getting non-existent job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._get_job("nonexistent")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_job_queue_unavailable(self, handler, mock_http_handler):
        """Test getting job when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._get_job("job-1")

        assert result.status_code == 503


class TestRetryJob:
    """Test POST /api/queue/jobs/:id/retry."""

    @pytest.mark.asyncio
    async def test_retry_job_queue_unavailable(self, handler, mock_http_handler):
        """Test retrying job when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._retry_job("job-1")

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_retry_job_not_found(self, handler, mock_queue, mock_http_handler):
        """Test retrying non-existent job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._retry_job("nonexistent")

        assert result.status_code == 404


class TestCancelJob:
    """Test DELETE /api/queue/jobs/:id."""

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, handler, mock_queue, mock_http_handler):
        """Test cancelling a pending job."""
        job = MockJob("job-1", MockJobStatus.PENDING)
        mock_queue._jobs["job-1"] = job

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._cancel_job("job-1")

        assert result is not None
        data = json.loads(result.body)
        assert data["status"] == "cancelled"
        assert mock_queue._jobs["job-1"].status == MockJobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, handler, mock_queue, mock_http_handler):
        """Test cancelling non-existent job."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._cancel_job("nonexistent")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_job_already_completed(self, handler, mock_queue, mock_http_handler):
        """Test cancelling completed job."""
        job = MockJob("job-1", MockJobStatus.COMPLETED)
        mock_queue._jobs["job-1"] = job

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler._cancel_job("job-1")

        # Should fail because job is already completed
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_cancel_job_queue_unavailable(self, handler, mock_http_handler):
        """Test cancelling job when queue unavailable."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler._cancel_job("job-1")

        assert result.status_code == 503
