"""
Comprehensive tests for aragora.server.handlers.queue - Queue Management Handler.

Tests cover:
1. Job submission endpoints (POST /api/queue/jobs)
2. Job status queries (GET /api/queue/jobs/:id)
3. Job listing with pagination
4. Worker status management
5. Retry logic and failure handling
6. Job cancellation
7. Queue statistics endpoints
8. Input validation for job submissions
9. Error handling for invalid jobs
10. Authentication/authorization checks
11. Dead-letter queue (DLQ) operations
12. Job cleanup operations
13. Stale job detection
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.base import JobStatus
from aragora.server.handlers.queue import QueueHandler, _get_queue


# ===========================================================================
# Test Fixtures and Helpers
# ===========================================================================


@dataclass
class MockJob:
    """Mock job for testing."""

    id: str = "job-123"
    status: JobStatus = JobStatus.PENDING
    payload: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 0
    error: str | None = None
    worker_id: str | None = None
    metadata: dict = field(default_factory=dict)
    job_type: str = "debate"


class MockStatusTracker:
    """Mock status tracker for testing."""

    def __init__(self, jobs: list[MockJob] | None = None):
        self.jobs = jobs or []

    async def list_jobs(self, status=None, limit=100):
        result = self.jobs
        if status is not None:
            result = [j for j in self.jobs if j.status == status]
        return result[:limit]

    async def get_counts_by_status(self):
        counts = {}
        for job in self.jobs:
            status = job.status.value if hasattr(job.status, "value") else str(job.status)
            counts[status] = counts.get(status, 0) + 1
        return counts


class MockQueue:
    """Mock queue for testing."""

    def __init__(self, jobs: list[MockJob] | None = None):
        self.jobs = {j.id: j for j in (jobs or [])}
        self._status_tracker = MockStatusTracker(jobs or [])
        self._redis = MagicMock()
        self.stream_key = "aragora:queue:stream"
        self.enqueued = []
        self.cancelled = []
        self.deleted = []

    async def get_queue_stats(self):
        return {
            "pending": 5,
            "processing": 2,
            "completed": 10,
            "failed": 1,
            "cancelled": 0,
            "retrying": 1,
            "stream_length": 19,
            "pending_in_group": 3,
        }

    async def get_status(self, job_id: str):
        return self.jobs.get(job_id)

    async def enqueue(self, job, priority=0):
        self.enqueued.append(job)
        return job.id if hasattr(job, "id") else "new-job-id"

    async def cancel(self, job_id: str):
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status not in (JobStatus.PENDING, JobStatus.RETRYING):
            return False
        self.cancelled.append(job_id)
        job.status = JobStatus.CANCELLED
        return True

    async def list_jobs(self, status=None, limit=100):
        return await self._status_tracker.list_jobs(status=status, limit=limit)

    async def delete(self, job_id: str):
        self.deleted.append(job_id)
        return True


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "admin"
    permissions: list = field(default_factory=lambda: ["queue:read", "queue:manage", "queue:admin"])
    workspace_id: str | None = None


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    return json.loads(body)


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
    path: str = "/api/queue/jobs",
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.path = path
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.headers["Content-Type"] = "application/json"
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.headers["Content-Length"] = "0"
        handler.rfile = BytesIO(b"")

    return handler


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {}


@pytest.fixture
def queue_handler(server_context):
    """Create a QueueHandler instance."""
    handler = QueueHandler(server_context)
    return handler


@pytest.fixture
def mock_auth_context():
    """Create a mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_queue():
    """Create a mock queue."""
    jobs = [
        MockJob(
            id="job-1",
            status=JobStatus.PENDING,
            payload={"question": "Test question 1"},
            created_at=time.time() - 3600,
        ),
        MockJob(
            id="job-2",
            status=JobStatus.PROCESSING,
            payload={"question": "Test question 2"},
            created_at=time.time() - 1800,
            started_at=time.time() - 900,
            worker_id="worker-1",
        ),
        MockJob(
            id="job-3",
            status=JobStatus.COMPLETED,
            payload={"question": "Test question 3"},
            created_at=time.time() - 7200,
            completed_at=time.time() - 3600,
        ),
    ]
    return MockQueue(jobs)


# ===========================================================================
# Test: Handler Structure and Routing
# ===========================================================================


class TestQueueHandlerStructure:
    """Tests for QueueHandler class structure."""

    def test_handler_has_routes(self, queue_handler):
        """Handler should have ROUTES defined."""
        assert hasattr(QueueHandler, "ROUTES")
        assert len(QueueHandler.ROUTES) > 0

    def test_routes_include_all_endpoints(self, queue_handler):
        """Routes should include all queue endpoints."""
        routes = QueueHandler.ROUTES
        assert "/api/queue/jobs" in routes
        assert "/api/queue/stats" in routes
        assert "/api/queue/workers" in routes
        assert "/api/queue/dlq" in routes

    def test_can_handle_queue_paths(self, queue_handler):
        """Handler should match queue paths."""
        assert queue_handler.can_handle("/api/queue/jobs", "GET")
        assert queue_handler.can_handle("/api/queue/stats", "GET")
        assert queue_handler.can_handle("/api/queue/workers", "GET")
        assert queue_handler.can_handle("/api/v1/queue/jobs", "GET")

    def test_rejects_non_queue_paths(self, queue_handler):
        """Handler should reject non-queue paths."""
        assert not queue_handler.can_handle("/api/debates", "GET")
        assert not queue_handler.can_handle("/api/agents", "GET")
        assert not queue_handler.can_handle("/health", "GET")

    def test_resource_type_is_queue(self, queue_handler):
        """Handler should have resource_type set to queue."""
        assert queue_handler.RESOURCE_TYPE == "queue"


# ===========================================================================
# Test: Authentication and Authorization
# ===========================================================================


class TestQueueAuthentication:
    """Tests for authentication requirements."""

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, queue_handler):
        """Requests without auth should return 401."""
        handler = make_mock_handler()
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(
            queue_handler,
            "get_auth_context",
            side_effect=UnauthorizedError("Not authenticated"),
        ):
            result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
            assert get_status(result) == 401

    @pytest.mark.asyncio
    async def test_missing_permission_returns_403(self, queue_handler):
        """Requests without proper permission should return 403."""
        handler = make_mock_handler()
        auth = MockAuthContext(permissions=[])
        from aragora.server.handlers.utils.auth import ForbiddenError

        with patch.object(queue_handler, "get_auth_context", return_value=auth):
            with patch.object(
                queue_handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ):
                result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                assert get_status(result) == 403

    @pytest.mark.asyncio
    async def test_read_permission_for_stats(self, queue_handler, mock_auth_context):
        """Stats endpoint should require queue:read permission."""
        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission") as mock_check:
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    mock_check.assert_called_once()
                    args = mock_check.call_args[0]
                    assert args[1] == "queue:read"

    @pytest.mark.asyncio
    async def test_read_permission_for_job_list(self, queue_handler, mock_auth_context):
        """Job listing endpoint should require queue:read permission."""
        handler = make_mock_handler(
            method="GET",
            path="/api/queue/jobs",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission") as mock_check:
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    mock_check.assert_called_once()
                    args = mock_check.call_args[0]
                    # Job listing is a read operation
                    assert args[1] == "queue:read"

    @pytest.mark.asyncio
    async def test_manage_permission_for_job_retry(
        self, queue_handler, mock_auth_context, mock_queue
    ):
        """Job retry should require queue:manage permission."""
        handler = make_mock_handler(
            method="POST",
            path="/api/queue/jobs/job-1/retry",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission") as mock_check:
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    with patch("aragora.server.handlers.queue.audit_admin"):
                        result = await queue_handler.handle(
                            "/api/queue/jobs/job-1/retry", "POST", handler
                        )
                        mock_check.assert_called_once()
                        args = mock_check.call_args[0]
                        # Retry is a manage operation
                        assert args[1] == "queue:manage"

    @pytest.mark.asyncio
    async def test_admin_permission_for_dlq(self, queue_handler, mock_auth_context):
        """DLQ endpoint should require queue:admin permission."""
        handler = make_mock_handler(path="/api/queue/dlq")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission") as mock_check:
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/dlq", "GET", handler)
                    mock_check.assert_called_once()
                    args = mock_check.call_args[0]
                    assert args[1] == "queue:admin"


# ===========================================================================
# Test: Queue Statistics Endpoint
# ===========================================================================


class TestQueueStats:
    """Tests for GET /api/queue/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, queue_handler, mock_auth_context, mock_queue):
        """Should return queue statistics."""
        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert "stats" in body
                    assert body["stats"]["pending"] == 5
                    assert body["stats"]["processing"] == 2
                    assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_get_stats_queue_unavailable(self, queue_handler, mock_auth_context):
        """Should return 503 when queue is unavailable."""
        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 503
                    body = get_body(result)
                    assert "error" in body
                    assert "stats" in body  # Should still have empty stats

    @pytest.mark.asyncio
    async def test_get_stats_connection_error(self, queue_handler, mock_auth_context):
        """Should handle connection errors gracefully."""
        handler = make_mock_handler(path="/api/queue/stats")
        mock_q = MagicMock()
        mock_q.get_queue_stats = AsyncMock(side_effect=ConnectionError("Redis down"))

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 503


# ===========================================================================
# Test: Worker Status Endpoint
# ===========================================================================


class TestWorkerStatus:
    """Tests for GET /api/queue/workers endpoint."""

    @pytest.mark.asyncio
    async def test_get_workers_success(self, queue_handler, mock_auth_context):
        """Should return worker information."""
        handler = make_mock_handler(path="/api/queue/workers")
        mock_q = MockQueue()
        mock_q._redis.xinfo_groups = AsyncMock(
            return_value=[
                {"name": "workers"},
            ]
        )
        mock_q._redis.xinfo_consumers = AsyncMock(
            return_value=[
                {"name": "worker-1", "pending": 2, "idle": 1000},
                {"name": "worker-2", "pending": 1, "idle": 500},
            ]
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/workers", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert "workers" in body
                    assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_get_workers_queue_unavailable(self, queue_handler, mock_auth_context):
        """Should return 503 when queue is unavailable."""
        handler = make_mock_handler(path="/api/queue/workers")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/workers", "GET", handler)
                    assert get_status(result) == 503

    @pytest.mark.asyncio
    async def test_get_workers_handles_connection_error(self, queue_handler, mock_auth_context):
        """Should handle connection errors gracefully."""
        handler = make_mock_handler(path="/api/queue/workers")
        mock_q = MockQueue()
        mock_q._redis.xinfo_groups = AsyncMock(side_effect=ConnectionError("Redis down"))

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/workers", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["workers"] == []


# ===========================================================================
# Test: Job Submission (POST /api/queue/jobs)
# ===========================================================================


class TestJobSubmission:
    """Tests for POST /api/queue/jobs endpoint."""

    @pytest.mark.asyncio
    async def test_submit_job_success(self, queue_handler, mock_auth_context, mock_queue):
        """Should submit a job successfully."""
        handler = make_mock_handler(
            body={"question": "What is the best programming language?"},
            method="POST",
            path="/api/queue/jobs",
        )

        mock_job = MagicMock()
        mock_job.id = "new-job-id"

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    with patch(
                        "aragora.queue.create_debate_job", return_value=mock_job
                    ) as mock_create:
                        with patch("aragora.server.handlers.queue.audit_data"):
                            result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                            assert get_status(result) == 202
                            body = get_body(result)
                            assert body["job_id"] == "new-job-id"
                            assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_submit_job_with_all_options(self, queue_handler, mock_auth_context, mock_queue):
        """Should accept all job options."""
        handler = make_mock_handler(
            body={
                "question": "Test question",
                "agents": ["claude", "gpt"],
                "rounds": 5,
                "consensus": "unanimous",
                "protocol": "adversarial",
                "priority": 10,
                "max_attempts": 5,
                "timeout_seconds": 600,
                "webhook_url": "https://example.com/webhook",
                "user_id": "user-456",
                "organization_id": "org-789",
                "metadata": {"custom": "data"},
            },
            method="POST",
            path="/api/queue/jobs",
        )

        mock_job = MagicMock()
        mock_job.id = "new-job-id"

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    with patch(
                        "aragora.queue.create_debate_job", return_value=mock_job
                    ) as mock_create:
                        with patch("aragora.server.handlers.queue.audit_data"):
                            result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                            assert get_status(result) == 202
                            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_job_missing_question(self, queue_handler, mock_auth_context, mock_queue):
        """Should return 400 when question is missing."""
        handler = make_mock_handler(
            body={"agents": ["claude", "gpt"]},
            method="POST",
            path="/api/queue/jobs",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                    assert get_status(result) == 400
                    body = get_body(result)
                    assert "question" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_submit_job_invalid_body(self, queue_handler, mock_auth_context, mock_queue):
        """Should return 400 for invalid JSON body."""
        handler = make_mock_handler(method="POST", path="/api/queue/jobs")
        handler.headers["Content-Length"] = "10"
        handler.rfile = BytesIO(b"not json{}")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_job_queue_unavailable(self, queue_handler, mock_auth_context):
        """Should return 503 when queue is unavailable."""
        handler = make_mock_handler(
            body={"question": "Test question"},
            method="POST",
            path="/api/queue/jobs",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                    assert get_status(result) == 503

    @pytest.mark.asyncio
    async def test_submit_job_connection_error(self, queue_handler, mock_auth_context):
        """Should handle connection errors during enqueue."""
        handler = make_mock_handler(
            body={"question": "Test question"},
            method="POST",
            path="/api/queue/jobs",
        )
        mock_q = MagicMock()
        mock_q.enqueue = AsyncMock(side_effect=ConnectionError("Redis down"))

        mock_job = MagicMock()

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    with patch("aragora.queue.create_debate_job", return_value=mock_job):
                        result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                        assert get_status(result) == 503


# ===========================================================================
# Test: Job Listing (GET /api/queue/jobs)
# ===========================================================================


class TestJobListing:
    """Tests for GET /api/queue/jobs endpoint."""

    @pytest.mark.asyncio
    async def test_list_jobs_success(self, queue_handler, mock_auth_context, mock_queue):
        """Should list jobs with pagination."""
        handler = make_mock_handler(path="/api/queue/jobs")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert "jobs" in body
                    assert "total" in body
                    assert "limit" in body
                    assert "offset" in body

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, queue_handler, mock_auth_context, mock_queue):
        """Should filter jobs by status."""
        handler = make_mock_handler(path="/api/queue/jobs?status=pending")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    assert get_status(result) == 200

    @pytest.mark.asyncio
    async def test_list_jobs_invalid_status(self, queue_handler, mock_auth_context, mock_queue):
        """Should return 400 for invalid status filter."""
        handler = make_mock_handler(path="/api/queue/jobs?status=invalid")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_list_jobs_with_pagination(self, queue_handler, mock_auth_context, mock_queue):
        """Should handle pagination parameters."""
        handler = make_mock_handler(path="/api/queue/jobs?limit=10&offset=5")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["limit"] == 10
                    assert body["offset"] == 5

    @pytest.mark.asyncio
    async def test_list_jobs_queue_unavailable(self, queue_handler, mock_auth_context):
        """Should return 503 when queue is unavailable."""
        handler = make_mock_handler(path="/api/queue/jobs")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    assert get_status(result) == 503


# ===========================================================================
# Test: Get Job Status (GET /api/queue/jobs/:id)
# ===========================================================================


class TestGetJobStatus:
    """Tests for GET /api/queue/jobs/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_job_success(self, queue_handler, mock_auth_context, mock_queue):
        """Should return job details."""
        handler = make_mock_handler(path="/api/queue/jobs/job-1")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs/job-1", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["job_id"] == "job-1"
                    assert "status" in body
                    assert "created_at" in body

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, queue_handler, mock_auth_context, mock_queue):
        """Should return 404 for non-existent job."""
        handler = make_mock_handler(path="/api/queue/jobs/nonexistent")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/nonexistent", "GET", handler
                    )
                    assert get_status(result) == 404

    @pytest.mark.asyncio
    async def test_get_job_with_valid_uuid(self, queue_handler, mock_auth_context, mock_queue):
        """Should accept valid UUID job IDs."""
        handler = make_mock_handler(path="/api/queue/jobs/123e4567-e89b-12d3-a456-426614174000")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/123e4567-e89b-12d3-a456-426614174000", "GET", handler
                    )
                    # Should be 404 not found (not 400 invalid ID)
                    assert get_status(result) == 404

    @pytest.mark.asyncio
    async def test_get_job_queue_unavailable(self, queue_handler, mock_auth_context):
        """Should return 503 when queue is unavailable."""
        handler = make_mock_handler(path="/api/queue/jobs/job-1")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/jobs/job-1", "GET", handler)
                    assert get_status(result) == 503


# ===========================================================================
# Test: Job Retry (POST /api/queue/jobs/:id/retry)
# ===========================================================================


class TestJobRetry:
    """Tests for POST /api/queue/jobs/:id/retry endpoint."""

    @pytest.mark.asyncio
    async def test_retry_failed_job_success(self, queue_handler, mock_auth_context):
        """Should retry a failed job."""
        failed_job = MockJob(id="failed-job", status=JobStatus.FAILED, error="Previous error")
        mock_q = MockQueue([failed_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/jobs/failed-job/retry",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    with patch("aragora.server.handlers.queue.audit_admin"):
                        result = await queue_handler.handle(
                            "/api/queue/jobs/failed-job/retry", "POST", handler
                        )
                        assert get_status(result) == 200
                        body = get_body(result)
                        assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_retry_cancelled_job_success(self, queue_handler, mock_auth_context):
        """Should retry a cancelled job."""
        cancelled_job = MockJob(id="cancelled-job", status=JobStatus.CANCELLED)
        mock_q = MockQueue([cancelled_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/jobs/cancelled-job/retry",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    with patch("aragora.server.handlers.queue.audit_admin"):
                        result = await queue_handler.handle(
                            "/api/queue/jobs/cancelled-job/retry", "POST", handler
                        )
                        assert get_status(result) == 200

    @pytest.mark.asyncio
    async def test_retry_completed_job_fails(self, queue_handler, mock_auth_context):
        """Should not allow retrying a completed job."""
        completed_job = MockJob(id="completed-job", status=JobStatus.COMPLETED)
        mock_q = MockQueue([completed_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/jobs/completed-job/retry",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/completed-job/retry", "POST", handler
                    )
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_retry_processing_job_fails(self, queue_handler, mock_auth_context):
        """Should not allow retrying a processing job."""
        processing_job = MockJob(id="processing-job", status=JobStatus.PROCESSING)
        mock_q = MockQueue([processing_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/jobs/processing-job/retry",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/processing-job/retry", "POST", handler
                    )
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_retry_nonexistent_job(self, queue_handler, mock_auth_context, mock_queue):
        """Should return 404 for non-existent job."""
        handler = make_mock_handler(
            method="POST",
            path="/api/queue/jobs/nonexistent/retry",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/nonexistent/retry", "POST", handler
                    )
                    assert get_status(result) == 404


# ===========================================================================
# Test: Job Cancellation (DELETE /api/queue/jobs/:id)
# ===========================================================================


class TestJobCancellation:
    """Tests for DELETE /api/queue/jobs/:id endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job_success(self, queue_handler, mock_auth_context, mock_queue):
        """Should cancel a pending job."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/queue/jobs/job-1",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    with patch("aragora.server.handlers.queue.audit_admin"):
                        result = await queue_handler.handle(
                            "/api/queue/jobs/job-1", "DELETE", handler
                        )
                        assert get_status(result) == 200
                        body = get_body(result)
                        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_processing_job_fails(self, queue_handler, mock_auth_context, mock_queue):
        """Should not allow cancelling a processing job."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/queue/jobs/job-2",  # This is processing
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs/job-2", "DELETE", handler)
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, queue_handler, mock_auth_context, mock_queue):
        """Should return 404 for non-existent job."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/queue/jobs/nonexistent",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/nonexistent", "DELETE", handler
                    )
                    assert get_status(result) == 404

    @pytest.mark.asyncio
    async def test_cancel_job_special_chars(self, queue_handler, mock_auth_context, mock_queue):
        """Should handle job IDs with special characters."""
        handler = make_mock_handler(
            method="DELETE",
            path="/api/queue/jobs/test-job-123",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/test-job-123", "DELETE", handler
                    )
                    # Should get 404 not found (valid ID format but doesn't exist)
                    assert get_status(result) == 404


# ===========================================================================
# Test: Dead-Letter Queue (DLQ) Operations
# ===========================================================================


class TestDLQOperations:
    """Tests for DLQ endpoints."""

    @pytest.mark.asyncio
    async def test_list_dlq_success(self, queue_handler, mock_auth_context):
        """Should list DLQ jobs."""
        dlq_job = MockJob(
            id="dlq-job",
            status=JobStatus.FAILED,
            attempts=3,
            max_attempts=3,
            error="Max retries exceeded",
            completed_at=time.time() - 3600,
        )
        mock_q = MockQueue([dlq_job])

        handler = make_mock_handler(path="/api/queue/dlq")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/dlq", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert "jobs" in body
                    assert "total" in body

    @pytest.mark.asyncio
    async def test_list_dlq_with_pagination(self, queue_handler, mock_auth_context):
        """Should paginate DLQ results."""
        dlq_jobs = [
            MockJob(
                id=f"dlq-job-{i}",
                status=JobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                completed_at=time.time() - 3600,
            )
            for i in range(10)
        ]
        mock_q = MockQueue(dlq_jobs)

        handler = make_mock_handler(path="/api/queue/dlq?limit=5&offset=2")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/dlq", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["limit"] == 5
                    assert body["offset"] == 2

    @pytest.mark.asyncio
    async def test_requeue_dlq_job_success(self, queue_handler, mock_auth_context):
        """Should requeue a specific DLQ job."""
        dlq_job = MockJob(
            id="dlq-job",
            status=JobStatus.FAILED,
            attempts=3,
            max_attempts=3,
            metadata={},
        )
        mock_q = MockQueue([dlq_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/dlq/dlq-job/requeue",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    with patch("aragora.server.handlers.queue.audit_admin"):
                        result = await queue_handler.handle(
                            "/api/queue/dlq/dlq-job/requeue", "POST", handler
                        )
                        assert get_status(result) == 200
                        body = get_body(result)
                        assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_requeue_non_failed_job_fails(self, queue_handler, mock_auth_context):
        """Should not requeue a non-failed job."""
        pending_job = MockJob(id="pending-job", status=JobStatus.PENDING)
        mock_q = MockQueue([pending_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/dlq/pending-job/requeue",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle(
                        "/api/queue/dlq/pending-job/requeue", "POST", handler
                    )
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_requeue_all_dlq_success(self, queue_handler, mock_auth_context):
        """Should requeue all DLQ jobs."""
        dlq_jobs = [
            MockJob(
                id=f"dlq-job-{i}",
                status=JobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                metadata={},
            )
            for i in range(3)
        ]
        mock_q = MockQueue(dlq_jobs)

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/dlq/requeue",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    with patch("aragora.server.handlers.queue.audit_admin"):
                        result = await queue_handler.handle(
                            "/api/queue/dlq/requeue", "POST", handler
                        )
                        assert get_status(result) == 200
                        body = get_body(result)
                        assert body["requeued"] == 3


# ===========================================================================
# Test: Job Cleanup Operations
# ===========================================================================


class TestJobCleanup:
    """Tests for POST /api/queue/cleanup endpoint."""

    @pytest.mark.asyncio
    async def test_cleanup_old_completed_jobs(self, queue_handler, mock_auth_context):
        """Should cleanup old completed jobs."""
        old_job = MockJob(
            id="old-job",
            status=JobStatus.COMPLETED,
            completed_at=time.time() - (8 * 86400),  # 8 days old
        )
        mock_q = MockQueue([old_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/cleanup?older_than_days=7",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/cleanup", "POST", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["deleted"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, queue_handler, mock_auth_context):
        """Should support dry run mode."""
        old_job = MockJob(
            id="old-job",
            status=JobStatus.COMPLETED,
            completed_at=time.time() - (8 * 86400),
        )
        mock_q = MockQueue([old_job])

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/cleanup?dry_run=true",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/cleanup", "POST", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["deleted"] == 0
                    assert body["would_delete"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_status_filter(self, queue_handler, mock_auth_context):
        """Should filter by status."""
        jobs = [
            MockJob(
                id="completed-job",
                status=JobStatus.COMPLETED,
                completed_at=time.time() - (8 * 86400),
            ),
            MockJob(
                id="failed-job",
                status=JobStatus.FAILED,
                completed_at=time.time() - (8 * 86400),
            ),
        ]
        mock_q = MockQueue(jobs)

        handler = make_mock_handler(
            method="POST",
            path="/api/queue/cleanup?status=failed",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/cleanup", "POST", handler)
                    assert get_status(result) == 200

    @pytest.mark.asyncio
    async def test_cleanup_invalid_status(self, queue_handler, mock_auth_context, mock_queue):
        """Should reject invalid status filter."""
        handler = make_mock_handler(
            method="POST",
            path="/api/queue/cleanup?status=invalid",
        )

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/cleanup", "POST", handler)
                    assert get_status(result) == 400


# ===========================================================================
# Test: Stale Job Detection
# ===========================================================================


class TestStaleJobDetection:
    """Tests for GET /api/queue/stale endpoint."""

    @pytest.mark.asyncio
    async def test_list_stale_jobs_success(self, queue_handler, mock_auth_context):
        """Should list stale jobs."""
        stale_job = MockJob(
            id="stale-job",
            status=JobStatus.PROCESSING,
            started_at=time.time() - (2 * 3600),  # 2 hours ago
            worker_id="dead-worker",
        )
        mock_q = MockQueue([stale_job])

        handler = make_mock_handler(path="/api/queue/stale?stale_minutes=60")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stale", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert "jobs" in body
                    assert len(body["jobs"]) == 1

    @pytest.mark.asyncio
    async def test_list_stale_jobs_no_stale(self, queue_handler, mock_auth_context):
        """Should return empty list when no stale jobs."""
        recent_job = MockJob(
            id="recent-job",
            status=JobStatus.PROCESSING,
            started_at=time.time() - 60,  # 1 minute ago
            worker_id="active-worker",
        )
        mock_q = MockQueue([recent_job])

        handler = make_mock_handler(path="/api/queue/stale?stale_minutes=60")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stale", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert len(body["jobs"]) == 0

    @pytest.mark.asyncio
    async def test_list_stale_jobs_sorted_by_duration(self, queue_handler, mock_auth_context):
        """Should sort stale jobs by duration (longest first)."""
        jobs = [
            MockJob(
                id="stale-1",
                status=JobStatus.PROCESSING,
                started_at=time.time() - (2 * 3600),  # 2 hours
                worker_id="worker-1",
            ),
            MockJob(
                id="stale-2",
                status=JobStatus.PROCESSING,
                started_at=time.time() - (4 * 3600),  # 4 hours (longest)
                worker_id="worker-2",
            ),
        ]
        mock_q = MockQueue(jobs)

        handler = make_mock_handler(path="/api/queue/stale?stale_minutes=60")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stale", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert len(body["jobs"]) == 2
                    # Longest duration first
                    assert body["jobs"][0]["job_id"] == "stale-2"


# ===========================================================================
# Test: Input Validation
# ===========================================================================


class TestInputValidation:
    """Tests for input validation across endpoints."""

    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, queue_handler, mock_auth_context, mock_queue):
        """Should reject SQL injection attempts in job IDs."""
        handler = make_mock_handler(path="/api/queue/jobs/'; DROP TABLE jobs;--")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/'; DROP TABLE jobs;--", "GET", handler
                    )
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_oversized_body_rejection(self, queue_handler, mock_auth_context, mock_queue):
        """Should reject oversized request bodies."""
        handler = make_mock_handler(method="POST", path="/api/queue/jobs")
        # Simulate large body
        handler.headers["Content-Length"] = str(20 * 1024 * 1024)  # 20MB

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/queue/jobs", "POST", handler)
                    assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_job_id_formats(self, queue_handler, mock_auth_context, mock_queue):
        """Should accept valid job ID formats."""
        valid_ids = [
            "job-123",
            "abc123",
            "test_job",
            "123e4567-e89b-12d3-a456-426614174000",  # UUID
        ]
        for job_id in valid_ids:
            handler = make_mock_handler(path=f"/api/queue/jobs/{job_id}")

            with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
                with patch.object(queue_handler, "check_permission"):
                    with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                        result = await queue_handler.handle(
                            f"/api/queue/jobs/{job_id}", "GET", handler
                        )
                        # Should return 404 (not found), not 400 (invalid ID)
                        assert get_status(result) == 404, f"Job ID {job_id} should be valid"


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, queue_handler, mock_auth_context):
        """Should handle timeout errors gracefully."""
        mock_q = MagicMock()
        mock_q.get_queue_stats = AsyncMock(side_effect=TimeoutError("Connection timed out"))

        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 503

    @pytest.mark.asyncio
    async def test_attribute_error_handling(self, queue_handler, mock_auth_context):
        """Should handle attribute errors gracefully."""
        mock_q = MagicMock()
        mock_q.get_queue_stats = AsyncMock(side_effect=AttributeError("Missing attribute"))

        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_handling(self, queue_handler, mock_auth_context):
        """Should handle OS errors gracefully."""
        mock_q = MagicMock()
        mock_q.get_queue_stats = AsyncMock(side_effect=OSError("File system error"))

        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 503


# ===========================================================================
# Test: Version Prefix Handling
# ===========================================================================


class TestVersionPrefixHandling:
    """Tests for API version prefix handling."""

    def test_can_handle_v1_paths(self, queue_handler):
        """Should handle /api/v1/ prefixed paths."""
        assert queue_handler.can_handle("/api/v1/queue/jobs", "GET")
        assert queue_handler.can_handle("/api/v1/queue/stats", "GET")
        assert queue_handler.can_handle("/api/v1/queue/workers", "GET")

    @pytest.mark.asyncio
    async def test_v1_stats_endpoint(self, queue_handler, mock_auth_context, mock_queue):
        """Should process /api/v1/queue/stats correctly."""
        handler = make_mock_handler(path="/api/v1/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
                    result = await queue_handler.handle("/api/v1/queue/stats", "GET", handler)
                    assert get_status(result) == 200


# ===========================================================================
# Test: Queue Instance Management
# ===========================================================================


class TestQueueInstanceManagement:
    """Tests for queue instance caching and management."""

    @pytest.mark.asyncio
    async def test_queue_unavailable_returns_503(self, queue_handler, mock_auth_context):
        """Should return 503 when queue cannot be created."""
        handler = make_mock_handler(path="/api/queue/stats")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=None):
                    result = await queue_handler.handle("/api/queue/stats", "GET", handler)
                    assert get_status(result) == 503


# ===========================================================================
# Test: Module-Level Functionality
# ===========================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_handler_can_be_imported(self):
        """Should be able to import QueueHandler."""
        from aragora.server.handlers.queue import QueueHandler

        assert QueueHandler is not None

    def test_get_queue_function_exported(self):
        """Should export _get_queue function."""
        assert callable(_get_queue)

    def test_job_status_can_be_imported(self):
        """Should be able to use JobStatus from queue module."""
        from aragora.queue.base import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.COMPLETED.value == "completed"


# ===========================================================================
# Test: Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_job_list(self, queue_handler, mock_auth_context):
        """Should handle empty job list gracefully."""
        mock_q = MockQueue([])

        handler = make_mock_handler(path="/api/queue/jobs")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/jobs", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["jobs"] == []

    @pytest.mark.asyncio
    async def test_empty_dlq(self, queue_handler, mock_auth_context):
        """Should handle empty DLQ gracefully."""
        mock_q = MockQueue([])

        handler = make_mock_handler(path="/api/queue/dlq")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle("/api/queue/dlq", "GET", handler)
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["jobs"] == []
                    assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_job_with_no_metadata(self, queue_handler, mock_auth_context):
        """Should handle jobs with empty metadata."""
        job = MockJob(id="no-meta-job", metadata={})
        mock_q = MockQueue([job])

        handler = make_mock_handler(path="/api/queue/jobs/no-meta-job")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/no-meta-job", "GET", handler
                    )
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert body["metadata"] == {}

    @pytest.mark.asyncio
    async def test_job_with_internal_metadata_hidden(self, queue_handler, mock_auth_context):
        """Should hide internal metadata fields (starting with _)."""
        job = MockJob(
            id="internal-meta-job",
            metadata={"visible": "yes", "_internal": "hidden", "_secret": "also hidden"},
        )
        mock_q = MockQueue([job])

        handler = make_mock_handler(path="/api/queue/jobs/internal-meta-job")

        with patch.object(queue_handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(queue_handler, "check_permission"):
                with patch("aragora.server.handlers.queue._get_queue", return_value=mock_q):
                    result = await queue_handler.handle(
                        "/api/queue/jobs/internal-meta-job", "GET", handler
                    )
                    assert get_status(result) == 200
                    body = get_body(result)
                    assert "visible" in body["metadata"]
                    assert "_internal" not in body["metadata"]
                    assert "_secret" not in body["metadata"]
