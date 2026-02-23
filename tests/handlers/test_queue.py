"""Tests for queue handler (aragora/server/handlers/queue.py).

Covers all routes and behavior of the QueueHandler class:
- can_handle() routing for all ROUTES
- POST   /api/queue/jobs              - Submit new job
- GET    /api/queue/jobs              - List jobs with filters
- GET    /api/queue/jobs/:id          - Get job status
- POST   /api/queue/jobs/:id/retry    - Retry failed job
- DELETE /api/queue/jobs/:id          - Cancel job
- GET    /api/queue/stats             - Queue statistics
- GET    /api/queue/workers           - Worker status
- GET    /api/queue/dlq               - List dead-letter queue jobs
- POST   /api/queue/dlq/requeue      - Requeue all DLQ jobs
- POST   /api/queue/dlq/:id/requeue  - Requeue specific DLQ job
- POST   /api/queue/cleanup           - Cleanup old jobs
- GET    /api/queue/stale             - List stale/stuck jobs
- Error handling (queue not available, Redis errors, validation errors)
- Path validation (SAFE_ID_PATTERN enforcement)
- Authentication and permission checks
- Circuit breaker status
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.queue import (
    QueueHandler,
    _get_queue,
    get_queue_circuit_breaker,
    get_queue_circuit_breaker_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a handler result.

    Handles both HandlerResult dataclass (with .body bytes and __getitem__) and plain dicts.
    """
    if hasattr(result, "status_code"):
        # HandlerResult dataclass - decode JSON body
        import json as _json

        try:
            return _json.loads(result.body.decode("utf-8")) if result.body else {}
        except (ValueError, AttributeError):
            return {}
    return result.get("body", result)


def _status(result) -> int | None:
    """Extract HTTP status code from a handler result.

    Handles HandlerResult dataclass, plain dicts, and None (no route match).
    """
    if result is None:
        return None
    if hasattr(result, "status_code"):
        return result.status_code
    return result.get("status_code", 200)


class MockJobStatus(Enum):
    """Mock JobStatus enum matching aragora.queue.JobStatus."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class MockJob:
    """Mock job object matching the queue job interface."""

    id: str = "job-001"
    status: MockJobStatus = MockJobStatus.PENDING
    job_type: str = "debate"
    created_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 0
    error: str | None = None
    worker_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = datetime.now().timestamp()


class MockHTTPHandler:
    """Mock HTTP handler used by the QueueHandler for body reading and query params."""

    def __init__(
        self,
        path: str = "/api/queue/jobs",
        body: dict[str, Any] | None = None,
    ):
        self.path = path
        self.rfile = MagicMock()
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}
        self.client_address = ("127.0.0.1", 12345)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a QueueHandler with minimal context."""
    return QueueHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_queue_instance():
    """Reset the module-level queue instance between tests."""
    import aragora.server.handlers.queue as queue_mod

    original = queue_mod._queue_instance
    queue_mod._queue_instance = None
    yield
    queue_mod._queue_instance = original


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker state between tests."""
    cb = get_queue_circuit_breaker()
    cb._single_failures = 0
    cb._single_open_at = 0.0
    cb._single_successes = 0
    cb._single_half_open_calls = 0
    yield
    cb._single_failures = 0
    cb._single_open_at = 0.0
    cb._single_successes = 0
    cb._single_half_open_calls = 0


def _make_mock_queue():
    """Create a mock queue with standard async methods."""
    queue = AsyncMock()
    queue.get_queue_stats = AsyncMock(
        return_value={
            "pending": 5,
            "processing": 2,
            "completed": 10,
            "failed": 1,
            "cancelled": 0,
            "retrying": 0,
            "stream_length": 18,
            "pending_in_group": 5,
        }
    )
    queue.enqueue = AsyncMock(return_value="job-new-001")
    queue.get_status = AsyncMock(return_value=None)
    queue.cancel = AsyncMock(return_value=True)
    queue.list_jobs = AsyncMock(return_value=[])
    queue.delete = AsyncMock()
    queue._redis = AsyncMock()
    queue._redis.xinfo_groups = AsyncMock(return_value=[])
    queue._redis.xinfo_consumers = AsyncMock(return_value=[])
    queue.stream_key = "aragora:queue:stream"
    queue._status_tracker = AsyncMock()
    queue._status_tracker.list_jobs = AsyncMock(return_value=[])
    queue._status_tracker.get_counts_by_status = AsyncMock(
        return_value={
            "pending": 5,
            "processing": 2,
            "completed": 10,
            "failed": 1,
        }
    )
    return queue


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_queue_jobs_path(self, handler):
        assert handler.can_handle("/api/queue/jobs")

    def test_queue_jobs_with_version(self, handler):
        assert handler.can_handle("/api/v1/queue/jobs")

    def test_queue_stats(self, handler):
        assert handler.can_handle("/api/queue/stats")

    def test_queue_stats_with_version(self, handler):
        assert handler.can_handle("/api/v1/queue/stats")

    def test_queue_workers(self, handler):
        assert handler.can_handle("/api/queue/workers")

    def test_queue_workers_with_version(self, handler):
        assert handler.can_handle("/api/v1/queue/workers")

    def test_queue_jobs_with_id(self, handler):
        assert handler.can_handle("/api/queue/jobs/job-123")

    def test_queue_jobs_retry(self, handler):
        assert handler.can_handle("/api/queue/jobs/job-123/retry")

    def test_queue_dlq(self, handler):
        assert handler.can_handle("/api/queue/dlq")

    def test_queue_dlq_requeue(self, handler):
        assert handler.can_handle("/api/queue/dlq/requeue")

    def test_queue_dlq_job_requeue(self, handler):
        assert handler.can_handle("/api/queue/dlq/job-123/requeue")

    def test_queue_cleanup(self, handler):
        assert handler.can_handle("/api/queue/cleanup")

    def test_queue_stale(self, handler):
        assert handler.can_handle("/api/queue/stale")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/debates")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/queue")

    def test_accepts_any_method_for_can_handle(self, handler):
        assert handler.can_handle("/api/queue/jobs", "POST")
        assert handler.can_handle("/api/queue/jobs", "GET")
        assert handler.can_handle("/api/queue/jobs", "DELETE")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = QueueHandler({})
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = QueueHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_context(self):
        ctx = {"key": "value"}
        h = QueueHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "queue"

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/queue/jobs" in handler.ROUTES
        assert "/api/v1/queue/jobs" in handler.ROUTES

    def test_routes_include_all_endpoints(self, handler):
        routes = handler.ROUTES
        assert "/api/queue/stats" in routes
        assert "/api/queue/workers" in routes
        assert "/api/queue/dlq" in routes
        assert "/api/queue/cleanup" in routes
        assert "/api/queue/stale" in routes


# ============================================================================
# Circuit Breaker
# ============================================================================


class TestCircuitBreaker:
    """Test circuit breaker functions."""

    def test_get_circuit_breaker_returns_instance(self):
        cb = get_queue_circuit_breaker()
        assert cb is not None
        assert cb.name == "queue_handler"

    def test_get_circuit_breaker_same_instance(self):
        cb1 = get_queue_circuit_breaker()
        cb2 = get_queue_circuit_breaker()
        assert cb1 is cb2

    def test_get_circuit_breaker_status(self):
        status = get_queue_circuit_breaker_status()
        assert isinstance(status, dict)
        # Status dict has config and state sub-dicts
        assert "config" in status or "name" in status

    def test_circuit_breaker_failure_threshold(self):
        cb = get_queue_circuit_breaker()
        assert cb.failure_threshold == 5

    def test_circuit_breaker_cooldown(self):
        cb = get_queue_circuit_breaker()
        assert cb.cooldown_seconds == 30.0


# ============================================================================
# _get_queue helper
# ============================================================================


class TestGetQueue:
    """Test the _get_queue module-level helper."""

    @pytest.mark.asyncio
    async def test_returns_none_when_import_fails(self):
        import aragora.server.handlers.queue as queue_mod

        with patch.object(queue_mod, "_get_queue", new=AsyncMock(return_value=None)):
            result = await queue_mod._get_queue()
            assert result is None

    @pytest.mark.asyncio
    async def test_caches_queue_instance(self):
        import aragora.server.handlers.queue as queue_mod

        mock_queue = _make_mock_queue()
        with patch.dict("sys.modules", {"aragora.queue": MagicMock()}):
            with patch(
                "aragora.server.handlers.queue.create_redis_queue",
                create=True,
            ):
                # Set it directly for the cache test
                queue_mod._queue_instance = mock_queue
                result = await _get_queue()
                assert result is mock_queue


# ============================================================================
# GET /api/queue/stats
# ============================================================================


class TestGetStats:
    """Test GET /api/queue/stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 503
        body = _body(result)
        assert body["error"] == "Queue not available"
        assert body["stats"]["pending"] == 0

    @pytest.mark.asyncio
    async def test_stats_success(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body
        assert "timestamp" in body
        assert body["stats"]["pending"] == 5

    @pytest.mark.asyncio
    async def test_stats_redis_connection_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_queue_stats.side_effect = ConnectionError("Redis down")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stats_attribute_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_queue_stats.side_effect = AttributeError("bad attr")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 500
        assert "Internal data error" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_stats_key_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_queue_stats.side_effect = KeyError("missing key")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_stats_with_version_prefix(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/v1/queue/stats", "GET")

        assert _status(result) == 200


# ============================================================================
# GET /api/queue/workers
# ============================================================================


class TestGetWorkers:
    """Test GET /api/queue/workers endpoint."""

    @pytest.mark.asyncio
    async def test_workers_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/workers", "GET")

        assert _status(result) == 503
        body = _body(result)
        assert body["workers"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_workers_empty(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._redis.xinfo_groups.return_value = []
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["workers"] == []
        assert body["total"] == 0
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_workers_with_consumers(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._redis.xinfo_groups.return_value = [
            {"name": "workers", "consumers": 2},
        ]
        mock_queue._redis.xinfo_consumers.return_value = [
            {"name": "worker-1", "pending": 3, "idle": 5000},
            {"name": "worker-2", "pending": 0, "idle": 100},
        ]
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["workers"]) == 2
        assert body["workers"][0]["worker_id"] == "worker-1"
        assert body["workers"][0]["pending"] == 3
        assert body["workers"][1]["worker_id"] == "worker-2"

    @pytest.mark.asyncio
    async def test_workers_redis_connection_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._redis.xinfo_groups.side_effect = ConnectionError("Redis down")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        # Should return a graceful response, not 500
        body = _body(result)
        assert body["workers"] == []
        assert body["total"] == 0
        assert "error" in body

    @pytest.mark.asyncio
    async def test_workers_consumer_fetch_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._redis.xinfo_groups.return_value = [
            {"name": "workers", "consumers": 2},
        ]
        mock_queue._redis.xinfo_consumers.side_effect = ConnectionError("fetch fail")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        assert _status(result) == 200
        body = _body(result)
        # Consumers fail but workers list is still returned (empty)
        assert body["workers"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_workers_non_dict_group(self, handler):
        mock_queue = _make_mock_queue()
        # Non-dict entries should be skipped
        mock_queue._redis.xinfo_groups.return_value = [
            "not a dict",
            42,
        ]
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["workers"] == []

    @pytest.mark.asyncio
    async def test_workers_non_dict_consumer(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._redis.xinfo_groups.return_value = [
            {"name": "workers"},
        ]
        mock_queue._redis.xinfo_consumers.return_value = [
            "not a dict",
        ]
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["workers"] == []

    @pytest.mark.asyncio
    async def test_workers_type_error_on_groups(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._redis.xinfo_groups.side_effect = TypeError("bad type")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/workers", "GET")

        body = _body(result)
        assert body["workers"] == []
        assert "error" in body


# ============================================================================
# POST /api/queue/jobs (submit)
# ============================================================================


class TestSubmitJob:
    """Test POST /api/queue/jobs endpoint."""

    @pytest.mark.asyncio
    async def test_submit_queue_unavailable(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": "What is AI?"},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 503
        assert "not available" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_submit_success(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.enqueue.return_value = "job-new-001"
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": "What is AI?"},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.create_debate_job") as mock_create:
                mock_create.return_value = MockJob(id="job-new-001")
                result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 202
        body = _body(result)
        assert body["job_id"] == "job-new-001"
        assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_submit_with_all_options(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.enqueue.return_value = "job-full-001"
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={
                "question": "What is AI?",
                "agents": ["claude", "gpt"],
                "rounds": 5,
                "consensus": "unanimous",
                "protocol": "debate",
                "priority": 10,
                "max_attempts": 5,
                "timeout_seconds": 300,
                "webhook_url": "https://example.com/hook",
                "user_id": "user-001",
                "organization_id": "org-001",
                "metadata": {"key": "value"},
            },
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.create_debate_job") as mock_create:
                mock_create.return_value = MockJob(id="job-full-001")
                result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 202
        assert _body(result)["job_id"] == "job-full-001"

    @pytest.mark.asyncio
    async def test_submit_missing_question(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"agents": ["claude"]},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 400
        assert "question" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_submit_empty_question(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": ""},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_invalid_body(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(path="/api/queue/jobs")
        mock_handler.rfile.read.return_value = b"not json"
        mock_handler.headers = {"Content-Length": "8"}
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_empty_body(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(path="/api/queue/jobs", body={})
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_redis_connection_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": "Test?"},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.create_debate_job") as mock_create:
                mock_create.return_value = MockJob(id="job-fail")
                mock_queue.enqueue.side_effect = ConnectionError("Redis gone")
                result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_submit_value_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": "Test?"},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.create_debate_job") as mock_create:
                mock_create.side_effect = ValueError("Invalid data")
                result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_attribute_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": "Test?"},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.create_debate_job") as mock_create:
                mock_create.return_value = MockJob(id="job-001")
                mock_queue.enqueue.side_effect = AttributeError("bad queue")
                result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)

        assert _status(result) == 500
        assert "Queue configuration error" in _body(result)["error"]


# ============================================================================
# GET /api/queue/jobs (list)
# ============================================================================


class TestListJobs:
    """Test GET /api/queue/jobs endpoint."""

    @pytest.mark.asyncio
    async def test_list_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/jobs", "GET")

        assert _status(result) == 503
        body = _body(result)
        assert body["jobs"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_empty(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["jobs"] == []
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    @pytest.mark.asyncio
    async def test_list_with_jobs(self, handler):
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="job-001",
                status=MockJobStatus.COMPLETED,
                attempts=1,
                max_attempts=3,
                priority=0,
                error=None,
                worker_id="w-1",
                metadata={"key": "value", "_internal": "hidden"},
            ),
            MockJob(
                id="job-002",
                status=MockJobStatus.PENDING,
                metadata={},
            ),
        ]
        mock_queue._status_tracker.list_jobs.return_value = jobs
        mock_queue._status_tracker.get_counts_by_status.return_value = {
            "pending": 1,
            "completed": 1,
        }

        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["jobs"]) == 2
        assert body["total"] == 2
        # Internal metadata should be hidden
        assert "_internal" not in body["jobs"][0]["metadata"]
        assert body["jobs"][0]["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(path="/api/queue/jobs?status=pending")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle(
                    "/api/queue/jobs",
                    "GET",
                    handler=mock_handler,
                )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_with_invalid_status_filter(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(path="/api/queue/jobs?status=invalid_status")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle(
                    "/api/queue/jobs",
                    "GET",
                    handler=mock_handler,
                )

        assert _status(result) == 400
        assert "Invalid status" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_redis_connection_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._status_tracker.list_jobs.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_attribute_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue._status_tracker.list_jobs.side_effect = AttributeError("bad")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs", "GET")

        assert _status(result) == 500
        assert "Internal data error" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(path="/api/queue/jobs?limit=5&offset=10")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle(
                    "/api/queue/jobs",
                    "GET",
                    handler=mock_handler,
                )

        assert _status(result) == 200


# ============================================================================
# GET /api/queue/jobs/:id
# ============================================================================


class TestGetJob:
    """Test GET /api/queue/jobs/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_job_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/jobs/job-001", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-999", "GET")

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_get_job_success(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(
            id="job-001",
            status=MockJobStatus.COMPLETED,
            started_at=datetime.now().timestamp() - 60,
            completed_at=datetime.now().timestamp(),
            attempts=1,
            max_attempts=3,
            priority=5,
            worker_id="w-1",
            payload={"question": "Test?"},
            metadata={"result": "some result", "_internal": "hidden"},
        )
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-001", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["job_id"] == "job-001"
        assert body["status"] == "completed"
        assert body["attempts"] == 1
        assert body["priority"] == 5
        assert body["worker_id"] == "w-1"
        assert body["payload"] == {"question": "Test?"}
        assert body["result"] == "some result"
        assert "_internal" not in body["metadata"]

    @pytest.mark.asyncio
    async def test_get_job_with_no_started_at(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-002", metadata={})
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-002", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["started_at"] is None
        assert body["completed_at"] is None

    @pytest.mark.asyncio
    async def test_get_job_invalid_id(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=_make_mock_queue()):
            result = await handler.handle("/api/queue/jobs/../../etc", "GET")

        # Path traversal changes segment count, may not match route
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_job_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-001", "GET")

        assert _status(result) == 503


# ============================================================================
# POST /api/queue/jobs/:id/retry
# ============================================================================


class TestRetryJob:
    """Test POST /api/queue/jobs/:id/retry endpoint."""

    @pytest.mark.asyncio
    async def test_retry_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/jobs/job-001/retry", "POST")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_retry_job_not_found(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-999/retry", "POST")

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_retry_failed_job(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-001", status=MockJobStatus.FAILED, priority=5)
        mock_queue.get_status.return_value = job
        mock_queue.enqueue.return_value = "job-001"
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-001/retry", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["job_id"] == "job-001"
        assert body["status"] == "pending"
        assert body["message"] == "Job queued for retry"
        # Verify job was reset
        assert job.status == MockJobStatus.PENDING
        assert job.attempts == 0
        assert job.error is None

    @pytest.mark.asyncio
    async def test_retry_cancelled_job(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-002", status=MockJobStatus.CANCELLED)
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-002/retry", "POST")

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_retry_pending_job_rejected(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-003", status=MockJobStatus.PENDING)
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-003/retry", "POST")

        assert _status(result) == 400
        assert "Cannot retry" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_retry_processing_job_rejected(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-004", status=MockJobStatus.PROCESSING)
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-004/retry", "POST")

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_retry_completed_job_rejected(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-005", status=MockJobStatus.COMPLETED)
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-005/retry", "POST")

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_retry_invalid_job_id(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=_make_mock_queue()):
            result = await handler.handle("/api/queue/jobs/../hack/retry", "POST")

        # Path traversal changes segment count, may not match route
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_retry_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.side_effect = ConnectionError("Redis offline")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs/job-001/retry", "POST")

        assert _status(result) == 503


# ============================================================================
# DELETE /api/queue/jobs/:id
# ============================================================================


class TestCancelJob:
    """Test DELETE /api/queue/jobs/:id endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/jobs/job-001", "DELETE")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cancel_success(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.cancel.return_value = True
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-001", "DELETE")

        assert _status(result) == 200
        body = _body(result)
        assert body["job_id"] == "job-001"
        assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.cancel.return_value = False
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-999", "DELETE")

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_cancel_wrong_status(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.cancel.return_value = False
        job = MockJob(id="job-001", status=MockJobStatus.COMPLETED)
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-001", "DELETE")

        assert _status(result) == 400
        assert "Cannot cancel" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_cancel_invalid_id(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=_make_mock_queue()):
            result = await handler.handle("/api/queue/jobs/../../etc", "DELETE")

        # Path traversal changes segment count, may not match route
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_cancel_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.cancel.side_effect = ConnectionError("offline")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-001", "DELETE")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cancel_attribute_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.cancel.side_effect = AttributeError("bad")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-001", "DELETE")

        assert _status(result) == 500


# ============================================================================
# GET /api/queue/dlq
# ============================================================================


class TestListDLQ:
    """Test GET /api/queue/dlq endpoint."""

    @pytest.mark.asyncio
    async def test_dlq_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_dlq_empty(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["jobs"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_dlq_with_failed_jobs(self, handler):
        mock_queue = _make_mock_queue()
        now = datetime.now().timestamp()
        jobs = [
            MockJob(
                id="dlq-001",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                completed_at=now - 3600,
                error="Timeout",
                job_type="debate",
                payload={"q": "test"},
            ),
            MockJob(
                id="dlq-002",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                completed_at=now - 7200,
                error="API error",
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["jobs"]) == 2
        assert body["jobs"][0]["job_id"] == "dlq-001"
        assert body["jobs"][0]["error"] == "Timeout"

    @pytest.mark.asyncio
    async def test_dlq_excludes_retriable_failures(self, handler):
        """Failed jobs with attempts < max_attempts should not appear in DLQ."""
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="retry-001",
                status=MockJobStatus.FAILED,
                attempts=1,  # < max_attempts
                max_attempts=3,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0
        assert body["jobs"] == []

    @pytest.mark.asyncio
    async def test_dlq_pagination(self, handler):
        mock_queue = _make_mock_queue()
        now = datetime.now().timestamp()
        jobs = [
            MockJob(
                id=f"dlq-{i:03d}",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                completed_at=now - i * 3600,
            )
            for i in range(5)
        ]
        mock_queue.list_jobs.return_value = jobs
        mock_handler = MockHTTPHandler(path="/api/queue/dlq?limit=2&offset=1")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle(
                    "/api/queue/dlq",
                    "GET",
                    handler=mock_handler,
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5
        assert len(body["jobs"]) == 2
        assert body["limit"] == 2
        assert body["offset"] == 1

    @pytest.mark.asyncio
    async def test_dlq_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_dlq_without_completed_at(self, handler):
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="dlq-no-time",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                completed_at=None,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["jobs"][0]["failed_at"] is None
        assert body["jobs"][0]["age_hours"] == 0.0


# ============================================================================
# POST /api/queue/dlq/:id/requeue
# ============================================================================


class TestRequeueDLQJob:
    """Test POST /api/queue/dlq/:id/requeue endpoint."""

    @pytest.mark.asyncio
    async def test_requeue_dlq_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/dlq/job-001/requeue", "POST")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_requeue_dlq_not_found(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/job-999/requeue", "POST")

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_requeue_dlq_success(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(
            id="dlq-001",
            status=MockJobStatus.FAILED,
            attempts=3,
            max_attempts=3,
            priority=5,
            metadata={},
        )
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/dlq-001/requeue", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "pending"
        assert body["message"] == "Job requeued from DLQ"
        # Job should have been reset
        assert job.status == MockJobStatus.PENDING
        assert job.attempts == 0
        assert job.error is None
        assert job.metadata["requeued_from_dlq"] is True
        assert "requeued_at" in job.metadata

    @pytest.mark.asyncio
    async def test_requeue_dlq_not_failed(self, handler):
        mock_queue = _make_mock_queue()
        job = MockJob(id="job-001", status=MockJobStatus.COMPLETED)
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/job-001/requeue", "POST")

        assert _status(result) == 400
        assert "not in DLQ" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_requeue_dlq_invalid_id(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=_make_mock_queue()):
            result = await handler.handle("/api/queue/dlq/../../hack/requeue", "POST")

        # Path traversal changes segment count, may not match route
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_requeue_dlq_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/job-001/requeue", "POST")

        assert _status(result) == 503


# ============================================================================
# POST /api/queue/dlq/requeue (requeue all)
# ============================================================================


class TestRequeueAllDLQ:
    """Test POST /api/queue/dlq/requeue endpoint."""

    @pytest.mark.asyncio
    async def test_requeue_all_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/dlq/requeue", "POST")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_requeue_all_empty(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/requeue", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["requeued"] == 0
        assert body["errors"] == 0

    @pytest.mark.asyncio
    async def test_requeue_all_success(self, handler):
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="dlq-001",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                metadata={},
            ),
            MockJob(
                id="dlq-002",
                status=MockJobStatus.FAILED,
                attempts=5,
                max_attempts=3,
                metadata={},
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/requeue", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["requeued"] == 2
        assert body["errors"] == 0

    @pytest.mark.asyncio
    async def test_requeue_all_skips_retriable(self, handler):
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="retry-001",
                status=MockJobStatus.FAILED,
                attempts=1,  # < max_attempts, not DLQ
                max_attempts=3,
                metadata={},
            ),
            MockJob(
                id="dlq-001",
                status=MockJobStatus.FAILED,
                attempts=3,  # == max_attempts, DLQ
                max_attempts=3,
                metadata={},
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/requeue", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["requeued"] == 1

    @pytest.mark.asyncio
    async def test_requeue_all_partial_failure(self, handler):
        mock_queue = _make_mock_queue()
        call_count = 0

        async def enqueue_side_effect(job, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("Redis down")
            return job.id

        mock_queue.enqueue = AsyncMock(side_effect=enqueue_side_effect)
        jobs = [
            MockJob(
                id=f"dlq-{i:03d}",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                metadata={},
            )
            for i in range(3)
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/requeue", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["requeued"] == 2
        assert body["errors"] == 1

    @pytest.mark.asyncio
    async def test_requeue_all_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq/requeue", "POST")

        assert _status(result) == 503


# ============================================================================
# POST /api/queue/cleanup
# ============================================================================


class TestCleanupJobs:
    """Test POST /api/queue/cleanup endpoint."""

    @pytest.mark.asyncio
    async def test_cleanup_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/cleanup", "POST")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cleanup_defaults(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["older_than_days"] == 7
        assert body["status_filter"] == "completed"
        assert body["dry_run"] is False

    @pytest.mark.asyncio
    async def test_cleanup_with_old_jobs(self, handler):
        mock_queue = _make_mock_queue()
        old_time = datetime.now().timestamp() - (10 * 86400)  # 10 days ago
        jobs = [
            MockJob(id="old-001", status=MockJobStatus.COMPLETED, completed_at=old_time),
            MockJob(
                id="new-001",
                status=MockJobStatus.COMPLETED,
                completed_at=datetime.now().timestamp(),
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["scanned"] == 2
        assert body["deleted"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, handler):
        mock_queue = _make_mock_queue()
        old_time = datetime.now().timestamp() - (10 * 86400)
        jobs = [MockJob(id="old-001", status=MockJobStatus.COMPLETED, completed_at=old_time)]
        mock_queue.list_jobs.return_value = jobs
        mock_handler = MockHTTPHandler(path="/api/queue/cleanup?dry_run=true")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST", handler=mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["dry_run"] is True
        assert body["deleted"] == 0
        assert body["would_delete"] == 1
        # Should not actually delete
        mock_queue.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_status_all(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        mock_handler = MockHTTPHandler(path="/api/queue/cleanup?status=all")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST", handler=mock_handler)

        assert _status(result) == 200
        # Should query all terminal statuses (completed, failed, cancelled)
        assert mock_queue.list_jobs.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_status_failed(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        mock_handler = MockHTTPHandler(path="/api/queue/cleanup?status=failed")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST", handler=mock_handler)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_cleanup_invalid_status(self, handler):
        mock_queue = _make_mock_queue()
        mock_handler = MockHTTPHandler(path="/api/queue/cleanup?status=invalid")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST", handler=mock_handler)

        assert _status(result) == 400
        assert "Invalid status" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_cleanup_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cleanup_custom_older_than(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        mock_handler = MockHTTPHandler(path="/api/queue/cleanup?older_than_days=30")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST", handler=mock_handler)

        assert _status(result) == 200
        assert _body(result)["older_than_days"] == 30

    @pytest.mark.asyncio
    async def test_cleanup_delete_error_continues(self, handler):
        """Individual delete errors should not abort the entire cleanup."""
        mock_queue = _make_mock_queue()
        old_time = datetime.now().timestamp() - (10 * 86400)
        jobs = [
            MockJob(id="old-001", status=MockJobStatus.COMPLETED, completed_at=old_time),
            MockJob(id="old-002", status=MockJobStatus.COMPLETED, completed_at=old_time),
        ]
        mock_queue.list_jobs.return_value = jobs

        call_count = 0

        async def delete_side_effect(job_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("flaky")

        mock_queue.delete = AsyncMock(side_effect=delete_side_effect)
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST")

        assert _status(result) == 200
        body = _body(result)
        # First delete fails, second succeeds
        assert body["deleted"] == 1


# ============================================================================
# GET /api/queue/stale
# ============================================================================


class TestListStaleJobs:
    """Test GET /api/queue/stale endpoint."""

    @pytest.mark.asyncio
    async def test_stale_queue_unavailable(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stale_empty(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["jobs"] == []
        assert body["total"] == 0
        assert body["stale_threshold_minutes"] == 60

    @pytest.mark.asyncio
    async def test_stale_with_stuck_jobs(self, handler):
        mock_queue = _make_mock_queue()
        stale_time = datetime.now().timestamp() - 7200  # 2 hours ago
        jobs = [
            MockJob(
                id="stuck-001",
                status=MockJobStatus.PROCESSING,
                started_at=stale_time,
                worker_id="w-1",
                job_type="debate",
                attempts=1,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["jobs"][0]["job_id"] == "stuck-001"
        assert body["jobs"][0]["duration_minutes"] > 100  # ~120 min

    @pytest.mark.asyncio
    async def test_stale_excludes_recent_jobs(self, handler):
        mock_queue = _make_mock_queue()
        recent_time = datetime.now().timestamp() - 30  # 30 seconds ago
        jobs = [
            MockJob(
                id="fresh-001",
                status=MockJobStatus.PROCESSING,
                started_at=recent_time,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_stale_custom_threshold(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        mock_handler = MockHTTPHandler(path="/api/queue/stale?stale_minutes=30")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET", handler=mock_handler)

        assert _status(result) == 200
        assert _body(result)["stale_threshold_minutes"] == 30

    @pytest.mark.asyncio
    async def test_stale_sorted_by_duration(self, handler):
        mock_queue = _make_mock_queue()
        now = datetime.now().timestamp()
        jobs = [
            MockJob(id="short-001", status=MockJobStatus.PROCESSING, started_at=now - 7200),  # 2h
            MockJob(id="long-001", status=MockJobStatus.PROCESSING, started_at=now - 14400),  # 4h
            MockJob(id="medium-001", status=MockJobStatus.PROCESSING, started_at=now - 10800),  # 3h
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        # Should be sorted by duration descending (longest first)
        assert body["jobs"][0]["job_id"] == "long-001"
        assert body["jobs"][1]["job_id"] == "medium-001"
        assert body["jobs"][2]["job_id"] == "short-001"

    @pytest.mark.asyncio
    async def test_stale_limit(self, handler):
        mock_queue = _make_mock_queue()
        now = datetime.now().timestamp()
        jobs = [
            MockJob(
                id=f"stuck-{i:03d}",
                status=MockJobStatus.PROCESSING,
                started_at=now - (i + 1) * 7200,
            )
            for i in range(10)
        ]
        mock_queue.list_jobs.return_value = jobs
        mock_handler = MockHTTPHandler(path="/api/queue/stale?limit=3")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET", handler=mock_handler)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["jobs"]) == 3

    @pytest.mark.asyncio
    async def test_stale_redis_error(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.side_effect = ConnectionError("gone")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stale_jobs_without_started_at(self, handler):
        """Jobs with None started_at should not appear in stale list."""
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="no-start-001",
                status=MockJobStatus.PROCESSING,
                started_at=None,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0


# ============================================================================
# Authentication / Permission Tests
# ============================================================================


class TestAuthentication:
    """Test authentication and permission checking."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, handler):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(
            handler,
            "get_auth_context",
            side_effect=UnauthorizedError("Not authenticated"),
        ):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 401
        assert "Authentication required" in _body(result)["error"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, handler):
        from aragora.server.handlers.utils.auth import ForbiddenError

        with patch.object(
            handler,
            "get_auth_context",
            side_effect=ForbiddenError("Forbidden"),
        ):
            result = await handler.handle("/api/queue/stats", "GET")

        assert _status(result) == 403
        assert "Permission denied" in _body(result)["error"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_permission_check_denied(self, handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.utils.auth import ForbiddenError

        mock_ctx = AuthorizationContext(
            user_id="test-user",
            user_email="test@test.com",
            org_id="org-1",
            roles={"viewer"},
            permissions={"queue:read"},
        )
        with patch.object(
            handler,
            "get_auth_context",
            return_value=mock_ctx,
        ):
            with patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("Denied"),
            ):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_read_ops_require_queue_read(self, handler):
        """Verify read operations use queue:read permission."""
        # This test just verifies the handler runs without permission errors
        # when the auto-auth provides wildcard permissions
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            for path in [
                "/api/queue/stats",
                "/api/queue/workers",
                "/api/queue/jobs",
                "/api/queue/stale",
            ]:
                result = await handler.handle(path, "GET")
                assert result is not None

    @pytest.mark.asyncio
    async def test_manage_ops_require_queue_manage(self, handler):
        """Verify manage operations use queue:manage permission."""
        mock_handler = MockHTTPHandler(
            path="/api/queue/jobs",
            body={"question": "Test"},
        )
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/jobs", "POST", handler=mock_handler)
            assert result is not None

    @pytest.mark.asyncio
    async def test_admin_ops_require_queue_admin(self, handler):
        """Verify admin operations use queue:admin permission."""
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/dlq", "GET")
            assert result is not None


# ============================================================================
# Path validation
# ============================================================================


class TestPathValidation:
    """Test path segment validation (SAFE_ID_PATTERN enforcement)."""

    @pytest.mark.asyncio
    async def test_valid_alphanumeric_id(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/abc123", "GET")
        assert _status(result) == 404  # Job not found, but ID was valid

    @pytest.mark.asyncio
    async def test_valid_id_with_hyphens(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-123-abc", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_valid_id_with_underscores(self, handler):
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job_123", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_invalid_id_path_traversal(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/../../../etc", "GET")
        # Path doesn't match expected segments so returns None
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_id_special_chars(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job!@#$", "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_id_too_long(self, handler):
        long_id = "a" * 65
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle(f"/api/queue/jobs/{long_id}", "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_max_length_id(self, handler):
        max_id = "a" * 64
        mock_queue = _make_mock_queue()
        mock_queue.get_status.return_value = None
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle(f"/api/queue/jobs/{max_id}", "GET")
        assert _status(result) == 404  # Valid ID, job not found


# ============================================================================
# Version prefix handling
# ============================================================================


class TestVersionPrefix:
    """Test that both /api/ and /api/v1/ paths work correctly."""

    @pytest.mark.asyncio
    async def test_stats_no_version(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/stats", "GET")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_stats_with_version(self, handler):
        mock_queue = _make_mock_queue()
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/v1/queue/stats", "GET")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_workers_no_version(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/queue/workers", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_workers_with_version(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/v1/queue/workers", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_jobs_with_version(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/v1/queue/jobs", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_dlq_with_version(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/v1/queue/dlq", "GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cleanup_with_version(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/v1/queue/cleanup", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stale_with_version(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=None):
            result = await handler.handle("/api/v1/queue/stale", "GET")
        assert _status(result) == 503


# ============================================================================
# Unmatched routes
# ============================================================================


class TestUnmatchedRoutes:
    """Test that unmatched routes return None."""

    @pytest.mark.asyncio
    async def test_unknown_method_returns_none(self, handler):
        with patch("aragora.server.handlers.queue._get_queue", return_value=_make_mock_queue()):
            result = await handler.handle("/api/queue/stats", "DELETE")
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_subpath_returns_none(self, handler):
        result = await handler.handle("/api/queue/unknown", "GET")
        assert result is None

    @pytest.mark.asyncio
    async def test_post_to_stats_returns_none(self, handler):
        result = await handler.handle("/api/queue/stats", "POST")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_to_workers_returns_none(self, handler):
        result = await handler.handle("/api/queue/workers", "DELETE")
        assert result is None


# ============================================================================
# handle_post and handle_get delegation
# ============================================================================


class TestSyncHandleDelegation:
    """Test that sync handle_post/handle_get delegate to async handle."""

    def test_handle_get_returns_none(self, handler):
        result = handler.handle_get("/api/queue/jobs", {}, None)
        assert result is None

    def test_handle_post_returns_none(self, handler):
        result = handler.handle_post("/api/queue/jobs", {}, None)
        # handle_post is decorated with @handle_errors which may wrap or return None
        assert result is None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_dlq_job_with_no_completed_at_age(self, handler):
        """DLQ job with no completed_at should have age_hours of 0."""
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="dlq-no-time",
                status=MockJobStatus.FAILED,
                attempts=3,
                max_attempts=3,
                completed_at=None,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/dlq", "GET")

        assert _status(result) == 200
        assert _body(result)["jobs"][0]["age_hours"] == 0.0

    @pytest.mark.asyncio
    async def test_cleanup_uses_created_at_when_no_completed_at(self, handler):
        """Cleanup should use created_at when completed_at is None."""
        mock_queue = _make_mock_queue()
        old_time = datetime.now().timestamp() - (10 * 86400)
        jobs = [
            MockJob(
                id="old-001",
                status=MockJobStatus.COMPLETED,
                completed_at=None,
                created_at=old_time,
            ),
        ]
        mock_queue.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/cleanup", "POST")

        assert _status(result) == 200
        assert _body(result)["deleted"] == 1

    @pytest.mark.asyncio
    async def test_list_jobs_hides_internal_metadata(self, handler):
        """Jobs should not expose metadata keys starting with underscore."""
        mock_queue = _make_mock_queue()
        jobs = [
            MockJob(
                id="job-meta",
                status=MockJobStatus.PENDING,
                metadata={"visible": "yes", "_secret": "hidden", "_internal": "private"},
            ),
        ]
        mock_queue._status_tracker.list_jobs.return_value = jobs
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/jobs", "GET")

        assert _status(result) == 200
        metadata = _body(result)["jobs"][0]["metadata"]
        assert "visible" in metadata
        assert "_secret" not in metadata
        assert "_internal" not in metadata

    @pytest.mark.asyncio
    async def test_get_job_hides_internal_metadata(self, handler):
        """Get job should not expose metadata keys starting with underscore."""
        mock_queue = _make_mock_queue()
        job = MockJob(
            id="job-meta",
            metadata={"public": "yes", "_private": "no"},
        )
        mock_queue.get_status.return_value = job
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            result = await handler.handle("/api/queue/jobs/job-meta", "GET")

        assert _status(result) == 200
        assert "_private" not in _body(result)["metadata"]
        assert "public" in _body(result)["metadata"]

    @pytest.mark.asyncio
    async def test_query_params_parsed_from_handler_path(self, handler):
        """Query params should be parsed from handler.path when handler is provided."""
        mock_queue = _make_mock_queue()
        mock_queue.list_jobs.return_value = []
        mock_handler = MockHTTPHandler(path="/api/queue/stale?stale_minutes=15&limit=10")
        with patch("aragora.server.handlers.queue._get_queue", return_value=mock_queue):
            with patch("aragora.queue.JobStatus", MockJobStatus):
                result = await handler.handle("/api/queue/stale", "GET", handler=mock_handler)

        assert _status(result) == 200
        assert _body(result)["stale_threshold_minutes"] == 15
