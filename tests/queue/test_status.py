"""
Tests for job status tracking module.

Tests cover:
- JobStatusTracker._job_key key construction
- create() stores job data in Redis hash with TTL
- update_status() updates fields atomically, adds timestamps
- get_job() reconstructs Job from Redis hash
- get_status() returns just the JobStatus
- delete() removes a job entry
- list_jobs() scans and filters by status
- get_counts_by_status() aggregates counts
- Handling of bytes vs string Redis responses
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.base import Job, JobStatus
from aragora.queue.config import QueueConfig, reset_queue_config, set_queue_config
from aragora.queue.status import JobStatusTracker


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure clean config for each test."""
    set_queue_config(QueueConfig())
    yield
    reset_queue_config()


def _make_tracker() -> tuple[JobStatusTracker, AsyncMock]:
    """Create a tracker with a mock Redis client."""
    mock_redis = AsyncMock()
    tracker = JobStatusTracker(mock_redis)
    return tracker, mock_redis


class TestJobKey:
    """Tests for _job_key construction."""

    def test_default_prefix(self):
        tracker, _ = _make_tracker()
        assert tracker._job_key("abc-123") == "aragora:queue:job:abc-123"

    def test_custom_prefix(self):
        set_queue_config(QueueConfig(key_prefix="custom:"))
        tracker, _ = _make_tracker()
        assert tracker._job_key("job-1") == "custom:job:job-1"


class TestCreate:
    """Tests for JobStatusTracker.create."""

    @pytest.mark.asyncio
    async def test_create_calls_hset_and_expire(self):
        tracker, mock_redis = _make_tracker()
        job = Job(id="j-1", payload={"q": "test"}, priority=5)

        await tracker.create(job)

        mock_redis.hset.assert_called_once()
        call_kwargs = mock_redis.hset.call_args
        key = call_kwargs[0][0]
        mapping = call_kwargs[1]["mapping"]

        assert key == "aragora:queue:job:j-1"
        assert mapping["id"] == "j-1"
        assert mapping["status"] == "pending"
        assert mapping["priority"] == "5"
        assert json.loads(mapping["payload"]) == {"q": "test"}

        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_sets_ttl(self):
        tracker, mock_redis = _make_tracker()
        job = Job(id="j-2", payload={})

        await tracker.create(job)

        key, ttl = mock_redis.expire.call_args[0]
        assert key == "aragora:queue:job:j-2"
        assert ttl == 7 * 86400  # default 7 days


class TestUpdateStatus:
    """Tests for JobStatusTracker.update_status."""

    @pytest.mark.asyncio
    async def test_update_existing_job(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        result = await tracker.update_status("j-1", JobStatus.PROCESSING, worker_id="w-1")

        assert result is True
        mock_redis.hset.assert_called_once()
        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert mapping["status"] == "processing"
        assert "started_at" in mapping
        assert mapping["worker_id"] == "w-1"

    @pytest.mark.asyncio
    async def test_update_nonexistent_job(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = False

        result = await tracker.update_status("j-missing", JobStatus.COMPLETED)
        assert result is False
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_completed_status_adds_completed_at(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.COMPLETED)

        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert "completed_at" in mapping

    @pytest.mark.asyncio
    async def test_failed_status_adds_completed_at(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.FAILED, error="boom")

        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert "completed_at" in mapping
        assert mapping["error"] == "boom"

    @pytest.mark.asyncio
    async def test_cancelled_status_adds_completed_at(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.CANCELLED)

        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert "completed_at" in mapping

    @pytest.mark.asyncio
    async def test_retrying_status_no_completed_at(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.RETRYING)

        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert "completed_at" not in mapping
        assert "started_at" not in mapping

    @pytest.mark.asyncio
    async def test_update_refreshes_ttl(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.PROCESSING)

        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_with_result(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.COMPLETED, result={"answer": "yes"})

        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert json.loads(mapping["result"]) == {"answer": "yes"}

    @pytest.mark.asyncio
    async def test_update_with_attempts(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.exists.return_value = True

        await tracker.update_status("j-1", JobStatus.PROCESSING, attempts=3)

        mapping = mock_redis.hset.call_args[1]["mapping"]
        assert mapping["attempts"] == "3"


class TestGetJob:
    """Tests for JobStatusTracker.get_job."""

    @pytest.mark.asyncio
    async def test_get_existing_job(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hgetall.return_value = {
            "id": "j-1",
            "payload": '{"q": "test"}',
            "status": "processing",
            "created_at": "1000.0",
            "started_at": "1001.0",
            "attempts": "2",
            "max_attempts": "5",
            "worker_id": "w-1",
            "priority": "3",
            "metadata": "{}",
        }

        job = await tracker.get_job("j-1")

        assert job is not None
        assert job.id == "j-1"
        assert job.payload == {"q": "test"}
        assert job.status == JobStatus.PROCESSING
        assert job.created_at == 1000.0
        assert job.started_at == 1001.0
        assert job.attempts == 2
        assert job.max_attempts == 5
        assert job.worker_id == "w-1"
        assert job.priority == 3

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hgetall.return_value = {}

        job = await tracker.get_job("j-missing")
        assert job is None

    @pytest.mark.asyncio
    async def test_get_job_with_bytes_keys(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hgetall.return_value = {
            b"id": b"j-bytes",
            b"payload": b'{"q": "bytes"}',
            b"status": b"pending",
            b"created_at": b"1000.0",
            b"attempts": b"0",
            b"max_attempts": b"3",
            b"priority": b"0",
            b"metadata": b"{}",
        }

        job = await tracker.get_job("j-bytes")

        assert job is not None
        assert job.id == "j-bytes"
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_job_handles_parse_error(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hgetall.return_value = {
            "id": "j-bad",
            "status": "invalid_status",
        }

        job = await tracker.get_job("j-bad")
        assert job is None  # should handle gracefully


class TestGetStatus:
    """Tests for JobStatusTracker.get_status."""

    @pytest.mark.asyncio
    async def test_get_status_string(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hget.return_value = "completed"

        status = await tracker.get_status("j-1")
        assert status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_status_bytes(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hget.return_value = b"failed"

        status = await tracker.get_status("j-1")
        assert status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_status_none(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.hget.return_value = None

        status = await tracker.get_status("j-missing")
        assert status is None


class TestDelete:
    """Tests for JobStatusTracker.delete."""

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.delete.return_value = 1

        result = await tracker.delete("j-1")
        assert result is True
        mock_redis.delete.assert_called_once_with("aragora:queue:job:j-1")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        tracker, mock_redis = _make_tracker()
        mock_redis.delete.return_value = 0

        result = await tracker.delete("j-missing")
        assert result is False


class TestGetCountsByStatus:
    """Tests for JobStatusTracker.get_counts_by_status."""

    @pytest.mark.asyncio
    async def test_counts_all_statuses(self):
        tracker, mock_redis = _make_tracker()

        # Simulate scan returning 3 keys
        async def mock_scan_iter(**kwargs):
            for key in [
                b"aragora:queue:job:j-1",
                b"aragora:queue:job:j-2",
                b"aragora:queue:job:j-3",
            ]:
                yield key

        mock_redis.scan_iter = mock_scan_iter
        mock_redis.hget.side_effect = [b"pending", b"completed", b"pending"]

        counts = await tracker.get_counts_by_status()

        assert counts["pending"] == 2
        assert counts["completed"] == 1
        assert counts["processing"] == 0
        assert counts["failed"] == 0

    @pytest.mark.asyncio
    async def test_counts_empty_queue(self):
        tracker, mock_redis = _make_tracker()

        async def mock_scan_iter(**kwargs):
            return
            yield  # make it an async generator

        mock_redis.scan_iter = mock_scan_iter

        counts = await tracker.get_counts_by_status()

        for status in JobStatus:
            assert counts[status.value] == 0
