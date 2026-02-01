"""
Tests for Redis Streams queue implementation.

Tests cover:
- RedisStreamsQueue construction and properties
- _ensure_initialized (creates consumer group, handles BUSYGROUP)
- enqueue (serializes job, adds to stream, tracks status)
- dequeue (reads from consumer group, reconstructs job, handles bytes)
- ack (acknowledges job, updates status)
- nack (retry vs permanent failure paths)
- cancel (only pending/retrying jobs)
- get_queue_stats (combines status tracker and stream info)
- claim_stale_jobs (xclaim with pending range)
- close (resets initialization flag)
- create_redis_queue factory function
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue.base import Job, JobStatus
from aragora.queue.config import QueueConfig, reset_queue_config, set_queue_config
from aragora.queue.streams import RedisStreamsQueue


@pytest.fixture(autouse=True)
def _reset_config():
    set_queue_config(QueueConfig())
    yield
    reset_queue_config()


def _make_queue(consumer_name: str = "test-consumer") -> tuple[RedisStreamsQueue, AsyncMock]:
    """Create a RedisStreamsQueue with a mock Redis client."""
    mock_redis = AsyncMock()
    config = QueueConfig()
    queue = RedisStreamsQueue(
        redis_client=mock_redis,
        consumer_name=consumer_name,
        config=config,
    )
    return queue, mock_redis


class TestConstruction:
    """Tests for RedisStreamsQueue construction."""

    def test_properties(self):
        queue, _ = _make_queue("worker-1")
        assert queue.stream_key == "aragora:queue:debates:stream"
        assert queue.group_name == "debate-workers"

    def test_not_initialized(self):
        queue, _ = _make_queue()
        assert queue._initialized is False


class TestEnsureInitialized:
    """Tests for _ensure_initialized."""

    @pytest.mark.asyncio
    async def test_creates_consumer_group(self):
        queue, mock_redis = _make_queue()

        await queue._ensure_initialized()

        mock_redis.xgroup_create.assert_called_once_with(
            queue.stream_key,
            queue.group_name,
            id="0",
            mkstream=True,
        )
        assert queue._initialized is True

    @pytest.mark.asyncio
    async def test_handles_busygroup_error(self):
        queue, mock_redis = _make_queue()
        mock_redis.xgroup_create.side_effect = Exception("BUSYGROUP Consumer group already exists")

        await queue._ensure_initialized()

        assert queue._initialized is True  # should still mark as initialized

    @pytest.mark.asyncio
    async def test_raises_on_other_error(self):
        queue, mock_redis = _make_queue()
        mock_redis.xgroup_create.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            await queue._ensure_initialized()

        assert queue._initialized is False

    @pytest.mark.asyncio
    async def test_skips_if_already_initialized(self):
        queue, mock_redis = _make_queue()
        queue._initialized = True

        await queue._ensure_initialized()

        mock_redis.xgroup_create.assert_not_called()


class TestEnqueue:
    """Tests for enqueue."""

    @pytest.mark.asyncio
    async def test_enqueue_adds_to_stream(self):
        queue, mock_redis = _make_queue()
        mock_redis.xgroup_create.side_effect = Exception("BUSYGROUP already exists")

        job = Job(id="j-enq-1", payload={"q": "test"}, priority=0)

        result = await queue.enqueue(job)

        assert result == "j-enq-1"
        mock_redis.xadd.assert_called_once()
        entry = mock_redis.xadd.call_args[0][1]
        assert entry["job_id"] == "j-enq-1"
        assert json.loads(entry["payload"]) == {"q": "test"}

    @pytest.mark.asyncio
    async def test_enqueue_sets_priority(self):
        queue, mock_redis = _make_queue()
        mock_redis.xgroup_create.side_effect = Exception("BUSYGROUP already exists")

        job = Job(id="j-pri", payload={})
        await queue.enqueue(job, priority=10)

        entry = mock_redis.xadd.call_args[0][1]
        assert entry["priority"] == "10"
        assert job.priority == 10


class TestDequeue:
    """Tests for dequeue."""

    @pytest.mark.asyncio
    async def test_dequeue_returns_job(self):
        queue, mock_redis = _make_queue("worker-1")
        queue._initialized = True

        mock_redis.xreadgroup.return_value = [
            (
                "aragora:queue:debates:stream",
                [
                    (
                        "1234-0",
                        {
                            "job_id": "j-deq-1",
                            "payload": '{"q": "dequeue test"}',
                            "priority": "5",
                            "max_attempts": "3",
                            "created_at": "1000.0",
                            "metadata": "{}",
                        },
                    )
                ],
            )
        ]

        job = await queue.dequeue("worker-1")

        assert job is not None
        assert job.id == "j-deq-1"
        assert job.payload == {"q": "dequeue test"}
        assert job.priority == 5
        assert job.status == JobStatus.PROCESSING
        assert job.worker_id == "worker-1"
        assert job.metadata["_stream_message_id"] == "1234-0"

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_on_empty(self):
        queue, mock_redis = _make_queue()
        queue._initialized = True
        mock_redis.xreadgroup.return_value = None

        job = await queue.dequeue("worker-1")
        assert job is None

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_on_empty_entries(self):
        queue, mock_redis = _make_queue()
        queue._initialized = True
        mock_redis.xreadgroup.return_value = [("stream", [])]

        job = await queue.dequeue("worker-1")
        assert job is None

    @pytest.mark.asyncio
    async def test_dequeue_handles_bytes(self):
        queue, mock_redis = _make_queue("worker-1")
        queue._initialized = True

        mock_redis.xreadgroup.return_value = [
            (
                b"aragora:queue:debates:stream",
                [
                    (
                        b"5678-0",
                        {
                            b"job_id": b"j-bytes",
                            b"payload": b'{"q": "bytes"}',
                            b"priority": b"0",
                            b"max_attempts": b"3",
                            b"created_at": b"1000.0",
                            b"metadata": b"{}",
                        },
                    )
                ],
            )
        ]

        job = await queue.dequeue("worker-1")

        assert job is not None
        assert job.id == "j-bytes"

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_on_error(self):
        queue, mock_redis = _make_queue()
        queue._initialized = True
        mock_redis.xreadgroup.side_effect = Exception("Redis connection lost")

        job = await queue.dequeue("worker-1")
        assert job is None


class TestAck:
    """Tests for ack."""

    @pytest.mark.asyncio
    async def test_ack_existing_job(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(
            id="j-ack-1",
            payload={},
            metadata={"_stream_message_id": "1234-0"},
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.ack("j-ack-1")

        assert result is True
        mock_redis.xack.assert_called_once_with(queue.stream_key, queue.group_name, "1234-0")
        queue._status_tracker.update_status.assert_called_once_with("j-ack-1", JobStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_ack_no_message_id(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(id="j-ack-2", payload={}, metadata={})
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.ack("j-ack-2")

        assert result is True
        mock_redis.xack.assert_not_called()

    @pytest.mark.asyncio
    async def test_ack_nonexistent_job(self):
        queue, _ = _make_queue()
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = None

        result = await queue.ack("j-missing")
        assert result is False


class TestNack:
    """Tests for nack."""

    @pytest.mark.asyncio
    async def test_nack_with_retry(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(
            id="j-nack-1",
            payload={},
            attempts=1,
            max_attempts=3,
            metadata={"_stream_message_id": "1234-0"},
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.nack("j-nack-1", requeue=True)

        assert result is True
        queue._status_tracker.update_status.assert_called_once_with(
            "j-nack-1", JobStatus.RETRYING, error=mock_job.error
        )
        mock_redis.xack.assert_not_called()  # not acknowledged yet

    @pytest.mark.asyncio
    async def test_nack_permanent_failure(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(
            id="j-nack-2",
            payload={},
            attempts=3,
            max_attempts=3,
            metadata={"_stream_message_id": "1234-0"},
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.nack("j-nack-2", requeue=True)

        assert result is True
        queue._status_tracker.update_status.assert_called_once_with(
            "j-nack-2", JobStatus.FAILED, error=mock_job.error
        )
        mock_redis.xack.assert_called_once()  # acknowledged to remove from pending

    @pytest.mark.asyncio
    async def test_nack_no_requeue(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(
            id="j-nack-3",
            payload={},
            attempts=1,
            max_attempts=3,
            metadata={"_stream_message_id": "1234-0"},
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.nack("j-nack-3", requeue=False)

        assert result is True
        queue._status_tracker.update_status.assert_called_once_with(
            "j-nack-3", JobStatus.FAILED, error=mock_job.error
        )

    @pytest.mark.asyncio
    async def test_nack_nonexistent_job(self):
        queue, _ = _make_queue()
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = None

        result = await queue.nack("j-missing")
        assert result is False


class TestCancel:
    """Tests for cancel."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(
            id="j-cancel-1",
            payload={},
            status=JobStatus.PENDING,
            metadata={"_stream_message_id": "1234-0"},
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.cancel("j-cancel-1")

        assert result is True
        queue._status_tracker.update_status.assert_called_once_with(
            "j-cancel-1", JobStatus.CANCELLED
        )
        mock_redis.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_retrying_job(self):
        queue, mock_redis = _make_queue()
        mock_job = Job(
            id="j-cancel-2",
            payload={},
            status=JobStatus.RETRYING,
            metadata={"_stream_message_id": "5678-0"},
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.cancel("j-cancel-2")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_processing_job_rejected(self):
        queue, _ = _make_queue()
        mock_job = Job(
            id="j-cancel-3",
            payload={},
            status=JobStatus.PROCESSING,
        )
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.cancel("j-cancel-3")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_job_rejected(self):
        queue, _ = _make_queue()
        mock_job = Job(id="j-cancel-4", payload={}, status=JobStatus.COMPLETED)
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.cancel("j-cancel-4")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self):
        queue, _ = _make_queue()
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = None

        result = await queue.cancel("j-missing")
        assert result is False


class TestGetQueueStats:
    """Tests for get_queue_stats."""

    @pytest.mark.asyncio
    async def test_returns_combined_stats(self):
        queue, mock_redis = _make_queue()
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_counts_by_status.return_value = {
            "pending": 5,
            "processing": 2,
            "completed": 10,
            "failed": 1,
            "cancelled": 0,
            "retrying": 1,
        }
        mock_redis.xlen.return_value = 19
        mock_redis.xpending.return_value = {"pending": 3}

        stats = await queue.get_queue_stats()

        assert stats["stream_length"] == 19
        assert stats["pending_in_group"] == 3
        assert stats["pending"] == 5
        assert stats["completed"] == 10

    @pytest.mark.asyncio
    async def test_handles_xpending_error(self):
        queue, mock_redis = _make_queue()
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_counts_by_status.return_value = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "retrying": 0,
        }
        mock_redis.xlen.return_value = 0
        mock_redis.xpending.side_effect = Exception("NOGROUP")

        stats = await queue.get_queue_stats()
        assert stats["pending_in_group"] == 0


class TestClaimStaleJobs:
    """Tests for claim_stale_jobs."""

    @pytest.mark.asyncio
    async def test_claims_stale_dict_format(self):
        queue, mock_redis = _make_queue("claimer")
        queue._initialized = True

        mock_redis.xpending_range.return_value = [
            {"message_id": "111-0", "idle": 120000},
            {"message_id": "222-0", "idle": 5000},  # not stale enough
        ]

        claimed = await queue.claim_stale_jobs(idle_ms=60000)

        assert claimed == 1
        mock_redis.xclaim.assert_called_once()

    @pytest.mark.asyncio
    async def test_claims_stale_tuple_format(self):
        queue, mock_redis = _make_queue("claimer")
        queue._initialized = True

        mock_redis.xpending_range.return_value = [
            ("333-0", "dead-worker", 90000, 2),
        ]

        claimed = await queue.claim_stale_jobs(idle_ms=60000)

        assert claimed == 1

    @pytest.mark.asyncio
    async def test_no_pending_messages(self):
        queue, mock_redis = _make_queue()
        queue._initialized = True
        mock_redis.xpending_range.return_value = []

        claimed = await queue.claim_stale_jobs(idle_ms=60000)
        assert claimed == 0

    @pytest.mark.asyncio
    async def test_handles_error(self):
        queue, mock_redis = _make_queue()
        queue._initialized = True
        mock_redis.xpending_range.side_effect = Exception("Redis error")

        claimed = await queue.claim_stale_jobs(idle_ms=60000)
        assert claimed == 0


class TestClose:
    """Tests for close."""

    @pytest.mark.asyncio
    async def test_close_resets_initialized(self):
        queue, _ = _make_queue()
        queue._initialized = True

        await queue.close()

        assert queue._initialized is False


class TestGetStatus:
    """Tests for get_status proxy."""

    @pytest.mark.asyncio
    async def test_delegates_to_tracker(self):
        queue, _ = _make_queue()
        mock_job = Job(id="j-st-1", payload={}, status=JobStatus.COMPLETED)
        queue._status_tracker = AsyncMock()
        queue._status_tracker.get_job.return_value = mock_job

        result = await queue.get_status("j-st-1")
        assert result is mock_job
