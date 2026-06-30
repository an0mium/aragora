"""
Tests for the queue module.

Tests the Redis Streams-based job queue, retry policies, and job management.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.queue import (
    Job,
    JobStatus,
    QueueConfig,
    RetryPolicy,
    create_debate_job,
    get_debate_payload,
    DebateJobPayload,
    DebateResult,
    is_retryable_error,
)
from aragora.queue.config import get_queue_config, reset_queue_config, set_queue_config
from aragora.queue.status import JobStatusTracker
from aragora.queue.streams import RedisStreamsQueue


class TestJob:
    """Tests for the Job dataclass."""

    def test_job_creation(self):
        """Test creating a job with default values."""
        job = Job(payload={"question": "test"})

        assert job.id is not None
        assert job.status == JobStatus.PENDING
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.payload == {"question": "test"}

    def test_job_to_dict(self):
        """Test serializing job to dictionary."""
        job = Job(
            id="test-id",
            payload={"question": "test"},
            status=JobStatus.PROCESSING,
            attempts=1,
        )

        data = job.to_dict()

        assert data["id"] == "test-id"
        assert data["status"] == "processing"
        assert data["attempts"] == 1
        assert data["payload"] == {"question": "test"}

    def test_job_from_dict(self):
        """Test deserializing job from dictionary."""
        data = {
            "id": "test-id",
            "payload": {"question": "test"},
            "status": "completed",
            "created_at": 1234567890.0,
            "attempts": 2,
            "max_attempts": 3,
        }

        job = Job.from_dict(data)

        assert job.id == "test-id"
        assert job.status == JobStatus.COMPLETED
        assert job.attempts == 2

    def test_mark_processing(self):
        """Test marking job as processing."""
        job = Job(payload={})
        job.mark_processing("worker-1")

        assert job.status == JobStatus.PROCESSING
        assert job.worker_id == "worker-1"
        assert job.attempts == 1
        assert job.started_at is not None

    def test_mark_completed(self):
        """Test marking job as completed."""
        job = Job(payload={})
        result = {"answer": "42"}
        job.mark_completed(result)

        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.metadata["result"] == result

    def test_mark_failed(self):
        """Test marking job as failed."""
        job = Job(payload={})
        job.mark_failed("Connection error")

        assert job.status == JobStatus.FAILED
        assert job.error == "Connection error"
        assert job.completed_at is not None

    def test_should_retry(self):
        """Test retry eligibility."""
        job = Job(payload={}, max_attempts=3)

        assert job.should_retry()  # 0 attempts

        job.attempts = 2
        assert job.should_retry()  # 2 < 3

        job.attempts = 3
        assert not job.should_retry()  # 3 >= 3


class TestRetryPolicy:
    """Tests for the RetryPolicy."""

    def test_default_policy(self):
        """Test default retry policy values."""
        policy = RetryPolicy()

        assert policy.max_attempts == 3
        assert policy.base_delay_seconds == 1.0
        assert policy.exponential_base == 2.0

    def test_get_delay_exponential(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert policy.get_delay(0) == 1.0  # 1 * 2^0
        assert policy.get_delay(1) == 2.0  # 1 * 2^1
        assert policy.get_delay(2) == 4.0  # 1 * 2^2

    def test_get_delay_capped(self):
        """Test delay is capped at max."""
        policy = RetryPolicy(
            base_delay_seconds=1.0,
            max_delay_seconds=5.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert policy.get_delay(10) == 5.0  # Capped at max

    def test_get_delay_with_jitter(self):
        """Test jitter adds randomness."""
        policy = RetryPolicy(
            base_delay_seconds=10.0,
            jitter=True,
        )

        # Get multiple delays
        delays = [policy.get_delay(0) for _ in range(10)]

        # Should have variation (±20%)
        assert min(delays) >= 8.0
        assert max(delays) <= 12.0
        # Not all the same
        assert len(set(delays)) > 1

    def test_should_retry_within_limit(self):
        """Test retry allowed within attempt limit."""
        policy = RetryPolicy(max_attempts=3)

        assert policy.should_retry(1)
        assert policy.should_retry(2)
        assert not policy.should_retry(3)

    def test_should_retry_non_retryable_error(self):
        """Test non-retryable errors."""
        policy = RetryPolicy()

        # ValueError is non-retryable
        assert not policy.should_retry(1, ValueError("invalid"))

        # TypeError is non-retryable
        assert not policy.should_retry(1, TypeError("wrong type"))


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_retryable_errors(self):
        """Test that general exceptions are retryable."""
        assert is_retryable_error(Exception("generic"))
        assert is_retryable_error(RuntimeError("runtime"))
        assert is_retryable_error(ConnectionError("connection"))

    def test_non_retryable_errors(self):
        """Test that validation errors are not retryable."""
        assert not is_retryable_error(ValueError("invalid"))
        assert not is_retryable_error(TypeError("wrong type"))
        assert not is_retryable_error(KeyError("missing"))

    def test_non_retryable_patterns(self):
        """Test error messages that indicate non-retryable conditions."""
        assert not is_retryable_error(Exception("Invalid request"))
        assert not is_retryable_error(Exception("not found"))
        assert not is_retryable_error(Exception("validation error"))


class TestQueueConfig:
    """Tests for queue configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_queue_config()

    def test_default_config(self):
        """Test default configuration values."""
        config = QueueConfig()

        assert config.redis_url == "redis://localhost:6379"
        assert config.retry_max_attempts == 3
        assert config.claim_idle_ms == 60000

    def test_stream_key(self):
        """Test stream key generation."""
        config = QueueConfig(key_prefix="test:")

        assert config.stream_key == "test:debates:stream"

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            QueueConfig(max_job_ttl_days=0)

        with pytest.raises(ValueError):
            QueueConfig(retry_max_attempts=100)

    def test_get_set_config(self):
        """Test global config get/set."""
        config = QueueConfig(key_prefix="custom:")
        set_queue_config(config)

        retrieved = get_queue_config()
        assert retrieved.key_prefix == "custom:"


class TestDebateJob:
    """Tests for debate job creation."""

    def test_create_debate_job(self):
        """Test creating a debate job."""
        job = create_debate_job(
            question="Should we use microservices?",
            agents=["claude", "gpt"],
            rounds=5,
        )

        assert job.id is not None
        assert job.payload["question"] == "Should we use microservices?"
        assert job.payload["agents"] == ["claude", "gpt"]
        assert job.payload["rounds"] == 5

    def test_get_debate_payload(self):
        """Test extracting payload from job."""
        job = create_debate_job(
            question="Test question",
            agents=["agent1"],
            consensus="unanimous",
        )

        payload = get_debate_payload(job)

        assert isinstance(payload, DebateJobPayload)
        assert payload.question == "Test question"
        assert payload.consensus == "unanimous"


class TestDebateResult:
    """Tests for debate result."""

    def test_result_to_dict(self):
        """Test serializing result to dictionary."""
        result = DebateResult(
            debate_id="test-123",
            consensus_reached=True,
            final_answer="Yes",
            confidence=0.95,
            rounds_used=3,
            participants=["claude", "gpt"],
            duration_seconds=120.5,
        )

        data = result.to_dict()

        assert data["debate_id"] == "test-123"
        assert data["consensus_reached"] is True
        assert data["confidence"] == 0.95

    def test_result_from_dict(self):
        """Test deserializing result from dictionary."""
        data = {
            "debate_id": "test-456",
            "consensus_reached": False,
            "final_answer": None,
            "confidence": 0.5,
            "rounds_used": 5,
            "participants": ["agent1"],
            "duration_seconds": 60.0,
        }

        result = DebateResult.from_dict(data)

        assert result.debate_id == "test-456"
        assert result.consensus_reached is False


class TestJobStatusTracker:
    """Tests for job status tracking."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.hset = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        redis.hget = AsyncMock(return_value=None)
        redis.exists = AsyncMock(return_value=True)
        redis.expire = AsyncMock()
        redis.delete = AsyncMock(return_value=1)
        return redis

    @pytest.fixture
    def tracker(self, mock_redis):
        """Create a status tracker with mock Redis."""
        with patch("aragora.queue.status.get_queue_config") as mock_config:
            mock_config.return_value = QueueConfig()
            return JobStatusTracker(mock_redis)

    @pytest.mark.asyncio
    async def test_create_job(self, tracker, mock_redis):
        """Test creating a job status entry."""
        job = Job(id="test-id", payload={"question": "test"})

        await tracker.create(job)

        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status(self, tracker, mock_redis):
        """Test updating job status."""
        result = await tracker.update_status("test-id", JobStatus.COMPLETED)

        assert result is True
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, tracker, mock_redis):
        """Test updating non-existent job."""
        mock_redis.exists.return_value = False

        result = await tracker.update_status("missing-id", JobStatus.COMPLETED)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_job(self, tracker, mock_redis):
        """Test getting job state."""
        mock_redis.hgetall.return_value = {
            b"id": b"test-id",
            b"payload": b'{"question": "test"}',
            b"status": b"pending",
            b"created_at": b"1234567890.0",
            b"attempts": b"0",
            b"max_attempts": b"3",
            b"priority": b"0",
            b"metadata": b"{}",
        }

        job = await tracker.get_job("test-id")

        assert job is not None
        assert job.id == "test-id"
        assert job.status == JobStatus.PENDING


class TestRedisStreamsQueue:
    """Tests for Redis Streams queue."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.xgroup_create = AsyncMock()
        redis.xadd = AsyncMock(return_value=b"1234567890-0")
        redis.xreadgroup = AsyncMock(return_value=[])
        redis.xack = AsyncMock()
        redis.xlen = AsyncMock(return_value=0)
        redis.xpending = AsyncMock(return_value={"pending": 0})
        redis.xpending_range = AsyncMock(return_value=[])
        redis.xclaim = AsyncMock()
        redis.hset = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        redis.hget = AsyncMock(return_value=None)
        redis.exists = AsyncMock(return_value=True)
        redis.expire = AsyncMock()
        redis.delete = AsyncMock(return_value=1)

        # Mock scan_iter as async generator
        async def mock_scan_iter(*args, **kwargs):
            return
            yield  # Make it a generator

        redis.scan_iter = mock_scan_iter
        return redis

    @pytest.fixture
    def queue(self, mock_redis):
        """Create a queue with mock Redis."""
        return RedisStreamsQueue(
            redis_client=mock_redis,
            consumer_name="test-worker",
            config=QueueConfig(),
        )

    @pytest.mark.asyncio
    async def test_enqueue(self, queue, mock_redis):
        """Test enqueueing a job."""
        job = create_debate_job(question="test")

        job_id = await queue.enqueue(job)

        assert job_id == job.id
        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_dequeue_empty(self, queue, mock_redis):
        """Test dequeueing from empty queue."""
        mock_redis.xreadgroup.return_value = []

        job = await queue.dequeue("worker-1", timeout_ms=100)

        assert job is None

    @pytest.mark.asyncio
    async def test_dequeue_success(self, queue, mock_redis):
        """Test successful dequeue."""
        mock_redis.xreadgroup.return_value = [
            (
                b"stream",
                [
                    (
                        b"1234-0",
                        {
                            b"job_id": b"test-job",
                            b"payload": b'{"question": "test"}',
                            b"priority": b"0",
                            b"max_attempts": b"3",
                            b"created_at": b"1234567890.0",
                            b"metadata": b"{}",
                        },
                    )
                ],
            )
        ]

        job = await queue.dequeue("worker-1")

        assert job is not None
        assert job.id == "test-job"
        assert job.status == JobStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_ack(self, queue, mock_redis):
        """Test acknowledging a job."""
        # Set up job with message ID
        mock_redis.hgetall.return_value = {
            b"id": b"test-id",
            b"payload": b"{}",
            b"status": b"processing",
            b"created_at": b"1234567890.0",
            b"attempts": b"1",
            b"max_attempts": b"3",
            b"priority": b"0",
            b"metadata": b'{"_stream_message_id": "1234-0"}',
        }

        result = await queue.ack("test-id")

        assert result is True
        mock_redis.xack.assert_called()

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, queue, mock_redis):
        """Test getting queue statistics."""
        mock_redis.xlen.return_value = 10

        stats = await queue.get_queue_stats()

        assert "stream_length" in stats
        assert stats["stream_length"] == 10


class TestQueueIntegration:
    """Integration-style tests (still using mocks but testing component interaction)."""

    @pytest.mark.asyncio
    async def test_job_lifecycle(self):
        """Test complete job lifecycle."""
        # Create job
        job = create_debate_job(
            question="Integration test",
            agents=["claude"],
            priority=5,
        )
        assert job.status == JobStatus.PENDING

        # Simulate processing
        job.mark_processing("worker-1")
        assert job.status == JobStatus.PROCESSING
        assert job.attempts == 1

        # Simulate completion
        job.mark_completed({"answer": "success"})
        assert job.status == JobStatus.COMPLETED
        assert "result" in job.metadata

    @pytest.mark.asyncio
    async def test_job_retry_lifecycle(self):
        """Test job retry lifecycle."""
        policy = RetryPolicy(max_attempts=3)
        job = create_debate_job(question="Retry test", max_attempts=3)

        # First attempt fails
        job.mark_processing("worker-1")
        job.mark_retrying("Temporary error")
        assert policy.should_retry(job.attempts)

        # Second attempt fails
        job.mark_processing("worker-1")
        job.mark_retrying("Still failing")
        assert policy.should_retry(job.attempts)

        # Third attempt fails - no more retries
        job.mark_processing("worker-1")
        job.mark_failed("Final failure")
        assert not policy.should_retry(job.attempts)
        assert job.status == JobStatus.FAILED
