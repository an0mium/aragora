"""
Tests for Queue Workers.

Tests for webhook delivery worker, batch worker, and retry policies.
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Retry Policy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_retry_policy_creation(self):
        """Test RetryPolicy dataclass creation."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(
            max_attempts=5,
            base_delay_seconds=2.0,
            max_delay_seconds=120.0,
        )

        assert policy.max_attempts == 5
        assert policy.base_delay_seconds == 2.0
        assert policy.max_delay_seconds == 120.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True

    def test_retry_policy_defaults(self):
        """Test default values."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy()

        assert policy.max_attempts == 3
        assert policy.base_delay_seconds == 1.0
        assert policy.max_delay_seconds == 300.0

    def test_get_delay_exponential_backoff(self):
        """Test exponential backoff calculation."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(
            base_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=False,  # Disable for predictable testing
        )

        # Attempt 0: 1 * 2^0 = 1
        assert policy.get_delay(0) == 1.0

        # Attempt 1: 1 * 2^1 = 2
        assert policy.get_delay(1) == 2.0

        # Attempt 2: 1 * 2^2 = 4
        assert policy.get_delay(2) == 4.0

        # Attempt 3: 1 * 2^3 = 8
        assert policy.get_delay(3) == 8.0

    def test_get_delay_caps_at_max(self):
        """Test delay is capped at max_delay_seconds."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(
            base_delay_seconds=10.0,
            max_delay_seconds=50.0,
            jitter=False,
        )

        # Attempt 5: 10 * 2^5 = 320, but capped at 50
        assert policy.get_delay(5) == 50.0

    def test_get_delay_with_jitter(self):
        """Test jitter adds randomness."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(
            base_delay_seconds=10.0,
            jitter=True,
        )

        # Run multiple times and check variance
        delays = [policy.get_delay(1) for _ in range(100)]

        # Should have some variance due to jitter
        assert min(delays) != max(delays)

        # Jitter is ±20%, so delays should be between 16 and 24 (20 * 0.8 to 20 * 1.2)
        for delay in delays:
            assert 16.0 <= delay <= 24.0

    def test_should_retry_within_limit(self):
        """Test should_retry returns True within limit."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(max_attempts=3)

        assert policy.should_retry(0) is True
        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True

    def test_should_retry_at_limit(self):
        """Test should_retry returns False at limit."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(max_attempts=3)

        assert policy.should_retry(3) is False
        assert policy.should_retry(4) is False

    def test_should_retry_skips_validation_errors(self):
        """Test should_retry returns False for validation errors."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(max_attempts=5)

        assert policy.should_retry(1, ValueError("invalid input")) is False
        assert policy.should_retry(1, TypeError("wrong type")) is False
        assert policy.should_retry(1, KeyError("missing key")) is False

    def test_should_retry_allows_runtime_errors(self):
        """Test should_retry returns True for runtime errors."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(max_attempts=5)

        assert policy.should_retry(1, RuntimeError("temporary failure")) is True
        assert policy.should_retry(1, ConnectionError("network issue")) is True
        assert policy.should_retry(1, TimeoutError("request timeout")) is True

    def test_get_remaining_attempts(self):
        """Test get_remaining_attempts calculation."""
        from aragora.queue.retry import RetryPolicy

        policy = RetryPolicy(max_attempts=5)

        assert policy.get_remaining_attempts(0) == 5
        assert policy.get_remaining_attempts(1) == 4
        assert policy.get_remaining_attempts(3) == 2
        assert policy.get_remaining_attempts(5) == 0
        assert policy.get_remaining_attempts(10) == 0  # Never negative


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_non_retryable_exceptions(self):
        """Test non-retryable exceptions."""
        from aragora.queue.retry import is_retryable_error

        assert is_retryable_error(ValueError("bad value")) is False
        assert is_retryable_error(TypeError("wrong type")) is False
        assert is_retryable_error(KeyError("missing")) is False
        assert is_retryable_error(AttributeError("no attr")) is False
        assert is_retryable_error(ImportError("no module")) is False

    def test_retryable_exceptions(self):
        """Test retryable exceptions."""
        from aragora.queue.retry import is_retryable_error

        assert is_retryable_error(RuntimeError("temporary failure")) is True
        assert is_retryable_error(ConnectionError("network issue")) is True
        assert is_retryable_error(TimeoutError("timeout")) is True
        assert is_retryable_error(OSError("io error")) is True

    def test_non_retryable_error_messages(self):
        """Test errors with non-retryable patterns in message."""
        from aragora.queue.retry import is_retryable_error

        assert is_retryable_error(Exception("Invalid token")) is False
        assert is_retryable_error(Exception("Resource not found")) is False
        assert is_retryable_error(Exception("Unauthorized access")) is False
        assert is_retryable_error(Exception("Forbidden action")) is False
        assert is_retryable_error(Exception("Bad request format")) is False


# =============================================================================
# Job and JobStatus Tests
# =============================================================================


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        """Test Job creation with defaults."""
        from aragora.queue.base import Job, JobStatus

        job = Job(payload={"task": "test"})

        assert job.payload == {"task": "test"}
        assert job.status == JobStatus.PENDING
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.error is None
        assert job.id is not None

    def test_job_to_dict(self):
        """Test Job.to_dict serialization."""
        from aragora.queue.base import Job

        job = Job(payload={"key": "value"}, priority=5)
        data = job.to_dict()

        assert data["payload"] == {"key": "value"}
        assert data["priority"] == 5
        assert data["status"] == "pending"

    def test_job_from_dict(self):
        """Test Job.from_dict deserialization."""
        from aragora.queue.base import Job, JobStatus

        data = {
            "id": "job-123",
            "payload": {"data": "test"},
            "status": "processing",
            "created_at": 1000.0,
            "attempts": 2,
            "max_attempts": 5,
            "priority": 10,
        }

        job = Job.from_dict(data)

        assert job.id == "job-123"
        assert job.payload == {"data": "test"}
        assert job.status == JobStatus.PROCESSING
        assert job.attempts == 2

    def test_job_mark_processing(self):
        """Test mark_processing method."""
        from aragora.queue.base import Job, JobStatus

        job = Job(payload={})
        job.mark_processing("worker-1")

        assert job.status == JobStatus.PROCESSING
        assert job.worker_id == "worker-1"
        assert job.attempts == 1
        assert job.started_at is not None

    def test_job_mark_completed(self):
        """Test mark_completed method."""
        from aragora.queue.base import Job, JobStatus

        job = Job(payload={})
        job.mark_completed(result={"output": "done"})

        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.metadata["result"] == {"output": "done"}

    def test_job_mark_failed(self):
        """Test mark_failed method."""
        from aragora.queue.base import Job, JobStatus

        job = Job(payload={})
        job.mark_failed("Something went wrong")

        assert job.status == JobStatus.FAILED
        assert job.error == "Something went wrong"
        assert job.completed_at is not None

    def test_job_should_retry(self):
        """Test should_retry method."""
        from aragora.queue.base import Job

        job = Job(payload={}, max_attempts=3)

        job.attempts = 0
        assert job.should_retry() is True

        job.attempts = 2
        assert job.should_retry() is True

        job.attempts = 3
        assert job.should_retry() is False


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """Test JobStatus enum values."""
        from aragora.queue.base import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.RETRYING.value == "retrying"


# =============================================================================
# Webhook Worker Tests
# =============================================================================


class TestDeliveryResult:
    """Tests for DeliveryResult dataclass."""

    def test_delivery_result_creation(self):
        """Test DeliveryResult creation."""
        from aragora.queue.webhook_worker import DeliveryResult

        result = DeliveryResult(
            webhook_id="wh_123",
            url="https://example.com/webhook",
            success=True,
            status_code=200,
            response_time_ms=150.5,
        )

        assert result.webhook_id == "wh_123"
        assert result.url == "https://example.com/webhook"
        assert result.success is True
        assert result.status_code == 200
        assert result.response_time_ms == 150.5

    def test_delivery_result_failure(self):
        """Test DeliveryResult for failed delivery."""
        from aragora.queue.webhook_worker import DeliveryResult

        result = DeliveryResult(
            webhook_id="wh_456",
            url="https://example.com/webhook",
            success=False,
            error="Connection refused",
            attempt=3,
        )

        assert result.success is False
        assert result.error == "Connection refused"
        assert result.attempt == 3


class TestEndpointHealth:
    """Tests for EndpointHealth dataclass."""

    def test_endpoint_health_creation(self):
        """Test EndpointHealth creation."""
        from aragora.queue.webhook_worker import EndpointHealth

        health = EndpointHealth(url="https://example.com/webhook")

        assert health.url == "https://example.com/webhook"
        assert health.total_deliveries == 0
        assert health.successful_deliveries == 0
        assert health.circuit_state == "closed"

    def test_success_rate_calculation(self):
        """Test success_rate property."""
        from aragora.queue.webhook_worker import EndpointHealth

        health = EndpointHealth(
            url="https://example.com",
            total_deliveries=100,
            successful_deliveries=95,
            failed_deliveries=5,
        )

        assert health.success_rate == 95.0

    def test_success_rate_no_deliveries(self):
        """Test success_rate with no deliveries."""
        from aragora.queue.webhook_worker import EndpointHealth

        health = EndpointHealth(url="https://example.com")
        assert health.success_rate == 100.0  # Default to 100% when no data


class TestWebhookDeliveryWorker:
    """Tests for WebhookDeliveryWorker."""

    def test_worker_initialization(self):
        """Test worker initialization."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker

        mock_queue = MagicMock()
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="webhook-worker-1",
            max_concurrent=20,
            request_timeout=15.0,
        )

        assert worker.worker_id == "webhook-worker-1"
        assert worker._max_concurrent == 20
        assert worker._request_timeout == 15.0
        assert worker.is_running is False

    def test_worker_get_stats(self):
        """Test get_stats method."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker

        mock_queue = MagicMock()
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="worker-1",
        )

        # Simulate some activity
        worker._deliveries_total = 100
        worker._deliveries_succeeded = 95
        worker._deliveries_failed = 5
        worker._start_time = time.time() - 3600

        stats = worker.get_stats()

        assert stats["worker_id"] == "worker-1"
        assert stats["deliveries_total"] == 100
        assert stats["deliveries_succeeded"] == 95
        assert stats["deliveries_failed"] == 5
        assert stats["uptime_seconds"] >= 3599

    def test_generate_signature(self):
        """Test HMAC signature generation."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker

        mock_queue = MagicMock()
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="worker-1",
        )

        payload = b'{"event": "test"}'
        secret = "my_secret_key"

        signature = worker._generate_signature(payload, secret)

        assert signature.startswith("sha256=")
        assert len(signature) == 71  # "sha256=" + 64 hex chars

    def test_get_endpoint_health(self):
        """Test endpoint health retrieval."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker, EndpointHealth

        mock_queue = MagicMock()
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="worker-1",
        )

        # Initially no health data
        assert worker.get_endpoint_health("https://example.com") is None

        # Add health data
        worker._endpoint_health["https://example.com"] = EndpointHealth(
            url="https://example.com",
            total_deliveries=10,
        )

        health = worker.get_endpoint_health("https://example.com")
        assert health is not None
        assert health.total_deliveries == 10

    def test_get_all_endpoint_health(self):
        """Test retrieving all endpoint health."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker, EndpointHealth

        mock_queue = MagicMock()
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="worker-1",
        )

        worker._endpoint_health["https://a.com"] = EndpointHealth(url="https://a.com")
        worker._endpoint_health["https://b.com"] = EndpointHealth(url="https://b.com")

        all_health = worker.get_all_endpoint_health()
        assert len(all_health) == 2

    @pytest.mark.asyncio
    async def test_worker_start_stop(self):
        """Test worker start and stop."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker

        mock_queue = AsyncMock()
        mock_queue.dequeue.return_value = None

        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="worker-1",
        )

        await worker.start()
        assert worker.is_running is True
        assert worker._start_time is not None

        await worker.stop(timeout=1.0)
        assert worker.is_running is False

    def test_update_metrics(self):
        """Test metrics update."""
        from aragora.queue.webhook_worker import WebhookDeliveryWorker, DeliveryResult

        mock_queue = MagicMock()
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="worker-1",
        )

        # Successful delivery
        result = DeliveryResult(
            webhook_id="wh_1",
            url="https://example.com",
            success=True,
        )
        worker._update_metrics(result)

        assert worker._deliveries_total == 1
        assert worker._deliveries_succeeded == 1
        assert worker._deliveries_failed == 0

        # Failed delivery
        result = DeliveryResult(
            webhook_id="wh_2",
            url="https://example.com",
            success=False,
        )
        worker._update_metrics(result)

        assert worker._deliveries_total == 2
        assert worker._deliveries_succeeded == 1
        assert worker._deliveries_failed == 1


class TestEnqueueWebhookDelivery:
    """Tests for enqueue_webhook_delivery function."""

    @pytest.mark.asyncio
    async def test_enqueue_webhook_delivery(self):
        """Test enqueueing a webhook delivery job."""
        from aragora.queue.webhook_worker import enqueue_webhook_delivery

        mock_queue = AsyncMock()

        job = await enqueue_webhook_delivery(
            queue=mock_queue,
            webhook_id="wh_123",
            url="https://example.com/webhook",
            secret="secret_key",
            event_type="debate.completed",
            event_data={"debate_id": "d_456"},
            priority=5,
        )

        assert job is not None
        assert job.payload["webhook_id"] == "wh_123"
        assert job.payload["url"] == "https://example.com/webhook"
        assert job.payload["event_type"] == "debate.completed"
        assert job.priority == 5

        mock_queue.enqueue.assert_called_once()


# =============================================================================
# Batch Worker Tests
# =============================================================================


class TestBatchJobProgress:
    """Tests for BatchJobProgress dataclass."""

    def test_batch_progress_creation(self):
        """Test BatchJobProgress creation."""
        from aragora.queue.batch_worker import BatchJobProgress

        progress = BatchJobProgress(job_id="batch-123", total=10)

        assert progress.job_id == "batch-123"
        assert progress.total == 10
        assert progress.processed == 0
        assert progress.succeeded == 0
        assert progress.failed == 0

    def test_completion_percent(self):
        """Test completion_percent property."""
        from aragora.queue.batch_worker import BatchJobProgress

        progress = BatchJobProgress(job_id="batch-123", total=10)
        progress.processed = 5

        assert progress.completion_percent == 50.0

    def test_completion_percent_empty(self):
        """Test completion_percent with empty batch."""
        from aragora.queue.batch_worker import BatchJobProgress

        progress = BatchJobProgress(job_id="batch-123", total=0)
        assert progress.completion_percent == 100.0

    def test_elapsed_seconds(self):
        """Test elapsed_seconds property."""
        from aragora.queue.batch_worker import BatchJobProgress

        progress = BatchJobProgress(job_id="batch-123", total=10)
        progress.started_at = time.time() - 60  # 60 seconds ago

        assert progress.elapsed_seconds >= 59.0


class TestBatchExplainabilityWorker:
    """Tests for BatchExplainabilityWorker."""

    def test_worker_initialization(self):
        """Test batch worker initialization."""
        from aragora.queue.batch_worker import BatchExplainabilityWorker

        mock_queue = MagicMock()
        mock_generator = AsyncMock()

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="batch-worker-1",
            explain_generator=mock_generator,
            max_concurrent_debates=10,
            max_concurrent_batches=3,
        )

        assert worker.worker_id == "batch-worker-1"
        assert worker._max_concurrent_debates == 10
        assert worker._max_concurrent_batches == 3
        assert worker.is_running is False

    def test_worker_get_stats(self):
        """Test get_stats method."""
        from aragora.queue.batch_worker import BatchExplainabilityWorker

        mock_queue = MagicMock()
        mock_generator = AsyncMock()

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="batch-worker-1",
            explain_generator=mock_generator,
        )

        worker._batches_processed = 5
        worker._debates_processed = 50
        worker._debates_failed = 3
        worker._start_time = time.time() - 1800

        stats = worker.get_stats()

        assert stats["worker_id"] == "batch-worker-1"
        assert stats["batches_processed"] == 5
        assert stats["debates_processed"] == 50
        assert stats["debates_failed"] == 3
        assert stats["uptime_seconds"] >= 1799

    def test_get_batch_progress(self):
        """Test get_batch_progress method."""
        from aragora.queue.batch_worker import BatchExplainabilityWorker, BatchJobProgress

        mock_queue = MagicMock()
        mock_generator = AsyncMock()

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="batch-worker-1",
            explain_generator=mock_generator,
        )

        # Initially no progress
        assert worker.get_batch_progress("batch-123") is None

        # Add progress
        worker._active_batches["batch-123"] = BatchJobProgress(
            job_id="batch-123",
            total=10,
            processed=5,
        )

        progress = worker.get_batch_progress("batch-123")
        assert progress is not None
        assert progress.processed == 5

    @pytest.mark.asyncio
    async def test_worker_start_stop(self):
        """Test batch worker start and stop."""
        from aragora.queue.batch_worker import BatchExplainabilityWorker

        mock_queue = AsyncMock()
        mock_queue.dequeue.return_value = None
        mock_generator = AsyncMock()

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="batch-worker-1",
            explain_generator=mock_generator,
        )

        await worker.start()
        assert worker.is_running is True

        await worker.stop(timeout=1.0)
        assert worker.is_running is False

    @pytest.mark.asyncio
    async def test_process_debate_success(self):
        """Test processing a single debate successfully."""
        from aragora.queue.batch_worker import BatchExplainabilityWorker, BatchJobProgress

        mock_queue = MagicMock()
        mock_generator = AsyncMock(return_value={"explanation": "Test explanation"})

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="batch-worker-1",
            explain_generator=mock_generator,
        )

        progress = BatchJobProgress(job_id="batch-1", total=1)
        semaphore = asyncio.Semaphore(1)

        await worker._process_debate("debate-123", {"option": "value"}, progress, semaphore)

        assert progress.processed == 1
        assert progress.succeeded == 1
        assert progress.failed == 0
        assert len(progress.results) == 1
        assert progress.results[0]["debate_id"] == "debate-123"
        mock_generator.assert_called_once_with("debate-123", {"option": "value"})

    @pytest.mark.asyncio
    async def test_process_debate_failure(self):
        """Test processing a debate that fails."""
        from aragora.queue.batch_worker import BatchExplainabilityWorker, BatchJobProgress

        mock_queue = MagicMock()
        mock_generator = AsyncMock(side_effect=RuntimeError("Explanation failed"))

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="batch-worker-1",
            explain_generator=mock_generator,
        )

        progress = BatchJobProgress(job_id="batch-1", total=1)
        semaphore = asyncio.Semaphore(1)

        await worker._process_debate("debate-456", {}, progress, semaphore)

        assert progress.processed == 1
        assert progress.succeeded == 0
        assert progress.failed == 1
        assert len(progress.errors) == 1
        assert progress.errors[0]["debate_id"] == "debate-456"


class TestCreateBatchJob:
    """Tests for create_batch_job function."""

    @pytest.mark.asyncio
    async def test_create_batch_job(self):
        """Test creating a batch job."""
        from aragora.queue.batch_worker import create_batch_job

        mock_queue = AsyncMock()

        job = await create_batch_job(
            queue=mock_queue,
            debate_ids=["d1", "d2", "d3"],
            options={"format": "detailed"},
            user_id="user_123",
            priority=10,
        )

        assert job is not None
        assert job.payload["debate_ids"] == ["d1", "d2", "d3"]
        assert job.payload["options"] == {"format": "detailed"}
        assert job.payload["user_id"] == "user_123"
        assert job.priority == 10
        assert job.metadata["debate_count"] == 3

        mock_queue.enqueue.assert_called_once()


# =============================================================================
# Package Import Tests
# =============================================================================


class TestQueuePackageImports:
    """Test that queue modules import correctly."""

    def test_base_imports(self):
        """Test base module imports."""
        from aragora.queue.base import Job, JobQueue, JobStatus

        assert Job is not None
        assert JobQueue is not None
        assert JobStatus is not None

    def test_retry_imports(self):
        """Test retry module imports."""
        from aragora.queue.retry import RetryPolicy, is_retryable_error

        assert RetryPolicy is not None
        assert is_retryable_error is not None

    def test_webhook_worker_imports(self):
        """Test webhook worker imports."""
        from aragora.queue.webhook_worker import (
            WebhookDeliveryWorker,
            DeliveryResult,
            EndpointHealth,
            enqueue_webhook_delivery,
        )

        assert WebhookDeliveryWorker is not None
        assert DeliveryResult is not None
        assert EndpointHealth is not None
        assert enqueue_webhook_delivery is not None

    def test_batch_worker_imports(self):
        """Test batch worker imports."""
        from aragora.queue.batch_worker import (
            BatchExplainabilityWorker,
            BatchJobProgress,
            create_batch_job,
        )

        assert BatchExplainabilityWorker is not None
        assert BatchJobProgress is not None
        assert create_batch_job is not None
