"""
Load tests for new features: Marketplace, Batch Explainability, and Webhooks.

These tests measure throughput and latency under load for the new feature workers.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.queue.base import Job, JobQueue, JobStatus
from aragora.queue.batch_worker import (
    BatchExplainabilityWorker,
    BatchJobProgress,
    create_batch_job,
)
from aragora.queue.webhook_worker import (
    DeliveryResult,
    WebhookDeliveryWorker,
    enqueue_webhook_delivery,
)


class MockJobQueue:
    """Mock job queue for testing."""

    def __init__(self) -> None:
        self.queues: Dict[str, List[Job]] = {}
        self.completed: List[str] = []
        self.failed: List[str] = []
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        job: Job,
        queue_name: str,
        delay_seconds: float = 0,
    ) -> None:
        """Add a job to the queue."""
        async with self._lock:
            if queue_name not in self.queues:
                self.queues[queue_name] = []
            self.queues[queue_name].append(job)

    async def dequeue(
        self,
        queue_name: str,
        worker_id: str,
    ) -> Job | None:
        """Get a job from the queue."""
        async with self._lock:
            if queue_name in self.queues and self.queues[queue_name]:
                return self.queues[queue_name].pop(0)
            return None

    async def complete(
        self,
        job_id: str,
        result: Any = None,
        status: JobStatus = JobStatus.COMPLETED,
    ) -> None:
        """Mark a job as complete."""
        self.completed.append(job_id)

    async def fail(
        self,
        job_id: str,
        error: str,
    ) -> None:
        """Mark a job as failed."""
        self.failed.append(job_id)


@pytest.fixture
def mock_queue() -> MockJobQueue:
    """Create a mock job queue."""
    return MockJobQueue()


class TestBatchExplainabilityLoad:
    """Load tests for batch explainability worker."""

    @pytest.mark.benchmark(group="batch_explainability")
    @pytest.mark.asyncio
    async def test_batch_processing_throughput(
        self,
        mock_queue: MockJobQueue,
        benchmark,
    ) -> None:
        """Test throughput of batch processing with many debates."""
        debate_count = 100
        debate_ids = [f"debate-{i}" for i in range(debate_count)]

        async def mock_explain(debate_id: str, options: Dict) -> Dict:
            """Mock explanation generator with minimal latency."""
            await asyncio.sleep(0.001)  # 1ms per debate
            return {"debate_id": debate_id, "explanation": "Test explanation"}

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="test-worker",
            explain_generator=mock_explain,
            max_concurrent_debates=20,
            max_concurrent_batches=2,
        )

        # Create batch job
        job = await create_batch_job(
            queue=mock_queue,
            debate_ids=debate_ids,
            options={"format": "summary"},
        )

        # Wrap async processing for benchmark
        elapsed_holder = [0.0]

        async def process_batch():
            start_time = time.perf_counter()
            await worker._process_batch(job)
            elapsed_holder[0] = time.perf_counter() - start_time

        def run_sync():
            asyncio.get_event_loop().run_until_complete(process_batch())

        benchmark.pedantic(run_sync, rounds=1, iterations=1)
        elapsed = elapsed_holder[0]

        # Calculate throughput
        throughput = debate_count / elapsed
        assert throughput > 50, f"Expected >50 debates/sec, got {throughput:.2f}"

        # Record metrics
        benchmark.extra_info["debates_processed"] = debate_count
        benchmark.extra_info["elapsed_seconds"] = elapsed
        benchmark.extra_info["throughput_per_second"] = throughput

    @pytest.mark.benchmark(group="batch_explainability")
    @pytest.mark.asyncio
    async def test_batch_concurrent_processing(
        self,
        mock_queue: MockJobQueue,
        benchmark,
    ) -> None:
        """Test concurrent batch processing."""
        batch_count = 5
        debates_per_batch = 20

        async def mock_explain(debate_id: str, options: Dict) -> Dict:
            await asyncio.sleep(0.005)
            return {"debate_id": debate_id, "explanation": "Test"}

        worker = BatchExplainabilityWorker(
            queue=mock_queue,
            worker_id="test-worker",
            explain_generator=mock_explain,
            max_concurrent_debates=10,
            max_concurrent_batches=5,
        )

        # Create multiple batch jobs
        jobs = []
        for i in range(batch_count):
            debate_ids = [f"batch-{i}-debate-{j}" for j in range(debates_per_batch)]
            job = await create_batch_job(
                queue=mock_queue,
                debate_ids=debate_ids,
            )
            jobs.append(job)

        # Wrap async processing for benchmark
        elapsed_holder = [0.0]

        async def process_all():
            start_time = time.perf_counter()
            await asyncio.gather(*[worker._process_batch(job) for job in jobs])
            elapsed_holder[0] = time.perf_counter() - start_time

        def run_sync():
            asyncio.get_event_loop().run_until_complete(process_all())

        benchmark.pedantic(run_sync, rounds=1, iterations=1)
        elapsed = elapsed_holder[0]

        total_debates = batch_count * debates_per_batch
        throughput = total_debates / elapsed

        benchmark.extra_info["batches"] = batch_count
        benchmark.extra_info["total_debates"] = total_debates
        benchmark.extra_info["elapsed_seconds"] = elapsed
        benchmark.extra_info["throughput_per_second"] = throughput


class TestWebhookDeliveryLoad:
    """Load tests for webhook delivery worker."""

    @pytest.mark.benchmark(group="webhook_delivery")
    @pytest.mark.asyncio
    async def test_webhook_delivery_throughput(
        self,
        mock_queue: MockJobQueue,
        benchmark,
    ) -> None:
        """Test throughput of webhook delivery."""
        delivery_count = 100
        delivered = []

        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="test-worker",
            max_concurrent=50,
            request_timeout=1.0,
        )

        # Mock the HTTP delivery
        async def mock_deliver(*args, **kwargs) -> DeliveryResult:
            await asyncio.sleep(0.001)  # 1ms latency
            delivered.append(1)
            return DeliveryResult(
                webhook_id="test",
                url="https://example.com/webhook",
                success=True,
                status_code=200,
                response_time_ms=1.0,
            )

        worker._deliver_webhook = mock_deliver

        # Create delivery jobs
        jobs = []
        for i in range(delivery_count):
            job = Job(
                payload={
                    "webhook_id": f"webhook-{i}",
                    "url": "https://example.com/webhook",
                    "secret": "test-secret",
                    "event_data": {"test": True},
                },
            )
            jobs.append(job)

        # Wrap async processing for benchmark
        elapsed_holder = [0.0]

        async def process_all():
            start_time = time.perf_counter()
            await asyncio.gather(*[worker._process_delivery(job) for job in jobs])
            elapsed_holder[0] = time.perf_counter() - start_time

        def run_sync():
            asyncio.get_event_loop().run_until_complete(process_all())

        benchmark.pedantic(run_sync, rounds=1, iterations=1)
        elapsed = elapsed_holder[0]

        throughput = len(delivered) / elapsed
        assert throughput > 100, f"Expected >100 deliveries/sec, got {throughput:.2f}"

        benchmark.extra_info["deliveries"] = len(delivered)
        benchmark.extra_info["elapsed_seconds"] = elapsed
        benchmark.extra_info["throughput_per_second"] = throughput

    @pytest.mark.benchmark(group="webhook_delivery")
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(
        self,
        mock_queue: MockJobQueue,
        benchmark,
    ) -> None:
        """Test circuit breaker behavior under failures."""
        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="test-worker",
            max_concurrent=10,
        )

        # Get circuit breaker for an endpoint
        circuit = worker._get_circuit_breaker("https://failing.example.com")
        failures_until_open = 0

        def record_failures():
            nonlocal failures_until_open
            # Reset for benchmark
            circuit._failure_count = 0
            circuit._state = "closed"
            failures_until_open = 0
            while circuit.state != "open":
                circuit.record_failure()
                failures_until_open += 1
                if failures_until_open > 100:
                    break

        # Use benchmark to measure the operation
        benchmark(record_failures)

        assert failures_until_open == worker.CIRCUIT_FAILURE_THRESHOLD
        assert circuit.state == "open"

        benchmark.extra_info["failures_until_open"] = failures_until_open

    @pytest.mark.benchmark(group="webhook_delivery")
    @pytest.mark.asyncio
    async def test_retry_scheduling_latency(
        self,
        mock_queue: MockJobQueue,
        benchmark,
    ) -> None:
        """Test latency of retry scheduling."""
        import asyncio

        worker = WebhookDeliveryWorker(
            queue=mock_queue,
            worker_id="test-worker",
        )

        retry_count = 50
        jobs = [
            Job(
                payload={
                    "webhook_id": f"webhook-{i}",
                    "url": "https://example.com/webhook",
                    "secret": "",
                    "event_data": {},
                },
            )
            for i in range(retry_count)
        ]

        async def schedule_retries():
            for job in jobs:
                await worker._schedule_retry(job, "Test error")

        # Run async function in benchmark using pedantic with setup
        def run_sync():
            asyncio.get_event_loop().run_until_complete(schedule_retries())

        result = benchmark.pedantic(run_sync, rounds=1, iterations=1)

        benchmark.extra_info["retries_scheduled"] = retry_count


class TestMarketplaceLoad:
    """Load tests for marketplace operations."""

    @pytest.mark.benchmark(group="marketplace")
    def test_template_serialization_throughput(self, benchmark) -> None:
        """Test throughput of template serialization."""
        import json

        template = {
            "template_id": "tmpl-123",
            "name": "Test Template",
            "description": "A test workflow template for benchmarking",
            "category": "workflow",
            "visibility": "public",
            "author_id": "user-456",
            "template_data": {
                "nodes": [{"id": f"node-{i}", "type": "task"} for i in range(20)],
                "edges": [{"from": f"node-{i}", "to": f"node-{i+1}"} for i in range(19)],
            },
            "tags": ["benchmark", "test", "workflow"],
        }

        def serialize():
            return json.dumps(template)

        result = benchmark(serialize)
        assert len(result) > 100

    @pytest.mark.benchmark(group="marketplace")
    def test_rating_calculation_throughput(self, benchmark) -> None:
        """Test throughput of rating calculations."""
        import statistics

        ratings = [4, 5, 3, 4, 5, 4, 3, 5, 4, 4] * 100  # 1000 ratings

        def calculate_stats():
            return {
                "avg": statistics.mean(ratings),
                "median": statistics.median(ratings),
                "stdev": statistics.stdev(ratings),
                "count": len(ratings),
            }

        result = benchmark(calculate_stats)
        assert result["avg"] > 3.0
        assert result["count"] == 1000


class TestMetricsLoad:
    """Load tests for metrics recording."""

    @pytest.mark.benchmark(group="metrics")
    def test_metrics_recording_throughput(self, benchmark) -> None:
        """Test throughput of metrics recording."""
        from aragora.observability.metrics import (
            _init_metrics,
            record_marketplace_download,
        )

        _init_metrics()

        def record_metrics():
            for i in range(100):
                record_marketplace_download(f"tmpl-{i}", "workflow")

        benchmark(record_metrics)

    @pytest.mark.benchmark(group="metrics")
    def test_histogram_observation_throughput(self, benchmark) -> None:
        """Test throughput of histogram observations."""
        from aragora.observability.metrics import (
            _init_metrics,
            record_webhook_delivery,
        )

        _init_metrics()

        def record_latencies():
            for i in range(100):
                record_webhook_delivery(
                    f"https://endpoint-{i % 10}.example.com",
                    True,
                    0.05 + (i % 10) * 0.01,
                )

        benchmark(record_latencies)
