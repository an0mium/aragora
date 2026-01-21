"""
Webhook Delivery Worker.

Background worker for reliable webhook delivery:
- Queued delivery for high throughput
- Retry with exponential backoff
- Circuit breaker per endpoint
- Delivery tracking and metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from aragora.queue.base import Job, JobQueue, JobStatus
from aragora.queue.config import get_queue_config
from aragora.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class DeliveryResult:
    """Result of a webhook delivery attempt."""

    webhook_id: str
    url: str
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0
    attempt: int = 1
    delivered_at: float = field(default_factory=time.time)


@dataclass
class EndpointHealth:
    """Health tracking for a webhook endpoint."""

    url: str
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    avg_response_time_ms: float = 0.0
    circuit_state: str = "closed"

    @property
    def success_rate(self) -> float:
        """Get success rate as a percentage."""
        if self.total_deliveries == 0:
            return 100.0
        return (self.successful_deliveries / self.total_deliveries) * 100


class WebhookDeliveryWorker:
    """
    Worker for reliable webhook delivery.

    Features:
    - Concurrent delivery to multiple endpoints
    - Per-endpoint circuit breakers
    - Exponential backoff retry
    - Delivery tracking and metrics
    - HMAC signature generation
    """

    QUEUE_NAME = "webhook_delivery"

    # Retry configuration
    MAX_RETRIES = 5
    INITIAL_BACKOFF_SECONDS = 1.0
    MAX_BACKOFF_SECONDS = 60.0
    BACKOFF_MULTIPLIER = 2.0

    # Circuit breaker configuration
    CIRCUIT_FAILURE_THRESHOLD = 5
    CIRCUIT_RESET_TIMEOUT = 60.0

    def __init__(
        self,
        queue: JobQueue,
        worker_id: str,
        max_concurrent: int = 10,
        request_timeout: float = 10.0,
    ) -> None:
        """
        Initialize the webhook delivery worker.

        Args:
            queue: The job queue
            worker_id: Unique worker identifier
            max_concurrent: Maximum concurrent deliveries
            request_timeout: HTTP request timeout in seconds
        """
        self._queue = queue
        self._worker_id = worker_id
        self._max_concurrent = max_concurrent
        self._request_timeout = request_timeout
        self._config = get_queue_config()

        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown_event = asyncio.Event()

        # Per-endpoint circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Endpoint health tracking
        self._endpoint_health: Dict[str, EndpointHealth] = {}

        # Metrics
        self._deliveries_total = 0
        self._deliveries_succeeded = 0
        self._deliveries_failed = 0
        self._start_time: Optional[float] = None

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "worker_id": self._worker_id,
            "is_running": self._running,
            "uptime_seconds": uptime,
            "deliveries_total": self._deliveries_total,
            "deliveries_succeeded": self._deliveries_succeeded,
            "deliveries_failed": self._deliveries_failed,
            "active_deliveries": self._max_concurrent - self._semaphore._value,
            "endpoints_tracked": len(self._endpoint_health),
            "queue_name": self.QUEUE_NAME,
        }

    def get_endpoint_health(self, url: str) -> Optional[EndpointHealth]:
        """Get health info for a specific endpoint."""
        return self._endpoint_health.get(url)

    def get_all_endpoint_health(self) -> List[EndpointHealth]:
        """Get health info for all tracked endpoints."""
        return list(self._endpoint_health.values())

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            logger.warning(f"Worker {self._worker_id} already running")
            return

        logger.info(f"Starting webhook delivery worker {self._worker_id}")
        self._running = True
        self._start_time = time.time()
        self._shutdown_event.clear()

        # Start the main processing loop
        asyncio.create_task(self._process_loop())

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        logger.info(f"Stopping webhook worker {self._worker_id}")
        self._running = False
        self._shutdown_event.set()

        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} active deliveries")
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            if pending:
                logger.warning(f"Cancelling {len(pending)} pending deliveries")
                for task in pending:
                    task.cancel()

        logger.info(f"Webhook worker {self._worker_id} stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        poll_interval = self._config.poll_interval_seconds

        while self._running:
            try:
                # Check for available capacity
                if self._semaphore.locked():
                    await asyncio.sleep(poll_interval)
                    continue

                # Try to get a job
                job = await self._queue.dequeue(
                    queue_name=self.QUEUE_NAME,
                    worker_id=self._worker_id,
                )

                if job is None:
                    await asyncio.sleep(poll_interval)
                    continue

                # Process the delivery
                await self._semaphore.acquire()
                task = asyncio.create_task(self._process_delivery(job))
                self._tasks.add(task)
                task.add_done_callback(self._on_task_done)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in webhook worker loop: {e}", exc_info=True)
                await asyncio.sleep(poll_interval)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback when a task completes."""
        self._tasks.discard(task)
        self._semaphore.release()

        if task.exception():
            logger.error(f"Delivery task failed: {task.exception()}")

    async def _process_delivery(self, job: Job) -> None:
        """Process a single webhook delivery job."""
        payload = job.payload
        webhook_id = payload.get("webhook_id", "unknown")
        url = payload.get("url", "")
        secret = payload.get("secret", "")
        event_data = payload.get("event_data", {})
        attempt = job.attempts + 1

        logger.debug(f"Processing webhook delivery {job.id} to {url} (attempt {attempt})")

        # Check circuit breaker
        circuit = self._get_circuit_breaker(url)
        if circuit.state == "open":
            logger.warning(f"Circuit breaker open for {url}, skipping delivery")
            await self._schedule_retry(job, "Circuit breaker open")
            return

        # Attempt delivery
        result = await self._deliver_webhook(
            webhook_id=webhook_id,
            url=url,
            secret=secret,
            event_data=event_data,
            attempt=attempt,
        )

        # Update metrics and health
        self._update_metrics(result)
        self._update_endpoint_health(result)

        if result.success:
            # Mark job as complete
            await self._queue.complete(
                job_id=job.id,
                result=result.__dict__,
                status=JobStatus.COMPLETED,
            )
            circuit.record_success()
        else:
            # Handle failure
            circuit.record_failure()

            if attempt < self.MAX_RETRIES:
                await self._schedule_retry(job, result.error)
            else:
                # Max retries exceeded
                await self._queue.fail(
                    job_id=job.id,
                    error=f"Max retries exceeded: {result.error}",
                )

    async def _deliver_webhook(
        self,
        webhook_id: str,
        url: str,
        secret: str,
        event_data: Dict[str, Any],
        attempt: int,
    ) -> DeliveryResult:
        """Perform the actual webhook delivery."""
        import aiohttp

        start_time = time.time()

        # Prepare payload
        payload_str = json.dumps(event_data, sort_keys=True)
        payload_bytes = payload_str.encode("utf-8")

        # Generate signature
        signature = self._generate_signature(payload_bytes, secret) if secret else ""

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Aragora-Webhooks/1.0",
            "X-Webhook-ID": webhook_id,
            "X-Delivery-Attempt": str(attempt),
        }
        if signature:
            headers["X-Signature-SHA256"] = signature

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=payload_bytes,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._request_timeout),
                ) as response:
                    response_time = (time.time() - start_time) * 1000

                    success = 200 <= response.status < 300
                    return DeliveryResult(
                        webhook_id=webhook_id,
                        url=url,
                        success=success,
                        status_code=response.status,
                        response_time_ms=response_time,
                        attempt=attempt,
                        error=None if success else f"HTTP {response.status}",
                    )

        except asyncio.TimeoutError:
            return DeliveryResult(
                webhook_id=webhook_id,
                url=url,
                success=False,
                error="Request timeout",
                response_time_ms=(time.time() - start_time) * 1000,
                attempt=attempt,
            )
        except aiohttp.ClientError as e:
            return DeliveryResult(
                webhook_id=webhook_id,
                url=url,
                success=False,
                error=str(e),
                response_time_ms=(time.time() - start_time) * 1000,
                attempt=attempt,
            )
        except Exception as e:
            logger.error(f"Unexpected error delivering to {url}: {e}", exc_info=True)
            return DeliveryResult(
                webhook_id=webhook_id,
                url=url,
                success=False,
                error=f"Unexpected error: {e}",
                response_time_ms=(time.time() - start_time) * 1000,
                attempt=attempt,
            )

    def _generate_signature(self, payload: bytes, secret: str) -> str:
        """Generate HMAC-SHA256 signature for payload."""
        signature = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """Get or create circuit breaker for an endpoint."""
        if url not in self._circuit_breakers:
            self._circuit_breakers[url] = CircuitBreaker(
                failure_threshold=self.CIRCUIT_FAILURE_THRESHOLD,
                cooldown_seconds=self.CIRCUIT_RESET_TIMEOUT,
            )
        return self._circuit_breakers[url]

    async def _schedule_retry(self, job: Job, error: Optional[str]) -> None:
        """Schedule a job for retry with exponential backoff."""
        attempt = job.attempts + 1
        backoff = min(
            self.INITIAL_BACKOFF_SECONDS * (self.BACKOFF_MULTIPLIER ** attempt),
            self.MAX_BACKOFF_SECONDS,
        )

        logger.info(f"Scheduling retry for job {job.id} in {backoff}s (attempt {attempt})")

        # Update job for retry
        job.attempts = attempt
        job.error = error
        job.status = JobStatus.RETRYING

        # Re-enqueue with delay
        await self._queue.enqueue(
            job=job,
            queue_name=self.QUEUE_NAME,
            delay_seconds=backoff,
        )

    def _update_metrics(self, result: DeliveryResult) -> None:
        """Update worker metrics."""
        self._deliveries_total += 1
        if result.success:
            self._deliveries_succeeded += 1
        else:
            self._deliveries_failed += 1

    def _update_endpoint_health(self, result: DeliveryResult) -> None:
        """Update endpoint health tracking."""
        url = result.url

        if url not in self._endpoint_health:
            self._endpoint_health[url] = EndpointHealth(url=url)

        health = self._endpoint_health[url]
        health.total_deliveries += 1

        if result.success:
            health.successful_deliveries += 1
            health.last_success_at = time.time()
        else:
            health.failed_deliveries += 1
            health.last_failure_at = time.time()

        # Update average response time
        n = health.total_deliveries
        health.avg_response_time_ms = (
            (health.avg_response_time_ms * (n - 1) + result.response_time_ms) / n
        )

        # Update circuit state
        circuit = self._circuit_breakers.get(url)
        if circuit:
            health.circuit_state = circuit.state


async def enqueue_webhook_delivery(
    queue: JobQueue,
    webhook_id: str,
    url: str,
    secret: str,
    event_type: str,
    event_data: Dict[str, Any],
    priority: int = 0,
) -> Job:
    """
    Enqueue a webhook delivery job.

    Args:
        queue: The job queue
        webhook_id: ID of the webhook configuration
        url: Destination URL
        secret: HMAC secret for signing
        event_type: Type of event being delivered
        event_data: Event payload
        priority: Job priority

    Returns:
        The created job
    """
    job = Job(
        payload={
            "webhook_id": webhook_id,
            "url": url,
            "secret": secret,
            "event_type": event_type,
            "event_data": event_data,
        },
        priority=priority,
        metadata={
            "type": "webhook_delivery",
            "event_type": event_type,
        },
    )

    await queue.enqueue(
        job=job,
        queue_name=WebhookDeliveryWorker.QUEUE_NAME,
    )

    return job


__all__ = [
    "WebhookDeliveryWorker",
    "DeliveryResult",
    "EndpointHealth",
    "enqueue_webhook_delivery",
]
