"""
Webhook Delivery Manager.

Provides reliable webhook delivery with:
- Delivery status tracking (pending, delivered, failed, dead-lettered)
- Retry queue with exponential backoff
- Dead-letter queue for consistently failing webhooks
- Delivery SLA metrics

Usage:
    from aragora.server.webhook_delivery import (
        WebhookDeliveryManager,
        get_delivery_manager,
        deliver_webhook,
    )

    manager = await get_delivery_manager()
    result = await manager.deliver(webhook_id, event_type, payload)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTERED = "dead_lettered"


@dataclass
class WebhookDelivery:
    """Represents a webhook delivery attempt."""

    delivery_id: str
    webhook_id: str
    event_type: str
    payload: Dict[str, Any]
    status: DeliveryStatus = DeliveryStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    max_attempts: int = 5
    next_retry_at: Optional[datetime] = None
    last_error: Optional[str] = None
    last_status_code: Optional[int] = None
    delivered_at: Optional[datetime] = None
    dead_lettered_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delivery_id": self.delivery_id,
            "webhook_id": self.webhook_id,
            "event_type": self.event_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "last_error": self.last_error,
            "last_status_code": self.last_status_code,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "dead_lettered_at": self.dead_lettered_at.isoformat() if self.dead_lettered_at else None,
        }


@dataclass
class DeliveryMetrics:
    """Metrics for webhook delivery."""

    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    dead_lettered: int = 0
    retries: int = 0
    total_latency_ms: float = 0.0
    endpoints_by_status: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Delivery success rate."""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100

    @property
    def avg_latency_ms(self) -> float:
        """Average delivery latency."""
        if self.successful_deliveries == 0:
            return 0.0
        return self.total_latency_ms / self.successful_deliveries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_deliveries": self.total_deliveries,
            "successful_deliveries": self.successful_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "dead_lettered": self.dead_lettered,
            "retries": self.retries,
            "success_rate": round(self.success_rate, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class WebhookDeliveryManager:
    """
    Manages webhook delivery with reliability guarantees.

    Features:
    - Async delivery with HTTP client
    - Exponential backoff retry
    - Dead-letter queue for failed deliveries
    - Delivery tracking and metrics
    - Circuit breaker per endpoint
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 300.0,
        timeout_seconds: float = 30.0,
        circuit_breaker_threshold: int = 5,
    ):
        """
        Initialize the delivery manager.

        Args:
            max_retries: Maximum retry attempts per delivery
            base_delay_seconds: Base delay for exponential backoff
            max_delay_seconds: Maximum delay between retries
            timeout_seconds: HTTP timeout for delivery
            circuit_breaker_threshold: Failures before circuit opens
        """
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._timeout = timeout_seconds
        self._circuit_threshold = circuit_breaker_threshold

        # In-memory stores (could be backed by Redis/SQLite for production)
        self._pending: Dict[str, WebhookDelivery] = {}
        self._retry_queue: Dict[str, WebhookDelivery] = {}
        self._dead_letter_queue: Dict[str, WebhookDelivery] = {}
        self._delivered: Dict[str, WebhookDelivery] = {}

        # Circuit breaker state per endpoint
        self._circuit_failures: Dict[str, int] = {}
        self._circuit_open_until: Dict[str, datetime] = {}

        # Metrics
        self._metrics = DeliveryMetrics()

        # Background retry processor
        self._retry_task: Optional[asyncio.Task] = None
        self._running = False

        # HTTP sender (can be mocked for testing)
        self._sender: Optional[Callable] = None

    async def start(self) -> None:
        """Start the background retry processor."""
        if self._running:
            return

        self._running = True
        self._retry_task = asyncio.create_task(self._process_retries())
        logger.info("Webhook delivery manager started")

    async def stop(self) -> None:
        """Stop the background retry processor."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        logger.info("Webhook delivery manager stopped")

    def set_sender(self, sender: Callable) -> None:
        """
        Set the HTTP sender function for testing.

        Args:
            sender: Async function(url, payload, headers) -> (status_code, response_body)
        """
        self._sender = sender

    async def deliver(
        self,
        webhook_id: str,
        event_type: str,
        payload: Dict[str, Any],
        url: str,
        secret: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WebhookDelivery:
        """
        Deliver a webhook with retry guarantees.

        Args:
            webhook_id: Webhook configuration ID
            event_type: Event type being delivered
            payload: Event payload
            url: Delivery URL
            secret: Signing secret for HMAC signature
            metadata: Additional metadata

        Returns:
            WebhookDelivery with status
        """
        delivery = WebhookDelivery(
            delivery_id=str(uuid.uuid4()),
            webhook_id=webhook_id,
            event_type=event_type,
            payload=payload,
            metadata=metadata or {},
        )

        self._pending[delivery.delivery_id] = delivery
        self._metrics.total_deliveries += 1

        # Check circuit breaker
        if self._is_circuit_open(url):
            delivery.status = DeliveryStatus.RETRYING
            delivery.last_error = "Circuit breaker open"
            self._schedule_retry(delivery, url, secret)
            return delivery

        # Attempt delivery
        success = await self._attempt_delivery(delivery, url, secret)

        if success:
            delivery.status = DeliveryStatus.DELIVERED
            delivery.delivered_at = datetime.utcnow()
            self._delivered[delivery.delivery_id] = delivery
            del self._pending[delivery.delivery_id]
            self._metrics.successful_deliveries += 1
            self._reset_circuit(url)
        else:
            self._record_circuit_failure(url)
            if delivery.attempts >= self._max_retries:
                self._move_to_dead_letter(delivery)
            else:
                self._schedule_retry(delivery, url, secret)

        return delivery

    async def _attempt_delivery(
        self,
        delivery: WebhookDelivery,
        url: str,
        secret: Optional[str],
    ) -> bool:
        """
        Attempt to deliver the webhook.

        Returns:
            True if delivery succeeded
        """
        delivery.status = DeliveryStatus.IN_PROGRESS
        delivery.attempts += 1
        delivery.updated_at = datetime.utcnow()

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Delivery-Id": delivery.delivery_id,
            "X-Webhook-Event-Type": delivery.event_type,
            "X-Webhook-Timestamp": str(int(time.time())),
        }

        # Add HMAC signature if secret provided
        if secret:
            payload_bytes = json.dumps(delivery.payload).encode()
            signature = hashlib.sha256(
                (headers["X-Webhook-Timestamp"] + "." + payload_bytes.decode()).encode()
                + secret.encode()
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        start_time = time.time()

        try:
            if self._sender:
                # Use mock sender for testing
                status_code, _ = await self._sender(url, delivery.payload, headers)
            else:
                # Use real HTTP client
                status_code = await self._http_send(url, delivery.payload, headers)

            latency_ms = (time.time() - start_time) * 1000
            delivery.last_status_code = status_code

            if 200 <= status_code < 300:
                self._metrics.total_latency_ms += latency_ms
                return True
            else:
                delivery.last_error = f"HTTP {status_code}"
                self._metrics.failed_deliveries += 1
                return False

        except asyncio.TimeoutError:
            delivery.last_error = "Timeout"
            self._metrics.failed_deliveries += 1
            return False
        except Exception as e:
            delivery.last_error = str(e)[:200]
            self._metrics.failed_deliveries += 1
            return False

    async def _http_send(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> int:
        """Send HTTP POST request."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    return response.status
        except ImportError:
            # Fallback to httpx if aiohttp not available
            try:
                import httpx

                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    return response.status_code
            except ImportError:
                logger.error("No HTTP client available (need aiohttp or httpx)")
                raise RuntimeError("No HTTP client available")

    def _schedule_retry(
        self,
        delivery: WebhookDelivery,
        url: str,
        secret: Optional[str],
    ) -> None:
        """Schedule delivery for retry with exponential backoff."""
        delivery.status = DeliveryStatus.RETRYING
        delivery.updated_at = datetime.utcnow()

        # Exponential backoff with jitter
        delay = min(
            self._base_delay * (2 ** (delivery.attempts - 1)),
            self._max_delay,
        )
        # Add jitter (±25%)
        import random
        delay *= 0.75 + random.random() * 0.5

        delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
        delivery.metadata["retry_url"] = url
        delivery.metadata["retry_secret"] = secret

        self._retry_queue[delivery.delivery_id] = delivery
        if delivery.delivery_id in self._pending:
            del self._pending[delivery.delivery_id]

        self._metrics.retries += 1
        logger.debug(
            f"Scheduled retry for {delivery.delivery_id} in {delay:.1f}s "
            f"(attempt {delivery.attempts}/{self._max_retries})"
        )

    def _move_to_dead_letter(self, delivery: WebhookDelivery) -> None:
        """Move delivery to dead-letter queue."""
        delivery.status = DeliveryStatus.DEAD_LETTERED
        delivery.dead_lettered_at = datetime.utcnow()
        delivery.updated_at = datetime.utcnow()

        self._dead_letter_queue[delivery.delivery_id] = delivery

        # Remove from other queues
        for queue in [self._pending, self._retry_queue]:
            if delivery.delivery_id in queue:
                del queue[delivery.delivery_id]

        self._metrics.dead_lettered += 1
        logger.warning(
            f"Moved delivery {delivery.delivery_id} to dead-letter queue "
            f"after {delivery.attempts} attempts: {delivery.last_error}"
        )

    async def _process_retries(self) -> None:
        """Background task to process retry queue."""
        while self._running:
            try:
                now = datetime.utcnow()

                # Find deliveries ready for retry
                ready = [
                    d for d in self._retry_queue.values()
                    if d.next_retry_at and d.next_retry_at <= now
                ]

                for delivery in ready:
                    url = delivery.metadata.get("retry_url", "")
                    secret = delivery.metadata.get("retry_secret")

                    if not url:
                        self._move_to_dead_letter(delivery)
                        continue

                    # Remove from retry queue before attempting
                    del self._retry_queue[delivery.delivery_id]

                    # Check circuit breaker
                    if self._is_circuit_open(url):
                        self._schedule_retry(delivery, url, secret)
                        continue

                    # Attempt delivery
                    success = await self._attempt_delivery(delivery, url, secret)

                    if success:
                        delivery.status = DeliveryStatus.DELIVERED
                        delivery.delivered_at = datetime.utcnow()
                        self._delivered[delivery.delivery_id] = delivery
                        self._metrics.successful_deliveries += 1
                        self._reset_circuit(url)
                    else:
                        self._record_circuit_failure(url)
                        if delivery.attempts >= self._max_retries:
                            self._move_to_dead_letter(delivery)
                        else:
                            self._schedule_retry(delivery, url, secret)

            except Exception as e:
                logger.error(f"Error processing retries: {e}")

            await asyncio.sleep(1.0)  # Check every second

    def _is_circuit_open(self, url: str) -> bool:
        """Check if circuit breaker is open for an endpoint."""
        if url in self._circuit_open_until:
            if datetime.utcnow() < self._circuit_open_until[url]:
                return True
            # Circuit timeout expired, reset
            del self._circuit_open_until[url]
            self._circuit_failures[url] = 0
        return False

    def _record_circuit_failure(self, url: str) -> None:
        """Record a failure for circuit breaker."""
        self._circuit_failures[url] = self._circuit_failures.get(url, 0) + 1

        if self._circuit_failures[url] >= self._circuit_threshold:
            # Open circuit for 30 seconds * failure count (max 5 minutes)
            timeout = min(30 * self._circuit_failures[url], 300)
            self._circuit_open_until[url] = datetime.utcnow() + timedelta(seconds=timeout)
            logger.warning(f"Circuit breaker opened for {url} for {timeout}s")

    def _reset_circuit(self, url: str) -> None:
        """Reset circuit breaker on success."""
        self._circuit_failures[url] = 0
        if url in self._circuit_open_until:
            del self._circuit_open_until[url]

    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery by ID."""
        for queue in [self._pending, self._retry_queue, self._dead_letter_queue, self._delivered]:
            if delivery_id in queue:
                return queue[delivery_id]
        return None

    async def get_pending_count(self) -> int:
        """Get count of pending deliveries."""
        return len(self._pending) + len(self._retry_queue)

    async def get_dead_letter_queue(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get deliveries in dead-letter queue."""
        return list(self._dead_letter_queue.values())[:limit]

    async def retry_dead_letter(self, delivery_id: str) -> bool:
        """
        Retry a dead-lettered delivery.

        Args:
            delivery_id: The delivery to retry

        Returns:
            True if moved back to retry queue
        """
        if delivery_id not in self._dead_letter_queue:
            return False

        delivery = self._dead_letter_queue[delivery_id]
        delivery.status = DeliveryStatus.RETRYING
        delivery.attempts = 0  # Reset attempts
        delivery.dead_lettered_at = None
        delivery.next_retry_at = datetime.utcnow()

        self._retry_queue[delivery_id] = delivery
        del self._dead_letter_queue[delivery_id]

        self._metrics.dead_lettered -= 1
        logger.info(f"Retrying dead-lettered delivery {delivery_id}")
        return True

    async def get_metrics(self) -> Dict[str, Any]:
        """Get delivery metrics."""
        return {
            **self._metrics.to_dict(),
            "pending_count": len(self._pending),
            "retry_queue_size": len(self._retry_queue),
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "delivered_count": len(self._delivered),
            "open_circuits": len(self._circuit_open_until),
        }


# Global manager instance
_manager: Optional[WebhookDeliveryManager] = None
_manager_lock = asyncio.Lock()


async def get_delivery_manager() -> WebhookDeliveryManager:
    """Get or create the global delivery manager."""
    global _manager
    async with _manager_lock:
        if _manager is None:
            _manager = WebhookDeliveryManager()
            await _manager.start()
        return _manager


async def deliver_webhook(
    webhook_id: str,
    event_type: str,
    payload: Dict[str, Any],
    url: str,
    secret: Optional[str] = None,
) -> WebhookDelivery:
    """
    Deliver a webhook with retry guarantees.

    Convenience function that uses the global manager.
    """
    manager = await get_delivery_manager()
    return await manager.deliver(webhook_id, event_type, payload, url, secret)


async def get_webhook_delivery_metrics() -> Dict[str, Any]:
    """Get webhook delivery metrics."""
    manager = await get_delivery_manager()
    return await manager.get_metrics()


def reset_delivery_manager() -> None:
    """Reset the global manager (for testing)."""
    global _manager
    _manager = None


__all__ = [
    "DeliveryStatus",
    "WebhookDelivery",
    "DeliveryMetrics",
    "WebhookDeliveryManager",
    "get_delivery_manager",
    "deliver_webhook",
    "get_webhook_delivery_metrics",
    "reset_delivery_manager",
]
