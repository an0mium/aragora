"""
Persistent Webhook Retry Queue with exponential backoff.

Provides reliable webhook delivery with:
- Exponential backoff retry (1s, 2s, 4s, 8s, 16s... capped at 5 minutes)
- Dead-letter queue for permanently failed deliveries
- Multiple storage backends (in-memory, Redis)
- Concurrency control with semaphore
- Integration with existing webhook infrastructure

Usage:
    from aragora.webhooks.retry_queue import WebhookRetryQueue, WebhookDelivery

    # Create queue with default in-memory store
    queue = WebhookRetryQueue()
    await queue.start()

    # Enqueue a delivery
    delivery = WebhookDelivery(
        id="delivery-123",
        url="https://example.com/webhook",
        payload={"event": "debate_end", "data": {...}},
    )
    await queue.enqueue(delivery)

    # Stop the queue
    await queue.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default maximum retry attempts
DEFAULT_MAX_ATTEMPTS = int(os.environ.get("ARAGORA_WEBHOOK_RETRY_MAX_ATTEMPTS", "5"))

# Default maximum concurrent deliveries
DEFAULT_MAX_CONCURRENT = int(os.environ.get("ARAGORA_WEBHOOK_RETRY_CONCURRENT", "10"))

# Default request timeout in seconds
DEFAULT_TIMEOUT = float(os.environ.get("ARAGORA_WEBHOOK_RETRY_TIMEOUT", "30.0"))

# Maximum backoff delay in seconds (5 minutes)
MAX_BACKOFF_SECONDS = 300

# Processing loop interval in seconds
PROCESS_INTERVAL = float(os.environ.get("ARAGORA_WEBHOOK_RETRY_INTERVAL", "1.0"))


# =============================================================================
# Delivery Status
# =============================================================================


class DeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""

    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


# =============================================================================
# Webhook Delivery
# =============================================================================


@dataclass
class WebhookDelivery:
    """
    Represents a webhook delivery request with retry state.

    Tracks the delivery lifecycle from initial enqueue through
    successful delivery or dead-letter status.
    """

    id: str
    url: str
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    next_retry_at: Optional[datetime] = None
    last_error: Optional[str] = None
    last_status_code: Optional[int] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Webhook configuration reference
    webhook_id: Optional[str] = None
    webhook_secret: Optional[str] = None

    def calculate_next_retry(self) -> datetime:
        """
        Calculate next retry time using exponential backoff.

        Backoff formula: min(2^attempts, MAX_BACKOFF_SECONDS)
        Delays: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 300s (capped)
        """
        delay = min(2**self.attempts, MAX_BACKOFF_SECONDS)
        return datetime.now(timezone.utc) + timedelta(seconds=delay)

    def is_ready_for_retry(self) -> bool:
        """Check if this delivery is ready for a retry attempt."""
        if self.status != DeliveryStatus.PENDING:
            return False
        if self.next_retry_at is None:
            return True
        return datetime.now(timezone.utc) >= self.next_retry_at

    def should_dead_letter(self) -> bool:
        """Check if this delivery should be moved to dead-letter queue."""
        return self.attempts >= self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delivery to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "payload": self.payload,
            "headers": self.headers,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "last_error": self.last_error,
            "last_status_code": self.last_status_code,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "webhook_id": self.webhook_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookDelivery":
        """Deserialize delivery from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        next_retry_at = data.get("next_retry_at")
        if isinstance(next_retry_at, str):
            next_retry_at = datetime.fromisoformat(next_retry_at)

        status = data.get("status", "pending")
        if isinstance(status, str):
            status = DeliveryStatus(status)

        return cls(
            id=data["id"],
            url=data["url"],
            payload=data.get("payload", {}),
            headers=data.get("headers", {}),
            created_at=created_at,
            status=status,
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", DEFAULT_MAX_ATTEMPTS),
            next_retry_at=next_retry_at,
            last_error=data.get("last_error"),
            last_status_code=data.get("last_status_code"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
            webhook_id=data.get("webhook_id"),
            webhook_secret=data.get("webhook_secret"),
        )

    def to_json(self) -> str:
        """Serialize delivery to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "WebhookDelivery":
        """Deserialize delivery from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Storage Backends
# =============================================================================


class WebhookDeliveryStore(ABC):
    """Abstract base class for webhook delivery storage backends."""

    @abstractmethod
    async def save(self, delivery: WebhookDelivery) -> None:
        """Save or update a delivery."""
        pass

    @abstractmethod
    async def get(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a delivery by ID."""
        pass

    @abstractmethod
    async def delete(self, delivery_id: str) -> bool:
        """Delete a delivery by ID."""
        pass

    @abstractmethod
    async def get_ready_for_retry(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get deliveries ready for retry (PENDING status with next_retry_at <= now)."""
        pass

    @abstractmethod
    async def get_by_status(
        self, status: DeliveryStatus, limit: int = 100
    ) -> List[WebhookDelivery]:
        """Get deliveries by status."""
        pass

    @abstractmethod
    async def get_dead_letters(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get dead-letter deliveries."""
        pass

    @abstractmethod
    async def count_by_status(self) -> Dict[DeliveryStatus, int]:
        """Count deliveries by status."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all deliveries. Returns count of deleted items."""
        pass

    async def close(self) -> None:
        """Close the store (optional to implement)."""
        pass


class InMemoryDeliveryStore(WebhookDeliveryStore):
    """
    In-memory webhook delivery store.

    Fast and simple, but not persistent. Suitable for development,
    testing, and single-instance deployments.
    """

    def __init__(self) -> None:
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._lock = asyncio.Lock()

    async def save(self, delivery: WebhookDelivery) -> None:
        """Save or update a delivery."""
        async with self._lock:
            self._deliveries[delivery.id] = delivery

    async def get(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a delivery by ID."""
        async with self._lock:
            return self._deliveries.get(delivery_id)

    async def delete(self, delivery_id: str) -> bool:
        """Delete a delivery by ID."""
        async with self._lock:
            if delivery_id in self._deliveries:
                del self._deliveries[delivery_id]
                return True
            return False

    async def get_ready_for_retry(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get deliveries ready for retry."""
        now = datetime.now(timezone.utc)
        async with self._lock:
            ready = []
            for delivery in self._deliveries.values():
                if delivery.status == DeliveryStatus.PENDING:
                    if delivery.next_retry_at is None or delivery.next_retry_at <= now:
                        ready.append(delivery)
                        if len(ready) >= limit:
                            break
            return ready

    async def get_by_status(
        self, status: DeliveryStatus, limit: int = 100
    ) -> List[WebhookDelivery]:
        """Get deliveries by status."""
        async with self._lock:
            result = []
            for delivery in self._deliveries.values():
                if delivery.status == status:
                    result.append(delivery)
                    if len(result) >= limit:
                        break
            return result

    async def get_dead_letters(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get dead-letter deliveries."""
        return await self.get_by_status(DeliveryStatus.DEAD_LETTER, limit)

    async def count_by_status(self) -> Dict[DeliveryStatus, int]:
        """Count deliveries by status."""
        async with self._lock:
            counts: Dict[DeliveryStatus, int] = {status: 0 for status in DeliveryStatus}
            for delivery in self._deliveries.values():
                counts[delivery.status] += 1
            return counts

    async def clear(self) -> int:
        """Clear all deliveries."""
        async with self._lock:
            count = len(self._deliveries)
            self._deliveries.clear()
            return count


class RedisDeliveryStore(WebhookDeliveryStore):
    """
    Redis-backed webhook delivery store.

    Persistent and distributed, suitable for multi-instance
    production deployments.
    """

    # Redis key prefixes
    PREFIX = "aragora:webhook_delivery"
    PENDING_KEY = f"{PREFIX}:pending"
    DEAD_LETTER_KEY = f"{PREFIX}:dead_letter"

    # TTL for delivery records (7 days)
    DELIVERY_TTL = 7 * 24 * 60 * 60

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis_url = redis_url or os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[Any] = None
        self._connected = False

    async def _get_redis(self) -> Any:
        """Get Redis connection (lazy initialization)."""
        if self._redis is not None:
            return self._redis

        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("RedisDeliveryStore connected")
            return self._redis
        except ImportError:
            raise ImportError(
                "redis package required for RedisDeliveryStore. Install with: pip install redis"
            )
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _delivery_key(self, delivery_id: str) -> str:
        """Get Redis key for a delivery."""
        return f"{self.PREFIX}:{delivery_id}"

    async def save(self, delivery: WebhookDelivery) -> None:
        """Save or update a delivery."""
        redis = await self._get_redis()

        # Save delivery data
        key = self._delivery_key(delivery.id)
        await redis.setex(key, self.DELIVERY_TTL, delivery.to_json())

        # Update index based on status
        if delivery.status == DeliveryStatus.PENDING:
            # Add to pending sorted set with next_retry_at as score
            score = delivery.next_retry_at.timestamp() if delivery.next_retry_at else time.time()
            await redis.zadd(self.PENDING_KEY, {delivery.id: score})
            await redis.zrem(self.DEAD_LETTER_KEY, delivery.id)
        elif delivery.status == DeliveryStatus.DEAD_LETTER:
            # Add to dead letter set
            await redis.zadd(self.DEAD_LETTER_KEY, {delivery.id: delivery.created_at.timestamp()})
            await redis.zrem(self.PENDING_KEY, delivery.id)
        else:
            # Remove from both indices (DELIVERED, FAILED, IN_FLIGHT)
            await redis.zrem(self.PENDING_KEY, delivery.id)
            await redis.zrem(self.DEAD_LETTER_KEY, delivery.id)

    async def get(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a delivery by ID."""
        redis = await self._get_redis()
        data = await redis.get(self._delivery_key(delivery_id))
        if data:
            return WebhookDelivery.from_json(data)
        return None

    async def delete(self, delivery_id: str) -> bool:
        """Delete a delivery by ID."""
        redis = await self._get_redis()
        key = self._delivery_key(delivery_id)

        # Remove from all indices
        await redis.zrem(self.PENDING_KEY, delivery_id)
        await redis.zrem(self.DEAD_LETTER_KEY, delivery_id)

        # Delete delivery data
        deleted = await redis.delete(key)
        return deleted > 0

    async def get_ready_for_retry(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get deliveries ready for retry."""
        redis = await self._get_redis()
        now = time.time()

        # Get delivery IDs with score <= now
        delivery_ids = await redis.zrangebyscore(self.PENDING_KEY, "-inf", now, start=0, num=limit)

        deliveries = []
        for delivery_id in delivery_ids:
            delivery = await self.get(delivery_id)
            if delivery and delivery.status == DeliveryStatus.PENDING:
                deliveries.append(delivery)

        return deliveries

    async def get_by_status(
        self, status: DeliveryStatus, limit: int = 100
    ) -> List[WebhookDelivery]:
        """Get deliveries by status."""
        redis = await self._get_redis()

        if status == DeliveryStatus.PENDING:
            delivery_ids = await redis.zrange(self.PENDING_KEY, 0, limit - 1)
        elif status == DeliveryStatus.DEAD_LETTER:
            delivery_ids = await redis.zrange(self.DEAD_LETTER_KEY, 0, limit - 1)
        else:
            # For other statuses, we'd need to scan - not efficient
            # In practice, we mostly care about PENDING and DEAD_LETTER
            return []

        deliveries = []
        for delivery_id in delivery_ids:
            delivery = await self.get(delivery_id)
            if delivery and delivery.status == status:
                deliveries.append(delivery)

        return deliveries

    async def get_dead_letters(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get dead-letter deliveries."""
        return await self.get_by_status(DeliveryStatus.DEAD_LETTER, limit)

    async def count_by_status(self) -> Dict[DeliveryStatus, int]:
        """Count deliveries by status."""
        redis = await self._get_redis()

        pending_count = await redis.zcard(self.PENDING_KEY)
        dead_letter_count = await redis.zcard(self.DEAD_LETTER_KEY)

        return {
            DeliveryStatus.PENDING: pending_count,
            DeliveryStatus.DEAD_LETTER: dead_letter_count,
            DeliveryStatus.IN_FLIGHT: 0,  # Tracked in memory
            DeliveryStatus.DELIVERED: 0,  # Not tracked after completion
            DeliveryStatus.FAILED: 0,  # Not tracked separately
        }

    async def clear(self) -> int:
        """Clear all deliveries."""
        redis = await self._get_redis()

        # Get all delivery IDs from both indices
        pending_ids = await redis.zrange(self.PENDING_KEY, 0, -1)
        dead_letter_ids = await redis.zrange(self.DEAD_LETTER_KEY, 0, -1)

        all_ids = set(pending_ids) | set(dead_letter_ids)
        count = len(all_ids)

        # Delete all delivery data
        if all_ids:
            keys = [self._delivery_key(d_id) for d_id in all_ids]
            await redis.delete(*keys)

        # Clear indices
        await redis.delete(self.PENDING_KEY)
        await redis.delete(self.DEAD_LETTER_KEY)

        return count

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._connected = False


# =============================================================================
# Webhook Retry Queue
# =============================================================================


# Type alias for delivery callback
DeliveryCallback = Callable[[WebhookDelivery], Awaitable[None]]


class WebhookRetryQueue:
    """
    Manages webhook delivery retries with persistence.

    Features:
    - Exponential backoff retry strategy
    - Configurable maximum retry attempts
    - Dead-letter queue for failed deliveries
    - Concurrent delivery with semaphore control
    - Pluggable storage backends (in-memory, Redis)

    Usage:
        queue = WebhookRetryQueue()
        await queue.start()

        delivery = WebhookDelivery(
            id="123",
            url="https://example.com/webhook",
            payload={"event": "test"},
        )
        await queue.enqueue(delivery)

        await queue.stop()
    """

    def __init__(
        self,
        store: Optional[WebhookDeliveryStore] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        dead_letter_callback: Optional[DeliveryCallback] = None,
        delivery_callback: Optional[DeliveryCallback] = None,
        request_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the retry queue.

        Args:
            store: Storage backend for delivery persistence.
                   Defaults to InMemoryDeliveryStore.
            max_concurrent: Maximum concurrent delivery attempts.
            dead_letter_callback: Called when a delivery is moved to dead-letter.
            delivery_callback: Called when a delivery succeeds.
            request_timeout: HTTP request timeout in seconds.
        """
        self._store = store or InMemoryDeliveryStore()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._dead_letter_callback = dead_letter_callback
        self._delivery_callback = delivery_callback
        self._request_timeout = request_timeout
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            "enqueued": 0,
            "delivered": 0,
            "failed": 0,
            "dead_lettered": 0,
            "retries": 0,
        }
        self._stats_lock = asyncio.Lock()

    @property
    def store(self) -> WebhookDeliveryStore:
        """Get the delivery store."""
        return self._store

    @property
    def is_running(self) -> bool:
        """Check if the queue processor is running."""
        return self._running

    async def enqueue(self, delivery: WebhookDelivery) -> str:
        """
        Add a webhook delivery to the retry queue.

        Args:
            delivery: The webhook delivery to enqueue.

        Returns:
            The delivery ID.
        """
        # Generate ID if not set
        if not delivery.id:
            delivery.id = str(uuid.uuid4())

        delivery.status = DeliveryStatus.PENDING
        await self._store.save(delivery)

        async with self._stats_lock:
            self._stats["enqueued"] += 1

        logger.debug(f"Enqueued webhook delivery {delivery.id} to {delivery.url}")
        return delivery.id

    async def start(self) -> None:
        """Start the retry queue processor."""
        if self._running:
            logger.warning("WebhookRetryQueue already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("WebhookRetryQueue started")

    async def stop(self, wait: bool = True) -> None:
        """
        Stop the retry queue processor.

        Args:
            wait: Whether to wait for the processor task to complete.
        """
        self._running = False

        if self._processor_task:
            if wait:
                # Give the task a chance to complete gracefully
                try:
                    await asyncio.wait_for(self._processor_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._processor_task.cancel()
                    try:
                        await self._processor_task
                    except asyncio.CancelledError:
                        pass
            else:
                self._processor_task.cancel()

            self._processor_task = None

        await self._store.close()
        logger.info("WebhookRetryQueue stopped")

    async def _process_loop(self) -> None:
        """Main processing loop for retry attempts."""
        while self._running:
            try:
                # Get deliveries ready for retry
                pending = await self._store.get_ready_for_retry(limit=100)

                # Process each delivery concurrently
                if pending:
                    tasks = [
                        asyncio.create_task(self._attempt_delivery(delivery))
                        for delivery in pending
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Sleep before next iteration
                await asyncio.sleep(PROCESS_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry queue process loop: {e}")
                await asyncio.sleep(PROCESS_INTERVAL)

    async def _attempt_delivery(self, delivery: WebhookDelivery) -> None:
        """
        Attempt to deliver a webhook.

        Args:
            delivery: The delivery to attempt.
        """
        async with self._semaphore:
            # Mark as in-flight
            delivery.attempts += 1
            delivery.status = DeliveryStatus.IN_FLIGHT
            await self._store.save(delivery)

            try:
                success, status_code, error = await self._send_webhook(delivery)

                if success:
                    delivery.status = DeliveryStatus.DELIVERED
                    delivery.last_status_code = status_code

                    async with self._stats_lock:
                        self._stats["delivered"] += 1

                    logger.info(
                        f"Webhook delivery {delivery.id} succeeded (attempt {delivery.attempts})"
                    )

                    # Call success callback
                    if self._delivery_callback:
                        try:
                            await self._delivery_callback(delivery)
                        except Exception as e:
                            logger.error(f"Delivery callback error: {e}")

                else:
                    delivery.last_error = error
                    delivery.last_status_code = status_code

                    # Check if we should dead-letter
                    if delivery.should_dead_letter():
                        delivery.status = DeliveryStatus.DEAD_LETTER

                        async with self._stats_lock:
                            self._stats["dead_lettered"] += 1

                        logger.warning(
                            f"Webhook delivery {delivery.id} moved to dead-letter "
                            f"after {delivery.attempts} attempts: {error}"
                        )

                        # Call dead-letter callback
                        if self._dead_letter_callback:
                            try:
                                await self._dead_letter_callback(delivery)
                            except Exception as e:
                                logger.error(f"Dead-letter callback error: {e}")
                    else:
                        # Schedule retry
                        delivery.status = DeliveryStatus.PENDING
                        delivery.next_retry_at = delivery.calculate_next_retry()

                        async with self._stats_lock:
                            self._stats["retries"] += 1

                        logger.info(
                            f"Webhook delivery {delivery.id} failed, "
                            f"scheduling retry {delivery.attempts + 1}/{delivery.max_attempts} "
                            f"at {delivery.next_retry_at.isoformat()}"
                        )

            except Exception as e:
                # Unexpected error - treat as failed attempt
                delivery.last_error = str(e)

                if delivery.should_dead_letter():
                    delivery.status = DeliveryStatus.DEAD_LETTER
                    async with self._stats_lock:
                        self._stats["dead_lettered"] += 1
                else:
                    delivery.status = DeliveryStatus.PENDING
                    delivery.next_retry_at = delivery.calculate_next_retry()
                    async with self._stats_lock:
                        self._stats["retries"] += 1

                logger.error(f"Unexpected error delivering webhook {delivery.id}: {e}")

            # Save final state
            await self._store.save(delivery)

            # Update failure stats
            if delivery.status in (DeliveryStatus.FAILED, DeliveryStatus.DEAD_LETTER):
                async with self._stats_lock:
                    self._stats["failed"] += 1

    async def _send_webhook(self, delivery: WebhookDelivery) -> tuple[bool, int, Optional[str]]:
        """
        Send the HTTP request for a webhook delivery.

        Returns:
            Tuple of (success, status_code, error_message)
        """
        try:
            import aiohttp
        except ImportError:
            # Fall back to synchronous delivery
            return await self._send_webhook_sync(delivery)

        try:
            # Import signature generator
            try:
                from aragora.server.handlers.webhooks import generate_signature
            except ImportError:
                generate_signature = None

            # Build headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Aragora-Webhooks/1.0",
                "X-Aragora-Event": delivery.payload.get("event", "unknown"),
                "X-Aragora-Delivery-ID": delivery.id,
                "X-Aragora-Timestamp": str(int(time.time())),
            }

            if delivery.correlation_id:
                headers["X-Aragora-Correlation-ID"] = delivery.correlation_id

            # Add custom headers
            headers.update(delivery.headers)

            # Add signature if secret is available
            payload_json = json.dumps(delivery.payload, default=str)
            if delivery.webhook_secret and generate_signature:
                signature = generate_signature(payload_json, delivery.webhook_secret)
                headers["X-Aragora-Signature"] = signature

            # Send request
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    delivery.url,
                    data=payload_json,
                    headers=headers,
                ) as response:
                    if response.status < 400:
                        return True, response.status, None
                    else:
                        return False, response.status, f"HTTP {response.status}"

        except aiohttp.ClientError as e:
            return False, 0, f"Connection error: {e}"
        except asyncio.TimeoutError:
            return False, 0, "Request timed out"
        except Exception as e:
            return False, 0, str(e)

    async def _send_webhook_sync(
        self, delivery: WebhookDelivery
    ) -> tuple[bool, int, Optional[str]]:
        """
        Fallback synchronous webhook delivery using urllib.

        Used when aiohttp is not available.
        """

        def _sync_send() -> tuple[bool, int, Optional[str]]:
            import json as _json
            from urllib.error import HTTPError, URLError
            from urllib.request import Request, urlopen

            try:
                # Import signature generator
                try:
                    from aragora.server.handlers.webhooks import generate_signature
                except ImportError:
                    generate_signature = None

                # Build headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Aragora-Webhooks/1.0",
                    "X-Aragora-Event": delivery.payload.get("event", "unknown"),
                    "X-Aragora-Delivery-ID": delivery.id,
                    "X-Aragora-Timestamp": str(int(time.time())),
                }

                if delivery.correlation_id:
                    headers["X-Aragora-Correlation-ID"] = delivery.correlation_id

                headers.update(delivery.headers)

                payload_json = _json.dumps(delivery.payload, default=str)
                if delivery.webhook_secret and generate_signature:
                    signature = generate_signature(payload_json, delivery.webhook_secret)
                    headers["X-Aragora-Signature"] = signature

                request = Request(
                    delivery.url,
                    data=payload_json.encode("utf-8"),
                    headers=headers,
                    method="POST",
                )

                with urlopen(request, timeout=self._request_timeout) as response:
                    return True, response.status, None

            except HTTPError as e:
                return False, e.code, f"HTTP {e.code}: {e.reason}"
            except URLError as e:
                return False, 0, f"Connection error: {e.reason}"
            except TimeoutError:
                return False, 0, "Request timed out"
            except Exception as e:
                return False, 0, str(e)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_send)

    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self._stats_lock:
            stats = dict(self._stats)

        # Add status counts from store
        status_counts = await self._store.count_by_status()
        stats["pending"] = status_counts.get(DeliveryStatus.PENDING, 0)
        stats["dead_letters"] = status_counts.get(DeliveryStatus.DEAD_LETTER, 0)
        stats["running"] = self._running

        return stats

    async def retry_dead_letter(self, delivery_id: str) -> bool:
        """
        Retry a dead-letter delivery.

        Resets the delivery to PENDING status with reset attempt count.

        Args:
            delivery_id: The delivery ID to retry.

        Returns:
            True if the delivery was found and retried.
        """
        delivery = await self._store.get(delivery_id)
        if not delivery:
            return False

        if delivery.status != DeliveryStatus.DEAD_LETTER:
            return False

        delivery.status = DeliveryStatus.PENDING
        delivery.attempts = 0
        delivery.next_retry_at = None
        delivery.last_error = None
        delivery.last_status_code = None

        await self._store.save(delivery)
        logger.info(f"Dead-letter delivery {delivery_id} queued for retry")
        return True

    async def get_dead_letters(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get dead-letter deliveries."""
        return await self._store.get_dead_letters(limit)

    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a delivery by ID."""
        return await self._store.get(delivery_id)

    async def cancel_delivery(self, delivery_id: str) -> bool:
        """
        Cancel a pending delivery.

        Args:
            delivery_id: The delivery ID to cancel.

        Returns:
            True if the delivery was found and cancelled.
        """
        delivery = await self._store.get(delivery_id)
        if not delivery:
            return False

        if delivery.status not in (DeliveryStatus.PENDING, DeliveryStatus.DEAD_LETTER):
            return False

        await self._store.delete(delivery_id)
        logger.info(f"Delivery {delivery_id} cancelled")
        return True


# =============================================================================
# Factory Functions
# =============================================================================


_global_queue: Optional[WebhookRetryQueue] = None


def get_retry_queue() -> WebhookRetryQueue:
    """
    Get or create the global webhook retry queue.

    Configuration via environment variables:
    - ARAGORA_WEBHOOK_RETRY_STORE: "memory" or "redis"
    - ARAGORA_REDIS_URL: Redis connection URL (for redis store)
    - ARAGORA_WEBHOOK_RETRY_MAX_ATTEMPTS: Maximum retry attempts (default: 5)
    - ARAGORA_WEBHOOK_RETRY_CONCURRENT: Maximum concurrent deliveries (default: 10)

    Returns:
        The global WebhookRetryQueue instance.
    """
    global _global_queue

    if _global_queue is not None:
        return _global_queue

    # Determine store type
    store_type = os.environ.get("ARAGORA_WEBHOOK_RETRY_STORE", "memory").lower()

    if store_type == "redis":
        store: WebhookDeliveryStore = RedisDeliveryStore()
    else:
        store = InMemoryDeliveryStore()

    _global_queue = WebhookRetryQueue(
        store=store,
        max_concurrent=DEFAULT_MAX_CONCURRENT,
    )

    return _global_queue


def set_retry_queue(queue: WebhookRetryQueue) -> None:
    """Set the global webhook retry queue."""
    global _global_queue
    _global_queue = queue


async def reset_retry_queue() -> None:
    """Reset the global webhook retry queue."""
    global _global_queue
    if _global_queue is not None:
        await _global_queue.stop()
        _global_queue = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "DeliveryStatus",
    # Data classes
    "WebhookDelivery",
    # Storage backends
    "WebhookDeliveryStore",
    "InMemoryDeliveryStore",
    "RedisDeliveryStore",
    # Queue
    "WebhookRetryQueue",
    # Factory functions
    "get_retry_queue",
    "set_retry_queue",
    "reset_retry_queue",
    # Type aliases
    "DeliveryCallback",
]
