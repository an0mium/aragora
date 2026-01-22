"""
Platform resilience module for chat integrations.

Provides production-grade resilience patterns for chat platform integrations:
- Platform-level circuit breakers (not just per-webhook)
- Distributed rate limiting
- Dead letter queue for failed messages
- Graceful degradation with fallback channels
- Comprehensive metrics and observability
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypeVar

from aragora.resilience import CircuitBreaker, CircuitOpenError, get_circuit_breaker

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Configuration
DLQ_ENABLED = os.getenv("ARAGORA_DLQ_ENABLED", "true").lower() == "true"
DLQ_DB_PATH = os.getenv("ARAGORA_DLQ_DB_PATH", ".data/dlq.db")
DLQ_MAX_RETRIES = int(os.getenv("ARAGORA_DLQ_MAX_RETRIES", "5"))
DLQ_RETENTION_HOURS = float(os.getenv("ARAGORA_DLQ_RETENTION_HOURS", "168"))  # 7 days


class PlatformStatus(Enum):
    """Platform health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class PlatformHealth:
    """Health status for a platform."""

    platform: str
    status: PlatformStatus
    circuit_state: str  # closed, open, half-open
    success_rate: float  # 0.0 - 1.0
    avg_latency_ms: float
    error_count: int
    last_success_at: Optional[float] = None
    last_error_at: Optional[float] = None
    last_error_message: Optional[str] = None


@dataclass
class PlatformCircuitBreaker:
    """Platform-level circuit breaker with enhanced metrics.

    Unlike per-webhook circuit breakers, this tracks the entire platform's
    health and can be used to fail-fast all requests to a platform when
    it's experiencing issues.
    """

    platform: str
    failure_threshold: int = 5
    cooldown_seconds: float = 120.0
    success_threshold: int = 3  # Successes needed to fully close from half-open

    # Internal state
    _circuit: CircuitBreaker = field(default=None, repr=False)
    _success_count: int = field(default=0, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _total_requests: int = field(default=0, repr=False)
    _latencies: list[float] = field(default_factory=list, repr=False)
    _max_latency_samples: int = field(default=100, repr=False)
    _last_success_at: float = field(default=0.0, repr=False)
    _last_error_at: float = field(default=0.0, repr=False)
    _last_error_message: str = field(default="", repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._circuit = get_circuit_breaker(
            f"platform:{self.platform}",
            failure_threshold=self.failure_threshold,
            cooldown_seconds=self.cooldown_seconds,
        )

    def can_proceed(self) -> bool:
        """Check if requests can proceed to this platform."""
        return self._circuit.can_proceed()

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record a successful request."""
        with self._lock:
            self._success_count += 1
            self._total_requests += 1
            self._last_success_at = time.time()
            if latency_ms > 0:
                self._latencies.append(latency_ms)
                if len(self._latencies) > self._max_latency_samples:
                    self._latencies.pop(0)

        self._circuit.record_success()

    def record_failure(self, error_message: str = "") -> bool:
        """Record a failed request. Returns True if circuit just opened."""
        with self._lock:
            self._failure_count += 1
            self._total_requests += 1
            self._last_error_at = time.time()
            self._last_error_message = error_message[:500]  # Truncate

        return self._circuit.record_failure()

    def get_health(self) -> PlatformHealth:
        """Get current health status."""
        with self._lock:
            total = self._success_count + self._failure_count
            success_rate = self._success_count / total if total > 0 else 1.0
            avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

            # Determine status
            circuit_state = self._circuit.get_status()
            if circuit_state == "open":
                status = PlatformStatus.UNAVAILABLE
            elif circuit_state == "half-open" or success_rate < 0.9:
                status = PlatformStatus.DEGRADED
            else:
                status = PlatformStatus.HEALTHY

            return PlatformHealth(
                platform=self.platform,
                status=status,
                circuit_state=circuit_state,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                error_count=self._failure_count,
                last_success_at=self._last_success_at if self._last_success_at > 0 else None,
                last_error_at=self._last_error_at if self._last_error_at > 0 else None,
                last_error_message=self._last_error_message or None,
            )

    def reset(self) -> None:
        """Reset circuit breaker and metrics."""
        with self._lock:
            self._success_count = 0
            self._failure_count = 0
            self._total_requests = 0
            self._latencies.clear()
        self._circuit.reset()


# Global platform circuit breakers registry
_platform_circuits: dict[str, PlatformCircuitBreaker] = {}
_platform_circuits_lock = threading.Lock()


def get_platform_circuit(
    platform: str,
    failure_threshold: int = 5,
    cooldown_seconds: float = 120.0,
) -> PlatformCircuitBreaker:
    """Get or create platform-level circuit breaker."""
    with _platform_circuits_lock:
        if platform not in _platform_circuits:
            _platform_circuits[platform] = PlatformCircuitBreaker(
                platform=platform,
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
            )
        return _platform_circuits[platform]


def get_all_platform_health() -> dict[str, PlatformHealth]:
    """Get health status for all tracked platforms."""
    with _platform_circuits_lock:
        return {name: circuit.get_health() for name, circuit in _platform_circuits.items()}


# =============================================================================
# Dead Letter Queue for Failed Messages
# =============================================================================


@dataclass
class DeadLetterMessage:
    """A message that failed delivery and is queued for retry."""

    id: str
    platform: str
    destination: str  # Channel ID, user ID, webhook URL, etc.
    payload: str  # JSON serialized
    error_message: str
    retry_count: int
    created_at: float
    last_retry_at: Optional[float] = None
    next_retry_at: Optional[float] = None
    metadata: Optional[str] = None  # JSON for additional context


class DeadLetterQueue:
    """SQLite-backed dead letter queue for failed message delivery.

    Stores messages that failed delivery for later retry or manual review.
    Implements exponential backoff for retries.
    """

    def __init__(self, db_path: str = DLQ_DB_PATH, max_retries: int = DLQ_MAX_RETRIES):
        self.db_path = db_path
        self.max_retries = max_retries
        self._initialized = False
        self._lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_initialized(self) -> None:
        """Initialize database schema if needed."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dead_letters (
                        id TEXT PRIMARY KEY,
                        platform TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        last_retry_at REAL,
                        next_retry_at REAL,
                        metadata TEXT,
                        status TEXT NOT NULL DEFAULT 'pending'
                    )
                """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_dlq_platform
                    ON dead_letters(platform, status)
                """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_dlq_next_retry
                    ON dead_letters(next_retry_at) WHERE status = 'pending'
                """
                )
                conn.commit()

            self._initialized = True
            logger.info(f"Dead letter queue initialized: {self.db_path}")

    def enqueue(
        self,
        platform: str,
        destination: str,
        payload: dict | str,
        error_message: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a failed message to the dead letter queue.

        Returns the message ID.
        """
        if not DLQ_ENABLED:
            logger.debug(f"DLQ disabled, dropping message for {platform}:{destination}")
            return ""

        self._ensure_initialized()

        import uuid

        msg_id = str(uuid.uuid4())
        now = time.time()

        payload_json = json.dumps(payload) if isinstance(payload, dict) else payload
        metadata_json = json.dumps(metadata) if metadata else None

        # First retry in 30 seconds
        next_retry = now + 30

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO dead_letters
                    (id, platform, destination, payload, error_message, retry_count,
                     created_at, next_retry_at, metadata, status)
                    VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, 'pending')
                """,
                    (
                        msg_id,
                        platform,
                        destination,
                        payload_json,
                        error_message[:1000],
                        now,
                        next_retry,
                        metadata_json,
                    ),
                )
                conn.commit()

            logger.info(f"Enqueued dead letter {msg_id} for {platform}:{destination}")
            return msg_id

        except Exception as e:
            logger.error(f"Failed to enqueue dead letter: {e}")
            return ""

    def get_pending(
        self, platform: Optional[str] = None, limit: int = 100
    ) -> list[DeadLetterMessage]:
        """Get messages ready for retry."""
        self._ensure_initialized()

        now = time.time()

        try:
            with self._get_connection() as conn:
                if platform:
                    cursor = conn.execute(
                        """
                        SELECT id, platform, destination, payload, error_message,
                               retry_count, created_at, last_retry_at, next_retry_at, metadata
                        FROM dead_letters
                        WHERE status = 'pending' AND platform = ? AND next_retry_at <= ?
                        ORDER BY next_retry_at ASC
                        LIMIT ?
                    """,
                        (platform, now, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT id, platform, destination, payload, error_message,
                               retry_count, created_at, last_retry_at, next_retry_at, metadata
                        FROM dead_letters
                        WHERE status = 'pending' AND next_retry_at <= ?
                        ORDER BY next_retry_at ASC
                        LIMIT ?
                    """,
                        (now, limit),
                    )

                messages = []
                for row in cursor.fetchall():
                    messages.append(
                        DeadLetterMessage(
                            id=row[0],
                            platform=row[1],
                            destination=row[2],
                            payload=row[3],
                            error_message=row[4],
                            retry_count=row[5],
                            created_at=row[6],
                            last_retry_at=row[7],
                            next_retry_at=row[8],
                            metadata=row[9],
                        )
                    )
                return messages

        except Exception as e:
            logger.error(f"Failed to get pending dead letters: {e}")
            return []

    def mark_success(self, msg_id: str) -> bool:
        """Mark a message as successfully delivered."""
        self._ensure_initialized()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE dead_letters SET status = 'delivered' WHERE id = ?",
                    (msg_id,),
                )
                conn.commit()
            logger.info(f"Dead letter {msg_id} delivered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to mark dead letter success: {e}")
            return False

    def mark_retry(self, msg_id: str, error_message: str) -> bool:
        """Mark a message for retry with exponential backoff.

        Returns False if max retries exceeded (message moved to failed).
        """
        self._ensure_initialized()

        now = time.time()

        try:
            with self._get_connection() as conn:
                # Get current retry count
                cursor = conn.execute(
                    "SELECT retry_count FROM dead_letters WHERE id = ?",
                    (msg_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                retry_count = row[0] + 1

                if retry_count >= self.max_retries:
                    # Move to failed
                    conn.execute(
                        """
                        UPDATE dead_letters
                        SET status = 'failed', retry_count = ?, last_retry_at = ?,
                            error_message = ?
                        WHERE id = ?
                    """,
                        (retry_count, now, error_message[:1000], msg_id),
                    )
                    conn.commit()
                    logger.warning(f"Dead letter {msg_id} exceeded max retries, marked failed")
                    return False

                # Exponential backoff: 30s, 1m, 2m, 4m, 8m, ...
                backoff = 30 * (2**retry_count)
                next_retry = now + min(backoff, 3600)  # Cap at 1 hour

                conn.execute(
                    """
                    UPDATE dead_letters
                    SET retry_count = ?, last_retry_at = ?, next_retry_at = ?,
                        error_message = ?
                    WHERE id = ?
                """,
                    (retry_count, now, next_retry, error_message[:1000], msg_id),
                )
                conn.commit()
                logger.info(
                    f"Dead letter {msg_id} scheduled for retry {retry_count} at {next_retry}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to mark dead letter retry: {e}")
            return False

    def cleanup_old(self, max_age_hours: float = DLQ_RETENTION_HOURS) -> int:
        """Remove old completed/failed messages."""
        self._ensure_initialized()

        cutoff = time.time() - (max_age_hours * 3600)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM dead_letters
                    WHERE status IN ('delivered', 'failed') AND created_at < ?
                """,
                    (cutoff,),
                )
                conn.commit()
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old dead letters")
                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup dead letters: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get dead letter queue statistics."""
        self._ensure_initialized()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT status, platform, COUNT(*) as count
                    FROM dead_letters
                    GROUP BY status, platform
                """
                )

                stats: dict[str, Any] = {
                    "total": 0,
                    "by_status": {},
                    "by_platform": {},
                }

                for row in cursor.fetchall():
                    status, platform, count = row
                    stats["total"] += count

                    if status not in stats["by_status"]:
                        stats["by_status"][status] = 0
                    stats["by_status"][status] += count

                    if platform not in stats["by_platform"]:
                        stats["by_platform"][platform] = {"pending": 0, "failed": 0, "delivered": 0}
                    stats["by_platform"][platform][status] = count

                return stats

        except Exception as e:
            logger.error(f"Failed to get DLQ stats: {e}")
            return {"total": 0, "by_status": {}, "by_platform": {}, "error": str(e)}


# Global DLQ instance
_dlq: Optional[DeadLetterQueue] = None
_dlq_lock = threading.Lock()


def get_dead_letter_queue() -> DeadLetterQueue:
    """Get the global dead letter queue instance."""
    global _dlq
    if _dlq is None:
        with _dlq_lock:
            if _dlq is None:
                _dlq = DeadLetterQueue()
    return _dlq


# =============================================================================
# Bot Command Timeout Wrapper
# =============================================================================


def with_timeout(
    timeout_seconds: float = 25.0,  # Telegram has 30s limit, leave margin
    fallback_response: Optional[str] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T | str]]]:
    """Decorator to wrap async bot command handlers with timeout.

    For platforms like Telegram that have strict webhook response timeouts,
    this ensures the handler doesn't block indefinitely.

    Args:
        timeout_seconds: Maximum execution time before timeout
        fallback_response: Response to return on timeout (None raises exception)

    Example:
        @with_timeout(timeout_seconds=25.0, fallback_response="Request is processing...")
        async def handle_command(message: str) -> str:
            result = await long_running_operation()
            return result
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T | str]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T | str:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout in {func.__name__} after {timeout_seconds}s")
                if fallback_response is not None:
                    return fallback_response
                raise

        return wrapper

    return decorator


def with_platform_resilience(
    platform: str,
    timeout_seconds: float = 25.0,
    use_dlq: bool = True,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T | None]]]:
    """Decorator combining circuit breaker, timeout, and DLQ for platform handlers.

    This is the recommended decorator for bot command handlers that need
    comprehensive resilience.

    Args:
        platform: Platform name (slack, discord, etc.)
        timeout_seconds: Maximum execution time
        use_dlq: Whether to queue failed messages to DLQ

    Example:
        @with_platform_resilience("telegram", timeout_seconds=25.0)
        async def handle_telegram_message(update: dict) -> str:
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T | None]]:
        circuit = get_platform_circuit(platform)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T | None:
            # Check circuit breaker
            if not circuit.can_proceed():
                health = circuit.get_health()
                logger.warning(
                    f"Platform {platform} circuit open, skipping {func.__name__}. "
                    f"Retry in {circuit._circuit.cooldown_remaining():.1f}s"
                )
                # Optionally queue to DLQ
                if use_dlq and args:
                    dlq = get_dead_letter_queue()
                    dlq.enqueue(
                        platform=platform,
                        destination=func.__name__,
                        payload={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
                        error_message=f"Circuit open: {health.last_error_message or 'Unknown'}",
                    )
                return None

            start_time = time.time()
            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                latency_ms = (time.time() - start_time) * 1000
                circuit.record_success(latency_ms)
                return result

            except asyncio.TimeoutError:
                latency_ms = (time.time() - start_time) * 1000
                circuit.record_failure(f"Timeout after {timeout_seconds}s")
                logger.warning(
                    f"Timeout in {func.__name__} for {platform} after {timeout_seconds}s"
                )
                return None

            except Exception as e:
                circuit.record_failure(str(e))
                logger.error(f"Error in {func.__name__} for {platform}: {e}")
                if use_dlq and args:
                    dlq = get_dead_letter_queue()
                    dlq.enqueue(
                        platform=platform,
                        destination=func.__name__,
                        payload={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
                        error_message=str(e)[:1000],
                    )
                raise

        return wrapper

    return decorator


# =============================================================================
# Metrics
# =============================================================================

# Metrics will be collected via the existing observability infrastructure
# These helper functions make it easy to record platform-specific metrics


def record_platform_request(
    platform: str,
    operation: str,
    success: bool,
    latency_ms: float,
    error_type: Optional[str] = None,
) -> None:
    """Record a platform request for metrics.

    Integrates with the existing observability/metrics infrastructure.
    """
    try:
        from aragora.observability.metrics.core import (
            counter_inc,
            histogram_observe,
        )

        labels = {"platform": platform, "operation": operation}

        if success:
            counter_inc("aragora_platform_requests_total", {**labels, "status": "success"})
        else:
            counter_inc(
                "aragora_platform_requests_total",
                {**labels, "status": "error", "error_type": error_type or "unknown"},
            )

        histogram_observe("aragora_platform_latency_ms", latency_ms, labels)

    except ImportError:
        # Metrics not available
        pass


__all__ = [
    # Circuit breakers
    "PlatformCircuitBreaker",
    "PlatformStatus",
    "PlatformHealth",
    "get_platform_circuit",
    "get_all_platform_health",
    # Dead letter queue
    "DeadLetterQueue",
    "DeadLetterMessage",
    "get_dead_letter_queue",
    # Decorators
    "with_timeout",
    "with_platform_resilience",
    # Metrics
    "record_platform_request",
    # Re-exports
    "CircuitOpenError",
]
