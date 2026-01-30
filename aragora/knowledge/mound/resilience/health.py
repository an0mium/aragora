"""Connection health monitoring."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for a storage backend."""

    healthy: bool
    last_check: datetime
    consecutive_failures: int = 0
    last_error: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
        }


class ConnectionHealthMonitor:
    """
    Monitors connection health and provides circuit breaker functionality.

    Tracks:
    - Connection success/failure rates
    - Latency metrics
    - Automatic health checks
    """

    def __init__(
        self,
        pool: Any,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        health_check_interval: float = 10.0,
    ):
        self._pool = pool
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._health_check_interval = health_check_interval

        self._status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
        )
        self._check_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start background health monitoring."""
        if self._check_task is None:
            self._check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Connection health monitor started")

    async def stop(self) -> None:
        """Stop background health monitoring."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

    async def check_health(self) -> HealthStatus:
        """Perform a health check."""
        start = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")

            latency = (time.monotonic() - start) * 1000
            self._status = HealthStatus(
                healthy=True,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=0,
                latency_ms=latency,
            )

        except (ConnectionError, TimeoutError, OSError) as e:
            self._status = HealthStatus(
                healthy=self._status.consecutive_failures < self._failure_threshold,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._status.consecutive_failures + 1,
                last_error=str(e),
            )
            logger.debug(f"Health check failed with expected error: {e}")
        except Exception as e:
            self._status = HealthStatus(
                healthy=self._status.consecutive_failures < self._failure_threshold,
                last_check=datetime.now(timezone.utc),
                consecutive_failures=self._status.consecutive_failures + 1,
                last_error=str(e),
            )
            logger.warning(f"Health check failed with unexpected error: {e}")

        return self._status

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.check_health()
            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Health check loop error (expected): {e}")
            except Exception as e:
                logger.warning(f"Health check loop error (unexpected): {e}")

    def record_success(self) -> None:
        """Record a successful operation."""
        if self._status.consecutive_failures > 0:
            self._status.consecutive_failures = 0
            self._status.healthy = True

    def record_failure(self, error: str) -> None:
        """Record a failed operation."""
        self._status.consecutive_failures += 1
        self._status.last_error = error
        if self._status.consecutive_failures >= self._failure_threshold:
            self._status.healthy = False
            logger.error(f"Connection unhealthy after {self._status.consecutive_failures} failures")

    def is_healthy(self) -> bool:
        """Check if connections are healthy."""
        return self._status.healthy

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        return self._status
