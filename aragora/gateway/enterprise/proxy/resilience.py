"""
Resilience patterns for the enterprise proxy.

Implements circuit breaker and bulkhead patterns for managing external
framework requests with proper failure isolation and concurrency control.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncGenerator

from .config import BulkheadSettings, CircuitBreakerSettings
from .exceptions import BulkheadFullError

logger = logging.getLogger(__name__)


class FrameworkCircuitBreaker:
    """Circuit breaker for a single external framework.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests allowed
    - OPEN: After failure threshold, requests blocked
    - HALF-OPEN: After cooldown, trial requests allowed
    """

    def __init__(self, framework: str, settings: CircuitBreakerSettings) -> None:
        """Initialize circuit breaker.

        Args:
            framework: Framework name for logging.
            settings: Circuit breaker settings.
        """
        self.framework = framework
        self.settings = settings

        self._failures = 0
        self._successes = 0
        self._open_at: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if self._open_at is None:
            return "closed"
        elapsed = time.time() - self._open_at
        if elapsed >= self.settings.cooldown_seconds:
            return "half-open"
        return "open"

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == "open"

    @property
    def cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self._open_at is None:
            return 0.0
        elapsed = time.time() - self._open_at
        remaining = self.settings.cooldown_seconds - elapsed
        return max(0.0, remaining)

    async def can_proceed(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            state = self.state

            if state == "closed":
                return True

            if state == "half-open":
                if self._half_open_calls < self.settings.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False  # open

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._failures = 0

            if self._open_at is not None:
                self._successes += 1
                if self._successes >= self.settings.success_threshold:
                    logger.info("Circuit breaker CLOSED for %s", self.framework)
                    self._open_at = None
                    self._successes = 0
                    self._half_open_calls = 0

    async def record_failure(self) -> bool:
        """Record a failed request. Returns True if circuit just opened."""
        async with self._lock:
            self._failures += 1
            self._successes = 0

            if self._failures >= self.settings.failure_threshold:
                if self._open_at is None:
                    self._open_at = time.time()
                    self._half_open_calls = 0
                    logger.warning(
                        "Circuit breaker OPEN for %s after %s failures",
                        self.framework,
                        self._failures,
                    )
                    return True

            return False

    async def reset(self) -> None:
        """Reset circuit breaker state."""
        async with self._lock:
            self._failures = 0
            self._successes = 0
            self._open_at = None
            self._half_open_calls = 0
            logger.info("Circuit breaker reset for %s", self.framework)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for monitoring."""
        return {
            "framework": self.framework,
            "state": self.state,
            "failures": self._failures,
            "successes": self._successes,
            "cooldown_remaining": self.cooldown_remaining,
            "half_open_calls": self._half_open_calls,
        }


class FrameworkBulkhead:
    """Bulkhead isolation for a single external framework.

    Limits concurrent requests to prevent resource exhaustion.
    """

    def __init__(self, framework: str, settings: BulkheadSettings) -> None:
        """Initialize bulkhead.

        Args:
            framework: Framework name for logging.
            settings: Bulkhead settings.
        """
        self.framework = framework
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        """Get number of active requests."""
        return self._active

    @property
    def available_slots(self) -> int:
        """Get number of available request slots."""
        return self.settings.max_concurrent - self._active

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None, None]:
        """Acquire a slot in the bulkhead.

        Raises:
            BulkheadFullError: If no slots available within timeout.
        """
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.wait_timeout,
            )
            if not acquired:
                raise BulkheadFullError(
                    self.framework,
                    self.settings.max_concurrent,
                )
        except asyncio.TimeoutError:
            raise BulkheadFullError(
                self.framework,
                self.settings.max_concurrent,
                {"wait_timeout": self.settings.wait_timeout},
            )

        async with self._lock:
            self._active += 1

        try:
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active -= 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for monitoring."""
        return {
            "framework": self.framework,
            "active": self._active,
            "max_concurrent": self.settings.max_concurrent,
            "available_slots": self.available_slots,
        }


__all__ = [
    "FrameworkCircuitBreaker",
    "FrameworkBulkhead",
]
