"""Simple thread-safe circuit breaker for feature handler modules.

Provides the same synchronous API used by marketplace, ecommerce, devops,
and CRM handlers.  Consolidates ~650 lines of duplicated code into one
canonical implementation.

Usage:
    from aragora.resilience.simple_circuit_breaker import SimpleCircuitBreaker

    cb = SimpleCircuitBreaker("marketplace", half_open_max_calls=3)
    if cb.can_proceed():
        try:
            result = do_work()
            cb.record_success()
        except Exception:
            cb.record_failure()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class SimpleCircuitBreaker:
    """Thread-safe circuit breaker with CLOSED -> OPEN -> HALF_OPEN state machine.

    Parameters match the defaults used across the feature handler modules.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("%s circuit breaker transitioning to HALF_OPEN", self.name)
        return self._state

    # ------------------------------------------------------------------
    # Gate methods
    # ------------------------------------------------------------------

    def can_proceed(self) -> bool:
        """Check if a call can proceed."""
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:
                return False

    def is_allowed(self) -> bool:
        """Alias for can_proceed (backwards-compat)."""
        return self.can_proceed()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("%s circuit breaker closed after successful recovery", self.name)
            elif self._state == self.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._success_count = 0
                logger.warning(
                    "%s circuit breaker reopened after failure in HALF_OPEN", self.name
                )
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        "%s circuit breaker opened after %d failures",
                        self.name,
                        self._failure_count,
                    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
