"""Circuit breaker for PagerDuty API access."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class DevOpsCircuitBreaker:
    """Circuit breaker for PagerDuty API access.

    Prevents cascading failures when PagerDuty API is unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    # State constants
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Time to wait before allowing test calls
            half_open_max_calls: Number of test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                logger.info("DevOps circuit breaker transitioning to HALF_OPEN")
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def is_allowed(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        with self._lock:
            state = self._check_state()

            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    logger.info("DevOps circuit breaker closing after successful tests")
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == self.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Failed during half-open, reopen circuit
                logger.warning("DevOps circuit breaker reopening after test failure")
                self._state = self.OPEN
                self._success_count = 0
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        "DevOps circuit breaker opening after %d failures",
                        self._failure_count,
                    )
                    self._state = self.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }


# Global circuit breaker instance
_devops_circuit_breaker: DevOpsCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def get_devops_circuit_breaker() -> DevOpsCircuitBreaker:
    """Get or create the global DevOps circuit breaker."""
    global _devops_circuit_breaker
    with _circuit_breaker_lock:
        if _devops_circuit_breaker is None:
            _devops_circuit_breaker = DevOpsCircuitBreaker()
        return _devops_circuit_breaker


def get_devops_circuit_breaker_status() -> dict[str, Any]:
    """Get the current status of the DevOps circuit breaker."""
    return get_devops_circuit_breaker().get_status()
