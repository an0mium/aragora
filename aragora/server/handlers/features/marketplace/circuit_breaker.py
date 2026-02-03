"""Marketplace circuit breaker for template loading resilience."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class MarketplaceCircuitBreaker:
    """Circuit breaker for marketplace template loading operations.

    Prevents cascading failures when template directory or YAML parsing is unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
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
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Marketplace circuit breaker transitioning to HALF_OPEN")
        return self._state

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
        """Alias for can_proceed (public API)."""
        return self.can_proceed()

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Marketplace circuit breaker closed after successful recovery")
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
                logger.warning("Marketplace circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Marketplace circuit breaker opened after {self._failure_count} failures"
                    )

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


# Global circuit breaker instance
_circuit_breaker: MarketplaceCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def _get_circuit_breaker() -> MarketplaceCircuitBreaker:
    """Get or create the marketplace circuit breaker."""
    global _circuit_breaker
    with _circuit_breaker_lock:
        if _circuit_breaker is None:
            _circuit_breaker = MarketplaceCircuitBreaker()
        return _circuit_breaker


def _get_marketplace_circuit_breaker() -> MarketplaceCircuitBreaker:
    """Public accessor for the marketplace circuit breaker (for testing)."""
    return _get_circuit_breaker()


def get_marketplace_circuit_breaker_status() -> dict[str, Any]:
    """Get the marketplace circuit breaker status."""
    return _get_marketplace_circuit_breaker().get_status()


def _reset_circuit_breaker() -> None:
    """Reset the global circuit breaker instance (for testing)."""
    global _circuit_breaker
    with _circuit_breaker_lock:
        _circuit_breaker = None
