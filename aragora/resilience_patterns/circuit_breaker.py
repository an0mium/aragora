"""
Unified Circuit Breaker for Aragora.

Provides a base circuit breaker implementation that can be extended for
specific use cases (agents, KM adapters, connectors, etc.).

The circuit breaker pattern prevents cascading failures by failing fast
when a service is unhealthy, then gradually allowing requests through
to test recovery.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing fast, no requests pass through
- HALF_OPEN: Testing recovery, limited requests allowed

Usage:
    from aragora.resilience_patterns import BaseCircuitBreaker, CircuitBreakerConfig

    # Direct usage
    cb = BaseCircuitBreaker("my_service")
    if cb.can_execute():
        try:
            result = call_service()
            cb.record_success()
        except Exception as e:
            cb.record_failure(e)
            raise

    # As decorator
    @with_circuit_breaker("my_service")
    async def call_service():
        ...
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        circuit_name: Optional[str] = None,
        cooldown_remaining: Optional[float] = None,
    ):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.cooldown_remaining = cooldown_remaining


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open before closing
        cooldown_seconds: Time to wait before entering half-open from open
        half_open_max_requests: Max concurrent requests in half-open state
        failure_rate_threshold: Alternative: open if failure rate exceeds this
        window_size: Time window for failure rate calculation (seconds)
        excluded_exceptions: Exceptions that don't count as failures
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    cooldown_seconds: float = 60.0
    half_open_max_requests: int = 3
    failure_rate_threshold: Optional[float] = None  # 0.0-1.0
    window_size: float = 60.0
    excluded_exceptions: tuple[type[Exception], ...] = ()
    on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    consecutive_failures: int
    consecutive_successes: int
    total_requests: int
    total_failures: int
    cooldown_remaining: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "cooldown_remaining": self.cooldown_remaining,
        }


class BaseCircuitBreaker:
    """Base circuit breaker implementation.

    Thread-safe circuit breaker with configurable thresholds and callbacks.
    Can be extended for specific use cases.

    Args:
        name: Unique name for this circuit breaker
        config: CircuitBreakerConfig instance
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._total_requests = 0
        self._total_failures = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._half_open_requests = 0
        self._last_accessed = time.time()

        # For failure rate calculation
        self._recent_results: list[tuple[float, bool]] = []  # (timestamp, success)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def failure_threshold(self) -> int:
        """Get failure threshold (backward compatibility)."""
        return self.config.failure_threshold

    @property
    def cooldown_seconds(self) -> float:
        """Get cooldown seconds (backward compatibility)."""
        return self.config.cooldown_seconds

    @property
    def failures(self) -> int:
        """Get consecutive failure count (backward compatibility)."""
        with self._lock:
            return self._consecutive_failures

    def get_status(self) -> str:
        """Get status string (backward compatibility)."""
        return self.state.value

    def can_proceed(self) -> bool:
        """Backward compatibility alias for can_execute()."""
        return self.can_execute()

    def can_execute(self) -> bool:
        """Check if a request can be executed.

        Returns:
            True if request is allowed, False if circuit is open
        """
        with self._lock:
            self._last_accessed = time.time()
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # HALF_OPEN: allow limited requests
            if self._half_open_requests < self.config.half_open_max_requests:
                self._half_open_requests += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._last_success_time = time.time()
            self._last_accessed = time.time()
            self._success_count += 1
            self._total_requests += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._add_result(True)

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed operation.

        Args:
            exception: The exception that caused the failure (optional)
        """
        # Check if exception should be excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            logger.debug(f"[{self.name}] Excluded exception, not counting as failure: {exception}")
            return

        with self._lock:
            self._last_failure_time = time.time()
            self._last_accessed = time.time()
            self._failure_count += 1
            self._total_requests += 1
            self._total_failures += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._add_result(False)

            # Check if we should open the circuit
            should_open = False

            # Threshold-based
            if self._consecutive_failures >= self.config.failure_threshold:
                should_open = True
                logger.warning(
                    f"[{self.name}] Opening circuit after {self._consecutive_failures} consecutive failures"
                )

            # Rate-based (if configured)
            if self.config.failure_rate_threshold is not None:
                rate = self._calculate_failure_rate()
                if rate >= self.config.failure_rate_threshold:
                    should_open = True
                    logger.warning(
                        f"[{self.name}] Opening circuit due to failure rate {rate:.2%} >= {self.config.failure_rate_threshold:.2%}"
                    )

            if should_open and self._state != CircuitState.OPEN:
                self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._half_open_requests = 0
            self._opened_at = None
            self._recent_results.clear()

            if old_state != CircuitState.CLOSED:
                logger.info(f"[{self.name}] Circuit reset from {old_state.value} to closed")
                self._notify_state_change(old_state, CircuitState.CLOSED)

    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        with self._lock:
            cooldown_remaining = None
            if self._state == CircuitState.OPEN and self._opened_at:
                elapsed = time.time() - self._opened_at
                cooldown_remaining = max(0, self.config.cooldown_seconds - elapsed)

            return CircuitBreakerStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                total_requests=self._total_requests,
                total_failures=self._total_failures,
                cooldown_remaining=cooldown_remaining,
            )

    def _check_state_transition(self) -> None:
        """Check if state should transition (must hold lock)."""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = time.time() - self._opened_at
            if elapsed >= self.config.cooldown_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state (must hold lock)."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        logger.info(f"[{self.name}] Circuit state: {old_state.value} -> {new_state.value}")

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_requests = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_requests = 0
            self._consecutive_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._opened_at = None

        self._notify_state_change(old_state, new_state)

    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """Notify callback of state change."""
        if self.config.on_state_change:
            try:
                self.config.on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.warning(f"[{self.name}] State change callback error: {e}")

    def _add_result(self, success: bool) -> None:
        """Add a result to the sliding window (must hold lock)."""
        now = time.time()
        self._recent_results.append((now, success))

        # Prune old results
        cutoff = now - self.config.window_size
        self._recent_results = [(t, s) for t, s in self._recent_results if t >= cutoff]

    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate in the current window (must hold lock)."""
        if not self._recent_results:
            return 0.0

        failures = sum(1 for _, success in self._recent_results if not success)
        return failures / len(self._recent_results)


def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    circuit_breaker: Optional[BaseCircuitBreaker] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for async functions with circuit breaker protection.

    Args:
        name: Circuit breaker name (used if circuit_breaker not provided)
        config: CircuitBreakerConfig instance
        circuit_breaker: Existing circuit breaker to use

    Returns:
        Decorator function

    Example:
        @with_circuit_breaker("external_api")
        async def call_api():
            ...
    """
    cb = circuit_breaker or BaseCircuitBreaker(name, config)

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not cb.can_execute():
                stats = cb.get_stats()
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{cb.name}' is open",
                    circuit_name=cb.name,
                    cooldown_remaining=stats.cooldown_remaining,
                )

            try:
                result = await func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure(e)
                raise

        return wrapper

    return decorator


def with_circuit_breaker_sync(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    circuit_breaker: Optional[BaseCircuitBreaker] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for sync functions with circuit breaker protection.

    Same as with_circuit_breaker but for synchronous functions.
    """
    cb = circuit_breaker or BaseCircuitBreaker(name, config)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not cb.can_execute():
                stats = cb.get_stats()
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{cb.name}' is open",
                    circuit_name=cb.name,
                    cooldown_remaining=stats.cooldown_remaining,
                )

            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure(e)
                raise

        return wrapper

    return decorator


# =============================================================================
# Global Circuit Breaker Registry (backward-compatible with aragora.resilience)
# =============================================================================

_circuit_breakers: dict[str, BaseCircuitBreaker] = {}
_circuit_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: Optional[int] = None,
    cooldown_seconds: Optional[float] = None,
    config: Optional[CircuitBreakerConfig] = None,
) -> BaseCircuitBreaker:
    """Get or create a named circuit breaker from the global registry.

    This ensures consistent circuit breaker state across components
    for the same service/agent.

    Args:
        name: Unique identifier for this circuit breaker
        failure_threshold: Failures before opening circuit
        cooldown_seconds: Seconds before attempting recovery
        config: Explicit CircuitBreakerConfig to use

    Returns:
        BaseCircuitBreaker instance (shared if already exists)

    Example:
        cb = get_circuit_breaker("anthropic_agent", failure_threshold=5)
        if cb.can_execute():
            try:
                result = await call_api()
                cb.record_success()
            except Exception as e:
                cb.record_failure(e)
                raise
    """
    with _circuit_breakers_lock:
        if name not in _circuit_breakers:
            # Build config from parameters
            if config is not None:
                resolved_config = config
            else:
                resolved_config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold or 5,
                    cooldown_seconds=cooldown_seconds or 60.0,
                )

            _circuit_breakers[name] = BaseCircuitBreaker(name, resolved_config)
            logger.debug(f"Created circuit breaker: {name}")

        return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all global circuit breakers. Useful for testing."""
    with _circuit_breakers_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
        count = len(_circuit_breakers)
    logger.info(f"Reset {count} circuit breakers")


def get_all_circuit_breakers() -> dict[str, BaseCircuitBreaker]:
    """Get all registered circuit breakers.

    Returns:
        Dict mapping circuit breaker names to instances.
    """
    with _circuit_breakers_lock:
        return dict(_circuit_breakers)
