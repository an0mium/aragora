"""Adapter circuit breaker for Knowledge Mound adapters."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import AsyncIterator

from aragora.knowledge.mound.resilience.health import ConnectionHealthMonitor

logger = logging.getLogger(__name__)


class AdapterCircuitState(str, Enum):
    """Circuit breaker states for adapters."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class AdapterCircuitBreakerConfig:
    """Configuration for adapter circuit breaker.

    Attributes:
        failure_threshold: Failures before opening circuit
        success_threshold: Successes in half-open to close circuit
        timeout_seconds: Time in open state before trying half-open
        half_open_max_calls: Max concurrent calls in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


@dataclass
class AdapterCircuitStats:
    """Statistics for an adapter circuit breaker."""

    adapter_name: str
    state: AdapterCircuitState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changed_at: float = field(default_factory=time.time)
    total_failures: int = 0
    total_successes: int = 0
    total_circuit_opens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_name": self.adapter_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "state_changed_at": self.state_changed_at,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_circuit_opens": self.total_circuit_opens,
        }


class AdapterCircuitBreaker:
    """
    Circuit breaker specifically designed for Knowledge Mound adapters.

    Provides per-adapter circuit breaker functionality with:
    - Configurable failure thresholds
    - Half-open state for gradual recovery
    - Metrics integration for monitoring
    - State persistence for recovery

    Usage:
        breaker = AdapterCircuitBreaker("continuum")

        async def operation():
            if not breaker.can_proceed():
                raise AdapterUnavailableError("Circuit open")
            try:
                result = await adapter.do_something()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure(str(e))
                raise
    """

    def __init__(
        self,
        adapter_name: str,
        config: AdapterCircuitBreakerConfig | None = None,
    ):
        """Initialize adapter circuit breaker.

        Args:
            adapter_name: Name of the adapter (continuum, consensus, etc.)
            config: Circuit breaker configuration
        """
        self.adapter_name = adapter_name
        self.config = config or AdapterCircuitBreakerConfig()
        self._state = AdapterCircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._state_changed_at = time.time()
        self._total_failures = 0
        self._total_successes = 0
        self._total_circuit_opens = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> AdapterCircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == AdapterCircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == AdapterCircuitState.CLOSED

    def can_proceed(self) -> bool:
        """Check if a request can proceed through the circuit.

        Returns:
            True if request is allowed, False if circuit is open
        """
        if self._state == AdapterCircuitState.CLOSED:
            return True

        if self._state == AdapterCircuitState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self._state_changed_at >= self.config.timeout_seconds:
                self._transition_to_half_open()
                return self._half_open_calls < self.config.half_open_max_calls
            return False

        if self._state == AdapterCircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        self._last_success_time = time.time()
        self._total_successes += 1

        if self._state == AdapterCircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls -= 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self._state == AdapterCircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

        self._record_metrics("success")

    def record_failure(self, error: str | None = None) -> bool:
        """Record a failed operation.

        Args:
            error: Optional error message for logging

        Returns:
            True if circuit just opened
        """
        self._last_failure_time = time.time()
        self._total_failures += 1
        circuit_opened = False

        if self._state == AdapterCircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._half_open_calls -= 1
            self._transition_to_open()
            circuit_opened = True
            logger.warning(f"Adapter circuit {self.adapter_name} reopened from half-open: {error}")
        elif self._state == AdapterCircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()
                circuit_opened = True
                logger.warning(
                    f"Adapter circuit {self.adapter_name} opened after "
                    f"{self._failure_count} failures: {error}"
                )

        self._record_metrics("failure")
        return circuit_opened

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = AdapterCircuitState.OPEN
        self._state_changed_at = time.time()
        self._total_circuit_opens += 1
        self._success_count = 0
        logger.info(f"Adapter circuit {self.adapter_name} -> OPEN")
        self._record_state_change()

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = AdapterCircuitState.HALF_OPEN
        self._state_changed_at = time.time()
        self._half_open_calls = 0
        self._success_count = 0
        logger.info(f"Adapter circuit {self.adapter_name} -> HALF_OPEN")
        self._record_state_change()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = AdapterCircuitState.CLOSED
        self._state_changed_at = time.time()
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"Adapter circuit {self.adapter_name} -> CLOSED")
        self._record_state_change()

    def reset(self) -> None:
        """Reset circuit to closed state."""
        self._state = AdapterCircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._state_changed_at = time.time()
        logger.info(f"Adapter circuit {self.adapter_name} reset to CLOSED")

    def get_stats(self) -> AdapterCircuitStats:
        """Get circuit breaker statistics."""
        return AdapterCircuitStats(
            adapter_name=self.adapter_name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            state_changed_at=self._state_changed_at,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            total_circuit_opens=self._total_circuit_opens,
        )

    def cooldown_remaining(self) -> float:
        """Get remaining time in cooldown (open state).

        Returns:
            Seconds remaining, or 0 if not in open state
        """
        if self._state != AdapterCircuitState.OPEN:
            return 0.0
        elapsed = time.time() - self._state_changed_at
        remaining = self.config.timeout_seconds - elapsed
        return max(0.0, remaining)

    def _record_metrics(self, event_type: str) -> None:
        """Record Prometheus metrics for circuit breaker events."""
        try:
            from aragora.observability.metrics.km import (
                record_km_adapter_sync,
            )

            success = event_type == "success"
            record_km_adapter_sync(self.adapter_name, "circuit_breaker", success)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to record circuit breaker metric: {e}")

    def _record_state_change(self) -> None:
        """Record Prometheus metrics for state changes."""
        try:
            # Map state to health status for logging
            state_map = {
                AdapterCircuitState.CLOSED: 3,  # healthy
                AdapterCircuitState.HALF_OPEN: 2,  # degraded
                AdapterCircuitState.OPEN: 1,  # unhealthy
            }
            logger.debug(
                f"Adapter {self.adapter_name} state: {self._state.value} "
                f"(health={state_map.get(self._state, 0)})"
            )
        except Exception as e:
            logger.debug(f"Failed to record state change metric: {e}")

    @asynccontextmanager
    async def protected_call(self) -> AsyncIterator[None]:
        """Context manager for circuit-breaker-protected async calls.

        Raises:
            AdapterUnavailableError: If circuit is open

        Usage:
            async with breaker.protected_call():
                result = await adapter.operation()
        """
        if not self.can_proceed():
            remaining = self.cooldown_remaining()
            raise AdapterUnavailableError(
                self.adapter_name,
                remaining,
                f"Circuit breaker open, retry in {remaining:.1f}s",
            )

        if self._state == AdapterCircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            yield
            self.record_success()
        except asyncio.CancelledError:
            # Don't count task cancellation as failure
            if self._state == AdapterCircuitState.HALF_OPEN:
                self._half_open_calls -= 1
            raise
        except Exception as e:
            self.record_failure(str(e))
            raise


class HealthAwareCircuitBreaker:
    """Circuit breaker that syncs state from a ConnectionHealthMonitor.

    Opens circuit automatically when the health monitor detects degradation.
    This bridges the gap between connection-level health and adapter-level
    circuit breakers.

    Usage:
        health_cb = HealthAwareCircuitBreaker("continuum", health_monitor)
        if health_cb.can_proceed():
            # Safe to call adapter
            ...
    """

    def __init__(
        self,
        adapter_name: str,
        health_monitor: ConnectionHealthMonitor,
        config: AdapterCircuitBreakerConfig | None = None,
    ):
        self.adapter_name = adapter_name
        self._health_monitor = health_monitor
        self._circuit = AdapterCircuitBreaker(adapter_name, config)

    def sync_from_health(self) -> None:
        """Sync circuit state from health monitor.

        If health monitor reports unhealthy, force circuit open.
        If healthy and circuit is open, allow half-open transition.
        """
        if not self._health_monitor.is_healthy():
            if self._circuit.is_closed:
                self._circuit._transition_to_open()
                logger.warning(
                    f"Health-aware circuit {self.adapter_name} opened due to health degradation"
                )

    def can_proceed(self) -> bool:
        """Check if request can proceed, syncing health first."""
        self.sync_from_health()
        return self._circuit.can_proceed()

    def record_success(self) -> None:
        """Record successful operation."""
        self._circuit.record_success()

    def record_failure(self, error: str | None = None) -> bool:
        """Record failed operation."""
        return self._circuit.record_failure(error)

    @property
    def state(self) -> AdapterCircuitState:
        """Get current circuit state."""
        return self._circuit.state

    def get_stats(self) -> dict[str, Any]:
        """Get combined health and circuit stats."""
        stats = self._circuit.get_stats().to_dict()
        stats["health_monitor_healthy"] = self._health_monitor.is_healthy()
        return stats


class LatencyTracker:
    """Tracks operation latencies and computes adaptive timeouts.

    Records latency samples and calculates P50/P90/P95/P99 percentiles
    for adaptive timeout configuration.

    Usage:
        tracker = LatencyTracker()
        tracker.record(150.0)  # 150ms
        tracker.record(200.0)
        timeout = tracker.get_adaptive_timeout()  # P95 * multiplier
    """

    def __init__(
        self,
        max_samples: int = 1000,
        timeout_multiplier: float = 1.5,
        min_timeout_ms: float = 100.0,
        max_timeout_ms: float = 30000.0,
    ):
        """Initialize latency tracker.

        Args:
            max_samples: Maximum number of latency samples to keep.
            timeout_multiplier: Multiplier for P95 to compute adaptive timeout.
            min_timeout_ms: Minimum adaptive timeout in milliseconds.
            max_timeout_ms: Maximum adaptive timeout in milliseconds.
        """
        self._samples: list[float] = []
        self._max_samples = max_samples
        self._timeout_multiplier = timeout_multiplier
        self._min_timeout_ms = min_timeout_ms
        self._max_timeout_ms = max_timeout_ms
        self._total_count: int = 0

    def record(self, latency_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._samples.append(latency_ms)
        self._total_count += 1
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]

    def get_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on P95 latency.

        Returns:
            Adaptive timeout in milliseconds, clamped to min/max bounds.
        """
        if not self._samples:
            return self._min_timeout_ms

        sorted_samples = sorted(self._samples)
        p95_idx = int(len(sorted_samples) * 0.95)
        p95 = sorted_samples[min(p95_idx, len(sorted_samples) - 1)]

        timeout = p95 * self._timeout_multiplier
        return max(self._min_timeout_ms, min(timeout, self._max_timeout_ms))

    def get_percentile(self, percentile: float) -> float:
        """Get a specific percentile value.

        Args:
            percentile: Percentile to compute (0.0-1.0).

        Returns:
            Latency at the given percentile in milliseconds.
        """
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * percentile)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def get_stats(self) -> dict[str, Any]:
        """Get latency statistics."""
        if not self._samples:
            return {
                "sample_count": 0,
                "total_count": self._total_count,
                "adaptive_timeout_ms": self._min_timeout_ms,
            }

        return {
            "sample_count": len(self._samples),
            "total_count": self._total_count,
            "p50_ms": self.get_percentile(0.50),
            "p90_ms": self.get_percentile(0.90),
            "p95_ms": self.get_percentile(0.95),
            "p99_ms": self.get_percentile(0.99),
            "min_ms": min(self._samples),
            "max_ms": max(self._samples),
            "adaptive_timeout_ms": self.get_adaptive_timeout(),
        }


class AdapterUnavailableError(Exception):
    """Raised when an adapter is unavailable due to circuit breaker."""

    def __init__(
        self,
        adapter_name: str,
        cooldown_remaining: float,
        message: str | None = None,
    ):
        self.adapter_name = adapter_name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            message or f"Adapter '{adapter_name}' unavailable. Retry in {cooldown_remaining:.1f}s"
        )


# Global registry of adapter circuit breakers
_adapter_circuits: dict[str, AdapterCircuitBreaker] = {}


def get_adapter_circuit_breaker(
    adapter_name: str,
    config: AdapterCircuitBreakerConfig | None = None,
) -> AdapterCircuitBreaker:
    """Get or create a circuit breaker for an adapter.

    Args:
        adapter_name: Name of the adapter
        config: Optional configuration (only used if creating new)

    Returns:
        AdapterCircuitBreaker instance
    """
    if adapter_name not in _adapter_circuits:
        _adapter_circuits[adapter_name] = AdapterCircuitBreaker(adapter_name, config)
    return _adapter_circuits[adapter_name]


def get_all_adapter_circuit_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all adapter circuit breakers.

    Returns:
        Dict mapping adapter names to their stats
    """
    return {name: cb.get_stats().to_dict() for name, cb in _adapter_circuits.items()}


def reset_adapter_circuit_breaker(adapter_name: str) -> bool:
    """Reset a specific adapter's circuit breaker.

    Args:
        adapter_name: Name of the adapter

    Returns:
        True if reset, False if adapter not found
    """
    if adapter_name in _adapter_circuits:
        _adapter_circuits[adapter_name].reset()
        return True
    return False


def reset_all_adapter_circuit_breakers() -> int:
    """Reset all adapter circuit breakers.

    Returns:
        Number of circuit breakers reset
    """
    count = 0
    for cb in _adapter_circuits.values():
        cb.reset()
        count += 1
    return count
