"""
Cross-subscriber management and administration.

Extracted from manager.py for maintainability.
Contains stats reporting, subscriber enable/disable, circuit breaker management,
sampling, filtering, retry configuration, and performance reporting.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

from aragora.events.subscribers.config import RetryConfig, SubscriberStats
from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)

# Import metrics (optional - graceful fallback)
try:
    from aragora.server.prometheus_cross_pollination import (
        set_circuit_breaker_state as _set_circuit_breaker_state,
    )

    _METRICS_AVAILABLE = True

    def _set_cb_state(handler: str, is_open: bool) -> None:
        _set_circuit_breaker_state(handler, is_open)

except ImportError:
    _METRICS_AVAILABLE = False

    def _set_cb_state(handler: str, is_open: bool) -> None:
        pass


class AdminMixin:
    """Mixin providing management and administration methods.

    Expects the consuming class to define:
    - self._subscribers: dict
    - self._stats: dict[str, SubscriberStats]
    - self._filters: dict[str, Callable]
    - self._circuit_breaker: CircuitBreaker
    - self._default_retry_config: RetryConfig
    """

    # Mixin attribute declarations (provided by the composed class)
    _subscribers: dict
    _stats: dict[str, SubscriberStats]
    _filters: dict[str, Callable[[StreamEvent], bool]]
    _circuit_breaker: Any
    _default_retry_config: RetryConfig

    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all subscribers including latency, sampling, retry, and circuit breaker metrics."""
        result = {}
        for name, stats in self._stats.items():
            # Get circuit breaker status for this handler
            cb_state = (
                self._circuit_breaker.get_status(name)
                if hasattr(self._circuit_breaker, "get_status")
                else "unknown"
            )
            cb_available = self._circuit_breaker.is_available(name)

            # Get retry config
            retry_cfg = stats.retry_config or self._default_retry_config

            result[name] = {
                "events_processed": stats.events_processed,
                "events_failed": stats.events_failed,
                "events_skipped": stats.events_skipped,
                "events_retried": stats.events_retried,
                "last_event": (
                    stats.last_event_time.isoformat() if stats.last_event_time else None
                ),
                "enabled": stats.enabled,
                "sample_rate": stats.sample_rate,
                "has_filter": name in self._filters,
                "latency_ms": {
                    "avg": round(stats.avg_latency_ms, 3),
                    "min": (
                        round(stats.min_latency_ms, 3)
                        if stats.min_latency_ms != float("inf")
                        else None
                    ),
                    "max": round(stats.max_latency_ms, 3),
                    "total": round(stats.total_latency_ms, 3),
                    "p50": (
                        round(stats.p50_latency_ms, 3) if stats.p50_latency_ms is not None else None
                    ),
                    "p90": (
                        round(stats.p90_latency_ms, 3) if stats.p90_latency_ms is not None else None
                    ),
                    "p99": (
                        round(stats.p99_latency_ms, 3) if stats.p99_latency_ms is not None else None
                    ),
                    "sample_count": len(stats.latency_samples),
                },
                "retry": {
                    "max_retries": retry_cfg.max_retries,
                    "base_delay_ms": retry_cfg.base_delay_ms,
                    "max_delay_ms": retry_cfg.max_delay_ms,
                },
                "circuit_breaker": {
                    "state": cb_state,
                    "available": cb_available,
                },
            }
        return result

    def enable_subscriber(self, name: str) -> bool:
        """Enable a subscriber by name."""
        if name in self._stats:
            self._stats[name].enabled = True
            return True
        return False

    def disable_subscriber(self, name: str) -> bool:
        """Disable a subscriber by name."""
        if name in self._stats:
            self._stats[name].enabled = False
            return True
        return False

    def reset_stats(self) -> None:
        """Reset all subscriber statistics."""
        for stats in self._stats.values():
            stats.events_processed = 0
            stats.events_failed = 0
            stats.events_skipped = 0
            stats.events_retried = 0
            stats.last_event_time = None
            stats.total_latency_ms = 0.0
            stats.min_latency_ms = float("inf")
            stats.max_latency_ms = 0.0
            stats.latency_samples.clear()

    def reset_circuit_breaker(self, name: str) -> bool:
        """Reset circuit breaker for a specific handler.

        Args:
            name: Handler name

        Returns:
            True if reset successful
        """
        try:
            self._circuit_breaker.reset(name)
            if _METRICS_AVAILABLE:
                _set_cb_state(name, False)  # Closed = not open
            return True
        except (RuntimeError, TypeError, AttributeError, ValueError, KeyError) as e:
            logger.debug("Circuit breaker reset failed for '%s': %s: %s", name, type(e).__name__, e)
            return False

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for name in self._stats:
            self.reset_circuit_breaker(name)

    def set_sample_rate(self, name: str, rate: float) -> bool:
        """Set sampling rate for a subscriber.

        Args:
            name: Subscriber name
            rate: Sample rate (0.0 to 1.0)

        Returns:
            True if set successfully
        """
        if name not in self._stats:
            return False
        if not 0.0 <= rate <= 1.0:
            return False
        self._stats[name].sample_rate = rate
        return True

    def set_filter(
        self,
        name: str,
        filter_func: Callable[[StreamEvent], bool],
    ) -> bool:
        """Set a filter function for a subscriber.

        Args:
            name: Subscriber name
            filter_func: Function that returns True if event should be processed

        Returns:
            True if set successfully
        """
        if name not in self._stats:
            return False
        self._filters[name] = filter_func
        return True

    def get_filter(self, name: str) -> Callable[[StreamEvent], bool] | None:
        """Get the filter function for a subscriber."""
        return self._filters.get(name)

    def set_retry_config(
        self,
        name: str,
        max_retries: int | None = None,
        base_delay_ms: int | None = None,
        max_delay_ms: int | None = None,
    ) -> bool:
        """Set retry configuration for a specific subscriber.

        Args:
            name: Subscriber name
            max_retries: Maximum retry attempts
            base_delay_ms: Base delay between retries
            max_delay_ms: Maximum delay between retries

        Returns:
            True if configuration was set
        """
        if name not in self._stats:
            return False

        stats = self._stats[name]

        # Create or update retry config
        if stats.retry_config is None:
            stats.retry_config = RetryConfig()

        if max_retries is not None:
            stats.retry_config.max_retries = max_retries
        if base_delay_ms is not None:
            stats.retry_config.base_delay_ms = base_delay_ms
        if max_delay_ms is not None:
            stats.retry_config.max_delay_ms = max_delay_ms

        return True

    def disable_retry(self, name: str) -> bool:
        """Disable retries for a specific subscriber.

        Args:
            name: Subscriber name

        Returns:
            True if disabled successfully
        """
        if name not in self._stats:
            return False

        self._stats[name].retry_config = RetryConfig(max_retries=0)
        return True

    def get_performance_report(self) -> dict:
        """Get a performance report summarizing all handlers.

        Returns:
            Dict with summary statistics
        """
        stats = self.get_stats()

        # Calculate totals
        total_processed = sum(s.get("events_processed", 0) for s in stats.values())
        total_failed = sum(s.get("events_failed", 0) for s in stats.values())
        total_skipped = sum(s.get("events_skipped", 0) for s in stats.values())
        total_retried = sum(s.get("events_retried", 0) for s in stats.values())

        # Find slowest handlers by P90 latency
        handlers_by_p90 = []
        for name, s in stats.items():
            p90 = s.get("latency_ms", {}).get("p90")
            if p90 is not None:
                handlers_by_p90.append((name, p90))
        handlers_by_p90.sort(key=lambda x: x[1], reverse=True)

        # Find handlers with highest error rates
        handlers_by_error_rate = []
        for name, s in stats.items():
            processed = s.get("events_processed", 0)
            failed = s.get("events_failed", 0)
            if processed > 0:
                error_rate = failed / (processed + failed)
                handlers_by_error_rate.append((name, error_rate))
        handlers_by_error_rate.sort(key=lambda x: x[1], reverse=True)

        # Circuit breaker summary
        circuits_open = sum(
            1 for s in stats.values() if not s.get("circuit_breaker", {}).get("available", True)
        )

        return {
            "summary": {
                "total_handlers": len(stats),
                "total_events_processed": total_processed,
                "total_events_failed": total_failed,
                "total_events_skipped": total_skipped,
                "total_events_retried": total_retried,
                "overall_error_rate": round(
                    total_failed / max(total_processed + total_failed, 1), 4
                ),
                "circuits_open": circuits_open,
            },
            "slowest_handlers": [
                {"name": name, "p90_latency_ms": lat} for name, lat in handlers_by_p90[:5]
            ],
            "highest_error_handlers": [
                {"name": name, "error_rate": round(rate, 4)}
                for name, rate in handlers_by_error_rate[:5]
                if rate > 0
            ],
            "per_handler": stats,
        }
