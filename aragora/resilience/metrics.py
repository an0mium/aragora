"""
Circuit breaker metrics and status functions.

Provides monitoring and observability support for circuit breakers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Metrics callback - set by prometheus module on import
_metrics_callback: Callable[[str, int], None] | None = None


def set_metrics_callback(callback: Callable[[str, int], None] | None) -> None:
    """Set the callback for circuit breaker state changes.

    The callback receives (circuit_name, state) where state is:
    - 0 = closed
    - 1 = open
    - 2 = half-open

    This is called by the prometheus module to wire up metrics.
    """
    global _metrics_callback
    _metrics_callback = callback
    if callback:
        logger.debug("Circuit breaker metrics callback registered")


def emit_metrics(circuit_name: str, state: int) -> None:
    """Emit metrics for circuit state change."""
    if _metrics_callback:
        try:
            _metrics_callback(circuit_name, state)
        except Exception as e:  # noqa: BLE001 - metrics emission must never break callers
            logger.debug("Error emitting circuit breaker metrics: %s", e)


def get_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all registered circuit breakers (thread-safe)."""
    from .registry import _circuit_breakers, _circuit_breakers_lock

    with _circuit_breakers_lock:
        return {
            "_registry_size": len(_circuit_breakers),
            **{
                name: {
                    "status": cb.get_status(),
                    "failures": cb.failures,
                    "last_accessed": getattr(cb, "_last_accessed", 0),
                }
                for name, cb in _circuit_breakers.items()
            },
        }


def get_circuit_breaker_metrics() -> dict[str, Any]:
    """Get comprehensive metrics for monitoring and observability.

    Returns metrics suitable for Prometheus/Grafana or other monitoring systems:
    - Summary counts (total, open, closed, half-open)
    - Per-circuit-breaker details with timing info
    - Configuration details
    - Health indicators for cascading failure detection

    Returns:
        Dict with structured metrics for monitoring integration.
    """
    from .registry import _circuit_breakers, _circuit_breakers_lock

    now = time.time()
    with _circuit_breakers_lock:
        metrics: dict[str, Any] = {
            "timestamp": now,
            "registry_size": len(_circuit_breakers),
            "summary": {
                "total": 0,
                "open": 0,
                "closed": 0,
                "half_open": 0,
                "total_failures": 0,
                "circuits_with_failures": 0,
            },
            "circuit_breakers": {},
            "health": {
                "status": "healthy",
                "open_circuits": [],
                "high_failure_circuits": [],
            },
        }

        for name, cb in _circuit_breakers.items():
            status = cb.get_status()
            failures = cb.failures
            last_accessed = getattr(cb, "_last_accessed", 0)
            age_seconds = now - last_accessed if last_accessed > 0 else 0

            # Calculate cooldown remaining if open
            cooldown_remaining = 0.0
            open_duration = 0.0
            if status == "open" or status == "half-open":
                if cb._single_open_at > 0:
                    open_duration = now - cb._single_open_at
                    cooldown_remaining = max(0, cb.cooldown_seconds - open_duration)

            circuit_metrics = {
                "status": status,
                "failures": failures,
                "failure_threshold": cb.failure_threshold,
                "cooldown_seconds": cb.cooldown_seconds,
                "cooldown_remaining": cooldown_remaining,
                "open_duration": open_duration,
                "last_accessed_seconds_ago": age_seconds,
                "entity_mode": {
                    "tracked_entities": len(cb._failures),
                    "open_entities": list(cb._circuit_open_at.keys()),
                },
            }
            metrics["circuit_breakers"][name] = circuit_metrics

            # Update summary
            metrics["summary"]["total"] += 1
            metrics["summary"]["total_failures"] += failures
            if failures > 0:
                metrics["summary"]["circuits_with_failures"] += 1

            if status == "open":
                metrics["summary"]["open"] += 1
                metrics["health"]["open_circuits"].append(name)
            elif status == "half-open":
                metrics["summary"]["half_open"] += 1
            else:
                metrics["summary"]["closed"] += 1

            # Flag high-failure circuits (>50% of threshold)
            if failures >= cb.failure_threshold * 0.5:
                metrics["health"]["high_failure_circuits"].append(
                    {
                        "name": name,
                        "failures": failures,
                        "threshold": cb.failure_threshold,
                        "percentage": round(failures / cb.failure_threshold * 100, 1),
                    }
                )

        # Determine overall health status
        if metrics["summary"]["open"] > 0:
            metrics["health"]["status"] = "degraded"
        if metrics["summary"]["open"] >= 3:
            metrics["health"]["status"] = "critical"

        return metrics


def get_all_circuit_breakers_status() -> dict[str, Any]:
    """Get status of all registered circuit breakers.

    Returns a dict with overall health and per-circuit-breaker details
    suitable for HTTP API response.

    Returns:
        Dict with structure:
        {
            "healthy": bool,  # True if no circuits are open
            "total_circuits": int,
            "open_circuits": int,
            "half_open_circuits": int,
            "closed_circuits": int,
            "circuits": {
                "circuit_name": {
                    "status": "closed" | "open" | "half-open",
                    "failures": int,
                    "config": {...},
                    ...
                }
            }
        }
    """
    from .registry import _circuit_breakers, _circuit_breakers_lock

    with _circuit_breakers_lock:
        circuits = {}
        open_count = 0
        half_open_count = 0
        closed_count = 0

        for name, cb in _circuit_breakers.items():
            status = cb.get_status()
            circuit_info = cb.to_dict()
            circuit_info["status"] = status
            circuit_info["name"] = name
            circuits[name] = circuit_info

            if status == "open":
                open_count += 1
            elif status == "half-open":
                half_open_count += 1
            else:
                closed_count += 1

        return {
            "healthy": open_count == 0,
            "total_circuits": len(circuits),
            "open_circuits": open_count,
            "half_open_circuits": half_open_count,
            "closed_circuits": closed_count,
            "circuits": circuits,
        }


def get_circuit_breaker_summary() -> dict[str, Any]:
    """Get a lightweight summary of circuit breaker health.

    Returns:
        Dict with structure:
        {
            "healthy": bool,
            "total": int,
            "open": list[str],  # Names of open circuits
            "half_open": list[str],  # Names of half-open circuits
        }
    """
    from .registry import _circuit_breakers, _circuit_breakers_lock

    with _circuit_breakers_lock:
        open_names = []
        half_open_names = []

        for name, cb in _circuit_breakers.items():
            status = cb.get_status()
            if status == "open":
                open_names.append(name)
            elif status == "half-open":
                half_open_names.append(name)

        return {
            "healthy": len(open_names) == 0,
            "total": len(_circuit_breakers),
            "open": open_names,
            "half_open": half_open_names,
        }


__all__ = [
    "set_metrics_callback",
    "emit_metrics",
    "get_circuit_breaker_status",
    "get_circuit_breaker_metrics",
    "get_all_circuit_breakers_status",
    "get_circuit_breaker_summary",
]
