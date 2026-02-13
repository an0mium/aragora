"""
System and infrastructure metrics.

Provides Prometheus metrics for tracking WebSocket connections,
server health, and core infrastructure components.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
WEBSOCKET_CONNECTIONS: Any = None

_initialized = False


def init_system_metrics() -> None:
    """Initialize system and infrastructure metrics."""
    global _initialized
    global WEBSOCKET_CONNECTIONS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Gauge

        WEBSOCKET_CONNECTIONS = Gauge(
            "aragora_websocket_connections",
            "Number of active WebSocket connections",
        )

        _initialized = True
        logger.debug("System metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global WEBSOCKET_CONNECTIONS

    WEBSOCKET_CONNECTIONS = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_system_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def track_websocket_connection(increment: bool = True) -> None:
    """Track WebSocket connections (increment or decrement).

    Args:
        increment: True to increment, False to decrement
    """
    _ensure_init()
    if increment:
        WEBSOCKET_CONNECTIONS.inc()
    else:
        WEBSOCKET_CONNECTIONS.dec()


def set_websocket_connections(count: int) -> None:
    """Set the number of active WebSocket connections.

    Args:
        count: Number of active connections
    """
    _ensure_init()
    WEBSOCKET_CONNECTIONS.set(count)


def increment_websocket_connections() -> None:
    """Increment WebSocket connections count."""
    _ensure_init()
    WEBSOCKET_CONNECTIONS.inc()


def decrement_websocket_connections() -> None:
    """Decrement WebSocket connections count."""
    _ensure_init()
    WEBSOCKET_CONNECTIONS.dec()


__all__ = [
    # Metrics
    "WEBSOCKET_CONNECTIONS",
    # Functions
    "init_system_metrics",
    "track_websocket_connection",
    "set_websocket_connections",
    "increment_websocket_connections",
    "decrement_websocket_connections",
]
