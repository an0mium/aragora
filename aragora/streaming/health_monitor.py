"""
Stream health monitor for production observability.

Provides ``StreamHealthMonitor``, a singleton that tracks active WebSocket
connections, error rates, reconnection counts, and message delivery rates
per debate.  Exposes data for the ``/api/v1/stream/health`` endpoint and
alerts on SLO violations.

Usage::

    from aragora.streaming.health_monitor import get_stream_health_monitor

    monitor = get_stream_health_monitor()
    monitor.record_connection("debate-1", "client-abc")
    monitor.record_error("debate-1", "send_timeout")
    monitor.record_message_delivered("debate-1")

    health = monitor.get_health()
    # {"status": "healthy", "active_connections": 5, ...}
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Defaults
_HEALTH_CHECK_INTERVAL = 30.0  # seconds
_ERROR_RATE_WINDOW = 300.0  # 5 minutes


@dataclass
class _DebateHealth:
    """Per-debate health tracking."""

    active_connections: int = 0
    total_connections: int = 0
    total_disconnections: int = 0
    reconnect_count: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    error_timestamps: list[float] = field(default_factory=list)
    last_error_type: str = ""
    last_activity_at: float = 0.0


@dataclass
class StreamHealthSnapshot:
    """Point-in-time snapshot of stream health.

    Suitable for serialization to JSON for the health endpoint.
    """

    status: str  # "healthy", "degraded", "unhealthy"
    active_connections: int
    active_debates: int
    total_messages_delivered: int
    total_messages_failed: int
    total_reconnects: int
    error_rate_5m: float  # errors per minute over 5min window
    message_delivery_rate: float  # ratio of delivered / (delivered + failed)
    slo_violations: list[dict[str, Any]]
    debates: dict[str, dict[str, Any]]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "status": self.status,
            "active_connections": self.active_connections,
            "active_debates": self.active_debates,
            "total_messages_delivered": self.total_messages_delivered,
            "total_messages_failed": self.total_messages_failed,
            "total_reconnects": self.total_reconnects,
            "error_rate_5m": round(self.error_rate_5m, 4),
            "message_delivery_rate": round(self.message_delivery_rate, 6),
            "slo_violations": self.slo_violations,
            "debates": self.debates,
            "timestamp": self.timestamp,
        }


class StreamHealthMonitor:
    """Singleton stream health monitor.

    Tracks connection, error, and delivery metrics per debate.
    Checks SLO compliance and emits warnings on violations.

    Thread-safe via ``threading.Lock``.
    """

    _instance: StreamHealthMonitor | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        *,
        error_rate_window: float = _ERROR_RATE_WINDOW,
        health_check_interval: float = _HEALTH_CHECK_INTERVAL,
    ) -> None:
        self._debates: dict[str, _DebateHealth] = {}
        self._lock = threading.Lock()
        self._error_rate_window = error_rate_window
        self._health_check_interval = health_check_interval
        self._started_at = time.monotonic()
        self._last_health_check = 0.0
        self._slo_violations: list[dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> StreamHealthMonitor:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = StreamHealthMonitor()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_connection(self, debate_id: str, client_id: str = "") -> None:
        """Record a new client connection for a debate."""
        with self._lock:
            dh = self._get_or_create(debate_id)
            dh.active_connections += 1
            dh.total_connections += 1
            dh.last_activity_at = time.monotonic()

    def record_disconnection(self, debate_id: str, client_id: str = "") -> None:
        """Record a client disconnection."""
        with self._lock:
            dh = self._debates.get(debate_id)
            if dh:
                dh.active_connections = max(0, dh.active_connections - 1)
                dh.total_disconnections += 1
                dh.last_activity_at = time.monotonic()

    def record_reconnect(self, debate_id: str, client_id: str = "") -> None:
        """Record a client reconnection."""
        with self._lock:
            dh = self._get_or_create(debate_id)
            dh.reconnect_count += 1
            dh.last_activity_at = time.monotonic()

    def record_message_delivered(self, debate_id: str, count: int = 1) -> None:
        """Record successfully delivered messages."""
        with self._lock:
            dh = self._get_or_create(debate_id)
            dh.messages_delivered += count
            dh.last_activity_at = time.monotonic()

    def record_message_failed(self, debate_id: str, count: int = 1) -> None:
        """Record failed message deliveries."""
        with self._lock:
            dh = self._get_or_create(debate_id)
            dh.messages_failed += count
            dh.last_activity_at = time.monotonic()

    def record_error(self, debate_id: str, error_type: str = "unknown") -> None:
        """Record a stream error."""
        now = time.monotonic()
        with self._lock:
            dh = self._get_or_create(debate_id)
            dh.error_timestamps.append(now)
            dh.last_error_type = error_type
            dh.last_activity_at = now

            # Prune old error timestamps
            cutoff = now - self._error_rate_window
            dh.error_timestamps = [
                ts for ts in dh.error_timestamps if ts > cutoff
            ]

    def remove_debate(self, debate_id: str) -> None:
        """Remove tracking for a finished debate."""
        with self._lock:
            self._debates.pop(debate_id, None)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def get_health(self) -> StreamHealthSnapshot:
        """Compute and return a health snapshot.

        Also runs SLO checks and emits warnings if necessary.
        """
        now = time.monotonic()

        with self._lock:
            total_conns = sum(dh.active_connections for dh in self._debates.values())
            total_delivered = sum(dh.messages_delivered for dh in self._debates.values())
            total_failed = sum(dh.messages_failed for dh in self._debates.values())
            total_reconnects = sum(dh.reconnect_count for dh in self._debates.values())

            # Calculate error rate over window
            cutoff = now - self._error_rate_window
            total_errors = 0
            for dh in self._debates.values():
                total_errors += len(
                    [ts for ts in dh.error_timestamps if ts > cutoff]
                )

            # Error rate per minute
            window_minutes = self._error_rate_window / 60.0
            error_rate = total_errors / window_minutes if window_minutes > 0 else 0.0

            # Message delivery rate
            total_attempted = total_delivered + total_failed
            delivery_rate = (
                total_delivered / total_attempted if total_attempted > 0 else 1.0
            )

            # Per-debate summary
            debate_summaries: dict[str, dict[str, Any]] = {}
            for did, dh in self._debates.items():
                d_attempted = dh.messages_delivered + dh.messages_failed
                d_rate = (
                    dh.messages_delivered / d_attempted if d_attempted > 0 else 1.0
                )
                debate_summaries[did] = {
                    "active_connections": dh.active_connections,
                    "messages_delivered": dh.messages_delivered,
                    "messages_failed": dh.messages_failed,
                    "delivery_rate": round(d_rate, 6),
                    "reconnect_count": dh.reconnect_count,
                    "last_error_type": dh.last_error_type,
                }

        # Check SLOs
        violations = self._check_slos(
            error_rate=error_rate,
            delivery_rate=delivery_rate,
            total_reconnects=total_reconnects,
        )

        # Determine overall status
        if violations:
            status = "degraded"
            # Critical if delivery rate below 95%
            if delivery_rate < 0.95:
                status = "unhealthy"
        else:
            status = "healthy"

        self._last_health_check = now

        return StreamHealthSnapshot(
            status=status,
            active_connections=total_conns,
            active_debates=len(debate_summaries),
            total_messages_delivered=total_delivered,
            total_messages_failed=total_failed,
            total_reconnects=total_reconnects,
            error_rate_5m=error_rate,
            message_delivery_rate=delivery_rate,
            slo_violations=violations,
            debates=debate_summaries,
            timestamp=time.time(),
        )

    def should_run_health_check(self) -> bool:
        """Whether enough time has elapsed for the next periodic health check."""
        return (
            time.monotonic() - self._last_health_check
            >= self._health_check_interval
        )

    # ------------------------------------------------------------------
    # SLO checks
    # ------------------------------------------------------------------

    def _check_slos(
        self,
        error_rate: float,
        delivery_rate: float,
        total_reconnects: int,
    ) -> list[dict[str, Any]]:
        """Check streaming SLOs and return a list of violations."""
        violations: list[dict[str, Any]] = []

        # stream_error_rate: target <= 0.5% per minute
        # Convert our errors/minute to a percentage rate approximation
        # (We use a simple threshold: >0.5 errors/min over the window)
        if error_rate > 0.5:
            v = {
                "slo": "stream_error_rate",
                "target": "<=0.5%",
                "current": round(error_rate, 4),
                "severity": "warning",
            }
            violations.append(v)
            logger.warning(
                "[StreamHealth] SLO violation: stream_error_rate=%.4f (target <=0.5%%)",
                error_rate,
            )

        # stream_message_delivery_rate: target >= 99.5%
        if delivery_rate < 0.995:
            severity = "warning" if delivery_rate >= 0.99 else "critical"
            v = {
                "slo": "stream_message_delivery_rate",
                "target": ">=99.5%",
                "current": round(delivery_rate, 6),
                "severity": severity,
            }
            violations.append(v)
            logger.warning(
                "[StreamHealth] SLO violation: delivery_rate=%.4f (target >=99.5%%)",
                delivery_rate,
            )

        self._slo_violations = violations
        self._emit_slo_metrics(violations)
        return violations

    def _emit_slo_metrics(self, violations: list[dict[str, Any]]) -> None:
        """Emit Prometheus metrics for SLO status."""
        try:
            from aragora.observability.metrics.base import (
                get_or_create_gauge,
            )

            gauge = get_or_create_gauge(
                "aragora_stream_slo_violations",
                "Number of active stream SLO violations",
                [],
            )
            gauge.set(len(violations))
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass  # Prometheus not available

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, debate_id: str) -> _DebateHealth:
        """Get or create debate health entry (caller must hold lock)."""
        if debate_id not in self._debates:
            self._debates[debate_id] = _DebateHealth(
                last_activity_at=time.monotonic()
            )
        return self._debates[debate_id]


def get_stream_health_monitor() -> StreamHealthMonitor:
    """Return the global stream health monitor singleton."""
    return StreamHealthMonitor.get_instance()


__all__ = [
    "StreamHealthMonitor",
    "StreamHealthSnapshot",
    "get_stream_health_monitor",
]
