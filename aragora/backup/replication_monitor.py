"""
Replication Health Monitor - Monitor replication lag and status.

Provides health monitoring for database replication:
- Monitor replication lag between primary and standby
- Track last successful replication timestamp
- Alert on replication lag exceeding thresholds
- Expose Prometheus metrics (replication_lag_seconds, replication_status)

Usage:
    from aragora.backup.replication_monitor import (
        ReplicationHealthMonitor,
        ReplicationStatus,
        get_replication_monitor,
    )

    # Get the global monitor
    monitor = get_replication_monitor()

    # Record a replication event
    monitor.record_replication(lag_seconds=0.5)

    # Check health status
    health = monitor.get_health()
    if not health.healthy:
        alert(f"Replication unhealthy: {health.error}")

SOC 2 Compliance: CC9.1, CC9.2 (Business Continuity)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class ReplicationStatus(str, Enum):
    """Status of replication health."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Lag above warning threshold but below critical
    CRITICAL = "critical"  # Lag above critical threshold
    UNKNOWN = "unknown"  # No replication data available
    STOPPED = "stopped"  # Replication has stopped


@dataclass
class ReplicationHealth:
    """Health status of replication."""

    status: ReplicationStatus
    healthy: bool
    lag_seconds: float | None = None
    last_replication_at: datetime | None = None
    standby_connected: bool = False
    primary_host: str | None = None
    standby_host: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "healthy": self.healthy,
            "lag_seconds": self.lag_seconds,
            "last_replication_at": (
                self.last_replication_at.isoformat() if self.last_replication_at else None
            ),
            "standby_connected": self.standby_connected,
            "primary_host": self.primary_host,
            "standby_host": self.standby_host,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ReplicationConfig:
    """Configuration for replication monitoring."""

    # Thresholds
    warning_lag_seconds: float = 10.0  # Warning when lag > this
    critical_lag_seconds: float = 60.0  # Critical when lag > this
    max_time_since_replication_seconds: float = 300.0  # Alert if no replication in 5 min

    # Monitoring intervals
    check_interval_seconds: float = 10.0  # How often to check replication status
    metrics_interval_seconds: float = 5.0  # How often to update metrics

    # Alerting
    enable_alerts: bool = True
    alert_cooldown_seconds: float = 300.0  # Don't repeat same alert within this time


@dataclass
class ReplicationMetrics:
    """Snapshot of replication metrics."""

    lag_seconds: float | None = None
    lag_bytes: int | None = None
    replication_rate_bytes_per_sec: float | None = None
    last_replication_at: datetime | None = None
    total_replications: int = 0
    failed_replications: int = 0
    average_lag_seconds: float | None = None


# Type alias for alert callback
AlertCallback = Callable[[str, ReplicationHealth], None]


# =============================================================================
# Prometheus Metrics
# =============================================================================

# Prometheus metrics - initialized lazily
_metrics_initialized = False

REPLICATION_LAG_SECONDS: Any = None
REPLICATION_STATUS: Any = None
REPLICATION_LAST_TIMESTAMP: Any = None
REPLICATION_TOTAL: Any = None
REPLICATION_FAILURES: Any = None
REPLICATION_RATE_BYTES: Any = None


def _init_replication_metrics() -> None:
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized
    global REPLICATION_LAG_SECONDS, REPLICATION_STATUS
    global REPLICATION_LAST_TIMESTAMP, REPLICATION_TOTAL
    global REPLICATION_FAILURES, REPLICATION_RATE_BYTES

    if _metrics_initialized:
        return

    try:
        from prometheus_client import Counter, Gauge

        REPLICATION_LAG_SECONDS = Gauge(
            "aragora_replication_lag_seconds",
            "Current replication lag in seconds",
            ["primary", "standby"],
        )

        REPLICATION_STATUS = Gauge(
            "aragora_replication_status",
            "Replication status (1=healthy, 0.5=degraded, 0=critical/stopped)",
            ["primary", "standby"],
        )

        REPLICATION_LAST_TIMESTAMP = Gauge(
            "aragora_replication_last_timestamp",
            "Unix timestamp of last successful replication",
            ["primary", "standby"],
        )

        REPLICATION_TOTAL = Counter(
            "aragora_replication_total",
            "Total number of replication events",
            ["primary", "standby", "status"],
        )

        REPLICATION_FAILURES = Counter(
            "aragora_replication_failures_total",
            "Total number of replication failures",
            ["primary", "standby", "error_type"],
        )

        REPLICATION_RATE_BYTES = Gauge(
            "aragora_replication_rate_bytes_per_second",
            "Current replication rate in bytes per second",
            ["primary", "standby"],
        )

        _metrics_initialized = True
        logger.info("Replication monitoring metrics initialized")

    except ImportError:
        logger.warning(
            "prometheus_client not installed, replication metrics disabled. "
            "Install with: pip install prometheus-client"
        )


# =============================================================================
# Replication Health Monitor
# =============================================================================


class ReplicationHealthMonitor:
    """
    Monitor replication health between primary and standby.

    Features:
    - Track replication lag in real-time
    - Alert when lag exceeds thresholds
    - Expose Prometheus metrics
    - Health check endpoint support
    """

    def __init__(
        self,
        config: ReplicationConfig | None = None,
        primary_host: str = "primary",
        standby_host: str = "standby",
        alert_callback: AlertCallback | None = None,
    ) -> None:
        """
        Initialize the replication health monitor.

        Args:
            config: Monitoring configuration
            primary_host: Identifier for primary host
            standby_host: Identifier for standby host
            alert_callback: Optional callback for alerts
        """
        self._config = config or ReplicationConfig()
        self._primary_host = primary_host
        self._standby_host = standby_host
        self._alert_callback = alert_callback

        # State
        self._current_lag: float | None = None
        self._current_lag_bytes: int | None = None
        self._last_replication_at: datetime | None = None
        self._standby_connected: bool = False
        self._replication_rate: float | None = None

        # Statistics
        self._total_replications: int = 0
        self._failed_replications: int = 0
        self._lag_history: list[tuple[float, float]] = []  # (timestamp, lag)
        self._max_history_size: int = 1000

        # Alerting
        self._last_alert_time: dict[str, float] = {}  # alert_type -> timestamp
        self._current_status: ReplicationStatus = ReplicationStatus.UNKNOWN

        # Monitoring task
        self._monitoring_task: asyncio.Task | None = None
        self._running: bool = False

        # Initialize metrics
        _init_replication_metrics()

    @property
    def primary_host(self) -> str:
        """Get primary host identifier."""
        return self._primary_host

    @property
    def standby_host(self) -> str:
        """Get standby host identifier."""
        return self._standby_host

    @property
    def config(self) -> ReplicationConfig:
        """Get current configuration."""
        return self._config

    def set_alert_callback(self, callback: AlertCallback) -> None:
        """Set the alert callback."""
        self._alert_callback = callback

    def record_replication(
        self,
        lag_seconds: float,
        lag_bytes: int | None = None,
        success: bool = True,
        error_type: str | None = None,
    ) -> None:
        """
        Record a replication event.

        Args:
            lag_seconds: Current replication lag in seconds
            lag_bytes: Optional lag in bytes
            success: Whether replication was successful
            error_type: Type of error if not successful
        """
        now = time.time()

        if success:
            self._current_lag = lag_seconds
            self._current_lag_bytes = lag_bytes
            self._last_replication_at = datetime.now(timezone.utc)
            self._standby_connected = True
            self._total_replications += 1

            # Calculate replication rate if we have lag bytes
            if lag_bytes is not None and self._lag_history:
                prev_time, _ = self._lag_history[-1]
                time_diff = now - prev_time
                if time_diff > 0:
                    self._replication_rate = lag_bytes / time_diff

            # Record in history
            self._lag_history.append((now, lag_seconds))
            if len(self._lag_history) > self._max_history_size:
                self._lag_history = self._lag_history[-self._max_history_size :]

            # Update status based on lag
            self._update_status()

            # Update metrics
            self._update_metrics()

        else:
            self._failed_replications += 1
            self._record_failure_metric(error_type or "unknown")

        logger.debug(
            "Replication recorded: lag=%ss, success=%s, status=%s",
            lag_seconds,
            success,
            self._current_status.value,
        )

    def record_standby_disconnect(self) -> None:
        """Record that the standby has disconnected."""
        self._standby_connected = False
        self._current_status = ReplicationStatus.STOPPED
        self._update_metrics()

        if self._config.enable_alerts:
            self._trigger_alert("standby_disconnected")

    def record_standby_connect(self) -> None:
        """Record that the standby has connected."""
        self._standby_connected = True
        self._current_status = ReplicationStatus.UNKNOWN
        self._update_metrics()

        logger.info("Standby %s connected to primary %s", self._standby_host, self._primary_host)

    def _update_status(self) -> None:
        """Update the replication status based on current lag."""
        if self._current_lag is None:
            self._current_status = ReplicationStatus.UNKNOWN
        elif self._current_lag >= self._config.critical_lag_seconds:
            prev_status = self._current_status
            self._current_status = ReplicationStatus.CRITICAL
            if prev_status != ReplicationStatus.CRITICAL:
                self._trigger_alert("critical_lag")
        elif self._current_lag >= self._config.warning_lag_seconds:
            prev_status = self._current_status
            self._current_status = ReplicationStatus.DEGRADED
            if prev_status == ReplicationStatus.HEALTHY:
                self._trigger_alert("warning_lag")
        else:
            self._current_status = ReplicationStatus.HEALTHY

    def _trigger_alert(self, alert_type: str) -> None:
        """Trigger an alert if cooldown has passed."""
        if not self._config.enable_alerts:
            return

        now = time.time()
        last_alert = self._last_alert_time.get(alert_type, 0)

        if now - last_alert < self._config.alert_cooldown_seconds:
            return

        self._last_alert_time[alert_type] = now
        health = self.get_health()

        logger.warning(
            "Replication alert: %s - lag=%ss, status=%s",
            alert_type,
            self._current_lag,
            self._current_status.value,
        )

        if self._alert_callback:
            try:
                self._alert_callback(alert_type, health)
            except (OSError, RuntimeError, ValueError) as e:
                logger.error("Failed to invoke alert callback: %s", e)

    def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        if not _metrics_initialized:
            return

        labels = {"primary": self._primary_host, "standby": self._standby_host}

        try:
            if self._current_lag is not None:
                REPLICATION_LAG_SECONDS.labels(**labels).set(self._current_lag)

            # Status: 1=healthy, 0.5=degraded, 0=critical/stopped
            status_value = {
                ReplicationStatus.HEALTHY: 1.0,
                ReplicationStatus.DEGRADED: 0.5,
                ReplicationStatus.CRITICAL: 0.0,
                ReplicationStatus.STOPPED: 0.0,
                ReplicationStatus.UNKNOWN: 0.0,
            }.get(self._current_status, 0.0)
            REPLICATION_STATUS.labels(**labels).set(status_value)

            if self._last_replication_at:
                REPLICATION_LAST_TIMESTAMP.labels(**labels).set(
                    self._last_replication_at.timestamp()
                )

            REPLICATION_TOTAL.labels(**labels, status="success").inc()

            if self._replication_rate is not None:
                REPLICATION_RATE_BYTES.labels(**labels).set(self._replication_rate)

        except (OSError, RuntimeError, ValueError) as e:
            logger.debug("Failed to update metrics: %s", e)

    def _record_failure_metric(self, error_type: str) -> None:
        """Record a replication failure metric."""
        if not _metrics_initialized:
            return

        try:
            labels = {"primary": self._primary_host, "standby": self._standby_host}
            REPLICATION_FAILURES.labels(**labels, error_type=error_type).inc()
            REPLICATION_TOTAL.labels(**labels, status="failure").inc()
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug("Failed to record failure metric: %s", e)

    def get_health(self) -> ReplicationHealth:
        """
        Get current replication health status.

        Returns:
            ReplicationHealth with current state
        """
        # Check if replication is stale
        error = None
        if self._last_replication_at:
            time_since = (datetime.now(timezone.utc) - self._last_replication_at).total_seconds()
            if time_since > self._config.max_time_since_replication_seconds:
                error = f"No replication in {time_since:.0f}s (threshold: {self._config.max_time_since_replication_seconds}s)"
                if self._current_status != ReplicationStatus.STOPPED:
                    self._current_status = ReplicationStatus.CRITICAL

        healthy = self._current_status == ReplicationStatus.HEALTHY

        return ReplicationHealth(
            status=self._current_status,
            healthy=healthy,
            lag_seconds=self._current_lag,
            last_replication_at=self._last_replication_at,
            standby_connected=self._standby_connected,
            primary_host=self._primary_host,
            standby_host=self._standby_host,
            error=error,
            metadata={
                "total_replications": self._total_replications,
                "failed_replications": self._failed_replications,
                "warning_threshold": self._config.warning_lag_seconds,
                "critical_threshold": self._config.critical_lag_seconds,
            },
        )

    def get_metrics(self) -> ReplicationMetrics:
        """
        Get current replication metrics snapshot.

        Returns:
            ReplicationMetrics with current values
        """
        avg_lag = None
        if self._lag_history:
            lags = [lag for _, lag in self._lag_history]
            avg_lag = sum(lags) / len(lags)

        return ReplicationMetrics(
            lag_seconds=self._current_lag,
            lag_bytes=self._current_lag_bytes,
            replication_rate_bytes_per_sec=self._replication_rate,
            last_replication_at=self._last_replication_at,
            total_replications=self._total_replications,
            failed_replications=self._failed_replications,
            average_lag_seconds=avg_lag,
        )

    async def start_monitoring(self) -> None:
        """Start background monitoring task."""
        if self._running:
            logger.warning("Replication monitoring already running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            "Started replication monitoring for %s -> %s", self._primary_host, self._standby_host
        )

    async def stop_monitoring(self) -> None:
        """Stop background monitoring task."""
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Stopped replication monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Check for stale replication
                if self._last_replication_at:
                    time_since = (
                        datetime.now(timezone.utc) - self._last_replication_at
                    ).total_seconds()
                    if time_since > self._config.max_time_since_replication_seconds:
                        if self._current_status not in (
                            ReplicationStatus.CRITICAL,
                            ReplicationStatus.STOPPED,
                        ):
                            self._current_status = ReplicationStatus.CRITICAL
                            self._trigger_alert("stale_replication")
                            self._update_metrics()

                await asyncio.sleep(self._config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except (OSError, RuntimeError, ConnectionError) as e:
                logger.error("Error in replication monitoring loop: %s", e)
                await asyncio.sleep(self._config.check_interval_seconds)

    def health_check(self) -> dict[str, Any]:
        """
        Health check endpoint for control plane integration.

        Returns:
            Dict with health status suitable for HTTP response
        """
        health = self.get_health()
        return {
            "service": "replication_monitor",
            "status": "healthy" if health.healthy else "unhealthy",
            "details": health.to_dict(),
        }


# =============================================================================
# Global Instance
# =============================================================================

_replication_monitor: ReplicationHealthMonitor | None = None


def get_replication_monitor() -> ReplicationHealthMonitor:
    """Get or create the global replication monitor."""
    global _replication_monitor
    if _replication_monitor is None:
        _replication_monitor = ReplicationHealthMonitor()
    return _replication_monitor


def set_replication_monitor(monitor: ReplicationHealthMonitor | None) -> None:
    """Set the global replication monitor."""
    global _replication_monitor
    _replication_monitor = monitor


def create_replication_monitor(
    config: ReplicationConfig | None = None,
    primary_host: str = "primary",
    standby_host: str = "standby",
    alert_callback: AlertCallback | None = None,
) -> ReplicationHealthMonitor:
    """
    Create and set the global replication monitor.

    Args:
        config: Monitoring configuration
        primary_host: Identifier for primary host
        standby_host: Identifier for standby host
        alert_callback: Optional callback for alerts

    Returns:
        The created ReplicationHealthMonitor
    """
    global _replication_monitor
    _replication_monitor = ReplicationHealthMonitor(
        config=config,
        primary_host=primary_host,
        standby_host=standby_host,
        alert_callback=alert_callback,
    )
    return _replication_monitor


__all__ = [
    # Classes
    "ReplicationHealthMonitor",
    "ReplicationHealth",
    "ReplicationConfig",
    "ReplicationMetrics",
    "ReplicationStatus",
    # Global access
    "get_replication_monitor",
    "set_replication_monitor",
    "create_replication_monitor",
    # Metrics (for direct access if needed)
    "REPLICATION_LAG_SECONDS",
    "REPLICATION_STATUS",
    "REPLICATION_LAST_TIMESTAMP",
    "REPLICATION_TOTAL",
    "REPLICATION_FAILURES",
    "REPLICATION_RATE_BYTES",
]
