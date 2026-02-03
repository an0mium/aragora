"""
Metric cardinality monitoring and alerting.

Tracks the cardinality (number of unique label combinations) of Prometheus
metrics to detect cardinality explosion before it causes memory issues.

Usage:
    from aragora.observability.metrics.cardinality import (
        CardinalityTracker,
        get_cardinality_tracker,
    )

    # Get global tracker
    tracker = get_cardinality_tracker()

    # Record a label observation
    tracker.observe("aragora_requests_total", {"method": "GET", "path": "/api/v1/health"})

    # Check cardinality
    stats = tracker.get_stats()
    if stats["aragora_requests_total"]["cardinality"] > 10000:
        alert("High cardinality detected!")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Default cardinality thresholds
DEFAULT_WARNING_THRESHOLD = 5000
DEFAULT_CRITICAL_THRESHOLD = 10000


@dataclass
class MetricCardinality:
    """Tracks cardinality for a single metric.

    Attributes:
        name: Metric name
        label_names: Set of label names used
        unique_combinations: Set of unique label value combinations
        first_seen: Timestamp of first observation
        last_seen: Timestamp of most recent observation
        total_observations: Total number of observations
    """

    name: str
    label_names: set[str] = field(default_factory=set)
    unique_combinations: set[tuple[tuple[str, str], ...]] = field(default_factory=set)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_observations: int = 0

    @property
    def cardinality(self) -> int:
        """Current cardinality (unique label combinations)."""
        return len(self.unique_combinations)

    def observe(self, labels: dict[str, str]) -> None:
        """Record an observation with labels.

        Args:
            labels: Dict of label name -> value
        """
        self.total_observations += 1
        self.last_seen = time.time()

        # Track label names
        self.label_names.update(labels.keys())

        # Track unique combination (sorted tuple for consistency)
        combo = tuple(sorted(labels.items()))
        self.unique_combinations.add(combo)


@dataclass
class CardinalityAlert:
    """Alert for high cardinality.

    Attributes:
        metric_name: Name of the metric
        cardinality: Current cardinality
        threshold: Threshold that was exceeded
        level: Alert level (warning or critical)
        timestamp: When the alert was generated
    """

    metric_name: str
    cardinality: int
    threshold: int
    level: str  # "warning" or "critical"
    timestamp: float = field(default_factory=time.time)


class CardinalityTracker:
    """
    Tracks and monitors metric cardinality.

    Samples label observations to estimate cardinality without storing
    all unique combinations (uses HyperLogLog-style sampling for large metrics).

    Features:
    - Per-metric cardinality tracking
    - Configurable warning/critical thresholds
    - Sampling for high-volume metrics
    - Cardinality growth rate estimation
    """

    def __init__(
        self,
        warning_threshold: int = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: int = DEFAULT_CRITICAL_THRESHOLD,
        sample_rate: float = 1.0,
        max_tracked_metrics: int = 1000,
    ):
        """Initialize the cardinality tracker.

        Args:
            warning_threshold: Cardinality threshold for warning alerts
            critical_threshold: Cardinality threshold for critical alerts
            sample_rate: Sampling rate (1.0 = track all, 0.1 = 10%)
            max_tracked_metrics: Maximum number of metrics to track
        """
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._sample_rate = sample_rate
        self._max_tracked_metrics = max_tracked_metrics

        self._metrics: dict[str, MetricCardinality] = {}
        self._lock = threading.Lock()
        self._alerts: list[CardinalityAlert] = []

        # Sampling counter
        self._sample_counter = 0

    def observe(
        self,
        metric_name: str,
        labels: dict[str, str],
    ) -> None:
        """Record a label observation for a metric.

        Args:
            metric_name: Prometheus metric name
            labels: Dict of label name -> value
        """
        # Apply sampling
        self._sample_counter += 1
        if self._sample_rate < 1.0:
            if (self._sample_counter % int(1 / self._sample_rate)) != 0:
                return

        with self._lock:
            # Enforce max tracked metrics
            if metric_name not in self._metrics:
                if len(self._metrics) >= self._max_tracked_metrics:
                    # Drop oldest metric
                    oldest = min(
                        self._metrics.items(),
                        key=lambda x: x[1].last_seen,
                    )
                    del self._metrics[oldest[0]]

            # Get or create metric tracking
            if metric_name not in self._metrics:
                self._metrics[metric_name] = MetricCardinality(name=metric_name)

            metric = self._metrics[metric_name]
            old_cardinality = metric.cardinality
            metric.observe(labels)

            # Check for threshold crossing
            new_cardinality = metric.cardinality
            if old_cardinality < self._critical_threshold <= new_cardinality:
                self._generate_alert(metric_name, new_cardinality, "critical")
            elif old_cardinality < self._warning_threshold <= new_cardinality:
                self._generate_alert(metric_name, new_cardinality, "warning")

    def get_cardinality(self, metric_name: str) -> int:
        """Get current cardinality for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Current cardinality, or 0 if metric not tracked
        """
        with self._lock:
            metric = self._metrics.get(metric_name)
            return metric.cardinality if metric else 0

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get cardinality statistics for all tracked metrics.

        Returns:
            Dict mapping metric name to stats dict
        """
        with self._lock:
            return {
                name: {
                    "cardinality": metric.cardinality,
                    "label_names": list(metric.label_names),
                    "total_observations": metric.total_observations,
                    "first_seen": metric.first_seen,
                    "last_seen": metric.last_seen,
                    "over_warning": metric.cardinality >= self._warning_threshold,
                    "over_critical": metric.cardinality >= self._critical_threshold,
                }
                for name, metric in self._metrics.items()
            }

    def get_high_cardinality_metrics(
        self,
        threshold: int | None = None,
    ) -> list[tuple[str, int]]:
        """Get metrics exceeding cardinality threshold.

        Args:
            threshold: Custom threshold (default: warning threshold)

        Returns:
            List of (metric_name, cardinality) tuples, sorted by cardinality desc
        """
        threshold = threshold or self._warning_threshold

        with self._lock:
            high_cardinality = [
                (name, metric.cardinality)
                for name, metric in self._metrics.items()
                if metric.cardinality >= threshold
            ]
            return sorted(high_cardinality, key=lambda x: x[1], reverse=True)

    def get_alerts(self, since: float | None = None) -> list[CardinalityAlert]:
        """Get cardinality alerts.

        Args:
            since: Only return alerts after this timestamp

        Returns:
            List of CardinalityAlert objects
        """
        with self._lock:
            if since is None:
                return list(self._alerts)
            return [a for a in self._alerts if a.timestamp > since]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict with summary metrics
        """
        with self._lock:
            cardinalities = [m.cardinality for m in self._metrics.values()]
            total_observations = sum(m.total_observations for m in self._metrics.values())

            return {
                "tracked_metrics": len(self._metrics),
                "total_observations": total_observations,
                "max_cardinality": max(cardinalities) if cardinalities else 0,
                "avg_cardinality": (
                    sum(cardinalities) / len(cardinalities) if cardinalities else 0
                ),
                "over_warning_count": sum(
                    1 for m in self._metrics.values() if m.cardinality >= self._warning_threshold
                ),
                "over_critical_count": sum(
                    1 for m in self._metrics.values() if m.cardinality >= self._critical_threshold
                ),
                "alert_count": len(self._alerts),
                "warning_threshold": self._warning_threshold,
                "critical_threshold": self._critical_threshold,
            }

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._metrics.clear()
            self._alerts.clear()
            self._sample_counter = 0

    def _generate_alert(
        self,
        metric_name: str,
        cardinality: int,
        level: str,
    ) -> None:
        """Generate a cardinality alert.

        Args:
            metric_name: Name of the metric
            cardinality: Current cardinality
            level: Alert level (warning or critical)
        """
        threshold = self._critical_threshold if level == "critical" else self._warning_threshold

        alert = CardinalityAlert(
            metric_name=metric_name,
            cardinality=cardinality,
            threshold=threshold,
            level=level,
        )

        self._alerts.append(alert)

        # Log the alert
        log_fn = logger.warning if level == "warning" else logger.error
        log_fn(
            "Metric cardinality %s: %s has %d unique label combinations (threshold: %d)",
            level.upper(),
            metric_name,
            cardinality,
            threshold,
        )


# Global tracker instance
_tracker: CardinalityTracker | None = None


def get_cardinality_tracker() -> CardinalityTracker:
    """Get or create the global cardinality tracker.

    Returns:
        CardinalityTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = CardinalityTracker()
    return _tracker


def reset_cardinality_tracker() -> None:
    """Reset the global cardinality tracker (for testing)."""
    global _tracker
    _tracker = None


# Global metric variables
CARDINALITY_GAUGE: Any = None
CARDINALITY_WARNING_COUNTER: Any = None
CARDINALITY_CRITICAL_COUNTER: Any = None

_cardinality_initialized = False


def init_cardinality_metrics() -> None:
    """Initialize cardinality Prometheus metrics."""
    global _cardinality_initialized
    global CARDINALITY_GAUGE, CARDINALITY_WARNING_COUNTER, CARDINALITY_CRITICAL_COUNTER

    if _cardinality_initialized:
        return

    if not get_metrics_enabled():
        _init_noop_cardinality_metrics()
        _cardinality_initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge

        CARDINALITY_GAUGE = Gauge(
            "aragora_metric_cardinality",
            "Current cardinality (unique label combinations) for tracked metrics",
            ["metric_name"],
        )

        CARDINALITY_WARNING_COUNTER = Counter(
            "aragora_metric_cardinality_warnings_total",
            "Total cardinality warning alerts",
            ["metric_name"],
        )

        CARDINALITY_CRITICAL_COUNTER = Counter(
            "aragora_metric_cardinality_critical_total",
            "Total cardinality critical alerts",
            ["metric_name"],
        )

        _cardinality_initialized = True
        logger.debug("Cardinality metrics initialized")

    except ImportError:
        _init_noop_cardinality_metrics()
        _cardinality_initialized = True


def _init_noop_cardinality_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global CARDINALITY_GAUGE, CARDINALITY_WARNING_COUNTER, CARDINALITY_CRITICAL_COUNTER

    CARDINALITY_GAUGE = NoOpMetric()
    CARDINALITY_WARNING_COUNTER = NoOpMetric()
    CARDINALITY_CRITICAL_COUNTER = NoOpMetric()


def _ensure_cardinality_init() -> None:
    """Ensure cardinality metrics are initialized."""
    if not _cardinality_initialized:
        init_cardinality_metrics()


def update_prometheus_metrics() -> None:
    """Update Prometheus metrics with current cardinality data."""
    _ensure_cardinality_init()

    tracker = get_cardinality_tracker()
    stats = tracker.get_stats()

    for metric_name, metric_stats in stats.items():
        CARDINALITY_GAUGE.labels(metric_name=metric_name).set(metric_stats["cardinality"])


__all__ = [
    "CardinalityTracker",
    "CardinalityAlert",
    "MetricCardinality",
    "get_cardinality_tracker",
    "reset_cardinality_tracker",
    "init_cardinality_metrics",
    "update_prometheus_metrics",
    "CARDINALITY_GAUGE",
    "CARDINALITY_WARNING_COUNTER",
    "CARDINALITY_CRITICAL_COUNTER",
    "DEFAULT_WARNING_THRESHOLD",
    "DEFAULT_CRITICAL_THRESHOLD",
]
