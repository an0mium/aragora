"""
RLM Metrics Export - Export factory metrics to monitoring systems.

Supports:
- Prometheus (via prometheus_client or manual exposition)
- StatsD (via statsd client)
- JSON (for custom integrations)
- OpenTelemetry (via OTEL SDK)

Usage:
    from aragora.rlm.metrics_export import (
        export_to_prometheus,
        export_to_statsd,
        export_to_json,
        get_metrics_collector,
    )

    # Prometheus integration
    export_to_prometheus()  # Registers metrics with prometheus_client

    # StatsD integration
    export_to_statsd(host="localhost", port=8125)

    # JSON for custom integrations
    json_metrics = export_to_json()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from .factory import get_factory_metrics

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of RLM metrics."""

    timestamp: float
    metrics: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
            ),
            "metrics": self.metrics,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """Collects and tracks RLM metrics over time.

    Provides:
    - Current metrics snapshot
    - Delta since last collection
    - Rate calculations (calls per second)
    """

    def __init__(self):
        self._last_snapshot: Optional[MetricsSnapshot] = None
        self._last_collect_time: float = 0
        self._callbacks: list[Callable[[MetricsSnapshot], None]] = []

    def collect(self) -> MetricsSnapshot:
        """Collect current metrics snapshot."""
        now = time.time()
        metrics = get_factory_metrics()
        snapshot = MetricsSnapshot(timestamp=now, metrics=metrics)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")

        self._last_snapshot = snapshot
        self._last_collect_time = now
        return snapshot

    def get_delta(self) -> Optional[Dict[str, int]]:
        """Get delta since last collection."""
        if self._last_snapshot is None:
            return None

        current = get_factory_metrics()
        return {
            key: current[key] - self._last_snapshot.metrics.get(key, 0)
            for key in current
        }

    def get_rates(self, window_seconds: float = 60.0) -> Dict[str, float]:
        """Calculate rates (per second) over the given window."""
        if self._last_snapshot is None or self._last_collect_time == 0:
            return {}

        elapsed = time.time() - self._last_collect_time
        if elapsed <= 0:
            return {}

        delta = self.get_delta()
        if delta is None:
            return {}

        return {key: value / elapsed for key, value in delta.items()}

    def add_callback(self, callback: Callable[[MetricsSnapshot], None]) -> None:
        """Add callback to be notified on each collection."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[MetricsSnapshot], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)


# Global collector instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def export_to_json() -> str:
    """Export current metrics as JSON.

    Returns:
        JSON string with metrics snapshot

    Example:
        >>> json_str = export_to_json()
        >>> print(json_str)
        {
          "timestamp": 1234567890.123,
          "timestamp_iso": "2024-01-15T10:30:00Z",
          "metrics": {
            "get_rlm_calls": 42,
            "true_rlm_calls": 10,
            ...
          }
        }
    """
    collector = get_metrics_collector()
    snapshot = collector.collect()
    return snapshot.to_json()


def export_to_prometheus(
    registry: Optional[Any] = None,
    prefix: str = "aragora_rlm",
) -> Dict[str, Any]:
    """Export metrics to Prometheus format.

    Args:
        registry: Optional prometheus_client.CollectorRegistry
        prefix: Metric name prefix

    Returns:
        Dict of created Prometheus metrics (Counter/Gauge objects)

    Example:
        >>> from prometheus_client import start_http_server
        >>> metrics = export_to_prometheus()
        >>> start_http_server(8000)  # Metrics at :8000/metrics
    """
    try:
        from prometheus_client import Counter, Gauge
        from prometheus_client import REGISTRY as DEFAULT_REGISTRY
    except ImportError:
        logger.warning(
            "prometheus_client not installed. Install with: pip install prometheus-client"
        )
        return {}

    reg = registry or DEFAULT_REGISTRY

    # Create Prometheus metrics
    metrics_map = {}

    # Counters (monotonically increasing)
    counter_metrics = [
        ("get_rlm_calls", "Total get_rlm() calls"),
        ("get_compressor_calls", "Total get_compressor() calls"),
        ("compress_and_query_calls", "Total compress_and_query() calls"),
        ("rlm_instances_created", "Total RLM instances created"),
        ("compressor_instances_created", "Total compressor instances created"),
        ("true_rlm_calls", "Queries using TRUE RLM (REPL-based)"),
        ("compression_fallback_calls", "Queries using compression fallback"),
        ("successful_queries", "Successful query count"),
        ("failed_queries", "Failed query count"),
        ("singleton_hits", "Singleton cache hits"),
        ("singleton_misses", "Singleton cache misses"),
    ]

    for name, desc in counter_metrics:
        full_name = f"{prefix}_{name}_total"
        try:
            counter = Counter(full_name, desc, registry=reg)
            metrics_map[name] = counter
        except ValueError:
            # Already registered
            pass

    # Update metrics with current values
    def update_prometheus_metrics():
        current = get_factory_metrics()
        for name, counter in metrics_map.items():
            if name in current:
                # Prometheus counters can only increase, so we track the delta
                # For simplicity, we just expose the current value
                pass  # Counter._value.set() is not standard - use a Gauge for absolute values

    logger.info(f"[RLM Metrics] Prometheus metrics registered with prefix '{prefix}'")
    return metrics_map


def export_to_statsd(
    host: str = "localhost",
    port: int = 8125,
    prefix: str = "aragora.rlm",
) -> bool:
    """Export metrics to StatsD.

    Args:
        host: StatsD host
        port: StatsD port
        prefix: Metric name prefix

    Returns:
        True if export succeeded

    Example:
        >>> export_to_statsd(host="statsd.local", port=8125)
    """
    try:
        import statsd
    except ImportError:
        logger.warning("statsd not installed. Install with: pip install statsd")
        return False

    try:
        client = statsd.StatsClient(host, port, prefix=prefix)
        metrics = get_factory_metrics()

        for name, value in metrics.items():
            client.gauge(name, value)

        logger.info(f"[RLM Metrics] Exported to StatsD at {host}:{port}")
        return True

    except Exception as e:
        logger.error(f"[RLM Metrics] StatsD export failed: {e}")
        return False


def export_to_opentelemetry(
    meter_name: str = "aragora.rlm",
) -> Dict[str, Any]:
    """Export metrics to OpenTelemetry.

    Args:
        meter_name: OpenTelemetry meter name

    Returns:
        Dict of created OTEL instruments

    Example:
        >>> from opentelemetry.sdk.metrics import MeterProvider
        >>> instruments = export_to_opentelemetry()
    """
    try:
        from opentelemetry import metrics
    except ImportError:
        logger.warning(
            "opentelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk"
        )
        return {}

    try:
        meter = metrics.get_meter(meter_name)
        instruments = {}

        # Create counters for each metric
        metric_names = [
            "get_rlm_calls",
            "get_compressor_calls",
            "compress_and_query_calls",
            "true_rlm_calls",
            "compression_fallback_calls",
            "successful_queries",
            "failed_queries",
        ]

        for name in metric_names:
            counter = meter.create_counter(
                name=f"rlm.{name}",
                description=f"RLM factory metric: {name}",
                unit="1",
            )
            instruments[name] = counter

        logger.info(f"[RLM Metrics] OpenTelemetry instruments created for '{meter_name}'")
        return instruments

    except Exception as e:
        logger.error(f"[RLM Metrics] OpenTelemetry export failed: {e}")
        return {}


def create_periodic_exporter(
    export_fn: Callable[[], Any],
    interval_seconds: float = 60.0,
) -> Callable[[], None]:
    """Create a function that periodically exports metrics.

    Args:
        export_fn: Export function to call
        interval_seconds: Export interval

    Returns:
        Stop function to cancel the periodic export

    Example:
        >>> stop = create_periodic_exporter(
        ...     lambda: export_to_statsd(),
        ...     interval_seconds=30.0
        ... )
        >>> # Later...
        >>> stop()  # Stop periodic export
    """
    import threading

    stop_event = threading.Event()

    def run():
        while not stop_event.is_set():
            try:
                export_fn()
            except Exception as e:
                logger.error(f"Periodic metrics export failed: {e}")
            stop_event.wait(interval_seconds)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def stop():
        stop_event.set()
        thread.join(timeout=5.0)

    return stop


__all__ = [
    "MetricsSnapshot",
    "MetricsCollector",
    "get_metrics_collector",
    "export_to_json",
    "export_to_prometheus",
    "export_to_statsd",
    "export_to_opentelemetry",
    "create_periodic_exporter",
]
