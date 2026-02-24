"""
SLA Instrumentation for Aragora.

Tracks request latency percentiles (p50, p95, p99), error rates per endpoint,
and uptime percentage for SLA reporting on the public status page.

All metrics are stored in-memory with configurable rolling windows and can be
queried for historical reporting.

Usage:
    from aragora.observability.sla_instrumentation import (
        get_sla_tracker,
        record_request_metric,
    )

    # Record a request
    record_request_metric(
        endpoint="/api/v1/debates",
        method="GET",
        status_code=200,
        latency_seconds=0.045,
    )

    # Get SLA metrics
    tracker = get_sla_tracker()
    metrics = tracker.get_sla_summary()
    print(f"Uptime: {metrics['uptime']['percentage']:.3f}%")
    print(f"p99 latency: {metrics['latency']['p99']:.3f}s")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LatencyPercentiles:
    """Latency percentile summary."""

    p50: float
    p95: float
    p99: float
    count: int
    mean: float
    min: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50": round(self.p50, 6),
            "p95": round(self.p95, 6),
            "p99": round(self.p99, 6),
            "count": self.count,
            "mean": round(self.mean, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
        }


@dataclass
class EndpointMetrics:
    """Metrics for a single endpoint."""

    endpoint: str
    total_requests: int = 0
    error_count: int = 0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "latency_p50": round(self.latency_p50, 6),
            "latency_p95": round(self.latency_p95, 6),
            "latency_p99": round(self.latency_p99, 6),
            "error_rate": round(self.error_rate, 6),
        }


@dataclass
class _RequestRecord:
    """Internal record of a single request."""

    timestamp: float
    endpoint: str
    method: str
    status_code: int
    latency_seconds: float


class SLATracker:
    """Tracks SLA metrics with rolling windows.

    Thread-safe. Maintains in-memory request records for computing
    latency percentiles, error rates, and uptime percentages.

    Args:
        max_records: Maximum number of records to keep in memory.
        window_24h: 24-hour window in seconds.
        window_7d: 7-day window in seconds.
        window_30d: 30-day window in seconds.
    """

    def __init__(
        self,
        max_records: int = 100_000,
        window_24h: float = 86_400.0,
        window_7d: float = 604_800.0,
        window_30d: float = 2_592_000.0,
    ):
        self._max_records = max_records
        self._window_24h = window_24h
        self._window_7d = window_7d
        self._window_30d = window_30d
        self._lock = threading.Lock()
        self._records: list[_RequestRecord] = []
        self._start_time = time.time()

        # Aggregate counters for efficiency (not reset on eviction)
        self._total_requests = 0
        self._total_errors = 0

    def record(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """Record a request metric.

        Args:
            endpoint: The API endpoint path.
            method: HTTP method (GET, POST, etc.).
            status_code: HTTP response status code.
            latency_seconds: Request latency in seconds.
        """
        record = _RequestRecord(
            timestamp=time.time(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_seconds=latency_seconds,
        )

        with self._lock:
            self._records.append(record)
            self._total_requests += 1
            if status_code >= 500:
                self._total_errors += 1

            # Evict oldest records if exceeding max
            if len(self._records) > self._max_records:
                self._records = self._records[-(self._max_records // 2) :]

    def _filter_records(
        self,
        window_seconds: float | None = None,
        endpoint: str | None = None,
    ) -> list[_RequestRecord]:
        """Filter records by time window and/or endpoint.

        Must be called with self._lock held.
        """
        now = time.time()
        result = self._records

        if window_seconds is not None:
            cutoff = now - window_seconds
            result = [r for r in result if r.timestamp >= cutoff]

        if endpoint is not None:
            result = [r for r in result if r.endpoint == endpoint]

        return result

    def _compute_percentiles(
        self,
        latencies: list[float],
    ) -> LatencyPercentiles:
        """Compute latency percentiles from a list of latency values."""
        if not latencies:
            return LatencyPercentiles(
                p50=0.0, p95=0.0, p99=0.0,
                count=0, mean=0.0, min=0.0, max=0.0,
            )

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def _percentile(pct: float) -> float:
            idx = (pct / 100.0) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            return sorted_latencies[lower] * (1 - frac) + sorted_latencies[upper] * frac

        return LatencyPercentiles(
            p50=_percentile(50),
            p95=_percentile(95),
            p99=_percentile(99),
            count=n,
            mean=sum(sorted_latencies) / n,
            min=sorted_latencies[0],
            max=sorted_latencies[-1],
        )

    def get_latency_percentiles(
        self,
        window_seconds: float | None = None,
        endpoint: str | None = None,
    ) -> LatencyPercentiles:
        """Get latency percentiles for a given window.

        Args:
            window_seconds: Time window in seconds (None = all time).
            endpoint: Filter by endpoint (None = all endpoints).

        Returns:
            LatencyPercentiles with p50, p95, p99 values.
        """
        with self._lock:
            records = self._filter_records(window_seconds, endpoint)
            latencies = [r.latency_seconds for r in records]

        return self._compute_percentiles(latencies)

    def get_error_rate(
        self,
        window_seconds: float | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """Get error rate for a given window.

        Args:
            window_seconds: Time window in seconds.
            endpoint: Filter by endpoint.

        Returns:
            Dict with total_requests, error_count, error_rate.
        """
        with self._lock:
            records = self._filter_records(window_seconds, endpoint)
            total = len(records)
            errors = sum(1 for r in records if r.status_code >= 500)

        error_rate = errors / total if total > 0 else 0.0

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(error_rate, 6),
        }

    def get_endpoint_metrics(
        self,
        window_seconds: float | None = None,
    ) -> list[EndpointMetrics]:
        """Get per-endpoint metrics.

        Args:
            window_seconds: Time window in seconds.

        Returns:
            List of EndpointMetrics for each endpoint with traffic.
        """
        with self._lock:
            records = self._filter_records(window_seconds)

        # Group by endpoint
        by_endpoint: dict[str, list[_RequestRecord]] = {}
        for r in records:
            by_endpoint.setdefault(r.endpoint, []).append(r)

        results = []
        for ep, ep_records in sorted(by_endpoint.items()):
            total = len(ep_records)
            errors = sum(1 for r in ep_records if r.status_code >= 500)
            latencies = [r.latency_seconds for r in ep_records]
            pct = self._compute_percentiles(latencies)

            results.append(EndpointMetrics(
                endpoint=ep,
                total_requests=total,
                error_count=errors,
                latency_p50=pct.p50,
                latency_p95=pct.p95,
                latency_p99=pct.p99,
                error_rate=errors / total if total > 0 else 0.0,
            ))

        return results

    def get_uptime(self) -> dict[str, Any]:
        """Get uptime percentages for 24h, 7d, 30d windows.

        Uptime is calculated as the ratio of non-5xx requests to total requests.
        If no requests are recorded in a window, uptime defaults to 100%.

        Returns:
            Dict with uptime percentages for each window.
        """
        with self._lock:
            uptime_data: dict[str, Any] = {}

            for label, window in [
                ("24h", self._window_24h),
                ("7d", self._window_7d),
                ("30d", self._window_30d),
            ]:
                records = self._filter_records(window)
                total = len(records)
                errors = sum(1 for r in records if r.status_code >= 500)
                uptime_pct = ((total - errors) / total * 100) if total > 0 else 100.0

                uptime_data[label] = {
                    "uptime_percent": round(uptime_pct, 4),
                    "total_requests": total,
                    "error_count": errors,
                    "incidents": errors,  # Simplified: each 5xx counts
                }

        return uptime_data

    def get_sla_summary(self) -> dict[str, Any]:
        """Get comprehensive SLA summary.

        Returns:
            Dict with latency, error_rate, uptime, and per-endpoint metrics.
        """
        now = datetime.now(timezone.utc)

        # Global latency percentiles (24h window)
        latency = self.get_latency_percentiles(window_seconds=self._window_24h)
        error_rate = self.get_error_rate(window_seconds=self._window_24h)
        uptime = self.get_uptime()
        endpoint_metrics = self.get_endpoint_metrics(window_seconds=self._window_24h)

        return {
            "timestamp": now.isoformat(),
            "latency": latency.to_dict(),
            "error_rate": error_rate,
            "uptime": uptime,
            "endpoints": [m.to_dict() for m in endpoint_metrics[:20]],  # Top 20
            "tracking_since": datetime.fromtimestamp(
                self._start_time, tz=timezone.utc
            ).isoformat(),
        }

    def reset(self) -> None:
        """Reset all tracked metrics. Primarily for testing."""
        with self._lock:
            self._records.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._start_time = time.time()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_tracker: SLATracker | None = None
_tracker_lock = threading.Lock()


def get_sla_tracker() -> SLATracker:
    """Get or create the global SLA tracker."""
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = SLATracker()
        return _global_tracker


def reset_sla_tracker() -> None:
    """Reset the global SLA tracker (for testing)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None


def record_request_metric(
    endpoint: str,
    method: str,
    status_code: int,
    latency_seconds: float,
) -> None:
    """Convenience function to record a request metric.

    Args:
        endpoint: The API endpoint path.
        method: HTTP method.
        status_code: HTTP response status code.
        latency_seconds: Request latency in seconds.
    """
    tracker = get_sla_tracker()
    tracker.record(endpoint, method, status_code, latency_seconds)


__all__ = [
    "SLATracker",
    "LatencyPercentiles",
    "EndpointMetrics",
    "get_sla_tracker",
    "reset_sla_tracker",
    "record_request_metric",
]
