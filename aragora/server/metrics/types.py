"""
Prometheus Metric Types for Aragora.

Provides thread-safe metric implementations:
- Counter: Monotonically increasing value
- Gauge: Value that can go up or down
- Histogram: Distribution of values with bucket counting

These are lightweight implementations compatible with Prometheus format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class Counter:
    """Simple counter metric."""

    name: str
    help: str
    label_names: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs: str) -> "LabeledCounter":
        """Get a labeled instance of this counter."""
        return LabeledCounter(self, tuple(sorted(kwargs.items())))

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the counter."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def get(self, **labels) -> float:
        """Get current value."""
        key = tuple(sorted(labels.items())) if labels else ()
        return self._values.get(key, 0)

    def collect(self) -> list[tuple[dict, float]]:
        """Collect all values for export."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


@dataclass
class LabeledCounter:
    """A counter with specific label values."""

    counter: Counter
    label_values: tuple

    def inc(self, value: float = 1.0) -> None:
        with self.counter._lock:
            self.counter._values[self.label_values] = (
                self.counter._values.get(self.label_values, 0) + value
            )


@dataclass
class Gauge:
    """Simple gauge metric (can go up or down)."""

    name: str
    help: str
    label_names: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs: str) -> "LabeledGauge":
        """Get a labeled instance of this gauge."""
        return LabeledGauge(self, tuple(sorted(kwargs.items())))

    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment gauge."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement gauge."""
        self.inc(-value, **labels)

    def get(self, **labels) -> float:
        """Get current value."""
        key = tuple(sorted(labels.items())) if labels else ()
        return self._values.get(key, 0)

    def collect(self) -> list[tuple[dict, float]]:
        """Collect all values for export."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


@dataclass
class LabeledGauge:
    """A gauge with specific label values."""

    gauge: Gauge
    label_values: tuple

    def set(self, value: float) -> None:
        with self.gauge._lock:
            self.gauge._values[self.label_values] = value

    def inc(self, value: float = 1.0) -> None:
        with self.gauge._lock:
            self.gauge._values[self.label_values] = (
                self.gauge._values.get(self.label_values, 0) + value
            )

    def dec(self, value: float = 1.0) -> None:
        self.inc(-value)


@dataclass
class Histogram:
    """Simple histogram for tracking distributions."""

    name: str
    help: str
    label_names: list[str] = field(default_factory=list)
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    _counts: dict[tuple, list[int]] = field(default_factory=dict)
    _sums: dict[tuple, float] = field(default_factory=dict)
    _totals: dict[tuple, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs: str) -> "LabeledHistogram":
        """Get a labeled instance of this histogram."""
        return LabeledHistogram(self, tuple(sorted(kwargs.items())))

    def observe(self, value: float, **labels) -> None:
        """Record an observation."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            if key not in self._counts:
                self._counts[key] = [0] * len(self.buckets)
                self._sums[key] = 0
                self._totals[key] = 0

            # Update bucket counts
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._counts[key][i] += 1

            self._sums[key] += value
            self._totals[key] += 1

    def collect(self) -> list[tuple[dict, dict]]:
        """Collect all values for export."""
        with self._lock:
            results = []
            for key, counts in self._counts.items():
                results.append(
                    (
                        dict(key),
                        {
                            "buckets": list(zip(self.buckets, counts)),
                            "sum": self._sums.get(key, 0),
                            "count": self._totals.get(key, 0),
                        },
                    )
                )
            return results


@dataclass
class LabeledHistogram:
    """A histogram with specific label values."""

    histogram: Histogram
    label_values: tuple

    def observe(self, value: float) -> None:
        key = self.label_values
        with self.histogram._lock:
            if key not in self.histogram._counts:
                self.histogram._counts[key] = [0] * len(self.histogram.buckets)
                self.histogram._sums[key] = 0
                self.histogram._totals[key] = 0

            for i, bucket in enumerate(self.histogram.buckets):
                if value <= bucket:
                    self.histogram._counts[key][i] += 1

            self.histogram._sums[key] += value
            self.histogram._totals[key] += 1


def get_percentile(histogram: Histogram, percentile: float, **labels) -> Optional[float]:
    """Estimate a percentile from histogram buckets.

    Uses linear interpolation between bucket boundaries.

    Args:
        histogram: The histogram to query
        percentile: The percentile to compute (0-100, e.g., 50, 95, 99)
        **labels: Label filters

    Returns:
        Estimated percentile value, or None if no data
    """
    key = tuple(sorted(labels.items())) if labels else ()

    with histogram._lock:
        if key not in histogram._totals or histogram._totals[key] == 0:
            return None

        total = histogram._totals[key]
        target = total * (percentile / 100.0)
        counts = histogram._counts[key]

        # Find the bucket containing the target percentile
        cumulative = 0
        for i, (bucket, count) in enumerate(zip(histogram.buckets, counts)):
            if cumulative + count >= target:
                if i == 0:
                    # In first bucket, estimate using bucket boundary
                    return bucket * (target / count) if count > 0 else bucket
                else:
                    # Linear interpolation between buckets
                    prev_bucket = histogram.buckets[i - 1]
                    prev_cumulative = cumulative
                    within_bucket = (target - prev_cumulative) / count if count > 0 else 0
                    return prev_bucket + (bucket - prev_bucket) * within_bucket
            cumulative += count

        # Above all buckets, return highest bucket boundary
        return histogram.buckets[-1]


def get_percentiles(histogram: Histogram, **labels) -> dict[str, Optional[float]]:
    """Get common percentiles (p50, p90, p95, p99) from a histogram.

    Args:
        histogram: The histogram to query
        **labels: Label filters

    Returns:
        Dict with keys 'p50', 'p90', 'p95', 'p99' and their values
    """
    return {
        "p50": get_percentile(histogram, 50, **labels),
        "p90": get_percentile(histogram, 90, **labels),
        "p95": get_percentile(histogram, 95, **labels),
        "p99": get_percentile(histogram, 99, **labels),
    }


__all__ = [
    "Counter",
    "LabeledCounter",
    "Gauge",
    "LabeledGauge",
    "Histogram",
    "LabeledHistogram",
    "get_percentile",
    "get_percentiles",
]
