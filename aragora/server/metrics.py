"""
Prometheus Metrics for Aragora.

Exposes metrics for monitoring:
- Billing and subscription events
- API usage and latency
- Debate throughput
- Agent performance

Usage:
    from aragora.server.metrics import (
        SUBSCRIPTION_EVENTS,
        API_REQUESTS,
        track_request,
    )

    # Track subscription event
    SUBSCRIPTION_EVENTS.labels(event="created", tier="starter").inc()

    # Track API request
    with track_request("/api/debates", "POST"):
        # handle request

Metrics endpoint: GET /metrics
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================


@dataclass
class Counter:
    """Simple counter metric."""

    name: str
    help: str
    labels: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs) -> "LabeledCounter":
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
    labels: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs) -> "LabeledGauge":
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
    labels: list[str] = field(default_factory=list)
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    _counts: dict[tuple, list[int]] = field(default_factory=dict)
    _sums: dict[tuple, float] = field(default_factory=dict)
    _totals: dict[tuple, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs) -> "LabeledHistogram":
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
                results.append((
                    dict(key),
                    {
                        "buckets": list(zip(self.buckets, counts)),
                        "sum": self._sums.get(key, 0),
                        "count": self._totals.get(key, 0),
                    },
                ))
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


# =============================================================================
# Billing Metrics
# =============================================================================

SUBSCRIPTION_EVENTS = Counter(
    name="aragora_subscription_events_total",
    help="Total subscription events by type and tier",
    labels=["event", "tier"],
)

SUBSCRIPTION_ACTIVE = Gauge(
    name="aragora_subscriptions_active",
    help="Currently active subscriptions by tier",
    labels=["tier"],
)

USAGE_DEBATES = Counter(
    name="aragora_debates_total",
    help="Total debates run by tier",
    labels=["tier", "org_id"],
)

USAGE_TOKENS = Counter(
    name="aragora_tokens_total",
    help="Total tokens used by provider",
    labels=["provider", "tier"],
)

BILLING_REVENUE = Counter(
    name="aragora_revenue_cents_total",
    help="Total revenue in cents by tier",
    labels=["tier"],
)

PAYMENT_FAILURES = Counter(
    name="aragora_payment_failures_total",
    help="Payment failure count by tier",
    labels=["tier"],
)


# =============================================================================
# API Metrics
# =============================================================================

API_REQUESTS = Counter(
    name="aragora_api_requests_total",
    help="Total API requests by endpoint and status",
    labels=["endpoint", "method", "status"],
)

API_LATENCY = Histogram(
    name="aragora_api_latency_seconds",
    help="API request latency in seconds",
    labels=["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_DEBATES = Gauge(
    name="aragora_active_debates",
    help="Currently running debates",
    labels=[],
)

WEBSOCKET_CONNECTIONS = Gauge(
    name="aragora_websocket_connections",
    help="Active WebSocket connections",
    labels=[],
)


# =============================================================================
# Agent Metrics
# =============================================================================

AGENT_REQUESTS = Counter(
    name="aragora_agent_requests_total",
    help="Agent API requests by agent and status",
    labels=["agent", "status"],
)

AGENT_LATENCY = Histogram(
    name="aragora_agent_latency_seconds",
    help="Agent response latency",
    labels=["agent"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

AGENT_TOKENS = Counter(
    name="aragora_agent_tokens_total",
    help="Tokens used by agent",
    labels=["agent", "direction"],  # direction: input/output
)


# =============================================================================
# Helpers
# =============================================================================


@contextmanager
def track_request(
    endpoint: str, method: str = "GET"
) -> Generator[None, None, None]:
    """Context manager to track request latency."""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        API_REQUESTS.inc(endpoint=endpoint, method=method, status=status)
        API_LATENCY.observe(duration, endpoint=endpoint, method=method)


def track_subscription_event(event: str, tier: str) -> None:
    """Track a subscription event."""
    SUBSCRIPTION_EVENTS.inc(event=event, tier=tier)


def track_debate(tier: str, org_id: str) -> None:
    """Track a debate execution."""
    USAGE_DEBATES.inc(tier=tier, org_id=org_id)


def track_tokens(provider: str, tier: str, count: int) -> None:
    """Track token usage."""
    USAGE_TOKENS.inc(count, provider=provider, tier=tier)


def track_agent_call(
    agent: str, latency: float, tokens_in: int, tokens_out: int, success: bool
) -> None:
    """Track an agent API call."""
    AGENT_REQUESTS.inc(agent=agent, status="success" if success else "error")
    AGENT_LATENCY.observe(latency, agent=agent)
    AGENT_TOKENS.inc(tokens_in, agent=agent, direction="input")
    AGENT_TOKENS.inc(tokens_out, agent=agent, direction="output")


# =============================================================================
# Prometheus Format Export
# =============================================================================


def _format_labels(labels: dict) -> str:
    """Format labels for Prometheus output."""
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


def generate_metrics() -> str:
    """Generate Prometheus-format metrics output."""
    lines = []

    # All metrics to export
    counters = [
        SUBSCRIPTION_EVENTS,
        USAGE_DEBATES,
        USAGE_TOKENS,
        BILLING_REVENUE,
        PAYMENT_FAILURES,
        API_REQUESTS,
        AGENT_REQUESTS,
        AGENT_TOKENS,
    ]

    gauges = [
        SUBSCRIPTION_ACTIVE,
        ACTIVE_DEBATES,
        WEBSOCKET_CONNECTIONS,
    ]

    histograms = [
        API_LATENCY,
        AGENT_LATENCY,
    ]

    # Export counters
    for counter in counters:
        lines.append(f"# HELP {counter.name} {counter.help}")
        lines.append(f"# TYPE {counter.name} counter")
        for labels, value in counter.collect():
            lines.append(f"{counter.name}{_format_labels(labels)} {value}")
        lines.append("")

    # Export gauges
    for gauge in gauges:
        lines.append(f"# HELP {gauge.name} {gauge.help}")
        lines.append(f"# TYPE {gauge.name} gauge")
        for labels, value in gauge.collect():
            lines.append(f"{gauge.name}{_format_labels(labels)} {value}")
        lines.append("")

    # Export histograms
    for histogram in histograms:
        lines.append(f"# HELP {histogram.name} {histogram.help}")
        lines.append(f"# TYPE {histogram.name} histogram")
        for labels, data in histogram.collect():
            for bucket, count in data["buckets"]:
                bucket_labels = {**labels, "le": str(bucket)}
                lines.append(
                    f"{histogram.name}_bucket{_format_labels(bucket_labels)} {count}"
                )
            inf_labels = {**labels, "le": "+Inf"}
            lines.append(
                f"{histogram.name}_bucket{_format_labels(inf_labels)} {data['count']}"
            )
            lines.append(f"{histogram.name}_sum{_format_labels(labels)} {data['sum']}")
            lines.append(
                f"{histogram.name}_count{_format_labels(labels)} {data['count']}"
            )
        lines.append("")

    return "\n".join(lines)


__all__ = [
    # Metric types
    "Counter",
    "Gauge",
    "Histogram",
    # Billing metrics
    "SUBSCRIPTION_EVENTS",
    "SUBSCRIPTION_ACTIVE",
    "USAGE_DEBATES",
    "USAGE_TOKENS",
    "BILLING_REVENUE",
    "PAYMENT_FAILURES",
    # API metrics
    "API_REQUESTS",
    "API_LATENCY",
    "ACTIVE_DEBATES",
    "WEBSOCKET_CONNECTIONS",
    # Agent metrics
    "AGENT_REQUESTS",
    "AGENT_LATENCY",
    "AGENT_TOKENS",
    # Helpers
    "track_request",
    "track_subscription_event",
    "track_debate",
    "track_tokens",
    "track_agent_call",
    # Export
    "generate_metrics",
]
