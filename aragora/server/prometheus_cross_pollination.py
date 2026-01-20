"""
Prometheus metrics for cross-pollination event system.

Provides OpenMetrics-compliant metrics for monitoring:
- Event dispatch (counts, latency per handler)
- Circuit breaker states
- Handler success/failure rates
"""

from typing import Dict, Optional

# Try to import prometheus_client, fall back to simple implementation
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Metric Definitions (when prometheus_client is available)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Event dispatch metrics
    CROSS_POLL_EVENTS_TOTAL = Counter(
        "aragora_cross_pollination_events_total",
        "Total cross-pollination events dispatched",
        ["event_type"],
    )

    CROSS_POLL_HANDLER_CALLS = Counter(
        "aragora_cross_pollination_handler_calls_total",
        "Total handler invocations",
        ["handler", "status"],  # status: success, failure, skipped
    )

    CROSS_POLL_HANDLER_DURATION = Histogram(
        "aragora_cross_pollination_handler_duration_seconds",
        "Handler execution time",
        ["handler"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    )

    CROSS_POLL_CIRCUIT_BREAKER = Gauge(
        "aragora_cross_pollination_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open)",
        ["handler"],
    )

    CROSS_POLL_SUBSCRIBERS = Gauge(
        "aragora_cross_pollination_subscribers",
        "Number of registered subscribers",
        ["event_type"],
    )


# ============================================================================
# Fallback Implementation (when prometheus_client not available)
# ============================================================================


class FallbackMetrics:
    """Simple metrics accumulator when prometheus_client is unavailable."""

    def __init__(self) -> None:
        self.events_total: Dict[str, int] = {}
        self.handler_calls: Dict[str, Dict[str, int]] = {}
        self.handler_durations: Dict[str, list] = {}
        self.circuit_breaker_states: Dict[str, int] = {}
        self.subscriber_counts: Dict[str, int] = {}

    def record_event(self, event_type: str) -> None:
        self.events_total[event_type] = self.events_total.get(event_type, 0) + 1

    def record_handler_call(self, handler: str, status: str) -> None:
        if handler not in self.handler_calls:
            self.handler_calls[handler] = {}
        self.handler_calls[handler][status] = (
            self.handler_calls[handler].get(status, 0) + 1
        )

    def record_handler_duration(self, handler: str, duration: float) -> None:
        if handler not in self.handler_durations:
            self.handler_durations[handler] = []
        self.handler_durations[handler].append(duration)
        # Keep only last 1000 samples
        if len(self.handler_durations[handler]) > 1000:
            self.handler_durations[handler] = self.handler_durations[handler][-1000:]

    def set_circuit_breaker_state(self, handler: str, state: int) -> None:
        self.circuit_breaker_states[handler] = state

    def set_subscriber_count(self, event_type: str, count: int) -> None:
        self.subscriber_counts[event_type] = count

    def get_metrics_text(self) -> str:
        """Generate OpenMetrics-style text output."""
        lines = []
        lines.append("# HELP aragora_cross_pollination_events_total Total cross-pollination events")
        lines.append("# TYPE aragora_cross_pollination_events_total counter")
        for event_type, count in self.events_total.items():
            lines.append(f'aragora_cross_pollination_events_total{{event_type="{event_type}"}} {count}')

        lines.append("")
        lines.append("# HELP aragora_cross_pollination_handler_calls_total Total handler calls")
        lines.append("# TYPE aragora_cross_pollination_handler_calls_total counter")
        for handler, statuses in self.handler_calls.items():
            for status, count in statuses.items():
                lines.append(
                    f'aragora_cross_pollination_handler_calls_total{{handler="{handler}",status="{status}"}} {count}'
                )

        lines.append("")
        lines.append("# HELP aragora_cross_pollination_circuit_breaker_state Circuit breaker state")
        lines.append("# TYPE aragora_cross_pollination_circuit_breaker_state gauge")
        for handler, state in self.circuit_breaker_states.items():
            lines.append(
                f'aragora_cross_pollination_circuit_breaker_state{{handler="{handler}"}} {state}'
            )

        lines.append("")
        lines.append("# HELP aragora_cross_pollination_subscribers Subscriber count")
        lines.append("# TYPE aragora_cross_pollination_subscribers gauge")
        for event_type, count in self.subscriber_counts.items():
            lines.append(
                f'aragora_cross_pollination_subscribers{{event_type="{event_type}"}} {count}'
            )

        return "\n".join(lines)


# Singleton fallback instance
_fallback_metrics: Optional[FallbackMetrics] = None


def get_fallback_metrics() -> FallbackMetrics:
    """Get or create the fallback metrics instance."""
    global _fallback_metrics
    if _fallback_metrics is None:
        _fallback_metrics = FallbackMetrics()
    return _fallback_metrics


# ============================================================================
# Unified API
# ============================================================================


def record_event_dispatched(event_type: str) -> None:
    """Record an event being dispatched."""
    if PROMETHEUS_AVAILABLE:
        CROSS_POLL_EVENTS_TOTAL.labels(event_type=event_type).inc()
    else:
        get_fallback_metrics().record_event(event_type)


def record_handler_call(handler: str, status: str, duration: Optional[float] = None) -> None:
    """Record a handler invocation.

    Args:
        handler: Handler name
        status: 'success', 'failure', or 'skipped'
        duration: Optional execution time in seconds
    """
    if PROMETHEUS_AVAILABLE:
        CROSS_POLL_HANDLER_CALLS.labels(handler=handler, status=status).inc()
        if duration is not None:
            CROSS_POLL_HANDLER_DURATION.labels(handler=handler).observe(duration)
    else:
        fallback = get_fallback_metrics()
        fallback.record_handler_call(handler, status)
        if duration is not None:
            fallback.record_handler_duration(handler, duration)


def set_circuit_breaker_state(handler: str, is_open: bool) -> None:
    """Set circuit breaker state for a handler.

    Args:
        handler: Handler name
        is_open: True if circuit is open (failing), False if closed (healthy)
    """
    state = 1 if is_open else 0
    if PROMETHEUS_AVAILABLE:
        CROSS_POLL_CIRCUIT_BREAKER.labels(handler=handler).set(state)
    else:
        get_fallback_metrics().set_circuit_breaker_state(handler, state)


def update_subscriber_count(event_type: str, count: int) -> None:
    """Update the subscriber count for an event type."""
    if PROMETHEUS_AVAILABLE:
        CROSS_POLL_SUBSCRIBERS.labels(event_type=event_type).set(count)
    else:
        get_fallback_metrics().set_subscriber_count(event_type, count)


def get_cross_pollination_metrics_text() -> str:
    """Get cross-pollination metrics as text (for non-prometheus endpoint)."""
    if PROMETHEUS_AVAILABLE:
        # Use prometheus_client's generate_latest for full output
        from prometheus_client import REGISTRY, generate_latest
        return generate_latest(REGISTRY).decode("utf-8")
    else:
        return get_fallback_metrics().get_metrics_text()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "record_event_dispatched",
    "record_handler_call",
    "set_circuit_breaker_state",
    "update_subscriber_count",
    "get_cross_pollination_metrics_text",
]
