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

    # ========================================================================
    # Knowledge Mound Bidirectional Flow Metrics
    # ========================================================================

    KM_INBOUND_EVENTS = Counter(
        "aragora_km_inbound_events_total",
        "Events flowing INTO Knowledge Mound from other subsystems",
        ["source", "event_type"],  # source: memory, belief, rlm, ranking, insights, etc.
    )

    KM_OUTBOUND_EVENTS = Counter(
        "aragora_km_outbound_events_total",
        "Events flowing OUT of Knowledge Mound to other subsystems",
        ["target", "event_type"],  # target: memory, belief, debate, trickster, etc.
    )

    KM_ADAPTER_SYNC = Counter(
        "aragora_km_adapter_sync_total",
        "Adapter sync operations to/from Knowledge Mound",
        ["adapter", "direction", "status"],  # direction: to_mound, from_mound
    )

    KM_ADAPTER_SYNC_DURATION = Histogram(
        "aragora_km_adapter_sync_duration_seconds",
        "Adapter sync operation duration",
        ["adapter", "direction"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30],
    )

    KM_STALENESS_CHECKS = Counter(
        "aragora_km_staleness_checks_total",
        "Staleness check operations",
        ["workspace", "status"],  # status: completed, failed, skipped
    )

    KM_STALE_NODES_FOUND = Gauge(
        "aragora_km_stale_nodes_found",
        "Number of stale nodes found in last check",
        ["workspace"],
    )

    KM_NODES_BY_SOURCE = Gauge(
        "aragora_km_nodes_by_source",
        "Knowledge nodes by source type",
        ["source"],
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
        # KM bidirectional flow metrics
        self.km_inbound_events: Dict[str, Dict[str, int]] = {}
        self.km_outbound_events: Dict[str, Dict[str, int]] = {}
        self.km_adapter_syncs: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.km_adapter_sync_durations: Dict[str, Dict[str, list]] = {}
        self.km_staleness_checks: Dict[str, Dict[str, int]] = {}
        self.km_stale_nodes_found: Dict[str, int] = {}
        self.km_nodes_by_source: Dict[str, int] = {}

    def record_event(self, event_type: str) -> None:
        self.events_total[event_type] = self.events_total.get(event_type, 0) + 1

    def record_handler_call(self, handler: str, status: str) -> None:
        if handler not in self.handler_calls:
            self.handler_calls[handler] = {}
        self.handler_calls[handler][status] = self.handler_calls[handler].get(status, 0) + 1

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

    def record_km_inbound_event(self, source: str, event_type: str) -> None:
        if source not in self.km_inbound_events:
            self.km_inbound_events[source] = {}
        self.km_inbound_events[source][event_type] = (
            self.km_inbound_events[source].get(event_type, 0) + 1
        )

    def record_km_outbound_event(self, target: str, event_type: str) -> None:
        if target not in self.km_outbound_events:
            self.km_outbound_events[target] = {}
        self.km_outbound_events[target][event_type] = (
            self.km_outbound_events[target].get(event_type, 0) + 1
        )

    def record_km_adapter_sync(
        self, adapter: str, direction: str, status: str, duration: float = None
    ) -> None:
        if adapter not in self.km_adapter_syncs:
            self.km_adapter_syncs[adapter] = {}
        if direction not in self.km_adapter_syncs[adapter]:
            self.km_adapter_syncs[adapter][direction] = {}
        self.km_adapter_syncs[adapter][direction][status] = (
            self.km_adapter_syncs[adapter][direction].get(status, 0) + 1
        )
        if duration is not None:
            if adapter not in self.km_adapter_sync_durations:
                self.km_adapter_sync_durations[adapter] = {}
            if direction not in self.km_adapter_sync_durations[adapter]:
                self.km_adapter_sync_durations[adapter][direction] = []
            self.km_adapter_sync_durations[adapter][direction].append(duration)
            # Keep only last 1000 samples
            if len(self.km_adapter_sync_durations[adapter][direction]) > 1000:
                self.km_adapter_sync_durations[adapter][direction] = self.km_adapter_sync_durations[
                    adapter
                ][direction][-1000:]

    def record_km_staleness_check(self, workspace: str, status: str) -> None:
        if workspace not in self.km_staleness_checks:
            self.km_staleness_checks[workspace] = {}
        self.km_staleness_checks[workspace][status] = (
            self.km_staleness_checks[workspace].get(status, 0) + 1
        )

    def set_km_stale_nodes_found(self, workspace: str, count: int) -> None:
        self.km_stale_nodes_found[workspace] = count

    def set_km_nodes_by_source(self, source: str, count: int) -> None:
        self.km_nodes_by_source[source] = count

    def get_metrics_text(self) -> str:
        """Generate OpenMetrics-style text output."""
        lines = []
        lines.append("# HELP aragora_cross_pollination_events_total Total cross-pollination events")
        lines.append("# TYPE aragora_cross_pollination_events_total counter")
        for event_type, count in self.events_total.items():
            lines.append(
                f'aragora_cross_pollination_events_total{{event_type="{event_type}"}} {count}'
            )

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

        # KM Inbound Events
        lines.append("")
        lines.append("# HELP aragora_km_inbound_events_total Events flowing INTO Knowledge Mound")
        lines.append("# TYPE aragora_km_inbound_events_total counter")
        for source, events in self.km_inbound_events.items():
            for event_type, count in events.items():
                lines.append(
                    f'aragora_km_inbound_events_total{{source="{source}",event_type="{event_type}"}} {count}'
                )

        # KM Outbound Events
        lines.append("")
        lines.append(
            "# HELP aragora_km_outbound_events_total Events flowing OUT of Knowledge Mound"
        )
        lines.append("# TYPE aragora_km_outbound_events_total counter")
        for target, events in self.km_outbound_events.items():
            for event_type, count in events.items():
                lines.append(
                    f'aragora_km_outbound_events_total{{target="{target}",event_type="{event_type}"}} {count}'
                )

        # KM Adapter Syncs
        lines.append("")
        lines.append("# HELP aragora_km_adapter_sync_total Adapter sync operations")
        lines.append("# TYPE aragora_km_adapter_sync_total counter")
        for adapter, directions in self.km_adapter_syncs.items():
            for direction, statuses in directions.items():
                for status, count in statuses.items():
                    lines.append(
                        f'aragora_km_adapter_sync_total{{adapter="{adapter}",direction="{direction}",status="{status}"}} {count}'
                    )

        # KM Staleness Checks
        lines.append("")
        lines.append("# HELP aragora_km_staleness_checks_total Staleness check operations")
        lines.append("# TYPE aragora_km_staleness_checks_total counter")
        for workspace, statuses in self.km_staleness_checks.items():
            for status, count in statuses.items():
                lines.append(
                    f'aragora_km_staleness_checks_total{{workspace="{workspace}",status="{status}"}} {count}'
                )

        # KM Stale Nodes Found
        lines.append("")
        lines.append("# HELP aragora_km_stale_nodes_found Stale nodes found in last check")
        lines.append("# TYPE aragora_km_stale_nodes_found gauge")
        for workspace, count in self.km_stale_nodes_found.items():
            lines.append(f'aragora_km_stale_nodes_found{{workspace="{workspace}"}} {count}')

        # KM Nodes by Source
        lines.append("")
        lines.append("# HELP aragora_km_nodes_by_source Knowledge nodes by source type")
        lines.append("# TYPE aragora_km_nodes_by_source gauge")
        for source, count in self.km_nodes_by_source.items():
            lines.append(f'aragora_km_nodes_by_source{{source="{source}"}} {count}')

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


# ============================================================================
# Knowledge Mound Bidirectional Flow API
# ============================================================================


def record_km_inbound_event(source: str, event_type: str) -> None:
    """Record an event flowing INTO Knowledge Mound.

    Args:
        source: Source subsystem (memory, belief, rlm, ranking, insights, etc.)
        event_type: Type of event being ingested
    """
    if PROMETHEUS_AVAILABLE:
        KM_INBOUND_EVENTS.labels(source=source, event_type=event_type).inc()
    else:
        get_fallback_metrics().record_km_inbound_event(source, event_type)


def record_km_outbound_event(target: str, event_type: str) -> None:
    """Record an event flowing OUT of Knowledge Mound.

    Args:
        target: Target subsystem (memory, belief, debate, trickster, etc.)
        event_type: Type of event being dispatched
    """
    if PROMETHEUS_AVAILABLE:
        KM_OUTBOUND_EVENTS.labels(target=target, event_type=event_type).inc()
    else:
        get_fallback_metrics().record_km_outbound_event(target, event_type)


def record_km_adapter_sync(
    adapter: str,
    direction: str,
    status: str,
    duration: float = None,
) -> None:
    """Record an adapter sync operation.

    Args:
        adapter: Adapter name (ranking, rlm, continuum, etc.)
        direction: 'to_mound' or 'from_mound'
        status: 'success', 'failure', or 'skipped'
        duration: Optional duration in seconds
    """
    if PROMETHEUS_AVAILABLE:
        KM_ADAPTER_SYNC.labels(adapter=adapter, direction=direction, status=status).inc()
        if duration is not None:
            KM_ADAPTER_SYNC_DURATION.labels(adapter=adapter, direction=direction).observe(duration)
    else:
        get_fallback_metrics().record_km_adapter_sync(adapter, direction, status, duration)


def record_km_staleness_check(workspace: str, status: str, stale_count: int = 0) -> None:
    """Record a staleness check operation.

    Args:
        workspace: Workspace ID
        status: 'completed', 'failed', or 'skipped'
        stale_count: Number of stale nodes found (for completed checks)
    """
    if PROMETHEUS_AVAILABLE:
        KM_STALENESS_CHECKS.labels(workspace=workspace, status=status).inc()
        if status == "completed":
            KM_STALE_NODES_FOUND.labels(workspace=workspace).set(stale_count)
    else:
        fallback = get_fallback_metrics()
        fallback.record_km_staleness_check(workspace, status)
        if status == "completed":
            fallback.set_km_stale_nodes_found(workspace, stale_count)


def update_km_nodes_by_source(source: str, count: int) -> None:
    """Update the count of knowledge nodes by source type.

    Args:
        source: Source type (fact, claim, memory, evidence, consensus, etc.)
        count: Current count of nodes from this source
    """
    if PROMETHEUS_AVAILABLE:
        KM_NODES_BY_SOURCE.labels(source=source).set(count)
    else:
        get_fallback_metrics().set_km_nodes_by_source(source, count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "record_event_dispatched",
    "record_handler_call",
    "set_circuit_breaker_state",
    "update_subscriber_count",
    "get_cross_pollination_metrics_text",
    # KM Bidirectional Flow Metrics
    "record_km_inbound_event",
    "record_km_outbound_event",
    "record_km_adapter_sync",
    "record_km_staleness_check",
    "update_km_nodes_by_source",
]
