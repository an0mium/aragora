"""
Observability package for Aragora.

Provides OpenTelemetry distributed tracing and Prometheus metrics.

Usage:
    from aragora.observability import tracer, trace_handler, trace_agent_call
    from aragora.observability import metrics

    @trace_handler("debates.create")
    def handle_create_debate(self, handler):
        ...

See docs/OBSERVABILITY.md for configuration and usage guide.
"""

from __future__ import annotations

from aragora.observability.config import (
    TracingConfig,
    MetricsConfig,
    get_tracing_config,
    get_metrics_config,
    is_tracing_enabled,
    is_metrics_enabled,
)
from aragora.observability.tracing import (
    get_tracer,
    trace_handler,
    trace_agent_call,
    trace_async_handler,
    add_span_attributes,
    record_exception,
)
from aragora.observability.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    AGENT_CALLS,
    AGENT_LATENCY,
    ACTIVE_DEBATES,
    CONSENSUS_RATE,
    record_request,
    record_agent_call,
    track_debate,
)

__all__ = [
    # Config
    "TracingConfig",
    "MetricsConfig",
    "get_tracing_config",
    "get_metrics_config",
    "is_tracing_enabled",
    "is_metrics_enabled",
    # Tracing
    "get_tracer",
    "trace_handler",
    "trace_agent_call",
    "trace_async_handler",
    "add_span_attributes",
    "record_exception",
    # Metrics
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "AGENT_CALLS",
    "AGENT_LATENCY",
    "ACTIVE_DEBATES",
    "CONSENSUS_RATE",
    "record_request",
    "record_agent_call",
    "track_debate",
]
