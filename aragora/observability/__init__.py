"""
Observability package for Aragora.

Provides structured logging, metrics, distributed tracing, and error tracking
for production deployments.

Usage:
    from aragora.observability import configure_logging, get_logger

    # Configure at startup
    configure_logging(environment="production", level="INFO")

    # Get a logger
    logger = get_logger(__name__)
    logger.info("debate_started", debate_id="123", agent_count=5)

Tracing:
    from aragora.observability import trace_handler, create_span, trace_debate_phase

    @trace_handler("debates.create")
    def handle_create_debate(self, handler):
        with create_span("validate_input"):
            ...

Metrics:
    from aragora.observability import record_request, record_agent_call, track_debate

    record_request("GET", "/api/debates", 200, 0.05)
    with track_debate():
        await arena.run()
"""

from aragora.observability.logging import (
    configure_logging,
    get_logger,
    StructuredLogger,
    LogConfig,
    set_correlation_id,
    get_correlation_id,
    correlation_context,
)

from aragora.observability.tracing import (
    get_tracer,
    create_span,
    trace_handler,
    trace_async_handler,
    trace_agent_call,
    trace_debate,
    trace_debate_phase,
    trace_consensus_check,
    trace_memory_operation,
    add_span_attributes,
    record_exception,
    shutdown as shutdown_tracing,
)

from aragora.observability.metrics import (
    start_metrics_server,
    record_request,
    record_agent_call,
    track_debate,
    set_consensus_rate,
    record_memory_operation,
    track_websocket_connection,
    measure_latency,
    measure_async_latency,
    record_debate_completion,
    record_phase_duration,
    record_agent_participation,
    track_phase,
    record_cache_hit,
    record_cache_miss,
)

from aragora.observability.config import (
    TracingConfig,
    MetricsConfig,
    get_tracing_config,
    get_metrics_config,
    is_tracing_enabled,
    is_metrics_enabled,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "StructuredLogger",
    "LogConfig",
    "set_correlation_id",
    "get_correlation_id",
    "correlation_context",
    # Tracing
    "get_tracer",
    "create_span",
    "trace_handler",
    "trace_async_handler",
    "trace_agent_call",
    "trace_debate",
    "trace_debate_phase",
    "trace_consensus_check",
    "trace_memory_operation",
    "add_span_attributes",
    "record_exception",
    "shutdown_tracing",
    # Metrics
    "start_metrics_server",
    "record_request",
    "record_agent_call",
    "track_debate",
    "set_consensus_rate",
    "record_memory_operation",
    "track_websocket_connection",
    "measure_latency",
    "measure_async_latency",
    "record_debate_completion",
    "record_phase_duration",
    "record_agent_participation",
    "track_phase",
    "record_cache_hit",
    "record_cache_miss",
    # Configuration
    "TracingConfig",
    "MetricsConfig",
    "get_tracing_config",
    "get_metrics_config",
    "is_tracing_enabled",
    "is_metrics_enabled",
]
