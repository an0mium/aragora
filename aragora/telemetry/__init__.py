"""
Telemetry module for Aragora.

This is a convenience re-export of aragora.observability for simpler imports.
All functionality is delegated to the observability package.

Usage:
    from aragora.telemetry import trace_handler, record_request, get_logger

    @trace_handler("debates.create")
    def handle_create_debate(self, handler):
        ...

See aragora.observability for full documentation.
"""

# Re-export everything from observability
from aragora.observability import (
    LogConfig,
    MetricsConfig,
    StructuredLogger,
    # Configuration
    TracingConfig,
    add_span_attributes,
    # Logging
    configure_logging,
    correlation_context,
    create_span,
    get_correlation_id,
    get_logger,
    get_metrics_config,
    # Tracing
    get_tracer,
    get_tracing_config,
    is_metrics_enabled,
    is_tracing_enabled,
    measure_async_latency,
    measure_latency,
    record_agent_call,
    record_agent_participation,
    record_cache_hit,
    record_cache_miss,
    record_debate_completion,
    record_exception,
    record_memory_operation,
    record_phase_duration,
    record_request,
    set_consensus_rate,
    set_correlation_id,
    shutdown_tracing,
    # Metrics
    start_metrics_server,
    trace_agent_call,
    trace_async_handler,
    trace_consensus_check,
    trace_debate,
    trace_debate_phase,
    trace_handler,
    trace_memory_operation,
    track_debate,
    track_phase,
    track_websocket_connection,
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
