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
    # Logging
    configure_logging,
    get_logger,
    StructuredLogger,
    LogConfig,
    set_correlation_id,
    get_correlation_id,
    correlation_context,
    # Tracing
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
    shutdown_tracing,
    # Metrics
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
    # Configuration
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
