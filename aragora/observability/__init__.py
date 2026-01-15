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

from aragora.observability.config import (
    MetricsConfig,
    TracingConfig,
    get_metrics_config,
    get_tracing_config,
    is_metrics_enabled,
    is_tracing_enabled,
)
from aragora.observability.logging import (
    LogConfig,
    StructuredLogger,
    configure_logging,
    correlation_context,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)
from aragora.observability.metrics import (
    measure_async_latency,
    measure_latency,
    record_agent_call,
    record_agent_participation,
    record_cache_hit,
    record_cache_miss,
    record_debate_completion,
    record_memory_operation,
    record_phase_duration,
    record_request,
    set_consensus_rate,
    start_metrics_server,
    track_debate,
    track_phase,
    track_websocket_connection,
)
from aragora.observability.tracing import (
    add_span_attributes,
    create_span,
    get_tracer,
    record_exception,
    trace_agent_call,
    trace_async_handler,
    trace_consensus_check,
    trace_debate,
    trace_debate_phase,
    trace_handler,
    trace_memory_operation,
)
from aragora.observability.tracing import (
    shutdown as shutdown_tracing,
)
from aragora.observability.siem import (
    SIEMBackend,
    SIEMConfig,
    SecurityEventType,
    SecurityEvent,
    SIEMClient,
    get_siem_client,
    emit_security_event,
    emit_auth_event,
    emit_data_access_event,
    emit_privacy_event,
    shutdown_siem,
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
    # SIEM
    "SIEMBackend",
    "SIEMConfig",
    "SecurityEventType",
    "SecurityEvent",
    "SIEMClient",
    "get_siem_client",
    "emit_security_event",
    "emit_auth_event",
    "emit_data_access_event",
    "emit_privacy_event",
    "shutdown_siem",
]
