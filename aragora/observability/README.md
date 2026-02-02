# Observability Module

Production-grade observability infrastructure for Aragora, providing structured logging, Prometheus metrics, OpenTelemetry distributed tracing, SIEM integration, and alerting systems.

## Overview

The observability module is a comprehensive monitoring and diagnostics system that enables:

- **Prometheus Metrics**: Request rates, latencies, agent performance, debate statistics, and 100+ custom metrics
- **Distributed Tracing**: OpenTelemetry-based tracing with automatic instrumentation for HTTP, databases, and LLM calls
- **Structured Logging**: JSON-formatted logs with correlation IDs for request tracing
- **SIEM Integration**: Security event forwarding to Splunk, Elastic, or custom backends
- **Alerting**: SLO-based alerting with Slack, email, and Prometheus AlertManager channels
- **Immutable Audit Logs**: Tamper-evident audit trails with SHA-256 hash chains

## Architecture

```
aragora/observability/
├── __init__.py              # Module exports and unified API
├── config.py                # Configuration for metrics and tracing
├── metrics.py               # Prometheus metrics facade (1700+ lines)
│   └── metrics/             # Specialized metrics submodules
│       ├── base.py          # NoOpMetric and base classes
│       ├── cache.py         # Cache hit/miss metrics
│       ├── consensus.py     # Consensus metrics
│       ├── convergence.py   # Convergence check metrics
│       ├── evidence.py      # Evidence storage metrics
│       ├── explainability.py # Batch explainability metrics
│       ├── governance.py    # Governance store metrics
│       ├── km.py            # Knowledge Mound metrics
│       ├── marketplace.py   # Template marketplace metrics
│       ├── memory.py        # Memory operation metrics
│       ├── notification.py  # Notification delivery metrics
│       ├── security.py      # Encryption and RBAC metrics
│       ├── tts.py           # Text-to-speech metrics
│       ├── webhook.py       # Webhook delivery metrics
│       └── workflow.py      # Workflow template metrics
├── tracing.py               # OpenTelemetry distributed tracing (1400+ lines)
├── otel.py                  # Unified OpenTelemetry setup
├── otlp_export.py           # OTLP exporter configuration
├── siem.py                  # SIEM integration (Splunk, Elastic)
├── alerting.py              # Alert rules and notification channels
├── slo.py                   # SLO monitoring and alerting
├── slo_alert_bridge.py      # SLO-to-alert bridge
├── immutable_log.py         # Tamper-evident audit logging
├── decision_metrics.py      # Decision routing metrics
├── memory_profiler.py       # Memory usage profiling
├── n1_detector.py           # N+1 query detection
├── query_analyzer.py        # SQL query plan analysis
└── trace_correlation.py     # Trace-to-metrics correlation
```

## Key Classes

### Metrics

- **`start_metrics_server(port)`**: Start Prometheus metrics endpoint on specified port
- **`record_request(method, endpoint, status, latency)`**: Record HTTP request metrics
- **`record_agent_call(agent, success, latency)`**: Record agent API call metrics
- **`track_debate()`**: Context manager for tracking debate execution
- **`track_phase(phase)`**: Context manager for tracking debate phase duration

### Tracing

- **`get_tracer()`**: Get the OpenTelemetry tracer instance
- **`create_span(name, attributes)`**: Context manager for creating spans
- **`@trace_handler(name)`**: Decorator to trace HTTP handler methods
- **`@trace_agent_call(agent_name)`**: Decorator to trace agent API calls
- **`@traced(name)`**: Universal decorator for tracing any function
- **`AutoInstrumentation`**: Auto-instrument httpx, aiohttp, asyncpg, redis

### SIEM

- **`SIEMClient`**: Client for forwarding security events
- **`emit_security_event(event_type, details)`**: Emit a security event
- **`emit_auth_event(user_id, action, success)`**: Emit authentication events
- **`emit_data_access_event(user_id, resource, action)`**: Emit data access events

### Alerting

- **`AlertManager`**: Manages alert rules and notifications
- **`AlertRule`**: Defines alert conditions and thresholds
- **`SlackNotificationChannel`**: Send alerts to Slack
- **`EmailNotificationChannel`**: Send alerts via email
- **`PrometheusAlertManagerChannel`**: Forward to Prometheus AlertManager

### Audit Logging

- **`ImmutableAuditLog`**: Tamper-evident audit log with hash chains
- **`AuditEntry`**: Single audit log entry
- **`LocalFileBackend`**: File-based audit storage
- **`S3ObjectLockBackend`**: S3 Object Lock for WORM compliance

## Usage Example

```python
from aragora.observability import (
    # Logging
    configure_logging,
    get_logger,
    # Metrics
    start_metrics_server,
    record_request,
    record_agent_call,
    track_debate,
    # Tracing
    get_tracer,
    create_span,
    trace_handler,
    trace_agent_call,
    traced,
    instrument_all,
    # SIEM
    emit_security_event,
    emit_auth_event,
    # Alerting
    init_alerting,
    create_critical_alert_rules,
)

# Configure logging at startup
configure_logging(environment="production", level="INFO")
logger = get_logger(__name__)

# Start metrics server
start_metrics_server(port=9090)

# Auto-instrument HTTP clients
instrument_all()

# Initialize alerting
init_alerting()

# Log with structured context
logger.info("debate_started", debate_id="123", agent_count=5)

# Record metrics
record_request("POST", "/api/debates", 200, 0.15)
record_agent_call("claude", success=True, latency=1.2)

# Track debate with context manager
with track_debate(outcome="consensus") as ctx:
    ctx["rounds"] = 3
    # ... run debate

# Create tracing span
with create_span("validate_input", {"user_id": "u123"}) as span:
    # ... validation logic
    span.set_attribute("valid", True)

# Use decorator for tracing
@trace_handler("debates.create")
def handle_create_debate(self, handler):
    with create_span("build_arena"):
        arena = Arena(...)
    return arena.run()

# Universal tracing decorator
@traced("user.create", record_args=True, record_result=True)
async def create_user(name: str, email: str) -> User:
    ...

# Emit security events to SIEM
emit_auth_event(user_id="u123", action="login", success=True)
emit_security_event(
    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
    details={"ip": "1.2.3.4", "reason": "multiple failed logins"}
)
```

## Integration Points

### With Debate Engine
- Traces entire debate lifecycle with `trace_debate`
- Records phase durations with `track_phase`
- Tracks agent participation and consensus rates

### With Agent System
- Traces agent API calls with `trace_agent_call`
- Records latency histograms per agent
- Tracks success/error rates

### With Knowledge Mound
- Custom metrics for KM operations
- Cache hit/miss tracking
- Federated query metrics

### With Authentication/RBAC
- Security event emission for auth flows
- RBAC decision metrics
- Audit logging for sensitive operations

### With External Services
- `trace_external_call` for third-party APIs
- `trace_llm_call` for LLM provider calls
- Auto-instrumentation for HTTP clients

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `METRICS_ENABLED` | Enable Prometheus metrics | `true` |
| `METRICS_PORT` | Port for /metrics endpoint | `9090` |
| `OTEL_ENABLED` | Enable OpenTelemetry tracing | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |
| `OTEL_SERVICE_NAME` | Service name for traces | `aragora` |
| `OTEL_SAMPLE_RATE` | Trace sampling rate (0-1) | `1.0` |
| `SIEM_BACKEND` | SIEM backend type | `none` |
| `SIEM_ENDPOINT` | SIEM endpoint URL | - |

## See Also

- `docs/OBSERVABILITY.md` - Full observability configuration guide
- `aragora/logging_config.py` - Structured logging configuration
- `aragora/server/middleware/tracing.py` - HTTP tracing middleware
