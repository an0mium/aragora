# Observability Setup Guide

This guide covers the complete observability stack for Aragora: metrics, logging, tracing, and alerting.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Observability Stack                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Metrics   │    │   Logging   │    │   Tracing   │         │
│  │ (Prometheus)│    │   (Loki)    │    │  (Jaeger)   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                     ┌──────▼──────┐                             │
│                     │   Grafana   │                             │
│                     │ Dashboards  │                             │
│                     └──────┬──────┘                             │
│                            │                                    │
│                     ┌──────▼──────┐                             │
│                     │  Alerting   │                             │
│                     │ (PagerDuty) │                             │
│                     └─────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics (Prometheus)

### Enabling Metrics

```python
# aragora/server/unified_server.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
REQUEST_COUNT = Counter(
    'aragora_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'aragora_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

# Business metrics
ACTIVE_DEBATES = Gauge('aragora_active_debates', 'Currently running debates')
WEBSOCKET_CONNECTIONS = Gauge('aragora_websocket_connections', 'Active WebSocket connections')
AGENT_CALLS = Counter('aragora_agent_calls_total', 'Total agent API calls', ['agent', 'status'])
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:8080']
    metrics_path: '/metrics'

  - job_name: 'aragora-workers'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: aragora-worker
        action: keep
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `aragora_requests_total` | Counter | Total HTTP requests |
| `aragora_request_duration_seconds` | Histogram | Request latency |
| `aragora_active_debates` | Gauge | Running debates |
| `aragora_websocket_connections` | Gauge | WebSocket connections |
| `aragora_agent_calls_total` | Counter | Agent API calls |
| `aragora_debate_duration_seconds` | Histogram | Debate completion time |
| `aragora_consensus_reached_total` | Counter | Successful consensus |
| `aragora_elo_updates_total` | Counter | ELO rating updates |

## Logging (Structured JSON)

### Log Configuration

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'trace_id'):
            log_entry['trace_id'] = record.trace_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id

        return json.dumps(log_entry)

# Configure root logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
```

### Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed debugging (disabled in production) |
| INFO | Normal operations, request logging |
| WARNING | Unexpected but handled situations |
| ERROR | Errors requiring attention |
| CRITICAL | System failures requiring immediate action |

### Loki Configuration

```yaml
# loki-config.yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2026-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/index
    cache_location: /loki/cache
  filesystem:
    directory: /loki/chunks
```

### Promtail Configuration

```yaml
# promtail-config.yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: aragora
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
    pipeline_stages:
      - json:
          expressions:
            level: level
            trace_id: trace_id
      - labels:
          level:
          trace_id:
```

## Tracing (OpenTelemetry)

### OpenTelemetry Setup

```python
# aragora/telemetry/setup.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

def setup_tracing():
    # Configure tracer provider
    provider = TracerProvider()

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
    )

    # Add batch processor
    provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument libraries
    AioHttpClientInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()

    return trace.get_tracer("aragora")
```

### Trace Sampling Strategy

```python
# Sampling configuration
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

# Sample 10% of traces in production
sampler = ParentBased(root=TraceIdRatioBased(0.1))

provider = TracerProvider(sampler=sampler)
```

### Adding Custom Spans

```python
from opentelemetry import trace

tracer = trace.get_tracer("aragora.debate")

async def run_debate(task: str, agents: list):
    with tracer.start_as_current_span("debate.run") as span:
        span.set_attribute("debate.task", task)
        span.set_attribute("debate.agent_count", len(agents))

        for round_num in range(3):
            with tracer.start_as_current_span(f"debate.round.{round_num}") as round_span:
                # Round logic here
                round_span.set_attribute("round.number", round_num)
```

### Jaeger Configuration

```yaml
# jaeger-config.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: aragora-jaeger
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
  ingress:
    enabled: true
```

## Grafana Dashboards

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     Aragora Overview                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Request/sec │  │  Error Rate  │  │   P99 Latency│      │
│  │     1,234    │  │    0.05%     │  │    245ms     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Active Debates│  │  WS Conns    │  │ Agent Calls  │      │
│  │      42      │  │    1,567     │  │   12,345     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│         Request Latency Over Time (Graph)                   │
│  ═══════════════════════════════════════════════════════   │
├─────────────────────────────────────────────────────────────┤
│         Error Rate by Endpoint (Graph)                      │
│  ═══════════════════════════════════════════════════════   │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard JSON

```json
{
  "dashboard": {
    "title": "Aragora Overview",
    "panels": [
      {
        "title": "Requests/sec",
        "type": "stat",
        "targets": [{
          "expr": "sum(rate(aragora_requests_total[5m]))"
        }]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [{
          "expr": "sum(rate(aragora_requests_total{status=~\"5..\"}[5m])) / sum(rate(aragora_requests_total[5m])) * 100"
        }]
      },
      {
        "title": "P99 Latency",
        "type": "stat",
        "targets": [{
          "expr": "histogram_quantile(0.99, sum(rate(aragora_request_duration_seconds_bucket[5m])) by (le))"
        }]
      },
      {
        "title": "Request Latency",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.50, sum(rate(aragora_request_duration_seconds_bucket[5m])) by (le))",
          "legendFormat": "P50"
        }, {
          "expr": "histogram_quantile(0.95, sum(rate(aragora_request_duration_seconds_bucket[5m])) by (le))",
          "legendFormat": "P95"
        }, {
          "expr": "histogram_quantile(0.99, sum(rate(aragora_request_duration_seconds_bucket[5m])) by (le))",
          "legendFormat": "P99"
        }]
      }
    ]
  }
}
```

### Importing Dashboards

```bash
# Import dashboard via API
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @dashboards/aragora-overview.json
```

## Alerting

### Alert Manager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#aragora-alerts'
        send_resolved: true

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '$PAGERDUTY_SERVICE_KEY'
        severity: critical

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#aragora-warnings'
        send_resolved: true
```

### Alert Rules

```yaml
# alert-rules.yml
groups:
  - name: aragora.availability
    rules:
      - alert: AragoraDown
        expr: up{job="aragora"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Aragora is down"
          description: "Aragora has been unreachable for more than 1 minute"

      - alert: HighErrorRate
        expr: |
          sum(rate(aragora_requests_total{status=~"5.."}[5m]))
          / sum(rate(aragora_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, sum(rate(aragora_request_duration_seconds_bucket[5m])) by (le)) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"

  - name: aragora.debates
    rules:
      - alert: DebateTimeouts
        expr: rate(aragora_debate_timeouts_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Debate timeouts increasing"

      - alert: LowConsensusRate
        expr: |
          sum(rate(aragora_consensus_reached_total[1h]))
          / sum(rate(aragora_debates_completed_total[1h])) < 0.5
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Low consensus rate"
```

## Environment Variables

```bash
# Metrics
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
METRICS_PORT=9090

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_OUTPUT=stdout

# Tracing
OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger
OTEL_EXPORTER_JAEGER_AGENT_PORT=6831
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1

# Alerting
ALERTMANAGER_URL=http://alertmanager:9093
PAGERDUTY_SERVICE_KEY=your-service-key
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
```

## Verification Checklist

### Metrics
- [ ] `/metrics` endpoint returns Prometheus format
- [ ] Key business metrics are exposed
- [ ] Prometheus scraping successfully
- [ ] Grafana dashboards load correctly

### Logging
- [ ] Logs in JSON format
- [ ] Trace IDs propagated
- [ ] Log levels appropriate
- [ ] Loki ingesting logs

### Tracing
- [ ] Spans created for key operations
- [ ] Trace IDs in logs match Jaeger
- [ ] Sampling rate configured
- [ ] Service map visible in Jaeger

### Alerting
- [ ] Alert rules loaded in Prometheus
- [ ] Test alerts fire correctly
- [ ] PagerDuty integration working
- [ ] Slack notifications arriving
