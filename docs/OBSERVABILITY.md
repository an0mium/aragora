# Observability Guide

This guide covers distributed tracing, metrics, and monitoring for Aragora deployments.

## Table of Contents

- [Overview](#overview)
- [OpenTelemetry Tracing](#opentelemetry-tracing)
- [Prometheus Metrics](#prometheus-metrics)
- [Grafana Dashboards](#grafana-dashboards)
- [Local Development](#local-development)
- [Production Setup](#production-setup)
- [Troubleshooting](#troubleshooting)

---

## Overview

Aragora provides comprehensive observability through:

1. **Distributed Tracing** - OpenTelemetry spans for request flows
2. **Prometheus Metrics** - Request rates, latencies, and business metrics
3. **Grafana Dashboards** - Pre-built visualizations

### Architecture

```
                                    ┌─────────────────┐
                                    │   Grafana       │
                                    │   Dashboard     │
                                    └────────┬────────┘
                                             │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
            ┌───────▼───────┐       ┌───────▼───────┐       ┌──────▼──────┐
            │  Prometheus   │       │    Jaeger     │       │   Alerting  │
            │   (Metrics)   │       │   (Traces)    │       │             │
            └───────▲───────┘       └───────▲───────┘       └─────────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Aragora Server      │
                    │  ┌─────────────────┐  │
                    │  │ Observability   │  │
                    │  │    Module       │  │
                    │  └─────────────────┘  │
                    └───────────────────────┘
```

---

## OpenTelemetry Tracing

### Configuration

Set environment variables to enable tracing:

```bash
# Enable tracing
export OTEL_ENABLED=true

# OTLP collector endpoint (Jaeger, Zipkin, etc.)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Service name in traces
export OTEL_SERVICE_NAME=aragora

# Sample rate (0.0-1.0, 1.0 = 100%)
export OTEL_SAMPLE_RATE=1.0
```

### Usage

#### Trace HTTP Handlers

```python
from aragora.observability import trace_handler

class DebatesHandler(BaseHandler):
    @trace_handler("debates.create")
    def _create_debate(self, handler):
        # Automatically creates span with HTTP attributes
        ...
```

#### Trace Agent Calls

```python
from aragora.observability import trace_agent_call

class ClaudeAgent:
    @trace_agent_call("anthropic")
    async def respond(self, prompt: str) -> str:
        # Automatically creates span with agent attributes
        ...
```

#### Manual Spans

```python
from aragora.observability import get_tracer, create_span

tracer = get_tracer()

# Using context manager
with create_span("custom_operation", {"key": "value"}) as span:
    result = do_work()
    span.set_attribute("result_size", len(result))

# Using tracer directly
with tracer.start_as_current_span("another_operation") as span:
    span.set_attribute("custom.attribute", "value")
    ...
```

### Span Attributes

Automatically captured attributes:

| Attribute | Description |
|-----------|-------------|
| `http.path` | Request path |
| `http.method` | HTTP method |
| `http.status_code` | Response status |
| `agent.name` | Agent identifier |
| `agent.prompt_length` | Input prompt size |
| `agent.response_length` | Output response size |

---

## Prometheus Metrics

### Available Metrics

#### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aragora_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `aragora_request_latency_seconds` | Histogram | endpoint | Request latency |

#### Agent Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aragora_agent_calls_total` | Counter | agent, status | Total agent API calls |
| `aragora_agent_latency_seconds` | Histogram | agent | Agent response latency |

#### Debate Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aragora_active_debates` | Gauge | - | Currently active debates |
| `aragora_consensus_rate` | Gauge | - | Rate of consensus reached |

#### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aragora_websocket_connections` | Gauge | - | Active WebSocket connections |
| `aragora_memory_operations_total` | Counter | operation, tier | Memory operations |

### Configuration

```bash
# Enable metrics (default: true)
export METRICS_ENABLED=true

# Metrics endpoint port (default: 9090)
export METRICS_PORT=9090
```

### Usage

```python
from aragora.observability import (
    record_request,
    record_agent_call,
    track_debate,
)

# Record an HTTP request
record_request("GET", "/api/debates", 200, 0.05)

# Record an agent call
record_agent_call("anthropic-api", success=True, latency=1.2)

# Track active debates
with track_debate():
    await arena.run()
```

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:9090']
    scrape_interval: 15s
```

---

## Grafana Dashboards

### Import Dashboard

1. Open Grafana
2. Go to Dashboards → Import
3. Upload `deploy/grafana/aragora-dashboard.json`
4. Select your Prometheus datasource
5. Click Import

### Dashboard Panels

| Panel | Description |
|-------|-------------|
| Request Rate | Requests per second |
| P95 Latency | 95th percentile response time |
| Active Debates | Currently running debates |
| Consensus Rate | Rate of successful consensus |
| Request Rate by Endpoint | Per-endpoint request rates |
| Request Latency by Endpoint | Per-endpoint latencies |
| Agent Calls by Status | Success/error rates by agent |
| Agent Response Latency | Per-agent latencies |
| Error Rates | 4xx and 5xx error percentages |
| WebSocket Connections | Active WebSocket clients |

---

## Local Development

### Quick Start with Docker Compose

Create `docker-compose.observability.yml`:

```yaml
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./deploy/grafana:/var/lib/grafana/dashboards
```

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['host.docker.internal:9090']
```

Start:

```bash
docker-compose -f docker-compose.observability.yml up -d
```

### Start Aragora with Observability

```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export METRICS_ENABLED=true

aragora serve --api-port 8080 --ws-port 8765
```

### View Traces

Open Jaeger UI: http://localhost:16686

### View Metrics

Open Grafana: http://localhost:3000 (admin/admin)

---

## Production Setup

### Kubernetes with Jaeger Operator

```yaml
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
```

### Prometheus Operator

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aragora
  labels:
    app: aragora
spec:
  selector:
    matchLabels:
      app: aragora
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

### Environment Variables

```yaml
# Kubernetes deployment
env:
  - name: OTEL_ENABLED
    value: "true"
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://jaeger-collector.observability:4317"
  - name: OTEL_SERVICE_NAME
    value: "aragora"
  - name: OTEL_SAMPLE_RATE
    value: "0.1"  # 10% sampling in production
  - name: METRICS_ENABLED
    value: "true"
  - name: METRICS_PORT
    value: "9090"
```

---

## Troubleshooting

### Traces Not Appearing

1. Check OTEL_ENABLED is "true"
2. Verify OTLP endpoint is reachable
3. Check Jaeger logs for errors
4. Verify sample rate is > 0

```bash
# Test OTLP endpoint
curl -v http://localhost:4317
```

### Metrics Not Scraped

1. Check METRICS_ENABLED is "true"
2. Verify port 9090 is accessible
3. Check Prometheus targets in UI
4. Verify network policies allow scraping

```bash
# Test metrics endpoint
curl http://localhost:9090/metrics
```

### High Latency from Tracing

1. Reduce sample rate (OTEL_SAMPLE_RATE=0.1)
2. Use batch export (default)
3. Increase export timeout

### Memory Issues

1. Limit trace context propagation
2. Reduce histogram bucket count
3. Enable metric aggregation

---

## See Also

- [KUBERNETES.md](./KUBERNETES.md) - Kubernetes deployment
- [RATE_LIMITING.md](./RATE_LIMITING.md) - Rate limiting configuration
- [SECURITY.md](./SECURITY.md) - Security configuration
