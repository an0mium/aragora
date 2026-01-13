# Aragora Observability Stack

This directory contains configuration for Prometheus, Grafana, and Alertmanager for monitoring Aragora in production.

## Quick Start

```bash
cd deploy/observability
docker-compose up -d
```

Access the tools:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

## Components

### Prometheus (`prometheus.yml`)
Metrics collection from Aragora endpoints:
- API server metrics at `/metrics`
- WebSocket server metrics
- Redis metrics (optional)

### Alert Rules (`alerts.rules`)
Pre-configured alerts for:
- **Availability**: Service down, high latency
- **Errors**: High error rate, rate limit exhaustion
- **Resources**: Memory, CPU, connections
- **Debates**: Timeout rate, consensus rate
- **Circuit Breakers**: Open breakers
- **Redis**: Down, high memory

### Grafana Dashboards
Located in `../grafana/`:
- `aragora-overview.json` - System health overview
- `debate-analytics.json` - Debate metrics
- `api-performance.json` - API performance

### Alertmanager (`alertmanager.yml`)
Alert routing and notification. Configure:
- Slack webhooks
- PagerDuty
- Email notifications

## Configuration

### Environment Variables

For Aragora server:
```bash
# Enable metrics endpoint
METRICS_ENABLED=true
METRICS_PORT=8080

# Sentry integration
SENTRY_DSN=https://xxx@sentry.io/yyy
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

### Production Setup

1. **Update Prometheus targets** in `prometheus.yml` to match your deployment
2. **Configure Alertmanager** with Slack/PagerDuty credentials
3. **Import dashboards** to your Grafana instance
4. **Set up alert routing** based on severity

## Redis Metrics (Optional)

To monitor Redis:
```bash
docker-compose --profile redis up -d
```

This starts the Redis exporter on port 9121.

## Useful Queries

### Request Rate
```promql
sum(rate(http_requests_total{job="aragora-api"}[5m])) by (status)
```

### P95 Latency
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Debate Success Rate
```promql
sum(rate(aragora_debates_total{status="completed"}[1h]))
/
sum(rate(aragora_debates_total[1h]))
```

### Active Debates
```promql
aragora_active_debates
```
