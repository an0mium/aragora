# Aragora Alerting Configuration

This directory contains alerting configuration for Aragora.

## Components

### Alertmanager (`alertmanager.yml`)
Routes alerts to appropriate channels based on severity and type.

### Prometheus Rules (`prometheus-rules.yml`)
Defines alert conditions based on metrics.

## Alert Routing

| Severity | Channel | Response Time |
|----------|---------|---------------|
| Critical | PagerDuty + Slack | 15 minutes |
| High | Slack (#alerts-high) + Email | 1 hour |
| Warning | Slack (#alerts) | 4 hours |

## Setup

### 1. Environment Variables

Set these in your deployment:

```bash
# Slack
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx

# PagerDuty
export PAGERDUTY_SERVICE_KEY=xxx

# Email (optional)
export SMTP_USERNAME=xxx
export SMTP_PASSWORD=xxx
```

### 2. Deploy Alertmanager

```bash
# Docker
docker run -d \
  -v $(pwd)/alertmanager.yml:/etc/alertmanager/alertmanager.yml \
  -v $(pwd)/templates:/etc/alertmanager/templates \
  -p 9093:9093 \
  prom/alertmanager

# Kubernetes
kubectl apply -f k8s/alertmanager.yaml
```

### 3. Configure Prometheus

Add to `prometheus.yml`:

```yaml
rule_files:
  - /etc/prometheus/rules/prometheus-rules.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

## Alert Channels

### Slack Channels

| Channel | Purpose |
|---------|---------|
| `#aragora-alerts` | Default notifications |
| `#aragora-alerts-high` | High/Critical alerts |
| `#aragora-api` | API-specific alerts |
| `#aragora-infrastructure` | DB/Redis alerts |

### PagerDuty

Critical alerts trigger PagerDuty incidents for immediate response.

## Testing Alerts

```bash
# Send test alert to Alertmanager
curl -X POST http://localhost:9093/api/v2/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning"
    },
    "annotations": {
      "summary": "Test alert",
      "description": "This is a test"
    }
  }]'
```

## Alert Definitions

### Critical Alerts
- `InstanceDown` - Backend instance unreachable
- `HealthCheckFailing` - Health endpoint returning errors
- `DatabasePoolExhausted` - No DB connections available

### High Severity Alerts
- `HighAPILatency` - p95 latency > 2s
- `APIErrorRate` - Error rate > 5%
- `AgentTimeoutRateHigh` - Agent timeouts > 10%
- `RedisConnectionIssues` - Redis connectivity problems

### Warning Alerts
- `HighMemoryUsage` - Memory > 1.5GB
- `HighCPUUsage` - CPU > 80%
- `ExternalAPIQuotaLow` - API quota running low
- `RateLimitingExcessive` - High rate limit rejections

## Silencing Alerts

During maintenance:

```bash
# Via Alertmanager API
curl -X POST http://localhost:9093/api/v2/silences \
  -H "Content-Type: application/json" \
  -d '{
    "matchers": [{"name": "alertname", "value": ".*", "isRegex": true}],
    "startsAt": "2026-01-20T10:00:00Z",
    "endsAt": "2026-01-20T12:00:00Z",
    "createdBy": "admin",
    "comment": "Maintenance window"
  }'
```

## Runbook Links

All alerts include `runbook_url` pointing to:
`https://docs.aragora.ai/runbook#<alert-name>`

See `docs/RUNBOOK.md` for detailed response procedures.
