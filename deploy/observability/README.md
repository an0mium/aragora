# Aragora Observability Stack

Complete monitoring, alerting, and logging infrastructure for Aragora production deployments.

## Quick Start

```bash
cd deploy/observability
docker-compose up -d
```

Access the tools:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

---

## Architecture

```
                                    ┌─────────────────┐
                                    │    PagerDuty    │
                                    │      Slack      │
                                    └────────▲────────┘
                                             │
                                    ┌────────┴────────┐
                                    │  Alertmanager   │
                                    │    :9093        │
                                    └────────▲────────┘
                                             │
┌──────────────┐                    ┌────────┴────────┐
│   Aragora    │──────metrics──────▶│   Prometheus    │
│   API/WS     │      /metrics      │    :9090        │
│   :8080      │                    └────────┬────────┘
└──────────────┘                             │
                                    ┌────────▼────────┐
┌──────────────┐                    │    Grafana      │
│    Redis     │──────metrics──────▶│    :3000        │
│   :6379      │      :9121         └─────────────────┘
└──────────────┘
```

---

## Components

### Prometheus (`prometheus.yml`)

Metrics collection from Aragora endpoints:
- API server metrics at `/metrics`
- WebSocket server metrics
- Redis metrics (via redis_exporter)
- Kubernetes pod/container metrics (via kube-state-metrics)

**Key Scrape Targets:**
```yaml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:8080']
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Alert Rules (`alerts.rules`)

Pre-configured alerts organized by category:

| Category | Alerts | Severity |
|----------|--------|----------|
| **Availability** | `AragoraDown`, `HighLatency`, `CriticalLatency` | Critical/Warning |
| **Errors** | `HighErrorRate`, `RateLimitExhaustion`, `DebateTimeoutRate` | Critical/Warning |
| **Resources** | `HighMemoryUsage`, `HighCPUUsage`, `TooManyOpenConnections` | Warning |
| **Debates** | `NoActiveDebates`, `LowConsensusRate`, `AgentHighErrorRate` | Warning/Info |
| **Circuit Breakers** | `CircuitBreakerOpen`, `MultipleCircuitBreakersOpen` | Warning/Critical |
| **Redis** | `RedisDown`, `RedisHighMemory` | Critical/Warning |
| **Database** | `DatabaseConnectionPoolExhausted`, `SlowQueries`, `PostgreSQLDown` | Critical/Warning |
| **Agents** | `AgentCircuitOpen`, `AllAgentsUnavailable`, `AgentQuotaExhausted` | Warning/Critical |
| **WebSocket** | `WebSocketConnectionSpike`, `WebSocketHighDropRate` | Warning |
| **Queue** | `JobQueueBacklog`, `JobQueueCritical`, `StaleJobs` | Warning/Critical |
| **SLO** | `SLOAvailabilityBreach`, `SLOLatencyBreach`, `SLODebateSuccessBreach` | Critical/Warning |
| **Certificates** | `TLSCertificateExpiringSoon`, `TLSCertificateExpired` | Critical/Warning |
| **Billing** | `StripeWebhookFailures`, `BillingSubscriptionSyncLag` | Warning |
| **Nomic Loop** | `NomicLoopStuck`, `NomicLoopFailureRate` | Warning |

### Grafana Dashboards

Located in `../grafana/dashboards/`:

| Dashboard | Description | Use Case |
|-----------|-------------|----------|
| `production-operations.json` | Unified on-call view | Incident response |
| `debate-metrics.json` | Debate analytics | Performance analysis |
| `api-latency.json` | API latency breakdown | Latency debugging |
| `agent-performance.json` | Per-agent metrics | Agent troubleshooting |
| `slo-tracking.json` | SLO compliance | Reliability reporting |

**Import to Grafana:**
```bash
# Via API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @../grafana/dashboards/production-operations.json

# Or use the Grafana UI: Dashboards > Import > Upload JSON
```

### Alertmanager (`alertmanager.yml`)

Alert routing and notification configuration.

**Configure receivers:**
```yaml
receivers:
  - name: 'slack-critical'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/XXX/YYY/ZZZ'
        channel: '#aragora-alerts-critical'
        send_resolved: true
        title: '{{ template "slack.title" . }}'
        text: '{{ template "slack.text" . }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'your-pagerduty-service-key'
        severity: '{{ .CommonLabels.severity }}'

route:
  receiver: 'slack-default'
  group_by: ['alertname', 'severity']
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack-critical'
```

---

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
5. **Enable blackbox exporter** for certificate monitoring

---

## Kubernetes Setup

### ServiceMonitor for Prometheus Operator

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aragora
  namespace: aragora
  labels:
    release: prometheus  # Match your Prometheus Operator selector
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: aragora
  namespaceSelector:
    matchNames:
      - aragora
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
```

### PrometheusRule for Alerts

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: aragora-alerts
  namespace: aragora
  labels:
    release: prometheus
spec:
  groups:
    # Copy groups from alerts.rules here
```

### Grafana Dashboard ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aragora-dashboards
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  production-operations.json: |
    # Paste dashboard JSON here
```

---

## Redis Metrics (Optional)

To monitor Redis:
```bash
docker-compose --profile redis up -d
```

This starts the Redis exporter on port 9121.

**Key Redis Metrics:**
- `redis_connected_clients` - Current client connections
- `redis_memory_used_bytes` - Memory consumption
- `redis_commands_processed_total` - Command throughput
- `redis_keyspace_hits_total` / `redis_keyspace_misses_total` - Cache hit rate

---

## Certificate Monitoring

### Blackbox Exporter Setup

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'blackbox-ssl'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - https://aragora.yourdomain.com
          - https://api.aragora.yourdomain.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### cert-manager Metrics

If using cert-manager in Kubernetes:
```yaml
scrape_configs:
  - job_name: 'cert-manager'
    static_configs:
      - targets: ['cert-manager:9402']
```

---

## Useful PromQL Queries

### Request Rate
```promql
sum(rate(aragora_http_requests_total[5m])) by (status)
```

### P95 Latency
```promql
histogram_quantile(0.95, sum(rate(aragora_http_request_duration_seconds_bucket[5m])) by (le))
```

### Error Rate
```promql
sum(rate(aragora_http_requests_total{status=~"5.."}[5m]))
/ sum(rate(aragora_http_requests_total[5m]))
```

### Debate Success Rate
```promql
sum(rate(aragora_debates_total{status="completed"}[1h]))
/ sum(rate(aragora_debates_total[1h]))
```

### Active Debates
```promql
aragora_debates_in_progress
```

### Agent Circuit Breaker Status
```promql
aragora_circuit_breaker_state
```

### Redis Cache Hit Rate
```promql
redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)
```

### Certificate Days Until Expiry
```promql
(probe_ssl_earliest_cert_expiry - time()) / 86400
```

---

## Runbook Links

| Alert | Runbook |
|-------|---------|
| `AragoraDown` | [docs/RUNBOOK.md#service-restart](../../docs/RUNBOOK.md#1-service-restart) |
| `HighErrorRate` | [docs/RUNBOOK.md#alerts-and-responses](../../docs/RUNBOOK.md#alerts-and-responses) |
| `CircuitBreakerOpen` | [docs/RUNBOOK.md#3-agent-timeout-handling](../../docs/RUNBOOK.md#3-agent-timeout-handling) |
| `RedisDown` | [docs/RUNBOOK.md#alert-redisdown](../../docs/RUNBOOK.md#alert-redisdown) |
| `DatabaseConnectionPoolExhausted` | [docs/RUNBOOK.md#4-database-connection-exhaustion](../../docs/RUNBOOK.md#4-database-connection-exhaustion) |
| `TLSCertificateExpiringSoon` | [docs/DEPLOYMENT.md#6-tls-configuration-with-cert-manager](../../docs/DEPLOYMENT.md#6-tls-configuration-with-cert-manager) |

---

## Troubleshooting

### Prometheus Not Scraping

```bash
# Check targets
curl http://localhost:9090/api/v1/targets

# Verify endpoint accessible
curl http://aragora:8080/metrics
```

### Alerts Not Firing

```bash
# Check alert rules loaded
curl http://localhost:9090/api/v1/rules

# Check Alertmanager status
curl http://localhost:9093/api/v2/status
```

### Grafana Dashboard Not Loading

1. Check datasource is configured: Settings > Data sources
2. Verify Prometheus URL is correct
3. Check dashboard UID matches variable references

### Missing Metrics

```bash
# List available metrics
curl -s http://aragora:8080/metrics | grep aragora_ | head -20

# Check specific metric
curl -s http://aragora:8080/metrics | grep aragora_debates_total
```

---

## SLO Targets

Based on `aragora/observability/slo.py`:

| SLO | Target | Window | Burn Rate Alert |
|-----|--------|--------|-----------------|
| API Availability | 99.9% | 30 days | 14.4x (critical), 1x (warning) |
| P99 Latency | <500ms | 5 min | When exceeded |
| Debate Success | >95% | 1 hour | When below |

Error budget calculation:
- 99.9% availability = 43 minutes/month downtime budget
- 99.5% availability = 3.6 hours/month downtime budget

---

## SLO Webhooks

The SLO system can send webhook notifications when operations violate or recover from SLO thresholds.

### Configuration

SLO webhooks are initialized at server startup. Configure via environment variables:

```bash
# Enable SLO webhooks
SLO_WEBHOOKS_ENABLED=true

# Minimum severity level for notifications (minor, moderate, major, critical)
SLO_WEBHOOKS_MIN_SEVERITY=minor

# Cooldown between notifications for same operation (seconds)
SLO_WEBHOOKS_COOLDOWN=60
```

### Programmatic Initialization

```python
from aragora.observability.metrics.slo import SLOWebhookConfig, init_slo_webhooks

config = SLOWebhookConfig(
    enabled=True,
    min_severity="moderate",  # Only notify on moderate+ violations
    cooldown_seconds=60.0,    # Prevent notification spam
)

if init_slo_webhooks(config):
    print("SLO webhooks initialized")
```

### Webhook Event Types

| Event Type | Description | Payload Fields |
|------------|-------------|----------------|
| `slo_violation` | Operation exceeded SLO threshold | `operation`, `percentile`, `severity`, `latency_ms`, `threshold_ms`, `margin_ms`, `margin_percent`, `context` |
| `slo_recovery` | Operation returned to compliance | `operation`, `percentile`, `latency_ms`, `threshold_ms`, `violation_duration_seconds`, `context` |

### Severity Levels

Based on how much the latency exceeds the threshold:

| Severity | Condition | Description |
|----------|-----------|-------------|
| `minor` | 1-1.5x threshold | Slightly over SLO |
| `moderate` | 1.5-2x threshold | Noticeably degraded |
| `major` | 2-3x threshold | Significant degradation |
| `critical` | 3x+ threshold | Severe performance issue |

### API Endpoints

Check SLO webhook status:
```bash
GET /api/webhooks/slo/status
```

Response:
```json
{
  "slo_webhooks": {
    "enabled": true,
    "initialized": true,
    "notifications_sent": 42,
    "recoveries_sent": 38
  },
  "violation_state": {
    "km_query": {
      "in_violation": true,
      "last_severity": "moderate"
    }
  },
  "active_violations": 1
}
```

Send a test notification:
```bash
POST /api/webhooks/slo/test
```

### Integration with Alertmanager

SLO webhooks can route through Alertmanager for consistent alerting:

```yaml
# alertmanager.yml
receivers:
  - name: 'slo-webhook-relay'
    webhook_configs:
      - url: 'http://aragora:8080/api/webhooks/receive'
        send_resolved: true

route:
  routes:
    - match:
        alertname: SLOViolation
      receiver: 'slo-webhook-relay'
```

### Grafana Alert Rules

See `grafana-alerts.yaml` for provisioned Grafana alerts including:
- `slo-violation-rate-high` - Overall violation rate exceeds 1%
- `slo-critical-violation` - Any critical severity violation
- `slo-km-query-latency` - Knowledge Mound query SLO breach
- `slo-consensus-ingestion-latency` - Consensus ingestion SLO breach
- `slo-error-budget-burn` - Rapid error budget consumption

### PromQL Queries for SLO Webhooks

```promql
# Violation rate by severity
sum(rate(aragora_slo_violations_total[5m])) by (severity)

# Operations currently in violation
aragora_slo_violation_state{in_violation="true"}

# Webhook notification success rate
sum(rate(aragora_webhook_delivery_success_total{event_type=~"slo_.*"}[5m]))
/ sum(rate(aragora_webhook_delivery_total{event_type=~"slo_.*"}[5m]))

# Average recovery time
avg(aragora_slo_recovery_duration_seconds) by (operation)
```

---

## Maintenance

### Rotating Prometheus Data

```bash
# Check TSDB status
curl http://localhost:9090/api/v1/status/tsdb

# Trigger compaction
curl -X POST http://localhost:9090/api/v1/admin/tsdb/compact
```

### Cleaning Old Dashboards

```bash
# List dashboards
curl http://admin:admin@localhost:3000/api/search

# Delete by UID
curl -X DELETE http://admin:admin@localhost:3000/api/dashboards/uid/OLD_UID
```

---

*Last updated: 2026-01-13*
