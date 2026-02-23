# Grafana Dashboard Templates

Pre-built Grafana dashboards for monitoring Aragora in production.

## Dashboards

| File | UID | Description |
|------|-----|-------------|
| `debate-performance.json` | `aragora-debate-performance` | Debate latency, throughput, consensus rates, agent quality scores, decision confidence |
| `system-health.json` | `aragora-system-health` | CPU, memory, file descriptors, WebSocket connections, Redis, PostgreSQL, SLO error budgets |
| `api-operations.json` | `aragora-api-operations` | HTTP request rate, error rate, p99 latency by endpoint, status code distribution, cache hit rates |
| `security.json` | `aragora-security` | Auth failures (401/403), rate limiting (429), circuit breaker states, agent errors, queue depth |

## Import Instructions

### Option 1: Automatic Provisioning (recommended)

The dashboards are auto-loaded when using the monitoring profile in Docker Compose:

```bash
cd deploy
docker compose --profile monitoring up -d
```

The Grafana provisioning config at `deploy/monitoring/grafana/provisioning/dashboards/dashboards.yaml` loads all JSON files from the provisioned directory. To include these dashboards, update the provisioning config:

```yaml
# deploy/monitoring/grafana/provisioning/dashboards/dashboards.yaml
apiVersion: 1

providers:
  - name: 'Aragora'
    orgId: 1
    folder: 'Aragora'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
```

Then mount the dashboards directory in your docker-compose:

```yaml
grafana:
  volumes:
    - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
```

### Option 2: Manual Import via Grafana UI

1. Open Grafana at `http://localhost:3001` (default credentials: admin/admin)
2. Navigate to **Dashboards** > **New** > **Import**
3. Click **Upload dashboard JSON file**
4. Select one of the JSON files from this directory
5. Select the **Prometheus** data source
6. Click **Import**

Repeat for each dashboard file.

### Option 3: Grafana API Import

```bash
GRAFANA_URL="http://localhost:3001"
GRAFANA_AUTH="admin:admin"

for dashboard in debate-performance system-health api-operations security; do
  curl -X POST "$GRAFANA_URL/api/dashboards/db" \
    -u "$GRAFANA_AUTH" \
    -H "Content-Type: application/json" \
    -d "{\"dashboard\": $(cat ${dashboard}.json), \"overwrite\": true}"
  echo ""
done
```

## Prerequisites

These dashboards require a Prometheus data source named `Prometheus` with UID `prometheus`. The default provisioned data source at `deploy/monitoring/grafana/provisioning/datasources/datasources.yaml` satisfies this.

### Required Metric Sources

| Dashboard | Required Jobs |
|-----------|--------------|
| Debate Performance | `aragora-backend` |
| System Health | `aragora-backend`, `node` (optional), `postgres` (optional), `redis` (optional) |
| API Operations | `aragora-backend` |
| Security | `aragora-backend` |

## Key Metrics Referenced

These dashboards use metrics from `aragora/observability/`:

| Metric | Type | Source |
|--------|------|--------|
| `aragora_http_requests_total` | Counter | `handler_instrumentation.py` |
| `aragora_http_request_duration_seconds` | Histogram | `handler_instrumentation.py` |
| `aragora_active_debates` | Gauge | `metrics` module |
| `aragora_debates_completed_total` | Counter | `metrics` module |
| `aragora_debate_duration_seconds` | Histogram | `metrics` module |
| `aragora_agent_calls_total` | Counter | `metrics` module |
| `aragora_agent_latency_seconds` | Histogram | `metrics` module |
| `aragora_decision_requests_total` | Counter | `decision_metrics.py` |
| `aragora_decision_latency_seconds` | Histogram | `decision_metrics.py` |
| `aragora_decision_confidence` | Histogram | `decision_metrics.py` |
| `aragora_decision_cache_hits_total` | Counter | `decision_metrics.py` |
| `aragora_decision_cache_misses_total` | Counter | `decision_metrics.py` |
| `aragora_decision_errors_total` | Counter | `decision_metrics.py` |
| `aragora_slo_compliance` | Gauge | `slo.py` |
| `aragora_slo_error_budget_remaining` | Gauge | `slo.py` |
| `aragora_slo_burn_rate` | Gauge | `slo.py` |
| `aragora_consensus_rate` | Gauge | `metrics` module |
| `aragora_websocket_connections` | Gauge | `metrics` module |
| `aragora_circuit_breaker_state` | Gauge | `resilience` module |
| `aragora_control_plane_queue_depth` | Gauge | `control_plane` module |

## Customization

All dashboards use a 30-second auto-refresh and 6-hour default time window. To adjust:

1. Open the dashboard in Grafana
2. Change the time range picker (top right)
3. Change the refresh interval (top right dropdown)
4. Save the dashboard (Ctrl+S)

Dashboard JSON can also be edited directly. The `schemaVersion` is 39 (Grafana 10.x compatible).
