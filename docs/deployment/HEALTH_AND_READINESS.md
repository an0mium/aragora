# Health and Readiness Endpoints

Aragora exposes a comprehensive set of health, readiness, and diagnostic
endpoints for use with Kubernetes probes, load balancers, monitoring systems,
and CI/CD pipelines.

## Quick Reference

| Endpoint | Auth | Latency | Purpose |
|----------|------|---------|---------|
| `GET /healthz` | None | <1ms | K8s liveness probe |
| `GET /readyz` | None | <10ms | K8s readiness probe (in-memory) |
| `GET /readyz/dependencies` | None | 2-5s | Full dependency validation |
| `GET /api/v1/health` | Optional | ~50ms | Basic health (minimal if unauth) |
| `GET /api/v1/health/detailed` | Required | ~200ms | Detailed with observer metrics |
| `GET /api/v1/health/deep` | Required | 1-5s | All external dependencies |
| `GET /metrics` | Required | ~10ms | Prometheus text format |
| `GET /health/build` | None | <1ms | Git SHA / deploy version |

## Endpoint Details

### Liveness Probe: `/healthz`

Returns 200 if the server process is running. Does not check external
dependencies. Returns 200 even in degraded mode (the container is alive and
should not be restarted).

```bash
curl -s http://localhost:8000/healthz
```

```json
{"status": "ok"}
```

When degraded:

```json
{
  "status": "ok",
  "degraded": true,
  "degraded_reason": "Missing required API keys",
  "note": "Server alive but degraded. Check /api/health for details."
}
```

### Readiness Probe (fast): `/readyz`

In-memory only checks, targets <10ms response. Returns 503 when the server
is not ready to accept traffic. Cached for 5 seconds.

```bash
curl -s http://localhost:8000/readyz
```

```json
{
  "status": "ready",
  "checks": {
    "degraded_mode": true,
    "startup_complete": true,
    "handlers_initialized": true,
    "storage_initialized": true,
    "elo_initialized": true,
    "redis_pool": "not_configured",
    "db_pool": "not_configured"
  },
  "latency_ms": 0.42,
  "fast_probe": true,
  "full_validation": "/readyz/dependencies"
}
```

### Readiness Probe (full): `/readyz/dependencies`

Validates all external connections (Redis, PostgreSQL, storage). May take
2-5 seconds. Cached for 1 second.

```bash
curl -s http://localhost:8000/readyz/dependencies
```

```json
{
  "status": "ready",
  "checks": {
    "storage": true,
    "elo_system": true,
    "redis": {"configured": false},
    "postgresql": {"configured": false},
    "api_keys": {
      "configured_count": 2,
      "providers": ["anthropic", "openai"]
    }
  },
  "latency_ms": 15.32
}
```

### Basic Health: `/api/v1/health`

When **unauthenticated**, returns only status and timestamp (prevents
information leakage):

```bash
curl -s http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "timestamp": "2026-02-24T12:00:00.000000Z"
}
```

When **authenticated** with `system.health.read` permission, returns the full
response including version, uptime, all subsystem checks, and response time.

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "version": "0.2.3",
  "uptime_seconds": 3600,
  "demo_mode": false,
  "checks": {
    "degraded_mode": {"healthy": true, "status": "normal"},
    "database": {"healthy": true, "latency_ms": 1.23},
    "elo_system": {"healthy": true},
    "filesystem": {"healthy": true},
    "redis": {"healthy": true, "configured": false},
    "ai_providers": {"any_available": true, "providers": ["anthropic"]},
    "websocket": {"healthy": true},
    "circuit_breakers": {"healthy": true, "open": 0},
    "rate_limiters": {"healthy": true}
  },
  "timestamp": "2026-02-24T12:00:00.000000Z",
  "response_time_ms": 45.12
}
```

Also available at `/api/health` (without version prefix).

### Detailed Health: `/api/v1/health/detailed`

Requires `system.health.read` permission. Adds observer metrics, maintenance
stats, memory usage, HTTP connector status, and production warnings (e.g.,
SQLite in production).

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/health/detailed
```

### Deep Health: `/api/v1/health/deep`

Requires `system.health.read` permission. The most comprehensive check --
verifies Supabase, user store, billing (Stripe), Slack, system resources
(CPU, memory, disk), email services, and dependency analyzer.

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/health/deep
```

### Build Info: `/health/build`

Public endpoint returning the git SHA, build time, and deploy version.
Used by CI/CD pipelines to verify deployments.

```bash
curl -s http://localhost:8000/health/build
```

Also available at `/api/v1/health/build`.

### Prometheus Metrics: `/metrics`

Requires `monitoring:metrics` permission. Returns metrics in Prometheus
text exposition format.

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/metrics
```

### Additional Health Endpoints

All require `system.health.read` permission:

| Endpoint | Description |
|----------|-------------|
| `/api/v1/health/stores` | Database and store connectivity |
| `/api/v1/health/database` | Database schema health |
| `/api/v1/health/sync` | Sync status |
| `/api/v1/health/circuits` | Circuit breaker states |
| `/api/v1/health/components` | Component health from HealthRegistry |
| `/api/v1/health/slow-debates` | Slow debate detection |
| `/api/v1/health/cross-pollination` | Cross-pollination feature status |
| `/api/v1/health/knowledge-mound` | Knowledge Mound health |
| `/api/v1/health/decay` | Confidence decay scheduler status |
| `/api/v1/health/startup` | Startup report and SLO status |
| `/api/v1/health/encryption` | Encryption service status |
| `/api/v1/health/platform` | Platform resilience status |
| `/api/v1/health/workers` | Background worker status |
| `/api/v1/health/job-queue` | Job queue health |
| `/api/v1/health/workers/all` | Combined workers + queue |
| `/api/v1/diagnostics` | Deployment diagnostics |
| `/api/v1/diagnostics/deployment` | Production readiness checklist |
| `/api/v1/gateway/health` | Gateway health (requires `gateway:health.read`) |

## Kubernetes Configuration

### Liveness Probe

Checks whether the container should be restarted. Uses `/healthz` which
always returns 200 if the process is alive, even in degraded mode.

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
  timeoutSeconds: 3
  failureThreshold: 3
```

### Readiness Probe

Checks whether the pod should receive traffic. Uses the fast `/readyz`
endpoint which only performs in-memory checks (<10ms).

```yaml
readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 2
```

For stricter validation (e.g., staging environments), use the full dependency
check instead:

```yaml
readinessProbe:
  httpGet:
    path: /readyz/dependencies
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3
```

### Startup Probe

For slow-starting pods (first boot, large model loads):

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 30  # 150 seconds max startup
```

## Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1
```

Or in `docker-compose.yml`:

```yaml
services:
  aragora:
    image: aragora:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
```

## Monitoring Integration

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'aragora'
    metrics_path: '/metrics'
    bearer_token: '<ARAGORA_API_TOKEN>'
    static_configs:
      - targets: ['aragora:8000']
    scrape_interval: 15s
```

### Grafana Dashboard

Use the Prometheus metrics endpoint for Grafana dashboards. Key metrics:

- `aragora_debates_total` -- total debates processed
- `aragora_agent_latency_seconds` -- agent response times
- `aragora_circuit_breaker_state` -- circuit breaker states
- `aragora_api_requests_total` -- API request counts

### Alertmanager Rules

```yaml
groups:
  - name: aragora
    rules:
      - alert: AragoraUnhealthy
        expr: probe_success{job="aragora-health"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Aragora instance unhealthy"

      - alert: AragoraCircuitBreakerOpen
        expr: aragora_circuit_breaker_state{state="open"} > 2
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Multiple circuit breakers open"
```

## Smoke Test Script

A stdlib-only smoke test script is provided for runtime validation:

```bash
# Default (localhost:8000)
python scripts/smoke_self_host_runtime.py

# Custom URL
python scripts/smoke_self_host_runtime.py --base-url http://aragora.internal:8080

# With auth token
python scripts/smoke_self_host_runtime.py --api-token "$ARAGORA_API_TOKEN"

# Quiet mode (exit code only)
python scripts/smoke_self_host_runtime.py --quiet
```

See `scripts/smoke_self_host_runtime.py` for full details.
