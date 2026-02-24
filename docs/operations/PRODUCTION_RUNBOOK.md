# Production Operations Runbook

Operational procedures for running Aragora in production. This runbook covers
health monitoring, scaling, troubleshooting, backup/recovery, key rotation,
observability, and step-by-step runbook procedures.

> **Prerequisites:** Familiarity with `deploy/docker-compose.production.yml`,
> the Kubernetes manifests in `deploy/kubernetes/`, and the environment
> variables documented in `deploy/.env.template`.

---

## Table of Contents

1. [Health Checks](#health-checks)
2. [Scaling Guide](#scaling-guide)
3. [Troubleshooting Matrix](#troubleshooting-matrix)
4. [Backup and Recovery](#backup-and-recovery)
5. [Key Rotation](#key-rotation)
6. [Monitoring](#monitoring)
7. [Runbook Procedures](#runbook-procedures)

---

## Health Checks

Aragora exposes several unauthenticated health endpoints used by Kubernetes
probes, load balancers, and monitoring systems.

### Endpoint Reference

| Endpoint | Auth | Purpose | Healthy Response |
|----------|------|---------|------------------|
| `GET /healthz` | None | **Liveness probe.** Returns 200 if the process is alive. | `{"status": "ok"}` (200) |
| `GET /readyz` | None | **Readiness probe.** Returns 200 only after the startup sequence completes (`mark_server_ready()` in `aragora/server/unified_server.py`). | `{"status": "ready"}` (200) |
| `GET /health` | API token | **Deep dependency check.** Validates database, Redis, Knowledge Mound, and API key availability via `run_startup_health_checks()` in `aragora/server/startup/health_check.py`. | `{"overall": "ok", "checks": {...}}` (200) |
| `GET /health/threads` | None | **Thread registry.** Returns active thread health from `aragora.server.lifecycle.get_thread_registry()`. | JSON with thread counts and status (200) |
| `GET /health/build` | None | **Build info.** Returns version, commit SHA, and build timestamp. | JSON with build metadata (200) |
| `GET /metrics` | None | **Prometheus metrics.** Scrape target at port 9090 (configurable via `ARAGORA_METRICS_PORT`). | Prometheus text format |

### Readiness Gate Behavior

The server sets the internal `_server_ready` flag to `True` at the end of the
startup sequence (after database migrations, Redis connection, KM initialization,
and background task spawning). Until that flag is set:

- `/readyz` returns `{"status": "not_ready", "reason": "startup in progress"}` with HTTP 503.
- Kubernetes will not route traffic to the pod.

**Start period configuration** (from `docker-compose.production.yml`):

```yaml
healthcheck:
  test: ["CMD", "curl", "-sf", "http://localhost:8080/healthz"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 30s
```

### Deep Health Check Subsystems

The `/health` endpoint (`aragora/server/startup/health_check.py`) runs four
checks at startup and on demand:

| Check | What It Validates | Failure Means |
|-------|-------------------|---------------|
| `api_keys` | At least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY` is set. | No LLM provider configured; debates cannot run. |
| `database` | SQLite file accessible or PostgreSQL DSN reachable (`DATABASE_URL` / `ARAGORA_POSTGRES_DSN`). | Persistence unavailable; API will return 500 on data operations. |
| `knowledge_mound` | KM store directory exists and is writable. | Knowledge features degraded; debates still run but without historical context. |
| `health_registry` | Internal `HealthChecker` registry is initialized. | Observability degraded; component-level health metrics unavailable. |

### Component Health Checker

Individual components (database, Redis, agents) use `HealthChecker` from
`aragora/resilience/health.py`:

- **Failure threshold:** 3 consecutive failures before marking unhealthy (configurable via `failure_threshold`).
- **Recovery threshold:** 2 consecutive successes to recover (configurable via `recovery_threshold`).
- **Latency tracking:** Rolling window of 10 samples (configurable via `latency_window`).
- **Health events:** Emitted on status transitions (degraded/recovered) via the event bus.

### Alert Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| `/healthz` returns non-200 | Immediate | K8s kills and restarts pod (liveness failure). |
| `/readyz` returns 503 for >90s | After `start_period` + `retries * interval` | K8s removes pod from Service endpoints. |
| Component `consecutive_failures >= 3` | 3 failures | Component marked unhealthy; circuit breaker may open. |
| All LLM provider health checks fail | Immediate | Debates cannot start; raise P1 alert. |

---

## Scaling Guide

### Architecture Overview

```
                    +-----------+
                    |   Nginx   |  (TLS termination, WebSocket upgrade)
                    |  / ALB    |
                    +-----+-----+
                          |
              +-----------+-----------+
              |           |           |
         +----+----+ +---+----+ +----+----+
         | Backend | | Backend| | Backend |  (stateless API + WS)
         | Pod 1   | | Pod 2  | | Pod N   |
         +---------+ +--------+ +---------+
              |           |           |
         +----+-----------+-----------+----+
         |          PostgreSQL              |  (shared state)
         +---------------------------------+
         |            Redis                |  (sessions, pub/sub, cache)
         +---------------------------------+
```

### Horizontal Scaling: API Servers

Backend pods are stateless. Scale by adding replicas.

**Docker Compose:**

```bash
# Scale to 3 API replicas (requires external load balancer)
docker compose -f deploy/docker-compose.production.yml up -d --scale aragora=3
```

**Kubernetes HPA** (from `deploy/kubernetes/hpa.yaml`):

```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

**Custom metrics HPA** (from `deploy/kubernetes/hpa-custom-metrics.yaml`):
Scale on `aragora_active_debates` or `aragora_websocket_connections` when the
Prometheus adapter is installed.

### Shared State Requirements

When running multiple API instances, configure shared PostgreSQL and Redis:

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATABASE_URL` | Shared PostgreSQL DSN | `postgresql://aragora:pw@rds:5432/aragora` |
| `ARAGORA_DB_BACKEND` | Force PostgreSQL (prevents SQLite fallback) | `postgresql` |
| `ARAGORA_REDIS_URL` | Shared Redis for sessions and pub/sub | `rediss://elasticache:6379` |
| `ARAGORA_REQUIRE_DISTRIBUTED` | Fail fast if shared state unavailable | `true` |
| `ARAGORA_MULTI_INSTANCE` | Enable multi-instance coordination | `true` |

**Database pool tuning** (per instance):

| Variable | Default | Guidance |
|----------|---------|----------|
| `ARAGORA_DB_POOL_SIZE` | 20 | 2 instances x 35 connections = 70 total (well within RDS default of 1600) |
| `ARAGORA_DB_POOL_OVERFLOW` | 15 | Burst capacity per instance |
| `ARAGORA_DB_COMMAND_TIMEOUT` | 60 | Seconds per query |
| `ARAGORA_DB_POOL_RECYCLE` | 1800 | Seconds before recycling idle connections |

See `deploy/SHARED_STATE_SETUP.md` for full multi-instance configuration.

### Session Affinity for Debates

Active debates maintain in-memory state during execution. While debate results
are persisted to PostgreSQL, in-flight debate rounds require the same server
instance.

**Strategy: Redis-backed debate state + sticky sessions.**

1. Set `ARAGORA_MULTI_INSTANCE=true` to enable Redis-based debate state sharing.
2. Configure load balancer sticky sessions (cookie-based or IP hash):

```nginx
# Nginx upstream with ip_hash for session affinity
upstream aragora_backend {
    ip_hash;
    server backend-1:8080;
    server backend-2:8080;
    server backend-3:8080;
}
```

For Kubernetes, use a session affinity annotation on the Service:

```yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 1800  # 30 minutes
```

### WebSocket Scaling

WebSocket connections are long-lived. Each backend pod handles its own
connections via `DebateStreamServer` (port 8765).

**Key considerations:**

- **Nginx/ALB must support WebSocket upgrade.** The production nginx config
  (`deploy/nginx/nginx.conf`) handles `Connection: Upgrade` headers.
- **Redis pub/sub** distributes events across instances so clients connected
  to any pod receive debate updates.
- **Connection limits:** Default `ARAGORA_MAX_CONCURRENT_DEBATES=10` per instance.
  Scale workers (via `--profile workers`) for debate-heavy workloads.

**Nginx WebSocket configuration:**

```nginx
location /ws {
    proxy_pass http://aragora_backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400s;  # 24h for long-lived connections
}
```

### Worker Scaling

Debate queue workers (`--profile workers`) run compute-intensive debate
orchestration separately from the API servers.

| Variable | Default | Purpose |
|----------|---------|---------|
| `WORKER_REPLICAS` | 2 | Number of worker containers |
| `WORKER_CONCURRENCY` | 3 | Simultaneous debates per worker |

```bash
# Scale workers independently
WORKER_REPLICAS=4 WORKER_CONCURRENCY=5 \
  docker compose -f deploy/docker-compose.production.yml --profile workers up -d
```

### Resource Limits

From `docker-compose.production.yml`:

| Service | CPU Limit | Memory Limit | CPU Reservation | Memory Reservation |
|---------|-----------|-------------|-----------------|-------------------|
| Backend (aragora) | 4 CPUs | 4 GB | 0.5 CPU | 512 MB |
| PostgreSQL | 2 CPUs | 2 GB | 0.25 CPU | 256 MB |
| Redis | 1 CPU | 1 GB | 0.1 CPU | 128 MB |
| Nginx | 1 CPU | 256 MB | 0.05 CPU | 32 MB |
| Worker | 2 CPUs | 2 GB | 0.25 CPU | 256 MB |
| Prometheus | 1 CPU | 1 GB | 0.1 CPU | 128 MB |
| Grafana | 1 CPU | 512 MB | 0.1 CPU | 64 MB |

---

## Troubleshooting Matrix

### Quick Diagnosis Commands

```bash
# Overall system status
docker compose -f deploy/docker-compose.production.yml ps
curl -s http://localhost:8080/healthz | jq
curl -s http://localhost:8080/readyz | jq

# Deep health check (requires API token)
curl -s -H "Authorization: Bearer $ARAGORA_API_TOKEN" http://localhost:8080/health | jq

# Check logs
docker compose -f deploy/docker-compose.production.yml logs aragora --tail 100
docker compose -f deploy/docker-compose.production.yml logs postgres --tail 50

# Database connectivity
docker exec aragora-postgres pg_isready -U aragora -d aragora

# Redis connectivity
docker exec aragora-redis redis-cli -a "$REDIS_PASSWORD" ping
```

### Symptom-to-Fix Table

| Symptom | Likely Root Cause | Diagnostic | Fix |
|---------|-------------------|------------|-----|
| **Debate stuck (no progress for >10m)** | Agent API timeout or circuit breaker open. | Check `aragora_active_debates` metric; look for `AragoraDebateRunningTooLong` alert. Inspect logs: `docker logs aragora-api \| grep "circuit.*open"`. | 1. Check agent provider status pages. 2. If circuit breaker open, wait for cooldown (default 60s) or restart. 3. If agent down, OpenRouter fallback triggers automatically on 429 errors. 4. The `init_stuck_debate_watchdog` background task auto-cancels debates exceeding the timeout. |
| **WebSocket disconnects** | Nginx proxy timeout, pod restart, or network interruption. | Check `aragora_websocket_connections` metric for sudden drops. Look for `AragoraWebSocketConnectionsDrop` alert (>10 drop in 5m). Check nginx logs for 502/504. | 1. Verify `proxy_read_timeout 86400s` in nginx config. 2. Check pod restarts: `kubectl get pods -n aragora`. 3. Enable `EventReplayBuffer` for reconnection support (clients replay missed events). 4. Ensure load balancer idle timeout >= 3600s. |
| **High API latency (p99 >500ms)** | Database slow queries, Redis latency, or too many concurrent debates. | Check `aragora_request_latency_seconds` histogram. Check `AragoraLatencySLOBreach` alert. Run `EXPLAIN ANALYZE` on slow queries. | 1. Scale API replicas (see Scaling Guide). 2. Check PostgreSQL connection pool saturation (`ARAGORA_DB_POOL_SIZE`). 3. Reduce `ARAGORA_MAX_CONCURRENT_DEBATES`. 4. Check Redis memory: `redis-cli INFO memory`. |
| **OOM killed (container restart with exit code 137)** | Memory leak or too many concurrent operations. | `docker inspect aragora-api \| jq '.[0].State'`. Check `AragoraHighMemoryUsage` alert (>2GB threshold). `docker stats`. | 1. Increase memory limit in compose file (default 4GB). 2. Reduce `ARAGORA_MAX_CONCURRENT_DEBATES`. 3. Reduce `WORKER_CONCURRENCY`. 4. Check for memory leaks in long-running debates. 5. Restart the container: `docker compose restart aragora`. |
| **502 Bad Gateway** | Backend crashed or not ready. | Check `/readyz` response. Check `docker compose ps` for unhealthy containers. | 1. Check backend logs for crash stack trace. 2. Wait for startup sequence (30s `start_period`). 3. Restart: `docker compose restart aragora`. |
| **Database connection refused** | PostgreSQL down or connection pool exhausted. | `pg_isready -U aragora -d aragora`. Check `AragoraDatabaseDown` alert. | 1. Restart PostgreSQL: `docker compose restart postgres`. 2. If data corruption, follow Backup and Recovery below. 3. Check `max_connections` in `deploy/postgres/postgresql.conf`. |
| **Redis connection errors** | Redis OOM or authentication failure. | `redis-cli -a "$REDIS_PASSWORD" INFO`. Check `REDIS_MAXMEMORY` (default 512MB). | 1. Increase `REDIS_MAXMEMORY`. 2. Verify `REDIS_PASSWORD` matches across services. 3. Check `volatile-lru` eviction policy is appropriate. |
| **No agent calls (debates fail to start)** | All API keys invalid or exhausted. | Check `AragoraNoAgentCalls` alert. Run `/health` deep check for `api_keys` status. | 1. Verify API keys in env/secrets. 2. Check provider status pages (Anthropic, OpenAI). 3. Add `OPENROUTER_API_KEY` as fallback. 4. Check billing/quota on provider dashboards. |
| **Rate limiting (429 responses)** | Client exceeding `ARAGORA_RATE_LIMIT` (default 100 req/min/IP). | Check `AragoraHighRateLimitRate` alert. | 1. Increase `ARAGORA_RATE_LIMIT` if legitimate traffic. 2. Investigate source IPs for abuse. 3. Scale API replicas to distribute load. |
| **Circuit breaker open** | Upstream provider returning errors consistently. | Check `AragoraAgentCircuitOpen` alert. Inspect `aragora_circuit_breaker_state` metric. | 1. The circuit breaker auto-recovers: CLOSED -> OPEN (after failures) -> HALF_OPEN (after cooldown) -> CLOSED (on success). 2. Check upstream provider. 3. OpenRouter fallback engages on 429. 4. If 3+ circuits open simultaneously (`AragoraManyCircuitsOpen`), treat as P1. |
| **Startup hangs (readyz never becomes ready)** | Database migration stuck or required service unreachable. | Check logs during startup for migration errors or connection timeouts. | 1. Verify PostgreSQL and Redis are healthy before starting backend. 2. Check `start_period` is sufficient (increase to 60s for slow networks). 3. Run `aragora doctor` to validate subsystems. |

### Circuit Breaker States

The circuit breaker (`aragora/resilience/circuit_breaker.py`) implements three states:

```
CLOSED  --[failure_threshold exceeded]--> OPEN
OPEN    --[cooldown_seconds elapsed]----> HALF_OPEN
HALF_OPEN --[success]-------------------> CLOSED
HALF_OPEN --[failure]-------------------> OPEN
```

Default configuration (`CircuitBreakerConfig`):
- `failure_threshold`: 5 failures to open
- `cooldown_seconds`: 60 seconds before half-open probe
- Metrics emitted on state changes via `_emit_metrics(circuit_name, state)`

---

## Backup and Recovery

### Automated Database Backups

The `--profile backup` container in `docker-compose.production.yml` runs
periodic PostgreSQL backups:

```bash
# Enable automated backups
docker compose -f deploy/docker-compose.production.yml --profile backup up -d
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `BACKUP_INTERVAL` | `86400` (24h) | Seconds between backups |
| `BACKUP_RETENTION_DAYS` | `7` | Days to retain local backups |

**Backup location:** `deploy/backups/aragora-YYYYMMDD_HHMMSS.sql.gz`

### Manual Database Backup

```bash
# Create an immediate backup
docker exec aragora-postgres pg_dump -U aragora aragora | gzip > \
  "backup-$(date +%Y%m%d_%H%M%S).sql.gz"

# Verify backup integrity
gunzip -t backup-*.sql.gz && echo "Integrity OK"
```

### Database Restore

```bash
# 1. Stop all services except PostgreSQL
docker compose -f deploy/docker-compose.production.yml stop aragora debate-worker

# 2. Drop and recreate the database
docker exec -i aragora-postgres psql -U aragora -c "DROP DATABASE IF EXISTS aragora;"
docker exec -i aragora-postgres psql -U aragora -c "CREATE DATABASE aragora;"

# 3. Restore from backup
gunzip -c backup-20260224_120000.sql.gz | \
  docker exec -i aragora-postgres psql -U aragora aragora

# 4. Restart services
docker compose -f deploy/docker-compose.production.yml up -d
```

### BackupManager (Application-Level)

The `BackupManager` in `aragora/backup/manager.py` provides application-level
backups with enhanced verification:

- **Backup types:** Full, Incremental, Differential
- **Storage backends:** Local filesystem, S3, GCS
- **Verification:** Checksum validation, schema comparison, referential
  integrity checks, row count validation, dry-run restore testing
- **Retention policy:** Configurable daily (7), weekly (4), monthly (3)
  retention with minimum backup guarantee

```python
from aragora.backup.manager import BackupManager, RetentionPolicy

manager = BackupManager(
    backup_dir="/app/data/backups",
    retention=RetentionPolicy(
        keep_daily=7,
        keep_weekly=4,
        keep_monthly=3,
        min_backups=1,
    ),
)

# Create verified backup
metadata = await manager.create_backup(backup_type="full", verify=True)
print(f"Backup {metadata.id}: {metadata.status}, size={metadata.size_bytes}")

# Verify existing backup
result = await manager.verify_backup(backup_id)
print(f"Verified: {result.verified}, checksum_valid: {result.checksum_valid}")
```

### Decision Receipt Backups

Decision receipts use cryptographic hashing (SHA-256) for audit trails
and are stored in PostgreSQL. They have a separate retention policy:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARAGORA_RECEIPT_RETENTION_DAYS` | `2555` (~7 years) | SOC 2 / compliance retention |
| `ARAGORA_RECEIPT_SIGNING_KEY` | Required | HMAC key for receipt integrity |

**Backing up receipts specifically:**

```bash
# Export receipts table only
docker exec aragora-postgres pg_dump -U aragora -t receipts -t decision_receipts aragora | \
  gzip > receipts-backup-$(date +%Y%m%d).sql.gz
```

### Knowledge Mound Backup

KM data resides in `ARAGORA_STORE_DIR` (default `/app/data/.aragora_beads`)
as a Docker volume (`aragora-data`).

```bash
# Backup the bead store volume
docker run --rm -v aragora_aragora-data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/km-data-$(date +%Y%m%d).tar.gz -C /data .aragora_beads

# Restore
docker run --rm -v aragora_aragora-data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/km-data-20260224.tar.gz -C /data
```

### Redis Backup

Redis is configured with AOF persistence (`--appendonly yes`) and periodic
RDB snapshots (`--save 60 1000`, `--save 300 100`).

```bash
# Trigger manual RDB save
docker exec aragora-redis redis-cli -a "$REDIS_PASSWORD" BGSAVE

# Copy RDB file
docker cp aragora-redis:/data/dump.rdb ./backups/redis-dump-$(date +%Y%m%d).rdb
```

### Disaster Recovery Reference

See `deploy/DISASTER_RECOVERY.md` for full recovery procedures including:

| Metric | Target |
|--------|--------|
| **RTO** (Recovery Time Objective) | 4 hours |
| **RPO** (Recovery Point Objective) | 1 hour |
| **Availability** | 99.9% |
| **MTTR** (Mean Time to Recovery) | 30 minutes |

---

## Key Rotation

Aragora uses the `KeyRotationScheduler` (`aragora/security/key_rotation.py`)
for automated encryption key management with SOC 2 compliance (CC6.1, CC6.7).

### Rotation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rotation_interval_days` | 90 | Days between automatic rotations |
| `check_interval_hours` | 6 | Hours between rotation checks |
| `key_overlap_days` | 7 | Days old key remains active after rotation |
| `re_encrypt_on_rotation` | `true` | Re-encrypt data with new key |
| `re_encrypt_batch_size` | 100 | Batch size for re-encryption |
| `re_encrypt_timeout` | 3600 | Seconds timeout for re-encryption |
| `max_retries` | 3 | Retry attempts on failure |
| `notify_days_before` | 7 | Days before rotation to send notification |

### Rotating API Keys (Zero Downtime)

API keys for LLM providers can be rotated without downtime because the
server reads them from environment variables on each agent call.

```bash
# 1. Update the secret in AWS Secrets Manager
aws secretsmanager update-secret \
  --secret-id aragora/production \
  --secret-string '{"ANTHROPIC_API_KEY": "sk-ant-new-key..."}'

# 2. Restart the backend to pick up new secrets (rolling restart for zero downtime)
# Docker Compose:
docker compose -f deploy/docker-compose.production.yml restart aragora

# Kubernetes (automatic rolling update):
kubectl rollout restart deployment/aragora-backend -n aragora
```

If not using AWS Secrets Manager, update the `.env` file and restart.

### Rotating JWT Secret (`ARAGORA_JWT_SECRET`)

JWT secret rotation invalidates all existing tokens. Use the overlap period:

```bash
# 1. Generate a new JWT secret
NEW_SECRET=$(openssl rand -base64 32)

# 2. Set ARAGORA_JWT_SECRET_PREVIOUS to the current secret (enables overlap)
# 3. Set ARAGORA_JWT_SECRET to the new secret
# 4. Deploy. The server accepts tokens signed with either secret.

# 5. After the overlap period (e.g., 24 hours), remove ARAGORA_JWT_SECRET_PREVIOUS
```

### Rotating Receipt Signing Key (`ARAGORA_RECEIPT_SIGNING_KEY`)

Receipt signing key rotation requires careful handling since receipts are
long-lived audit artifacts:

1. Generate a new key: `openssl rand -base64 32`
2. Store the old key in `ARAGORA_RECEIPT_SIGNING_KEY_PREVIOUS`
3. Set the new key in `ARAGORA_RECEIPT_SIGNING_KEY`
4. Deploy. New receipts use the new key; verification checks both keys.
5. Old key should be retained indefinitely (receipts are retained ~7 years).

### Rotating Database Password

```bash
# 1. Set new password in PostgreSQL
docker exec -i aragora-postgres psql -U aragora -c \
  "ALTER USER aragora PASSWORD 'new-strong-password';"

# 2. Update POSTGRES_PASSWORD in .env or Secrets Manager

# 3. Rolling restart of all services that connect to PostgreSQL
docker compose -f deploy/docker-compose.production.yml restart aragora debate-worker
```

### Rotating Redis Password

```bash
# 1. Set new password in Redis (requires current auth)
docker exec aragora-redis redis-cli -a "$REDIS_PASSWORD" \
  CONFIG SET requirepass "new-strong-password"

# 2. Persist the change
docker exec aragora-redis redis-cli -a "new-strong-password" CONFIG REWRITE

# 3. Update REDIS_PASSWORD in .env or Secrets Manager

# 4. Rolling restart of all services that connect to Redis
docker compose -f deploy/docker-compose.production.yml restart aragora debate-worker
```

### Encryption Key Rotation (AES-256-GCM)

Data-at-rest encryption keys are managed by the `KeyRotationScheduler`:

```python
from aragora.security.key_rotation import KeyRotationScheduler, KeyRotationConfig

scheduler = KeyRotationScheduler(
    config=KeyRotationConfig(
        rotation_interval_days=90,
        auto_rotate_kms_keys=True,
        re_encrypt_on_rotation=True,
        stores_to_re_encrypt=[
            "integrations",
            "webhooks",
            "gmail_tokens",
            "enterprise_sync",
        ],
    ),
)

# Start automated rotation
await scheduler.start()

# Trigger manual rotation
await scheduler.rotate_now()
```

The scheduler re-encrypts data in the configured stores using the new key,
with a 7-day overlap period where both old and new keys are valid.

---

## Monitoring

### Prometheus Metrics

Enable the monitoring profile:

```bash
docker compose -f deploy/docker-compose.production.yml \
  --profile monitoring up -d
```

This starts Prometheus (port 9090), Alertmanager (port 9093), and
Grafana (port 3000).

**Scrape configuration** (`deploy/monitoring/prometheus.yml`):

| Job | Target | Interval | Path |
|-----|--------|----------|------|
| `aragora-backend` | `backend:8080` | 10s | `/metrics` |
| `aragora-websocket` | `backend:8080` | 15s | `/metrics/websocket` |
| `postgres` | `postgres-exporter:9187` | 30s | `/metrics` |
| `redis` | `redis-exporter:9121` | 15s | `/metrics` |
| `node` | `node-exporter:9100` | 30s | `/metrics` |

### Key Metrics to Watch

#### Application Metrics

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|-----------------|
| `aragora_requests_total` | Counter | Total HTTP requests by status code | 5xx rate > 0.1% of total |
| `aragora_request_latency_seconds` | Histogram | Request latency distribution | p99 > 500ms |
| `aragora_active_debates` | Gauge | Currently running debates | > `ARAGORA_MAX_CONCURRENT_DEBATES` |
| `aragora_debate_duration_seconds` | Histogram | Debate completion time | Error outcome rate > 5% |
| `aragora_agent_calls_total` | Counter | LLM API calls by agent and status | Error rate > 10% per agent |
| `aragora_agent_latency_seconds` | Histogram | LLM API call latency by agent | p95 > 30s |
| `aragora_websocket_connections` | Gauge | Active WebSocket connections | Drop > 10 in 5 minutes |
| `aragora_circuit_breaker_state` | Gauge | Circuit breaker state per agent | Any in "open" state for >5m |

#### Infrastructure Metrics

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| `process_resident_memory_bytes` | > 2 GB for 10 minutes | Investigate memory leak or reduce concurrency |
| PostgreSQL active connections | > 80% of `max_connections` | Increase pool size or scale instances |
| Redis memory usage | > 80% of `REDIS_MAXMEMORY` | Increase limit or review eviction policy |
| Disk usage on data volume | > 85% | Extend volume or run retention cleanup |

### SLO Definitions

From `aragora/observability/slo.py` and `deploy/monitoring/alerts.yaml`:

| SLO | Target | Measurement Window | Override Variable |
|-----|--------|-------------------|-------------------|
| API Availability | 99.9% (3 nines) | Rolling 5m / 15m | `SLO_AVAILABILITY_TARGET` |
| p99 Latency | < 500ms | Rolling 5m | `SLO_LATENCY_P99_TARGET_MS` |
| Debate Success Rate | > 95% | Rolling 1h | `SLO_DEBATE_SUCCESS_TARGET` |

### Alert Rules Reference

From `deploy/monitoring/alerts.yaml`:

| Alert | Severity | Condition | For |
|-------|----------|-----------|-----|
| `AragoraAvailabilitySLOBreach` | Critical | 5xx rate > 0.1% (5m window) | 5m |
| `AragoraAvailabilitySLOWarning` | Warning | 5xx rate > 0.05% (15m window) | 10m |
| `AragoraLatencySLOBreach` | Critical | p99 > 500ms | 5m |
| `AragoraLatencySLOWarning` | Warning | p99 > 400ms | 10m |
| `AragoraDebateSuccessSLOBreach` | Critical | Debate error rate > 5% (1h) | 30m |
| `AragoraDatabaseDown` | Critical | Database SLO compliance = 0 | 1m |
| `AragoraAgentCircuitOpen` | Warning | Any circuit breaker open | 5m |
| `AragoraManyCircuitsOpen` | Critical | 3+ circuit breakers open | 2m |
| `AragoraWebSocketConnectionsDrop` | Warning | Connections drop > 10 in 5m | 2m |
| `AragoraHighMemoryUsage` | Warning | Process memory > 2 GB | 10m |
| `AragoraHighRateLimitRate` | Warning | 429 responses > 10/s | 5m |
| `AragoraAgentLatencyHigh` | Warning | Agent p95 > 30s | 10m |
| `AragoraAgentErrorRateHigh` | Warning | Agent error rate > 10% | 5m |
| `AragoraNoAgentCalls` | Warning | Zero agent calls in 15m | 15m |
| `AragoraDebateRunningTooLong` | Warning | Debate running > 10m | 1m |

### Grafana Dashboards

Pre-built dashboards are located in `deploy/grafana/`:

| Dashboard | File | Purpose |
|-----------|------|---------|
| Aragora Overview | `aragora-overview.json` | High-level system health |
| API Performance | `api-performance.json` | Request latency, throughput, error rates |
| Debate Analytics | `debate-analytics.json` | Debate duration, consensus rates, agent performance |
| SLO Dashboard | `slo-dashboard.json` | SLO compliance and burn rate |
| Receipt Retention | `receipt-retention.json` | Decision receipt storage and compliance |

**Grafana access:** `https://<domain>/grafana/` (credentials from
`GRAFANA_USER` / `GRAFANA_PASSWORD`).

**Dashboard provisioning** is automated via
`deploy/self-hosted/grafana/provisioning/`.

### OpenTelemetry Tracing

```bash
# Enable OTLP export
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
```

Jaeger UI is available at port 16686 when the monitoring profile is active.
Traces cover HTTP requests, database queries, agent calls, and debate phases.

### Log Aggregation

Production logs use JSON format (`ARAGORA_LOG_FORMAT=json`). Loki + Promtail
are included in the monitoring profile for centralized log search.

```bash
# View structured logs
docker compose -f deploy/docker-compose.production.yml logs aragora --tail 50 | jq
```

---

## Runbook Procedures

### Procedure 1: Deploy a New Version

**Docker Compose:**

```bash
# 1. Pull the latest image
docker compose -f deploy/docker-compose.production.yml pull aragora

# 2. Rolling restart (one container at a time if scaled)
docker compose -f deploy/docker-compose.production.yml up -d aragora

# 3. Verify health
sleep 30
curl -sf http://localhost:8080/healthz | jq
curl -sf http://localhost:8080/readyz | jq

# 4. Check logs for errors
docker compose -f deploy/docker-compose.production.yml logs aragora --tail 50

# 5. Run smoke test
curl -sf http://localhost:8080/api/v1/status | jq
```

**Kubernetes (with Argo Rollouts canary):**

```bash
# 1. Update image tag
kubectl argo rollouts set image aragora aragora=ghcr.io/an0mium/aragora:v2.8.1 -n aragora

# 2. Monitor canary progress (from deploy/argo-rollouts/rollout.yaml)
kubectl argo rollouts get rollout aragora -n aragora --watch

# 3. Promote canary to full deployment
kubectl argo rollouts promote aragora -n aragora

# 4. If issues detected, abort
kubectl argo rollouts abort aragora -n aragora
```

### Procedure 2: Rollback

**Docker Compose:**

```bash
# 1. Identify the previous version
docker compose -f deploy/docker-compose.production.yml logs aragora 2>&1 | head -5

# 2. Roll back to a specific image tag
ARAGORA_IMAGE=ghcr.io/an0mium/aragora:v2.8.0 \
  docker compose -f deploy/docker-compose.production.yml up -d aragora

# 3. Verify
curl -sf http://localhost:8080/healthz | jq
```

**Kubernetes:**

```bash
# Argo Rollouts rollback
kubectl argo rollouts undo aragora -n aragora

# Standard deployment rollback
kubectl rollout undo deployment/aragora-backend -n aragora

# Rollback to specific revision
kubectl rollout undo deployment/aragora-backend -n aragora --to-revision=3
```

### Procedure 3: Scale Up

```bash
# Docker Compose: scale API servers
docker compose -f deploy/docker-compose.production.yml up -d --scale aragora=3

# Docker Compose: scale workers
WORKER_REPLICAS=4 docker compose -f deploy/docker-compose.production.yml \
  --profile workers up -d

# Kubernetes: manual scale
kubectl scale deployment aragora-backend -n aragora --replicas=5

# Kubernetes: adjust HPA limits
kubectl patch hpa aragora-backend-hpa -n aragora \
  --type='json' -p='[{"op": "replace", "path": "/spec/maxReplicas", "value": 15}]'
```

### Procedure 4: Drain a Node

```bash
# 1. Cordon the node (prevent new pod scheduling)
kubectl cordon <node-name>

# 2. Drain pods gracefully (respects PDB: minAvailable=1 from deploy/kubernetes/pdb.yaml)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data --grace-period=60

# 3. Verify pods rescheduled
kubectl get pods -n aragora -o wide

# 4. Perform maintenance

# 5. Uncordon when ready
kubectl uncordon <node-name>
```

### Procedure 5: Database Migration

```bash
# 1. Check pending migrations
docker exec aragora-api python -m aragora.db.migrate --check

# 2. Create a backup FIRST
docker exec aragora-postgres pg_dump -U aragora aragora | gzip > \
  pre-migration-$(date +%Y%m%d_%H%M%S).sql.gz

# 3. Run migrations (uses advisory lock for safety; see aragora/db/ migration runner)
docker exec aragora-api python -m aragora.db.migrate --upgrade

# 4. Verify
docker exec aragora-api python -m aragora.db.migrate --check
curl -sf -H "Authorization: Bearer $ARAGORA_API_TOKEN" http://localhost:8080/health | jq
```

The migration runner acquires a PostgreSQL advisory lock (`pg_try_advisory_lock(2089872453)`)
to prevent concurrent migrations across instances. Per-migration defensive
checks skip already-applied versions (safe for crashed-pod edge cases).

### Procedure 6: Restart a Stuck Debate

The `init_stuck_debate_watchdog` background task automatically detects and
cancels debates that exceed the timeout. For manual intervention:

```bash
# 1. Identify stuck debates
curl -sf -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
  http://localhost:8080/api/v1/debates?status=active | jq

# 2. Cancel a specific debate
curl -sf -X POST -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
  http://localhost:8080/api/v1/debates/<debate-id>/cancel | jq

# 3. If the API is unresponsive, restart the backend
docker compose -f deploy/docker-compose.production.yml restart aragora
```

### Procedure 7: Investigate High Error Rate

```bash
# 1. Check SLO compliance
curl -sf http://localhost:9090/api/v1/query?query=aragora_requests_total | jq

# 2. Identify which endpoints are failing
curl -sf "http://localhost:9090/api/v1/query?query=topk(10,sum(rate(aragora_requests_total{status=~\"5..\"}[5m]))by(handler))" | jq

# 3. Check circuit breaker states
curl -sf "http://localhost:9090/api/v1/query?query=aragora_circuit_breaker_state" | jq

# 4. Check agent-specific errors
docker compose -f deploy/docker-compose.production.yml logs aragora --tail 200 | \
  grep -i "error\|exception\|circuit" | tail -20

# 5. Check upstream provider status
# Anthropic: https://status.anthropic.com
# OpenAI: https://status.openai.com
# OpenRouter: https://openrouter.ai/status
```

### Procedure 8: Emergency Full Stack Restart

```bash
# 1. Stop all services
docker compose -f deploy/docker-compose.production.yml down

# 2. Verify no orphan containers
docker ps -a --filter name=aragora

# 3. Start infrastructure first
docker compose -f deploy/docker-compose.production.yml up -d postgres redis
sleep 15

# 4. Verify infrastructure health
docker exec aragora-postgres pg_isready -U aragora
docker exec aragora-redis redis-cli -a "$REDIS_PASSWORD" ping

# 5. Start application
docker compose -f deploy/docker-compose.production.yml up -d aragora
sleep 30

# 6. Verify application health
curl -sf http://localhost:8080/healthz | jq
curl -sf http://localhost:8080/readyz | jq

# 7. Start optional profiles
docker compose -f deploy/docker-compose.production.yml \
  --profile workers --profile monitoring --profile backup up -d
```

### Procedure 9: Rotate All Secrets

For a scheduled full rotation (e.g., quarterly):

```bash
# 1. Generate new secrets
NEW_JWT=$(openssl rand -base64 32)
NEW_RECEIPT_KEY=$(openssl rand -base64 32)
NEW_PG_PASS=$(openssl rand -base64 24 | tr -d '/+=')
NEW_REDIS_PASS=$(openssl rand -base64 24 | tr -d '/+=')

# 2. Update PostgreSQL password (while old one still works)
docker exec -i aragora-postgres psql -U aragora -c \
  "ALTER USER aragora PASSWORD '$NEW_PG_PASS';"

# 3. Update Redis password
docker exec aragora-redis redis-cli -a "$REDIS_PASSWORD" \
  CONFIG SET requirepass "$NEW_REDIS_PASS"
docker exec aragora-redis redis-cli -a "$NEW_REDIS_PASS" CONFIG REWRITE

# 4. Update .env or Secrets Manager with ALL new values
# Set ARAGORA_JWT_SECRET_PREVIOUS=<old_jwt> for overlap period
# Set ARAGORA_JWT_SECRET=$NEW_JWT
# Set ARAGORA_RECEIPT_SIGNING_KEY_PREVIOUS=<old_receipt_key>
# Set ARAGORA_RECEIPT_SIGNING_KEY=$NEW_RECEIPT_KEY
# Set POSTGRES_PASSWORD=$NEW_PG_PASS
# Set REDIS_PASSWORD=$NEW_REDIS_PASS

# 5. Rolling restart
docker compose -f deploy/docker-compose.production.yml restart aragora debate-worker

# 6. Verify
curl -sf http://localhost:8080/healthz | jq
curl -sf -H "Authorization: Bearer $ARAGORA_API_TOKEN" http://localhost:8080/health | jq

# 7. After 24h overlap, remove _PREVIOUS env vars
```

---

## Quick Reference: Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | One LLM key required | - | Anthropic Claude provider |
| `OPENAI_API_KEY` | - | - | OpenAI GPT provider |
| `OPENROUTER_API_KEY` | Recommended | - | Fallback on 429 rate limits |
| `ARAGORA_JWT_SECRET` | Yes (production) | - | JWT signing |
| `ARAGORA_API_TOKEN` | Recommended | - | Service-to-service auth |
| `ARAGORA_RECEIPT_SIGNING_KEY` | Yes (production) | - | Receipt HMAC |
| `POSTGRES_PASSWORD` | Yes | - | Database auth |
| `REDIS_PASSWORD` | Yes | - | Redis auth |
| `ARAGORA_ENV` | - | `production` | Environment mode |
| `ARAGORA_LOG_LEVEL` | - | `INFO` | Log verbosity |
| `ARAGORA_LOG_FORMAT` | - | `json` | `json` or `text` |
| `ARAGORA_RATE_LIMIT` | - | `100` | Requests per minute per IP |
| `ARAGORA_MAX_CONCURRENT_DEBATES` | - | `10` | Max parallel debates |
| `ARAGORA_METRICS_ENABLED` | - | `true` | Prometheus metrics |
| `ARAGORA_METRICS_PORT` | - | `9090` | Metrics scrape port |
| `ARAGORA_ALLOWED_ORIGINS` | - | - | CORS allowed origins |
| `ARAGORA_TRUSTED_PROXIES` | - | `10.0.0.0/8,...` | Trusted reverse proxy CIDRs |
| `ARAGORA_USE_SECRETS_MANAGER` | - | `false` | Load secrets from AWS SM |
| `ARAGORA_REQUIRE_DISTRIBUTED` | - | `false` | Fail if shared state unavailable |
| `ARAGORA_MULTI_INSTANCE` | - | `false` | Enable multi-instance coordination |

---

## Port Reference

| Port | Service | Protocol | Exposure |
|------|---------|----------|----------|
| 80 | Nginx HTTP | HTTP | Public |
| 443 | Nginx HTTPS | HTTPS | Public |
| 8080 | Backend REST API | HTTP | Internal |
| 8765 | WebSocket Streaming | WS | Internal (proxied via Nginx) |
| 9090 | Prometheus / Metrics | HTTP | Internal |
| 9093 | Alertmanager | HTTP | Internal |
| 3000 | Grafana | HTTP | Internal (proxied via Nginx) |
| 5432 | PostgreSQL | TCP | Internal only |
| 6379 | Redis | TCP | Internal only |
| 16686 | Jaeger UI | HTTP | Internal |
