---
title: Aragora Deployment Guide
description: Aragora Deployment Guide
---

# Aragora Deployment Guide

**Version**: 2.0.0
**Last Updated**: January 18, 2026

This guide covers deploying Aragora to production environments.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.25+ (for K8s deployment)
- At least one AI provider API key (Anthropic, OpenAI, etc.)

## Quick Start: Docker Compose

The simplest way to run Aragora in production:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your API keys
vim .env

# 3. Start services
docker compose up -d

# 4. Check health
curl http://localhost:8080/api/health
```

## Kubernetes Deployment

### 1. Prepare Secrets

First, create your secrets file from the template:

```bash
cp deploy/k8s/secret.yaml deploy/k8s/secret-local.yaml
# Edit with your actual values
vim deploy/k8s/secret-local.yaml
```

For production, use [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets):

```bash
kubeseal --format yaml < deploy/k8s/secret-local.yaml > deploy/k8s/sealed-secret.yaml
```

### 2. Build and Push Docker Image

```bash
# Build production image
docker build -t your-registry/aragora:latest .

# Push to registry
docker push your-registry/aragora:latest
```

### 3. Update Kustomization

Edit `deploy/k8s/kustomization.yaml`:

```yaml
images:
  - name: aragora
    newName: your-registry/aragora
    newTag: v1.0.0
```

### 4. Deploy

```bash
# Apply all resources
kubectl apply -k deploy/k8s/

# Watch rollout
kubectl -n aragora rollout status deployment/aragora

# Check pods
kubectl -n aragora get pods
```

### 5. Configure Ingress

Edit `deploy/k8s/ingress.yaml` with your domain:

```yaml
spec:
  tls:
    - hosts:
        - aragora.yourdomain.com
      secretName: aragora-tls
  rules:
    - host: aragora.yourdomain.com
```

### 6. TLS Configuration with cert-manager

Aragora includes cert-manager ClusterIssuers for automatic TLS certificate management.

#### Install cert-manager

```bash
# Install cert-manager (v1.14.0+)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=120s
kubectl wait --for=condition=ready pod -l app=webhook -n cert-manager --timeout=120s
kubectl wait --for=condition=ready pod -l app=cainjector -n cert-manager --timeout=120s
```

#### Apply ClusterIssuers

The `cert-manager.yaml` file includes three issuers:

| Issuer | Use Case |
|--------|----------|
| `letsencrypt-staging` | Testing (avoids rate limits, issues untrusted certs) |
| `letsencrypt-prod` | Production (real trusted certificates) |
| `selfsigned-issuer` | Local/dev environments |

```bash
# Update email in cert-manager.yaml first
vim deploy/k8s/cert-manager.yaml  # Change admin@aragora.ai to your email

# Apply ClusterIssuers
kubectl apply -f deploy/k8s/cert-manager.yaml
```

#### Configure Ingress for TLS

The ingress is already configured to use cert-manager. Update the domain:

```yaml
# deploy/k8s/ingress.yaml
metadata:
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"  # or letsencrypt-staging for testing
spec:
  tls:
    - hosts:
        - aragora.yourdomain.com
      secretName: aragora-tls
  rules:
    - host: aragora.yourdomain.com
```

#### Verify TLS Setup

```bash
# Check ClusterIssuers are ready
kubectl get clusterissuers

# Check certificate is issued
kubectl -n aragora get certificate

# Check certificate secret
kubectl -n aragora get secret aragora-tls

# Test HTTPS
curl -v https://aragora.yourdomain.com/api/health
```

#### Troubleshooting TLS

```bash
# Check certificate status
kubectl -n aragora describe certificate aragora-tls

# Check cert-manager logs
kubectl -n cert-manager logs -l app=cert-manager

# Check ACME challenges
kubectl -n aragora get challenges
```

**Common Issues:**

1. **Challenge failed**: Ensure DNS points to your ingress controller
2. **Rate limited**: Switch to `letsencrypt-staging` while testing
3. **Webhook timeout**: Restart cert-manager pods

### 7. PostgreSQL Configuration (Multi-Instance Required)

For production multi-instance deployments, PostgreSQL is required instead of SQLite.

#### Deploy PostgreSQL StatefulSet

```bash
# Apply PostgreSQL resources
kubectl apply -f deploy/k8s/postgres-statefulset.yaml

# Wait for PostgreSQL to be ready
kubectl -n aragora wait --for=condition=ready pod postgres-0 --timeout=120s
```

#### Or Use Managed PostgreSQL

For production, consider managed services:
- **AWS RDS**: `postgresql://user:pass@rds-instance.region.rds.amazonaws.com:5432/aragora?sslmode=require`
- **Google Cloud SQL**: Use Cloud SQL Auth Proxy
- **Supabase**: `postgresql://postgres.project-ref:password@aws-0-region.pooler.supabase.com:6543/postgres`

Configure via secrets:

```yaml
# In aragora-secrets
stringData:
  ARAGORA_POSTGRES_DSN: "postgresql://aragora:password@postgres-primary:5432/aragora?sslmode=require"
```

#### Initialize Schema

```bash
# Run schema initialization
kubectl -n aragora exec -it deploy/aragora -- python scripts/init_postgres_db.py

# Verify tables
kubectl -n aragora exec -it deploy/aragora -- python scripts/init_postgres_db.py --verify
```

### 9. Database Migrations

For PostgreSQL deployments, run migrations before starting the application:

```bash
# Option 1: Manual migration (before first deploy)
kubectl apply -f deploy/k8s/migration-job.yaml
kubectl -n aragora wait --for=condition=complete job/aragora-migrate --timeout=120s
kubectl -n aragora logs job/aragora-migrate

# Option 2: With Argo CD (automatic)
# The migration job has PreSync hook annotations - runs automatically before each sync

# Check migration status
kubectl apply -f deploy/k8s/migration-job.yaml --dry-run=client -o yaml | \
  grep -A 100 'name: aragora-migrate-status' | kubectl apply -f -
kubectl -n aragora logs job/aragora-migrate-status
```

For more database setup details, see [POSTGRESQL_MIGRATION.md](../POSTGRESQL_MIGRATION.md).

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `OPENAI_API_KEY` | OpenAI API key (alternative to Anthropic) |

### Recommended

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_REDIS_URL` | `redis://localhost:6379/0` | Redis for rate limiting and caching |
| `REDIS_URL` | `redis://localhost:6379` | Legacy Redis URL used by queues/oauth/token revocation |
| `ARAGORA_JWT_SECRET` | (required for auth) | 32+ character secret for JWT tokens |
| `ARAGORA_API_TOKEN` | (optional) | API token for authenticated endpoints |

### Optional Providers

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter for fallback |
| `GEMINI_API_KEY` | Google Gemini |
| `XAI_API_KEY` | xAI Grok |
| `MISTRAL_API_KEY` | Mistral AI |

### Billing (Optional)

| Variable | Description |
|----------|-------------|
| `STRIPE_SECRET_KEY` | Stripe API key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |

### Multi-Tenant Configuration (v2.0.0+)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_MULTI_TENANT` | `false` | Enable multi-tenant isolation |
| `ARAGORA_DEFAULT_TENANT` | `default` | Default tenant ID for legacy requests |
| `ARAGORA_TENANT_HEADER` | `X-Tenant-ID` | HTTP header for tenant identification |

#### Tenant Quotas

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_QUOTA_API_CALLS` | `100000` | Monthly API call limit per tenant |
| `ARAGORA_QUOTA_TOKENS` | `10000000` | Monthly token limit per tenant |
| `ARAGORA_QUOTA_STORAGE_GB` | `100` | Storage limit in GB per tenant |
| `ARAGORA_QUOTA_DEBATES` | `1000` | Monthly debate limit per tenant |

### API Versioning (v2.0.0+)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_API_VERSION` | `v2` | Current API version |
| `ARAGORA_API_LEGACY_ENABLED` | `true` | Support legacy unversioned endpoints |
| `ARAGORA_API_V1_SUNSET` | `2026-12-31` | Sunset date for API v1 |

The API supports both URL prefix versioning (`/api/v2/debates`) and header-based versioning (`X-API-Version: v2`).

### Metering & Usage Tracking (v2.0.0+)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_METERING_ENABLED` | `true` | Enable usage metering |
| `ARAGORA_METERING_FLUSH_INTERVAL` | `60` | Seconds between metering flushes |
| `ARAGORA_METERING_BACKEND` | `prometheus` | Metering backend (prometheus/statsd) |

## Resource Requirements

### Minimum (Development)

- CPU: 0.5 cores
- Memory: 512MB
- Storage: 1GB

### Recommended (Production)

- CPU: 2 cores
- Memory: 2GB
- Storage: 10GB
- Redis: 256MB

### Scaling Guidelines

| Concurrent Debates | Replicas | CPU | Memory |
|--------------------|----------|-----|--------|
| 1-5 | 1 | 1 core | 1GB |
| 5-20 | 2-3 | 2 cores | 2GB |
| 20-50 | 3-5 | 4 cores | 4GB |
| 50+ | 5-10 | 8 cores | 8GB |

## Health Checks

### Liveness Probe
```
GET /api/v2/health
```
Returns 200 if server is running.

### Readiness Probe
```
GET /api/v2/health/ready
```
Returns 200 if server can accept requests.

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` (port 8080):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:8080']
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `aragora_debates_total` | Total debates run |
| `aragora_debate_duration_seconds` | Debate duration histogram |
| `aragora_agent_errors_total` | Agent error count by type |
| `aragora_consensus_rate` | Consensus achievement rate |

### v2.0.0 Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `aragora_rlm_compression_ratio` | RLM context compression | < 0.5 |
| `aragora_tenant_requests_total` | Per-tenant request rate | - |
| `aragora_connector_sync_duration_seconds` | Connector sync time | p95 > 60s |
| `aragora_billing_events_total` | Billing events by tenant | - |
| `aragora_quota_usage_ratio` | Quota utilization per tenant | > 0.9 |

### Grafana Dashboard

Import the pre-built dashboard from `k8s/monitoring/aragora-dashboard.json`:

```bash
# Port-forward Grafana
kubectl -n monitoring port-forward svc/grafana 3000:3000

# Import via Grafana UI:
# 1. Go to Dashboards > Import
# 2. Upload k8s/monitoring/aragora-dashboard.json
# 3. Select Prometheus data source
```

### Alerting Rules

Apply alert rules from `k8s/monitoring/alerts.yaml`:

```bash
kubectl apply -f k8s/monitoring/alerts.yaml
```

Key alerts included:
- `AragoraHighErrorRate` - Agent error rate > 10/min
- `AragoraSlowDebates` - p95 debate duration > 5min
- `AragoraQuotaNearLimit` - Tenant quota > 90%
- `AragoraConnectorSyncFailed` - Connector sync failures

See `docs/RUNBOOK_METRICS.md` for alert response procedures.

## Troubleshooting

### Pod CrashLoopBackOff

1. Check logs: `kubectl -n aragora logs deploy/aragora`
2. Verify secrets: `kubectl -n aragora get secret aragora-secrets -o yaml`
3. Check resource limits

### Redis Connection Failed

1. Verify Redis is running: `kubectl -n aragora get pods -l app.kubernetes.io/name=aragora-redis`
2. Check service: `kubectl -n aragora get svc aragora-redis`
3. Test connection: `kubectl -n aragora exec -it deploy/aragora -- redis-cli -h aragora-redis ping`

### High Memory Usage

1. Check debate queue: Reduce `ARAGORA_MAX_CONCURRENT_DEBATES`
2. Enable memory limits in deployment
3. Consider horizontal scaling via HPA

## High Availability Deployment

For production deployments requiring high availability, Aragora includes pre-configured manifests for multi-replica, zone-distributed deployments.

### HA Architecture

```
                    ┌─────────────────┐
                    │   Ingress/LB    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Aragora  │  │  Aragora  │  │  Aragora  │
        │ Replica 1 │  │ Replica 2 │  │ Replica 3 │
        │  Zone A   │  │  Zone B   │  │  Zone C   │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │     Redis       │
                    │  (sessions/     │
                    │   token store)  │
                    └─────────────────┘
```

### Deploy HA Configuration

```bash
# Apply HA deployment (uses deploy/kubernetes/)
kubectl apply -k deploy/kubernetes/

# Verify replicas
kubectl -n aragora get pods -l app.kubernetes.io/name=aragora

# Check HPA status
kubectl -n aragora get hpa

# Check PodDisruptionBudget
kubectl -n aragora get pdb
```

### Key HA Components

#### 1. Horizontal Pod Autoscaler (HPA)

Automatically scales pods based on load:

```yaml
# deploy/kubernetes/hpa.yaml
spec:
  minReplicas: 2       # Minimum for HA
  maxReplicas: 10      # Scale up under load
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 70
```

#### 2. Pod Disruption Budget (PDB)

Ensures availability during node maintenance:

```yaml
# deploy/kubernetes/pdb.yaml
spec:
  minAvailable: 1    # At least 1 pod always running
  # OR: maxUnavailable: 1
```

#### 3. Pod Anti-Affinity

Spreads pods across nodes/zones:

```yaml
# In deployment.yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app.kubernetes.io/name: aragora
          topologyKey: kubernetes.io/hostname
```

#### 4. Topology Spread Constraints

Distributes across availability zones:

```yaml
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
```

### Redis for Shared State

The HA deployment uses Redis for:
- Session storage (enables sticky-session-free load balancing)
- Token blacklist (for logout across all replicas)
- Rate limiting state

Deploy Redis:

```bash
kubectl apply -f deploy/k8s/redis/statefulset.yaml
kubectl apply -f deploy/k8s/redis/service.yaml
```

For Redis HA in production, consider:
- Redis Sentinel for automatic failover
- Redis Cluster for horizontal scaling
- Managed Redis (AWS ElastiCache, GCP Memorystore)

### Load Testing

Verify HA setup with included Locust tests:

```bash
# Install locust
pip install locust

# Run load test (headless)
locust -f tests/load/locustfile.py --host=https://aragora.yourdomain.com \
    --headless -u 100 -r 10 --run-time 5m

# Or with web UI
locust -f tests/load/locustfile.py --host=https://aragora.yourdomain.com
# Open http://localhost:8089
```

### HA Checklist

- [ ] At least 2 replicas running
- [ ] HPA configured and active
- [ ] PDB prevents total outage during updates
- [ ] Redis deployed for shared state
- [ ] Pods spread across zones (check with `kubectl get pods -o wide`)
- [ ] Health checks passing (`/healthz`, `/readyz`)
- [ ] Load tested with expected traffic

## Rollback Procedures (v2.0.0+)

### Kubernetes Rollback

```bash
# View rollout history
kubectl -n aragora rollout history deployment/aragora

# Rollback to previous version
kubectl -n aragora rollout undo deployment/aragora

# Rollback to specific revision
kubectl -n aragora rollout undo deployment/aragora --to-revision=2

# Verify rollback
kubectl -n aragora rollout status deployment/aragora
```

### Database Rollback

```bash
# Rollback one alembic migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade abc123

# Restore from backup
pg_restore -d aragora backup_20260118.dump
```

### API Version Rollback

If you need to revert API changes while maintaining v2.0.0 server:

```bash
# Set environment to use legacy endpoints
export ARAGORA_API_VERSION=v1
export ARAGORA_API_LEGACY_ENABLED=true

# Apply config change
kubectl -n aragora set env deployment/aragora ARAGORA_API_VERSION=v1
```

## Security Recommendations

1. **Use Sealed Secrets or External Secrets** for API keys
2. **Enable TLS** via cert-manager or your ingress controller
3. **Set resource limits** to prevent resource exhaustion
4. **Use NetworkPolicies** to restrict traffic
5. **Enable Pod Security Standards** (restricted profile)
6. **Enable audit logging** for multi-tenant environments
7. **Configure tenant isolation** for shared deployments

## Backup and Recovery

### Database Backup

```bash
# SQLite backup (if using default storage)
kubectl -n aragora exec deploy/aragora -- sqlite3 /app/data/aragora.db ".backup /tmp/backup.db"
kubectl -n aragora cp aragora-0:/tmp/backup.db ./aragora-backup.db
```

### Redis Backup

```bash
# Trigger RDB snapshot
kubectl -n aragora exec aragora-redis-0 -- redis-cli BGSAVE
```
