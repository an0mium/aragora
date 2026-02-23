# Deployment Guide

Unified entry point for deploying Aragora. Choose a path based on your needs.

## Quick Start (Docker Compose)

The fastest way to run Aragora locally:

```bash
# 1. Clone and configure
git clone https://github.com/an0mium/aragora.git
cd aragora
cp .env.example .env

# 2. Add at least one AI provider API key to .env
#    ANTHROPIC_API_KEY=sk-ant-...
#    or OPENAI_API_KEY=sk-...

# 3. Start (SQLite, no external deps)
docker compose -f docker-compose.simple.yml up

# 4. Verify
curl http://localhost:8080/api/health
```

For a full-stack local environment with PostgreSQL and Redis:

```bash
docker compose up
```

See [DOCKER_COMPOSE_GUIDE.md](DOCKER_COMPOSE_GUIDE.md) for the complete decision matrix
of all 6 Compose files and when to use each one.

## Deployment Paths

| Path | Guide | When to Use |
|------|-------|-------------|
| Docker Compose | [DOCKER_COMPOSE_GUIDE.md](DOCKER_COMPOSE_GUIDE.md) | Local dev, small teams, evaluation |
| Self-Hosted | [SELF_HOSTED_GUIDE.md](SELF_HOSTED_GUIDE.md) | Deploy on your own infra in <15 min |
| Kubernetes | [KUBERNETES.md](KUBERNETES.md) | Production clusters, horizontal scaling |
| Production (bare metal) | [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) | Direct Python + Supabase deployment |

## Kubernetes Deployment

Helm charts are available at `deploy/helm/aragora/`:

```bash
# Install from local chart
helm install aragora deploy/helm/aragora/ \
  --namespace aragora --create-namespace \
  --set secrets.anthropicApiKey=sk-ant-... \
  -f deploy/helm/aragora/values-production.yaml

# Or with staging values
helm install aragora deploy/helm/aragora/ \
  --namespace aragora --create-namespace \
  -f deploy/helm/aragora/values-staging.yaml
```

Additional Helm value files:

| File | Purpose |
|------|---------|
| `values.yaml` | Default configuration |
| `values-production.yaml` | Production-hardened settings |
| `values-staging.yaml` | Staging environment |
| `values-supabase.yaml` | Supabase backend |

For multi-region deployment, see `deploy/multi-region/helm/`.

For the Aragora Kubernetes Operator, see `aragora-operator/helm/aragora-operator/`.

Full Kubernetes guide: [KUBERNETES.md](KUBERNETES.md)

## Environment Variables

Required (at least one AI provider):

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude) |
| `OPENAI_API_KEY` | OpenAI API key (GPT) |

Recommended:

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Fallback provider (auto-used on 429 errors) |
| `ARAGORA_POSTGRES_DSN` | PostgreSQL connection string |
| `ARAGORA_REDIS_URL` | Redis URL for caching and pub/sub |

Full reference: [../reference/ENVIRONMENT.md](../reference/ENVIRONMENT.md)

## Scaling and Production

| Topic | Guide |
|-------|-------|
| Scaling limits and tuning | [SCALING.md](SCALING.md) |
| Performance tuning | [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) |
| Capacity planning | [CAPACITY_PLANNING.md](CAPACITY_PLANNING.md) |
| Production checklist | [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) |
| Production readiness | [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) |
| TLS configuration | [TLS.md](TLS.md) |
| Security hardening | [SECURITY_DEPLOYMENT.md](SECURITY_DEPLOYMENT.md) |

## Observability

| Topic | Guide |
|-------|-------|
| Monitoring stack | `deploy/monitoring/docker-compose.observability.yml` |
| Observability stack | `deploy/observability/docker-compose.yml` |
| Uptime monitoring | `deploy/uptime-kuma/docker-compose.yml` |
| Alert runbooks | [ALERT_RUNBOOKS.md](ALERT_RUNBOOKS.md) |
| Runbook metrics | [RUNBOOK_METRICS.md](RUNBOOK_METRICS.md) |

## Operations

| Topic | Guide |
|-------|-------|
| Operational runbook | [RUNBOOK.md](RUNBOOK.md) |
| Incident response | [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) |
| Incident playbooks | [INCIDENT_RESPONSE_PLAYBOOKS.md](INCIDENT_RESPONSE_PLAYBOOKS.md) |
| Incident communication | [INCIDENT_COMMUNICATION.md](INCIDENT_COMMUNICATION.md) |
| Disaster recovery | [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) |
| DR drill procedures | [DR_DRILL_PROCEDURES.md](DR_DRILL_PROCEDURES.md) |
| Launch checklist | [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md) |
| Staging validation | [STAGING_VALIDATION.md](STAGING_VALIDATION.md) |

## High Availability

| Component | Guide |
|-----------|-------|
| PostgreSQL HA | [POSTGRES_HA.md](POSTGRES_HA.md) |
| Redis HA | [REDIS_HA.md](REDIS_HA.md) |
| Streaming (Kafka/RabbitMQ) | [STREAMING_DEPLOYMENT.md](STREAMING_DEPLOYMENT.md) |
| Async gateway | [ASYNC_GATEWAY.md](ASYNC_GATEWAY.md) |
| Container volumes | [CONTAINER_VOLUMES.md](CONTAINER_VOLUMES.md) |

## CI/CD

| Topic | Guide |
|-------|-------|
| GitHub Actions | [GITHUB_ACTIONS.md](GITHUB_ACTIONS.md) |
| Release notes | [RELEASE_NOTES.md](RELEASE_NOTES.md) |
| Upgrade roadmap | [UPGRADE_ROADMAP.md](UPGRADE_ROADMAP.md) |
