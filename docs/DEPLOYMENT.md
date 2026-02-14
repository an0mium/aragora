# Deployment Guide

Three deployment paths, from simplest to production-grade.

## 1. Local Development (no Docker)

```bash
pip install aragora

# Offline mode — SQLite, no external services, no API keys needed
aragora serve --offline

# With API keys — full functionality
export ANTHROPIC_API_KEY=your-key
aragora serve
```

**Minimum requirements:** Python 3.10+, 4GB RAM

**Offline mode** sets SQLite backend, enables demo mode, and skips all external service connections. Good for testing and development.

## 2. Docker Compose (recommended for production)

```bash
cd deploy
cp docker-compose.yml docker-compose.override.yml  # customize if needed
docker compose up -d
```

**Services started:** Backend (port 8080 + WS 8765), Redis, PostgreSQL (optional profile)

**Environment variables:**

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes (one LLM key) | LLM provider |
| `ARAGORA_API_TOKEN` | Yes | API authentication |
| `OPENAI_API_KEY` | No | Additional LLM provider |
| `OPENROUTER_API_KEY` | No | Fallback on quota errors |
| `GMAIL_CLIENT_ID` | No | Gmail integration |
| `GMAIL_CLIENT_SECRET` | No | Gmail integration |
| `STRIPE_SECRET_KEY` | No | Billing (Stripe) |

### Secrets Management

**Production:** Use AWS Secrets Manager (already integrated):

```yaml
# docker-compose.yml
environment:
  - ARAGORA_USE_SECRETS_MANAGER=true
  - ARAGORA_SECRET_NAME=aragora/production
  - AWS_REGION=us-east-1
  - ARAGORA_SECRETS_STRICT=true
volumes:
  - ~/.aws:/home/aragora/.aws:ro  # mount AWS credentials
```

All API keys, OAuth credentials, and tokens are loaded from AWS Secrets Manager at runtime. The `.env` file contains only non-secret configuration (AWS region, database name).

**Development:** Use `.env` file (gitignored):

```bash
cp .env.template .env  # fill in API keys
```

### Health Checks

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /healthz` | Liveness probe (K8s) | None |
| `GET /readyz` | Readiness probe (K8s) | None |
| `GET /health` | Detailed dependency check | API token |

### Build Variants

The Dockerfile supports three installation levels:

```dockerfile
# Minimal (no Redis/Postgres drivers)
ARG INSTALL_VARIANT=minimal
pip install .

# With PostgreSQL + Redis
ARG INSTALL_VARIANT=postgres
pip install ".[postgres,redis]"

# Full (all optional dependencies)
ARG INSTALL_VARIANT=full
pip install ".[persistence,redis,monitoring,observability,postgres,rlm]"
```

Default in `deploy/Dockerfile` is full.

## 3. Kubernetes

Helm charts and manifests are in `deploy/kubernetes/`.

```bash
# Apply manifests
kubectl apply -f deploy/kubernetes/

# Or use the Helm chart
helm install aragora deploy/kubernetes/helm/
```

**Key K8s features:**
- Liveness/readiness probes at `/healthz` and `/readyz`
- Horizontal pod autoscaler based on CPU/memory
- Secrets mounted from AWS Secrets Manager via External Secrets Operator
- PostgreSQL via CloudNativePG or RDS
- Redis via Elasticache or Bitnami chart

## Platform-Specific: Mac Studio Deployment

For running Aragora as an always-on operations server on macOS (Apple Silicon):

```bash
cd deploy/liftmode
chmod +x setup.sh
./setup.sh
```

This setup script:
1. Validates Docker + AWS CLI prerequisites
2. Creates/verifies AWS Secrets Manager secret with your API keys
3. Starts Docker Compose (backend + Redis + PostgreSQL)
4. Guides through Gmail OAuth setup
5. Installs daily briefing via macOS launchd (7:00 AM)

**Hardware:** Tested on Mac Studio M3 Ultra (96GB). Runs comfortably on any Apple Silicon Mac with 16GB+.

## Port Reference

| Port | Service | Protocol |
|------|---------|----------|
| 8080 | REST API | HTTP |
| 8765 | WebSocket | WS |
| 5432 | PostgreSQL | TCP |
| 6379 | Redis | TCP |
| 9090 | Prometheus metrics | HTTP |

## Monitoring

Enable the monitoring profile for Prometheus + Grafana:

```bash
docker compose --profile monitoring up -d
```

This adds:
- Prometheus (port 9090) — scrapes `/metrics` endpoint
- Grafana (port 3001) — pre-configured dashboards
- Jaeger (port 16686) — distributed tracing via OpenTelemetry
- Loki + Promtail — log aggregation
