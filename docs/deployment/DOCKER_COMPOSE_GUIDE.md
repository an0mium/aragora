# Docker Compose Decision Matrix

Quick reference for choosing the right Docker Compose configuration.

## Root-Level Compose Files

| File | Use Case | Database | Redis | Command |
|------|----------|----------|-------|---------|
| `docker-compose.yml` | **Default** — full stack local dev | PostgreSQL | Yes | `docker compose up` |
| `docker-compose.simple.yml` | **Quickest start** — SQLite, no external deps | SQLite | No | `docker compose -f docker-compose.simple.yml up` |
| `docker-compose.quickstart.yml` | **Demo with UI** — frontend + backend | SQLite | No | `docker compose -f docker-compose.quickstart.yml up` |
| `docker-compose.dev.yml` | **Full dev** — hot reload, debug ports | PostgreSQL | Yes | `docker compose -f docker-compose.dev.yml up` |
| `docker-compose.sme.yml` | **SME evaluation** — business-focused config | PostgreSQL | Yes | `docker compose -f docker-compose.sme.yml up` |
| `docker-compose.production.yml` | **Production** — hardened, resource limits | PostgreSQL 16 | Redis 7 | `docker compose -f docker-compose.production.yml up -d` |

## Decision Flow

```
Need to run Aragora?
  |
  +-- Just trying it out? --> docker-compose.simple.yml
  |
  +-- Want the UI demo? --> docker-compose.quickstart.yml
  |
  +-- Local development?
  |     +-- Quick iteration? --> docker-compose.yml
  |     +-- Full debug/hot-reload? --> docker-compose.dev.yml
  |
  +-- Evaluating for business? --> docker-compose.sme.yml
  |
  +-- Deploying to production?
        +-- Single server? --> docker-compose.production.yml
        +-- Kubernetes? --> deploy/kubernetes/
        +-- Multi-region? --> deploy/docker-compose.production.yml
```

## Deploy Directory Variants

| File | Purpose |
|------|---------|
| `deploy/docker-compose.yml` | Base deployment template |
| `deploy/docker-compose.production.yml` | Enterprise production (Nginx, pgBouncer, monitoring profiles) |
| `deploy/demo/docker-compose.yml` | Lightweight demo/trial deployment |
| `deploy/self-hosted/docker-compose.yml` | Self-hosted customer deployment |

## Specialized Stacks

| File | Purpose |
|------|---------|
| `deploy/openclaw/docker-compose.yml` | OpenClaw gateway standalone |
| `deploy/openclaw/docker-compose.standalone.yml` | Minimal OpenClaw (no Aragora dependency) |
| `deploy/openclaw/docker-compose.pr-reviewer.yml` | GitHub PR review bot |
| `deploy/openclaw/docker-compose.devops.yml` | DevOps automation agent |
| `deploy/monitoring/docker-compose.observability.yml` | Prometheus + Grafana + alerting |
| `deploy/observability/docker-compose.yml` | OpenTelemetry collector stack |
| `deploy/uptime-kuma/docker-compose.yml` | Uptime monitoring |
| `deploy/liftmode/docker-compose.yml` | Liftmode integration |

## Production Profiles

The production compose (`docker-compose.production.yml`) supports optional profiles:

```bash
# Base stack (API + PostgreSQL + Redis)
docker compose -f docker-compose.production.yml up -d

# With monitoring (adds Prometheus + Grafana + Alertmanager)
docker compose -f docker-compose.production.yml --profile monitoring up -d

# With workers (adds horizontal scaling)
docker compose -f docker-compose.production.yml --profile workers up -d

# With backups (adds pg_dump cronjob)
docker compose -f docker-compose.production.yml --profile backup up -d

# Everything
docker compose -f docker-compose.production.yml --profile monitoring --profile workers --profile backup up -d
```

## Environment Variables

All compose files read from `.env` in the project root. Copy `.env.example` to get started:

```bash
cp .env.example .env
# Edit .env with your API keys (at minimum: ANTHROPIC_API_KEY or OPENAI_API_KEY)
```

For production, use `.env.production.template` as your starting point.
