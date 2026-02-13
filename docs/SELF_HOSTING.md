# Self-Hosting Aragora

Run the Aragora Decision Integrity Platform on your own infrastructure.

## Quick Start (SQLite, no dependencies)

The fastest way to get running. Uses SQLite for storage -- no PostgreSQL or Redis needed.

```bash
# 1. Clone the repository
git clone https://github.com/an0mium/aragora.git
cd aragora

# 2. Set your API key (at least one required)
export ANTHROPIC_API_KEY=sk-ant-...
# or: export OPENAI_API_KEY=sk-...

# 3. Start with Docker Compose
docker compose -f docker-compose.simple.yml up -d

# 4. Verify it's running
curl http://localhost:8080/healthz
```

The server is now available at `http://localhost:8080`.

## Deployment Tiers

| Tier | Compose File | Storage | Best For |
|------|-------------|---------|----------|
| **Simple** | `docker-compose.simple.yml` | SQLite | Evaluation, small teams |
| **Standard** | `docker-compose.yml` | PostgreSQL + Redis | Production use |
| **SME** | `docker-compose.sme.yml` | PostgreSQL + Redis + Grafana | SMBs with monitoring |
| **Production** | `docker-compose.production.yml` | PostgreSQL + Redis Sentinel + Traefik | Enterprise HA |

## Standard Deployment (PostgreSQL + Redis)

For production use with durable storage and caching:

```bash
# 1. Set required variables
export ANTHROPIC_API_KEY=sk-ant-...
export POSTGRES_PASSWORD=$(openssl rand -base64 24)

# 2. Start all services
docker compose up -d

# 3. Verify
curl http://localhost:8080/healthz
```

This starts three containers: `aragora` (API server), `db` (PostgreSQL 15), and `redis` (Redis 7).

## Environment Variables

### Required (at least one AI provider)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI GPT API key |

### Optional AI Providers

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter fallback (auto-used on 429 rate limits) |
| `MISTRAL_API_KEY` | Mistral AI (Large, Codestral) |
| `GEMINI_API_KEY` | Google Gemini |
| `XAI_API_KEY` | xAI Grok |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_BIND_HOST` | `0.0.0.0` | Server bind address (set in Dockerfile) |
| `ARAGORA_API_PORT` | `8080` | HTTP API port |
| `ARAGORA_WS_PORT` | `8765` | WebSocket port |
| `ARAGORA_ENV` | `production` | Environment (`development`, `production`) |
| `ARAGORA_DB_BACKEND` | `auto` | Storage backend (`sqlite`, `postgres`, `auto`) |
| `ARAGORA_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ARAGORA_ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_API_TOKEN` | *(none)* | API token for authentication |
| `ARAGORA_SECRET_KEY` | *(none)* | JWT signing key (generate: `openssl rand -base64 32`) |
| `ARAGORA_JWT_SECRET` | *(none)* | Alias for SECRET_KEY |

### Database (Standard/Production tiers)

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | `aragora` | PostgreSQL password |
| `DATABASE_URL` | *(auto)* | Full PostgreSQL DSN (auto-configured in compose) |
| `ARAGORA_REDIS_URL` | *(auto)* | Redis connection URL (auto-configured in compose) |

## Persistent Data

All compose files mount a named volume for persistent data:

| Volume | Container Path | Contents |
|--------|---------------|----------|
| `aragora-data` | `/app/data` | SQLite databases, ELO ratings, debate history |
| `aragora-data/.aragora_beads` | `/app/data/.aragora_beads` | Bead and convoy store |
| `postgres-data` | `/var/lib/postgresql/data` | PostgreSQL data (Standard+ tiers) |
| `redis-data` | `/data` | Redis append-only file (Standard+ tiers) |

To back up your data:

```bash
# SQLite tier
docker cp aragora:/app/data ./backup-data

# PostgreSQL tier
docker exec aragora-db pg_dump -U aragora aragora > backup.sql
```

## Health Checks

The server exposes these health endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/healthz` | Liveness probe (is the server running?) |
| `/readyz` | Readiness probe (can the server handle requests?) |
| `/health` | Alias for `/healthz` |
| `/metrics` | Prometheus metrics (when `ARAGORA_METRICS_ENABLED=true`) |

## Offline / Demo Mode

Run without any API keys for evaluation:

```bash
docker compose -f docker-compose.simple.yml run --rm \
  -e ARAGORA_OFFLINE=true -e ARAGORA_DEMO_MODE=true \
  aragora python -m aragora.server --offline
```

Or use the quickstart script:

```bash
bash deploy/quickstart.sh --offline
```

## Upgrading

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose -f docker-compose.simple.yml up -d --build
```

Data volumes are preserved across rebuilds.

## Troubleshooting

**Container starts but health check fails:**

Check the logs:
```bash
docker compose -f docker-compose.simple.yml logs aragora
```

Common causes:
- No API key set -- at least `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is required
- Port conflict -- change `ARAGORA_API_PORT` if 8080 is in use

**Database connection errors (Standard tier):**

Wait for PostgreSQL to finish initializing:
```bash
docker compose logs db
```

The server retries database connections on startup and will fall back to degraded mode if migrations fail.

**Permission denied on data volume:**

The container runs as UID 1000. If your host volume has different ownership:
```bash
sudo chown -R 1000:1000 ./data
```

## Architecture

```
                    +------------------+
                    |   Load Balancer  |  (optional: Traefik in production tier)
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     +--------v---------+        +---------v--------+
     |  Aragora Server  |        |  Aragora Server  |  (replicas in production)
     |  :8080 (HTTP)    |        |  :8080 (HTTP)    |
     |  :8765 (WS)      |        |  :8765 (WS)      |
     +--------+---------+        +---------+--------+
              |                             |
     +--------v-----------------------------v--------+
     |              Data Layer                        |
     |  SQLite (simple) | PostgreSQL + Redis (std+)   |
     +------------------------------------------------+
```
