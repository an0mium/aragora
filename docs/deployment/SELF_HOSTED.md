# Self-Hosted Deployment Guide

Deploy Aragora on your own infrastructure with a single command.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Configuration Reference](#configuration-reference)
5. [TLS / HTTPS](#tls--https)
6. [Profiles](#profiles)
7. [Backup and Restore](#backup-and-restore)
8. [Upgrading](#upgrading)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)
11. [Production Checklist](#production-checklist)
12. [Uninstalling](#uninstalling)

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Docker | 20.10+ | 25.0+ |
| Docker Compose | v2 (plugin) | v2.24+ |
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4 cores |
| Disk | 10 GB | 40 GB (SSD) |
| OS | Linux, macOS, WSL2 | Ubuntu 22.04 LTS |
| Network | Outbound HTTPS | Static IP or domain |

**AI Provider:** At least one API key from Anthropic, OpenAI, or OpenRouter.

Verify Docker is installed:

```bash
docker --version          # Docker 20.10+
docker compose version    # Docker Compose v2+
```

---

## Quick Start

### One-Command Install

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
bash scripts/install.sh
```

The installer will:
1. Check prerequisites (Docker, Docker Compose, RAM, disk)
2. Generate `.env` with cryptographically random secrets
3. Prompt for your AI provider API key
4. Generate a self-signed TLS certificate
5. Pull/build container images
6. Start all services (PostgreSQL, Redis, Aragora, Nginx)
7. Run health checks and print access URLs

### Install Options

```bash
# Configure only, do not start containers
bash scripts/install.sh --no-start

# Start with monitoring (Prometheus + Grafana)
bash scripts/install.sh --profile monitoring

# Start with workers and monitoring
bash scripts/install.sh --profile monitoring --profile workers

# Full stack (monitoring + workers + daily backups)
bash scripts/install.sh --profile monitoring --profile workers --profile backup

# Regenerate .env (overwrites existing)
bash scripts/install.sh --force-env
```

### Verify Installation

```bash
# Health check (from the host)
curl -k https://localhost/healthz

# Service status
docker compose -f deploy/docker-compose.production.yml ps

# View logs
docker compose -f deploy/docker-compose.production.yml logs -f aragora

# Run a test debate
curl -k -X POST https://localhost/api/v1/debates \
  -H "Content-Type: application/json" \
  -d '{"topic": "What is the best database for startups?", "rounds": 2}'
```

---

## Architecture Overview

```
                         Internet
                            |
                     +------+------+
                     |    Nginx    |  :80 (HTTP -> HTTPS redirect)
                     | TLS Termn. |  :443 (HTTPS)
                     +------+------+
                            |
               +------------+------------+
               |                         |
        +------+------+          +-------+-------+
        | Aragora API |          | Aragora WS    |
        |   :8080     |          |   :8765       |
        +------+------+          +-------+-------+
               |                         |
        +------+------+          +-------+-------+
        |  PostgreSQL |          |    Redis      |
        |   :5432     |          |    :6379      |
        +-------------+          +---------------+

  Optional:
    +------------------+   +------------------+   +---------+
    | Debate Workers   |   |   Prometheus     |   | Grafana |
    | (--profile       |   | (--profile       |   |         |
    |  workers)        |   |  monitoring)     |   |         |
    +------------------+   +------------------+   +---------+
```

| Service | Purpose | Image |
|---------|---------|-------|
| nginx | TLS termination, rate limiting, reverse proxy | `nginx:1.25-alpine` |
| aragora | API server + WebSocket server | `ghcr.io/an0mium/aragora:latest` |
| postgres | Primary database | `postgres:16-alpine` |
| redis | Caching, session store, pub/sub | `redis:7-alpine` |
| debate-worker | Async debate processing (optional) | Same as aragora |
| prometheus | Metrics collection (optional) | `prom/prometheus:v2.51.0` |
| alertmanager | Alert routing and notifications (optional) | `prom/alertmanager:v0.27.0` |
| grafana | Dashboards and alerting (optional) | `grafana/grafana:10.4.0` |
| backup | Automated database backups (optional) | `postgres:16-alpine` |

---

## Configuration Reference

Configuration is stored in `deploy/.env`. The installer generates this from
`deploy/.env.template` with random secrets pre-filled.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key (or another provider) | `sk-ant-api03-...` |
| `POSTGRES_PASSWORD` | PostgreSQL password (auto-generated) | `aB3x...` |
| `REDIS_PASSWORD` | Redis password (auto-generated) | `rK9m...` |
| `ARAGORA_JWT_SECRET` | JWT signing secret (auto-generated) | `eF7q...` |

### AI Providers

At least one is required. Multiple providers enable agent diversity and
automatic fallback on rate-limit errors.

| Variable | Provider | Notes |
|----------|----------|-------|
| `ANTHROPIC_API_KEY` | Anthropic Claude | Recommended primary |
| `OPENAI_API_KEY` | OpenAI GPT | Secondary |
| `OPENROUTER_API_KEY` | OpenRouter | Auto-fallback on 429 errors |
| `MISTRAL_API_KEY` | Mistral AI | Large, Codestral models |
| `GEMINI_API_KEY` | Google Gemini | Optional |
| `XAI_API_KEY` | xAI Grok | Optional |

### Server

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_ENV` | Environment mode | `production` |
| `ARAGORA_PORT` | HTTP API port (internal) | `8080` |
| `ARAGORA_WS_PORT` | WebSocket port (internal) | `8765` |
| `ARAGORA_LOG_LEVEL` | Log verbosity | `INFO` |
| `ARAGORA_LOG_FORMAT` | `json` or `text` | `json` |
| `ARAGORA_MAX_CONCURRENT_DEBATES` | Max parallel debates | `10` |
| `ARAGORA_RATE_LIMIT` | Requests/min per IP | `100` |

### Security

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_JWT_SECRET` | JWT signing key | (generated) |
| `ARAGORA_API_TOKEN` | Service-to-service auth | (empty) |
| `ARAGORA_RECEIPT_SIGNING_KEY` | HMAC key for decision receipts | (generated) |
| `ARAGORA_ALLOWED_ORIGINS` | CORS allowed origins | `https://yourdomain.com` |
| `ARAGORA_TRUSTED_PROXIES` | Trusted proxy CIDRs | Private ranges |
| `ARAGORA_COOKIE_SECURE` | Secure cookie flag | `true` |
| `ARAGORA_COOKIE_HTTPONLY` | HttpOnly cookie flag | `true` |
| `ARAGORA_COOKIE_SAMESITE` | SameSite cookie policy | `Strict` |

### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | PostgreSQL username | `aragora` |
| `POSTGRES_PASSWORD` | PostgreSQL password | (generated) |
| `POSTGRES_DB` | Database name | `aragora` |

### Redis

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_PASSWORD` | Redis password | (generated) |
| `REDIS_MAXMEMORY` | Max memory allocation | `512mb` |

### Monitoring

| Variable | Description | Default |
|----------|-------------|---------|
| `GRAFANA_USER` | Grafana admin username | `admin` |
| `GRAFANA_PASSWORD` | Grafana admin password | (generated) |
| `PROMETHEUS_RETENTION` | Metrics retention period | `15d` |
| `ARAGORA_METRICS_ENABLED` | Enable Prometheus metrics | `true` |

### TLS / Domain

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_DOMAIN` | Public domain name | `aragora.example.com` |
| `ACME_EMAIL` | Let's Encrypt email | `admin@example.com` |
| `TLS_MODE` | `selfsigned` or `letsencrypt` | `selfsigned` |

### Workers

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKER_REPLICAS` | Number of worker containers | `2` |
| `WORKER_CONCURRENCY` | Debates per worker | `3` |

### Backup

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKUP_RETENTION_DAYS` | Days to keep backups | `7` |
| `BACKUP_INTERVAL` | Seconds between backups | `86400` (daily) |

### Integrations (Optional)

| Variable | Description |
|----------|-------------|
| `SLACK_BOT_TOKEN` | Slack bot token |
| `SLACK_SIGNING_SECRET` | Slack request signing |
| `TEAMS_APP_ID` | Microsoft Teams app ID |
| `TEAMS_APP_PASSWORD` | Microsoft Teams password |
| `ARAGORA_SSO_ENABLED` | Enable SSO (OIDC/SAML) |
| `SENTRY_DSN` | Sentry error tracking |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry endpoint |

---

## TLS / HTTPS

### Default: Self-Signed Certificate

The installer generates a self-signed certificate for immediate HTTPS. This
is suitable for internal use and testing, but browsers will show a warning.

Certificates are stored at:
```
deploy/nginx/certs/server.crt
deploy/nginx/certs/server.key
```

### Let's Encrypt (Recommended for Production)

For automated, trusted TLS certificates:

1. Point your domain's DNS A record to your server's IP address.

2. Install certbot on the host:
   ```bash
   # Ubuntu/Debian
   sudo apt install certbot

   # macOS
   brew install certbot
   ```

3. Obtain a certificate:
   ```bash
   sudo certbot certonly --standalone -d aragora.yourdomain.com
   ```

4. Copy or symlink the certificates:
   ```bash
   sudo cp /etc/letsencrypt/live/aragora.yourdomain.com/fullchain.pem \
       deploy/nginx/certs/server.crt
   sudo cp /etc/letsencrypt/live/aragora.yourdomain.com/privkey.pem \
       deploy/nginx/certs/server.key
   ```

5. Restart nginx:
   ```bash
   docker compose -f deploy/docker-compose.production.yml restart nginx
   ```

6. Set up auto-renewal:
   ```bash
   # Add to crontab
   0 0 1 * * certbot renew --quiet && \
       cp /etc/letsencrypt/live/aragora.yourdomain.com/fullchain.pem deploy/nginx/certs/server.crt && \
       cp /etc/letsencrypt/live/aragora.yourdomain.com/privkey.pem deploy/nginx/certs/server.key && \
       docker compose -f deploy/docker-compose.production.yml restart nginx
   ```

### Cloudflare Origin Certificate

If you use Cloudflare as a proxy:
1. In Cloudflare dashboard: SSL/TLS > Origin Server > Create Certificate
2. Download the certificate and private key
3. Copy to `deploy/nginx/certs/server.crt` and `deploy/nginx/certs/server.key`
4. Set Cloudflare SSL mode to "Full (strict)"

---

## Profiles

Docker Compose profiles enable optional services without modifying the
compose file.

| Profile | Services Added | Use Case |
|---------|---------------|----------|
| `monitoring` | Prometheus, Alertmanager, Grafana | Metrics, SLO alerts, dashboards |
| `workers` | Debate workers (2 replicas) | High-throughput debate processing |
| `backup` | Automated pg_dump | Daily database backups |

### Enable Profiles

```bash
# Single profile
docker compose -f deploy/docker-compose.production.yml --profile monitoring up -d

# Multiple profiles
docker compose -f deploy/docker-compose.production.yml \
    --profile monitoring --profile workers --profile backup up -d
```

### Monitoring Access

When the `monitoring` profile is enabled:
- Grafana is available at `https://localhost/grafana/`
- Default credentials: admin / (your GRAFANA_PASSWORD from .env)
- Prometheus loads rule files from `deploy/monitoring/alerts.yaml`
- Alertmanager is reachable internally at `alertmanager:9093` and receives Prometheus alerts
- Pre-provisioned dashboards for debate metrics, API latency, queue health, and costs

---

## Backup and Restore

### Automated Backups

Enable the backup profile for daily PostgreSQL dumps:

```bash
docker compose -f deploy/docker-compose.production.yml --profile backup up -d
```

Backups are stored in `deploy/backups/` with the naming pattern:
```
aragora-20260212_030000.sql.gz
```

Retention is controlled by `BACKUP_RETENTION_DAYS` (default: 7 days).

### Manual Backup

```bash
# Full database dump
docker exec aragora-postgres \
    pg_dump -U aragora aragora | gzip > backup-$(date +%Y%m%d).sql.gz

# Backup Aragora data volume
docker run --rm -v aragora-data:/data -v $(pwd):/backup \
    alpine tar czf /backup/aragora-data-$(date +%Y%m%d).tar.gz -C /data .
```

### Restore from Backup

```bash
# Stop Aragora to prevent writes during restore
docker compose -f deploy/docker-compose.production.yml stop aragora

# Restore database
gunzip < backup-20260212.sql.gz | \
    docker exec -i aragora-postgres psql -U aragora aragora

# Restart
docker compose -f deploy/docker-compose.production.yml start aragora
```

### Off-Site Backup

For disaster recovery, copy backups to an external location:

```bash
# Example: AWS S3
aws s3 sync deploy/backups/ s3://your-bucket/aragora-backups/

# Example: rsync to another server
rsync -az deploy/backups/ backup-server:/opt/aragora-backups/
```

---

## Upgrading

### Minor Updates (Patch / Feature)

```bash
# Pull latest code
cd aragora
git pull

# Rebuild and restart (zero-downtime for stateless containers)
docker compose -f deploy/docker-compose.production.yml up -d --build

# Verify health
curl -k https://localhost/healthz

# Check migration logs
docker compose -f deploy/docker-compose.production.yml logs aragora | grep -i migration
```

### Major Updates (Breaking Changes)

1. **Read the changelog** for breaking changes and migration notes.

2. **Create a backup** before upgrading:
   ```bash
   docker exec aragora-postgres pg_dump -U aragora aragora | gzip > pre-upgrade.sql.gz
   ```

3. **Pull and rebuild:**
   ```bash
   git pull
   docker compose -f deploy/docker-compose.production.yml up -d --build
   ```

4. **Verify** the upgrade:
   ```bash
   curl -k https://localhost/healthz
   docker compose -f deploy/docker-compose.production.yml logs --tail 50 aragora
   ```

5. **Rollback** if needed:
   ```bash
   git checkout <previous-tag>
   docker compose -f deploy/docker-compose.production.yml up -d --build
   gunzip < pre-upgrade.sql.gz | docker exec -i aragora-postgres psql -U aragora aragora
   ```

### Updating Individual Services

```bash
# Update only the Aragora container
docker compose -f deploy/docker-compose.production.yml up -d --build aragora

# Update only PostgreSQL (careful: test first)
docker compose -f deploy/docker-compose.production.yml pull postgres
docker compose -f deploy/docker-compose.production.yml up -d postgres
```

---

## Scaling

### Horizontal Scaling (Workers)

For high-throughput environments, add more debate workers:

```bash
# Scale workers to 4
docker compose -f deploy/docker-compose.production.yml \
    --profile workers up -d --scale debate-worker=4
```

### Increase Concurrent Debates

Edit `deploy/.env`:
```bash
ARAGORA_MAX_CONCURRENT_DEBATES=20
WORKER_CONCURRENCY=5
```

Then restart:
```bash
docker compose -f deploy/docker-compose.production.yml up -d
```

### Resource Tuning

The default resource limits are conservative. Adjust in the compose file or
override with a `docker-compose.override.yml`:

```yaml
services:
  aragora:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 8G
  postgres:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## Troubleshooting

### Service Will Not Start

```bash
# Check service status
docker compose -f deploy/docker-compose.production.yml ps

# View recent logs
docker compose -f deploy/docker-compose.production.yml logs --tail 100 aragora

# Validate compose file
docker compose -f deploy/docker-compose.production.yml config --quiet

# Validate environment
docker compose -f deploy/docker-compose.production.yml config | head -20
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
docker compose -f deploy/docker-compose.production.yml exec postgres pg_isready -U aragora

# Test connection from Aragora container
docker compose -f deploy/docker-compose.production.yml exec aragora \
    python -c "import asyncpg; print('asyncpg available')"

# Check database logs
docker compose -f deploy/docker-compose.production.yml logs postgres
```

### Redis Connection Errors

```bash
# Ping Redis
docker compose -f deploy/docker-compose.production.yml exec redis \
    redis-cli -a "$REDIS_PASSWORD" ping

# Check memory usage
docker compose -f deploy/docker-compose.production.yml exec redis \
    redis-cli -a "$REDIS_PASSWORD" info memory
```

### API Returns 500 Errors

1. Check AI provider API keys are valid and have available quota.
2. Verify database connectivity (see above).
3. Check logs for specific error messages:
   ```bash
   docker compose -f deploy/docker-compose.production.yml logs --tail 200 aragora | grep -i error
   ```

### TLS Certificate Errors

```bash
# Check certificate validity
openssl x509 -in deploy/nginx/certs/server.crt -text -noout | grep -A2 "Validity"

# Verify key matches certificate
openssl x509 -noout -modulus -in deploy/nginx/certs/server.crt | openssl md5
openssl rsa -noout -modulus -in deploy/nginx/certs/server.key | openssl md5
# Both should output the same hash

# Test TLS handshake
openssl s_client -connect localhost:443 -servername localhost </dev/null
```

### WebSocket Connection Failures

```bash
# Test WebSocket endpoint
curl -k -i -N \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket" \
    -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
    https://localhost/ws
```

Ensure nginx is forwarding Upgrade headers (the default config does this).

### Out of Memory

```bash
# Check container memory usage
docker stats --no-stream

# Increase limits in .env or compose override
# See "Scaling > Resource Tuning" above
```

### Port Conflicts

If ports 80 or 443 are in use:

```bash
# Check what is using the port
sudo lsof -i :80
sudo lsof -i :443
```

To use different ports, add a `docker-compose.override.yml`:

```yaml
services:
  nginx:
    ports:
      - "8080:80"
      - "8443:443"
```

### Container Logs

```bash
# All services
docker compose -f deploy/docker-compose.production.yml logs

# Specific service with timestamps
docker compose -f deploy/docker-compose.production.yml logs -t aragora

# Follow logs in real time
docker compose -f deploy/docker-compose.production.yml logs -f

# Last 100 lines from all services
docker compose -f deploy/docker-compose.production.yml logs --tail 100
```

---

## Production Checklist

Before going to production, verify each item:

### Security

- [ ] Replace self-signed TLS certificate with a trusted certificate
- [ ] Set `ARAGORA_ALLOWED_ORIGINS` to your actual domain(s) (not `*`)
- [ ] Set a strong, unique `ARAGORA_API_TOKEN` for service-to-service auth
- [ ] Verify `ARAGORA_COOKIE_SECURE=true`
- [ ] Review and restrict `ARAGORA_TRUSTED_PROXIES` to your actual proxy IPs
- [ ] Firewall: only expose ports 80 and 443 to the internet
- [ ] Firewall: block direct access to PostgreSQL (5432) and Redis (6379)
- [ ] API keys stored in .env are not committed to version control

### Data

- [ ] Enable automated backups (`--profile backup`)
- [ ] Test backup restore procedure at least once
- [ ] Set up off-site backup replication (S3, rsync, etc.)
- [ ] Verify `BACKUP_RETENTION_DAYS` meets your compliance requirements

### Monitoring

- [ ] Enable monitoring profile (`--profile monitoring`)
- [ ] Change default Grafana password
- [ ] Configure alerting (Grafana alerts or Alertmanager)
- [ ] Set up uptime monitoring (external probe to `/healthz`)

### Performance

- [ ] Set `ARAGORA_MAX_CONCURRENT_DEBATES` based on expected load
- [ ] Enable workers profile if running more than 10 concurrent debates
- [ ] Verify PostgreSQL `shared_buffers` matches allocated RAM (25% of container memory)
- [ ] Verify Redis `maxmemory` is set appropriately

### Operations

- [ ] Document your deployment in your team's runbook
- [ ] Set up log aggregation (Loki, CloudWatch, Datadog, etc.)
- [ ] Test the upgrade procedure in a staging environment
- [ ] Verify DNS and domain configuration
- [ ] Test WebSocket connectivity from client networks

---

## Uninstalling

```bash
# Stop all services and remove containers
docker compose -f deploy/docker-compose.production.yml \
    --profile monitoring --profile workers --profile backup down

# Remove all data volumes (DESTRUCTIVE)
docker compose -f deploy/docker-compose.production.yml \
    --profile monitoring --profile workers --profile backup down -v

# Remove generated files
rm -f deploy/.env
rm -rf deploy/nginx/certs
rm -rf deploy/backups
```
