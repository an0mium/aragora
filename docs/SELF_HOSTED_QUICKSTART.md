# Self-Hosted Quickstart Guide

**Target:** Running Aragora in <5 minutes, complete setup in <30 minutes.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Docker | 20.10+ | 24.0+ |
| Docker Compose | 2.0+ | 2.20+ |
| Memory | 2 GB | 4 GB |
| Disk | 5 GB | 20 GB |
| API Key | 1 provider | 2+ providers |

---

## Quick Start (5 minutes)

### Step 1: Clone the Repository

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
```

### Step 2: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env and add at least one API key
# Required: ANTHROPIC_API_KEY or OPENAI_API_KEY
nano .env
```

**Minimum configuration:**

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
# OR
OPENAI_API_KEY=sk-...
```

### Step 3: Start Aragora

```bash
# Simple mode (SQLite, no dependencies)
docker compose -f docker-compose.simple.yml up -d
```

### Step 4: Verify

```bash
# Check health
curl http://localhost:8080/api/health

# Expected response:
# {"status": "healthy", "version": "2.1.15"}
```

### Step 5: Run Your First Debate

```bash
# Via CLI (if installed)
aragora ask "What is the best programming language for beginners?"

# Via API
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{"topic": "What is the best programming language for beginners?"}'
```

**Done! Aragora is running at http://localhost:8080**

---

## Production Setup (30 minutes)

For production deployments with PostgreSQL, Redis, and persistent storage.

### Step 1: Configure Production Environment

```bash
cp .env.example .env
nano .env
```

**Production configuration:**

```bash
# .env

# Environment
ARAGORA_ENVIRONMENT=production

# AI Providers (at least one required)
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...

# Database (PostgreSQL)
POSTGRES_USER=aragora
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=aragora

# JWT Secret (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
ARAGORA_JWT_SECRET=your-secure-jwt-secret-minimum-32-characters

# CORS (your frontend domain)
ARAGORA_ALLOWED_ORIGINS=https://your-domain.com

# Optional: Additional providers for diverse debates
GEMINI_API_KEY=...
XAI_API_KEY=...
OPENROUTER_API_KEY=...
```

### Step 2: Start with PostgreSQL

```bash
# Production mode with PostgreSQL
docker compose --profile postgres up -d

# Initialize database tables
docker compose exec aragora python scripts/init_postgres_db.py

# Verify database
docker compose exec aragora python scripts/init_postgres_db.py --verify
```

### Step 3: Enable Workers (Horizontal Scaling)

```bash
# Start with debate workers for parallel processing
docker compose --profile postgres --profile with-workers up -d

# Check worker status
docker compose logs debate-worker
```

### Step 4: Configure TLS (Required for Production)

Option A: Use a reverse proxy (recommended)

```nginx
# /etc/nginx/sites-available/aragora
server {
    listen 443 ssl http2;
    server_name api.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/api.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Option B: Use Caddy (automatic TLS)

```caddyfile
# Caddyfile
api.your-domain.com {
    reverse_proxy localhost:8080
}
```

### Step 5: Verify Production Setup

```bash
# Health check
curl https://api.your-domain.com/api/health

# Database connection
curl https://api.your-domain.com/api/health/db

# Run a test debate
curl -X POST https://api.your-domain.com/api/debates \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
  -d '{"topic": "Test debate"}'
```

---

## Configuration Reference

### Core Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | One of | Anthropic Claude API key |
| `OPENAI_API_KEY` | these | OpenAI API key |
| `ARAGORA_ENVIRONMENT` | Prod | `development` or `production` |
| `ARAGORA_JWT_SECRET` | Prod | JWT signing secret (32+ chars) |
| `ARAGORA_ALLOWED_ORIGINS` | Prod | CORS allowed origins |

### Database Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `aragora` | PostgreSQL username |
| `POSTGRES_PASSWORD` | - | PostgreSQL password |
| `POSTGRES_DB` | `aragora` | PostgreSQL database name |
| `DATABASE_URL` | - | Full connection string (alternative) |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_MAX_CONCURRENT_DEBATES` | `5` | Max parallel debates |
| `ARAGORA_DEBATE_TIMEOUT` | `900` | Debate timeout (seconds) |
| `ARAGORA_AGENT_TIMEOUT` | `240` | Per-agent timeout (seconds) |
| `ARAGORA_DB_POOL_SIZE` | `10` | Database connection pool |

---

## Monitoring

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Basic health check |
| `GET /api/health/db` | Database connectivity |
| `GET /api/health/redis` | Redis connectivity |
| `GET /api/metrics` | Prometheus metrics |

### Logs

```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f aragora

# View worker logs
docker compose logs -f debate-worker
```

### Prometheus Metrics

Add to your Prometheus configuration:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/api/metrics'
```

---

## Backup & Restore

### Backup Database

```bash
# PostgreSQL backup
docker compose exec postgres pg_dump -U aragora aragora > backup.sql

# With timestamp
docker compose exec postgres pg_dump -U aragora aragora > "backup_$(date +%Y%m%d_%H%M%S).sql"
```

### Backup Data Volumes

```bash
# Stop services first
docker compose down

# Backup volumes
docker run --rm -v aragora_aragora-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/aragora-data.tar.gz -C /data .

docker run --rm -v aragora_postgres-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres-data.tar.gz -C /data .
```

### Restore Database

```bash
# Restore from SQL dump
cat backup.sql | docker compose exec -T postgres psql -U aragora aragora
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs aragora

# Common issues:
# - Missing API key: Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env
# - Port in use: Change ARAGORA_PORT in .env
# - Memory: Increase Docker memory limit
```

### Database Connection Failed

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Check connection string
docker compose exec aragora python -c "from aragora.db import get_db; print(get_db())"

# Reinitialize database
docker compose exec aragora python scripts/init_postgres_db.py
```

### Debate Timeouts

```bash
# Increase timeouts in .env
ARAGORA_DEBATE_TIMEOUT=1800
ARAGORA_AGENT_TIMEOUT=480

# Restart
docker compose restart aragora
```

### High Memory Usage

```bash
# Check memory
docker stats

# Limit container memory in docker-compose.yml
services:
  aragora:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## Upgrades

### Standard Upgrade

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker compose build
docker compose up -d

# Run migrations
docker compose exec aragora python scripts/init_postgres_db.py
```

### Major Version Upgrade

```bash
# Backup first
docker compose exec postgres pg_dump -U aragora aragora > pre_upgrade_backup.sql

# Pull and rebuild
git pull origin main
docker compose build

# Stop old version
docker compose down

# Start new version
docker compose up -d

# Run migrations
docker compose exec aragora alembic upgrade head
```

---

## Related Documentation

- [ENVIRONMENT.md](ENVIRONMENT.md) - Full environment variable reference
- [SUPABASE_SETUP.md](SUPABASE_SETUP.md) - Supabase configuration
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [BACKLOG_Q1_2026.md](BACKLOG_Q1_2026.md) - Roadmap

---

*Created: 2026-01-24*
*Version: 2.1.15*
