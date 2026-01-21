# Aragora Production Deployment Guide

This guide covers deploying aragora.ai to production with Supabase as the database backend.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for the live dashboard)
- A Supabase project (free tier works for getting started)
- At least one AI provider API key (Anthropic, OpenAI, etc.)

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/aragora.git
cd aragora
pip install -e ".[postgres]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your credentials (see Configuration below)

# 3. Initialize database
python scripts/init_postgres_db.py

# 4. Verify setup
python scripts/init_postgres_db.py --verify

# 5. Start server
python -m aragora.server.unified_server --port 8080
```

## Configuration

### Supabase Setup

1. **Create a Supabase Project**
   - Go to [supabase.com](https://supabase.com) and create a new project
   - Note your project reference (e.g., `abcdefghijklmnop`)

2. **Get API Credentials** (Settings > API)
   ```bash
   SUPABASE_URL=https://[project-ref].supabase.co
   SUPABASE_KEY=[your-service-role-key]  # Use service role for server-side
   ```

3. **Get PostgreSQL Connection String** (Settings > Database > Connection string)
   - Select "Transaction" pooler mode for best performance
   - Copy the connection string:
   ```bash
   ARAGORA_POSTGRES_DSN=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
   ```

### Required Environment Variables

```bash
# Production mode
ARAGORA_ENVIRONMENT=production

# Authentication (generate a secure secret)
ARAGORA_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# CORS - your production domain
ARAGORA_ALLOWED_ORIGINS=https://aragora.ai,https://www.aragora.ai

# Supabase (real-time features)
SUPABASE_URL=https://[project-ref].supabase.co
SUPABASE_KEY=[service-role-key]

# Supabase PostgreSQL (database storage)
ARAGORA_POSTGRES_DSN=postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres

# At least one AI provider
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
```

### Optional but Recommended

```bash
# SSL/TLS
ARAGORA_SSL_ENABLED=true
ARAGORA_SSL_CERT=/path/to/cert.pem
ARAGORA_SSL_KEY=/path/to/key.pem

# Metrics (Prometheus)
METRICS_ENABLED=true
METRICS_PORT=9090

# Rate limiting with Redis
ARAGORA_REDIS_URL=redis://localhost:6379/0

# Error tracking
SENTRY_DSN=https://...@sentry.io/...
```

## Database Initialization

The initialization script creates all required tables in your Supabase PostgreSQL database:

```bash
# Initialize all tables
python scripts/init_postgres_db.py

# Verify tables exist (dry run)
python scripts/init_postgres_db.py --verify

# Verbose output
python scripts/init_postgres_db.py -v
```

### Tables Created

| Table | Purpose |
|-------|---------|
| `webhook_configs` | Webhook endpoint configurations |
| `integrations` | Third-party integration settings |
| `gmail_tokens` | Gmail OAuth tokens |
| `finding_workflows` | Research workflow state |
| `gauntlet_runs` | Challenge/benchmark runs |
| `job_queue` | Background job queue |
| `governance_artifacts` | Governance decisions |
| `marketplace_items` | Marketplace templates |
| `federation_nodes` | Federation registry |
| `approval_requests` | Approval workflow |
| `token_blacklist` | Revoked JWT tokens |
| `users` | User accounts |
| `webhooks` | Webhook delivery tracking |

## Deployment Options

### Option 1: Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[postgres]"

# Initialize database on startup
CMD python scripts/init_postgres_db.py && \
    python -m aragora.server.unified_server --port 8080
```

### Option 2: Systemd Service

```ini
# /etc/systemd/system/aragora.service
[Unit]
Description=Aragora Debate Server
After=network.target

[Service]
Type=simple
User=aragora
WorkingDirectory=/opt/aragora
EnvironmentFile=/opt/aragora/.env
ExecStart=/opt/aragora/venv/bin/python -m aragora.server.unified_server --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Option 3: Cloud Platforms

**Render.com:**
```yaml
# render.yaml
services:
  - type: web
    name: aragora
    env: python
    buildCommand: pip install -e ".[postgres]" && python scripts/init_postgres_db.py
    startCommand: python -m aragora.server.unified_server --port $PORT
    envVars:
      - key: ARAGORA_POSTGRES_DSN
        sync: false
      - key: SUPABASE_URL
        sync: false
```

**Railway:**
```bash
# Procfile
web: python scripts/init_postgres_db.py && python -m aragora.server.unified_server --port $PORT
```

## Health Checks

The server exposes health endpoints:

```bash
# Basic health check
curl https://aragora.ai/api/health

# Detailed health with database status
curl https://aragora.ai/api/health/detailed

# Readiness probe (for Kubernetes)
curl https://aragora.ai/api/health/ready
```

## Monitoring

### Prometheus Metrics

Enable metrics endpoint:
```bash
METRICS_ENABLED=true
METRICS_PORT=9090
```

Scrape config:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:9090']
```

### Key Metrics

- `aragora_debates_total` - Total debates completed
- `aragora_consensus_rate` - Consensus achievement rate
- `aragora_agent_latency_seconds` - Agent response times
- `aragora_db_pool_size` - Database connection pool usage

## Troubleshooting

### Connection Issues

```bash
# Test Supabase connection
python -c "
import asyncio
from aragora.storage.postgres_store import get_postgres_pool

async def test():
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchval('SELECT 1')
        print(f'Connection successful: {result}')

asyncio.run(test())
"
```

### Table Verification

```bash
# List all tables
python scripts/init_postgres_db.py --verify
```

### Logs

```bash
# Enable debug logging
ARAGORA_LOG_LEVEL=DEBUG python -m aragora.server.unified_server
```

## Security Checklist

- [ ] `ARAGORA_ENVIRONMENT=production` is set
- [ ] `ARAGORA_JWT_SECRET` is a strong, unique secret
- [ ] `ARAGORA_ALLOWED_ORIGINS` only includes your domains
- [ ] SSL/TLS is enabled with valid certificates
- [ ] `ARAGORA_CSP_REPORT_ONLY=false` (enforce CSP)
- [ ] Rate limiting is configured appropriately
- [ ] Supabase service role key is not exposed to clients
- [ ] Database connection uses SSL (`?sslmode=require`)

## Scaling

### Horizontal Scaling

For multiple server instances:

```bash
# Use Redis for shared state
ARAGORA_STATE_BACKEND=redis
ARAGORA_REDIS_URL=redis://your-redis:6379

# Unique instance ID (auto-generated if not set)
ARAGORA_INSTANCE_ID=instance-1
```

### Connection Pooling

Supabase connection limits by tier:
- Free: 20 direct connections
- Pro: 60 direct connections
- Team: 200 direct connections

Configure pool size accordingly:
```bash
ARAGORA_DB_POOL_SIZE=10  # Keep under your tier limit
```

## Backup & Recovery

Supabase provides automatic daily backups. For point-in-time recovery:

1. Go to Supabase Dashboard > Database > Backups
2. Enable Point-in-Time Recovery (Pro plan)
3. Restore from any point in the last 7 days

## Support

- Documentation: [docs/](../docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/aragora/issues)
- Community: [Discord](https://discord.gg/aragora)
