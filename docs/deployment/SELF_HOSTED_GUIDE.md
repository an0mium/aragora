# Aragora Self-Hosted Deployment Guide

Deploy Aragora on your own infrastructure in under 15 minutes.

## Prerequisites

- Docker and Docker Compose installed
- At least one AI provider API key (Anthropic, OpenAI, or OpenRouter)
- 2GB+ RAM, 2 CPU cores recommended
- 10GB+ disk space

## Quick Start (5 Minutes)

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/an0mium/aragora.git
cd aragora

# Copy the environment template
cp .env.example .env
```

### 2. Add Your API Key

Edit `.env` and add at least one AI provider key:

```bash
# Required: At least one of these
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
# or
OPENROUTER_API_KEY=sk-or-...
```

### 3. Start Aragora

```bash
# Simple deployment (SQLite, single container)
docker compose -f docker-compose.simple.yml up -d

# Check health
curl http://localhost:8080/api/health
```

### 4. Verify Installation

```bash
# Run a test debate
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "What is the best programming language for beginners?",
    "rounds": 2
  }'
```

## Deployment Options

### Option 1: Simple (Development/Testing)

Best for: Personal use, testing, small teams

```bash
docker compose -f docker-compose.simple.yml up -d
```

Features:
- SQLite database (no external dependencies)
- Single container
- Data persisted in Docker volume

### Option 2: SME (Small Business)

Best for: Small businesses, startups, teams up to 50 users

```bash
docker compose -f docker-compose.sme.yml up -d
```

Features:
- PostgreSQL database
- Redis for caching
- Separate web and worker containers
- Better performance for concurrent users

### Option 3: Production (Enterprise)

Best for: Large organizations, high availability requirements

```bash
docker compose -f docker-compose.production.yml up -d
```

Features:
- PostgreSQL with replication
- Redis cluster
- Prometheus/Grafana monitoring
- TLS termination
- Horizontal scaling

## Configuration Reference

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | `sk-ant-...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENROUTER_API_KEY` | OpenRouter fallback key | `sk-or-...` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_ENVIRONMENT` | Runtime environment | `development` |
| `ARAGORA_DB_BACKEND` | Database backend | `sqlite` |
| `ARAGORA_API_TOKEN` | API authentication token | None |
| `ARAGORA_ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000` |
| `ARAGORA_DEFAULT_ROUNDS` | Default debate rounds | `3` |
| `ARAGORA_DEBATE_TIMEOUT` | Debate timeout (seconds) | `900` |

### Database Configuration

For PostgreSQL:

```bash
ARAGORA_DB_BACKEND=postgres
DATABASE_URL=postgresql://user:password@localhost:5432/aragora
```

For Redis caching:

```bash
REDIS_URL=redis://localhost:6379/0
```

## Data Management

### Backup

```bash
# Create backup
docker exec aragora aragora backup create

# Download backup
docker cp aragora:/app/data/backups/latest.tar.gz ./backup.tar.gz
```

### Restore

```bash
# Upload backup
docker cp ./backup.tar.gz aragora:/app/data/backups/

# Restore
docker exec aragora aragora backup restore latest.tar.gz
```

### Data Location

Data is stored in the `aragora-data` Docker volume:

- `/app/data/aragora.db` - SQLite database (if using SQLite)
- `/app/data/elo_ratings.json` - Agent ELO ratings
- `/app/data/memory/` - Continuum memory tiers
- `/app/data/backups/` - Automated backups

## Monitoring

### Health Check

```bash
curl http://localhost:8080/api/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.1.14",
  "database": "connected",
  "agents_available": 15
}
```

### Logs

```bash
# View logs
docker logs aragora

# Follow logs
docker logs -f aragora

# Last 100 lines
docker logs --tail 100 aragora
```

### Metrics (Production)

Prometheus metrics available at:
```
http://localhost:8080/metrics
```

## Upgrading

### Minor Updates

```bash
# Pull latest
git pull

# Rebuild and restart
docker compose -f docker-compose.simple.yml up -d --build
```

### Major Updates

1. Create a backup first
2. Check the [CHANGELOG](../../CHANGELOG.md) for breaking changes
3. Update environment variables if needed
4. Rebuild and restart

```bash
# Backup
docker exec aragora aragora backup create

# Update
git pull
docker compose -f docker-compose.simple.yml up -d --build

# Verify
curl http://localhost:8080/api/health
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs for errors
docker logs aragora

# Verify API keys are set
docker exec aragora env | grep API_KEY
```

### Database Connection Issues

```bash
# SQLite: Check file permissions
docker exec aragora ls -la /app/data/

# PostgreSQL: Test connection
docker exec aragora python -c "from aragora.storage import check_db; check_db()"
```

### AI Provider Errors

1. Verify API key is valid
2. Check rate limits
3. Try with OpenRouter fallback

```bash
# Test API key
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/models
```

### Out of Memory

Increase container memory limits:

```yaml
services:
  aragora:
    deploy:
      resources:
        limits:
          memory: 4G
```

## Security Recommendations

### For Production

1. **Enable HTTPS**
   - Use a reverse proxy (nginx, Traefik)
   - Configure TLS certificates

2. **Set API Token**
   ```bash
   ARAGORA_API_TOKEN=your-secure-token-here
   ```

3. **Restrict CORS**
   ```bash
   ARAGORA_ALLOWED_ORIGINS=https://your-domain.com
   ```

4. **Use External Database**
   - PostgreSQL with encryption at rest
   - Regular automated backups

5. **Network Isolation**
   - Run in private subnet
   - Use firewall rules

## Support

- Documentation: [docs/](../../docs/)
- Issues: [GitHub Issues](https://github.com/an0mium/aragora/issues)
- Security: security@aragora.ai
