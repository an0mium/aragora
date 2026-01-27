# Aragora Self-Hosted Deployment

Deploy Aragora on your own infrastructure in under 15 minutes.

## Prerequisites

- Docker 20.10+ and Docker Compose v2
- 4GB RAM minimum (8GB recommended)
- At least one AI provider API key (Anthropic recommended)

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora/deploy/self-hosted

# Quick setup with auto-generated secrets
./init.sh

# Or with auto-verify after setup
./init.sh --verify

# Manual setup: Copy and edit configuration
cp .env.example .env
nano .env  # or your preferred editor
```

**Required settings:**
- `POSTGRES_PASSWORD` - Strong database password
- `ARAGORA_JWT_SECRET` - Generate with `openssl rand -base64 32`
- `REDIS_PASSWORD` - Redis authentication password
- At least one AI provider key (e.g., `ANTHROPIC_API_KEY`)

### 2. Start Services

```bash
# Core services only (API + Redis + Postgres)
docker compose up -d

# With monitoring (Prometheus + Grafana)
docker compose --profile monitoring up -d

# With queue workers (for high-volume)
docker compose --profile workers up -d

# Full stack
docker compose --profile monitoring --profile workers up -d
```

### 3. Verify Deployment

```bash
# Run the smoke test suite
./smoke_test.sh

# Or run quick tests only
./smoke_test.sh --quick

# With verbose output
./smoke_test.sh --verbose
```

The smoke test validates:
- Container health (Aragora, PostgreSQL, Redis)
- Liveness and readiness probes (`/healthz`, `/readyz`)
- Database connectivity
- Redis connectivity
- WebSocket endpoint
- API endpoints
- Prometheus metrics (if monitoring profile enabled)

**Alternative: Quick Verification**

```bash
# Check service health
docker compose ps

# View logs
docker compose logs -f aragora

# Test API
curl http://localhost:8080/healthz
```

### 4. Access

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Aragora API | http://localhost:8080 | N/A |
| Grafana | http://localhost:3001 | admin / (your GRAFANA_PASSWORD) |
| Prometheus | http://localhost:9091 | N/A |

## Configuration

### Environment Variables

See `.env.example` for all available options. Key settings:

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Yes (or another provider) |
| `POSTGRES_PASSWORD` | Database password | Yes |
| `ARAGORA_JWT_SECRET` | JWT signing secret | Yes |
| `REDIS_PASSWORD` | Redis password | Yes |

### Scaling

```bash
# Add more debate workers
docker compose --profile workers up -d --scale debate-worker=4

# Increase concurrent debates
# Edit .env: ARAGORA_MAX_CONCURRENT_DEBATES=20
docker compose up -d
```

## Backup & Restore

### Enable Automated Backups

```bash
docker compose --profile backup up -d
```

Backups are stored in `./backups/` with 7-day retention.

### Using the Backup CLI

For advanced backup operations with verification:

```bash
# Create a backup with integrity verification
python scripts/backup_cli.py backup /app/data/aragora.db

# List all verified backups
python scripts/backup_cli.py list --status verified

# Verify backup integrity (comprehensive)
python scripts/backup_cli.py verify abc123 --comprehensive

# Restore a backup (dry-run first)
python scripts/backup_cli.py restore abc123 ./restored.db --dry-run
python scripts/backup_cli.py restore abc123 ./restored.db

# Clean up old backups per retention policy
python scripts/backup_cli.py cleanup --dry-run
python scripts/backup_cli.py cleanup
```

### Manual PostgreSQL Backup

```bash
docker exec aragora-postgres pg_dump -U aragora aragora | gzip > backup.sql.gz
```

### Restore from PostgreSQL Backup

```bash
gunzip < backup.sql.gz | docker exec -i aragora-postgres psql -U aragora aragora
```

## Upgrading

```bash
# Pull latest images
docker compose pull

# Apply upgrade
docker compose up -d

# Check logs for migration status
docker compose logs aragora | grep -i migration
```

## Troubleshooting

### Service won't start

```bash
# Check logs
docker compose logs aragora

# Verify environment
docker compose config

# Check database connection
docker exec aragora-postgres pg_isready -U aragora
```

### API returns 500 errors

1. Check AI provider API key is valid
2. Verify Redis connection: `docker exec aragora-redis redis-cli -a $REDIS_PASSWORD ping`
3. Check database: `docker exec aragora-postgres psql -U aragora -c "SELECT 1"`

### Performance issues

1. Increase worker replicas: `--scale debate-worker=4`
2. Increase RAM allocation in Docker settings
3. Check Grafana dashboards for bottlenecks

## Architecture

```
                    ┌─────────────┐
                    │   Clients   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Aragora   │──────┐
                    │    API      │      │
                    └──────┬──────┘      │
              ┌────────────┼────────────┐│
              │            │            ││
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │ PostgreSQL│ │  Redis  │ │  Workers  │
        └───────────┘ └─────────┘ └───────────┘
```

## TLS/SSL Configuration

For production deployments, always use HTTPS. Two approaches:

### Option 1: Reverse Proxy (Recommended)

Use nginx, Traefik, or Caddy with Let's Encrypt:

```yaml
# Add to docker-compose.yml (Traefik example)
services:
  traefik:
    image: traefik:v3.0
    command:
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.le.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.le.acme.email=your@email.com"
      - "--certificatesresolvers.le.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./letsencrypt:/letsencrypt
      - /var/run/docker.sock:/var/run/docker.sock:ro

  aragora:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.aragora.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.aragora.entrypoints=websecure"
      - "traefik.http.routers.aragora.tls.certresolver=le"
```

### Option 2: Direct SSL

Add to `.env`:
```bash
ARAGORA_SSL_ENABLED=true
ARAGORA_SSL_CERT=/etc/ssl/certs/aragora.pem
ARAGORA_SSL_KEY=/etc/ssl/private/aragora-key.pem
```

Mount certificates in docker-compose:
```yaml
  aragora:
    volumes:
      - /path/to/cert.pem:/etc/ssl/certs/aragora.pem:ro
      - /path/to/key.pem:/etc/ssl/private/aragora-key.pem:ro
```

Generate self-signed cert for testing:
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

## Security Recommendations

1. **Change default passwords** - Never use example passwords in production
2. **Enable TLS** - Use a reverse proxy (nginx/Traefik) with Let's Encrypt
3. **Restrict network access** - Limit exposed ports to necessary services
4. **Regular backups** - Enable automated backups with off-site storage
5. **Monitor logs** - Check for unusual activity in Grafana/Prometheus

## Support

- Documentation: https://docs.aragora.ai
- Issues: https://github.com/an0mium/aragora/issues
- Enterprise Support: enterprise@aragora.ai
