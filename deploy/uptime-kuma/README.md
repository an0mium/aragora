# Aragora Status Page (Uptime Kuma)

Public status page for Aragora services at `status.aragora.ai`.

**SOC 2 Control:** A1-01 - Public availability information

## Quick Start

```bash
# Start the status page
docker-compose up -d

# Access at http://localhost:3001
# Create admin account on first login
```

## Initial Configuration

After first login, configure these monitors:

### API Health Monitor

| Setting | Value |
|---------|-------|
| Name | API - Health |
| Type | HTTP(s) |
| URL | https://api.aragora.ai/api/health |
| Method | GET |
| Expected Status | 200 |
| Interval | 60 seconds |
| Timeout | 10 seconds |
| Retries | 3 |

### API Detailed Health

| Setting | Value |
|---------|-------|
| Name | API - Services |
| Type | HTTP(s) - Keyword |
| URL | https://api.aragora.ai/api/health/detailed |
| Keyword | "healthy" |
| Interval | 120 seconds |

### WebSocket Server

| Setting | Value |
|---------|-------|
| Name | WebSocket |
| Type | HTTP(s) |
| URL | https://api.aragora.ai/api/ws/stats |
| Expected Status | 200 |
| Interval | 60 seconds |

### Database (Internal)

| Setting | Value |
|---------|-------|
| Name | Database |
| Type | HTTP(s) - Keyword |
| URL | https://api.aragora.ai/api/health/detailed |
| Keyword | "database.*healthy" (regex) |
| Interval | 120 seconds |
| Group | Internal (hidden from public) |

## Status Page Configuration

1. **Create Status Page:**
   - Go to Status Pages > Add New
   - Name: Aragora Status
   - Slug: aragora
   - Theme: Dark

2. **Add Groups:**
   - Core Services (API, WebSocket)
   - Infrastructure (Database, internal monitors)

3. **Configure Incident Templates:**
   - Investigating
   - Identified
   - Monitoring
   - Resolved

## Notification Channels

Configure alerts for:
- **Email**: ops@aragora.ai
- **Slack**: #ops-alerts channel
- **PagerDuty**: For P1 incidents

## Production Deployment

### With Traefik (recommended)

Ensure Traefik is configured with labels in docker-compose.yml:

```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.status.rule=Host(`status.aragora.ai`)"
  - "traefik.http.routers.status.entrypoints=websecure"
  - "traefik.http.routers.status.tls.certresolver=letsencrypt"
```

### With nginx

```nginx
server {
    listen 443 ssl http2;
    server_name status.aragora.ai;

    ssl_certificate /etc/letsencrypt/live/status.aragora.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/status.aragora.ai/privkey.pem;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Backup and Restore

### Backup

```bash
# Stop container
docker-compose stop

# Backup data volume
docker run --rm \
  -v aragora-status_uptime-kuma-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/uptime-kuma-$(date +%Y%m%d).tar.gz /data

# Start container
docker-compose start
```

### Restore

```bash
docker-compose stop

docker run --rm \
  -v aragora-status_uptime-kuma-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/uptime-kuma-YYYYMMDD.tar.gz -C /

docker-compose start
```

## Maintenance

### Update Uptime Kuma

```bash
docker-compose pull
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f
```

### Reset Admin Password

If you forget the admin password:

```bash
docker-compose exec uptime-kuma sqlite3 /app/data/kuma.db \
  "UPDATE user SET password='<bcrypt_hash>' WHERE id=1;"
```

## API Access

Uptime Kuma provides API endpoints for integration:

```bash
# Get status page data
curl https://status.aragora.ai/api/status-page/aragora

# Get monitor list (requires auth)
curl -H "Authorization: Bearer <api-key>" \
  https://status.aragora.ai/api/getMonitorList
```

## Metrics Export

For Prometheus integration:

```bash
# Enable in Settings > Monitoring > Prometheus
# Scrape endpoint: http://localhost:3001/metrics
```

## Related Documentation

- [SLA.md](../../docs/SLA.md) - Service level agreements
- [RUNBOOK.md](../../docs/RUNBOOK.md) - Operational procedures
- [DISASTER_RECOVERY.md](../../docs/DISASTER_RECOVERY.md) - Recovery procedures
