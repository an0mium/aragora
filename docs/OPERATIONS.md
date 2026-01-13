# Aragora Operations Runbook

This document provides operational guidance for running, monitoring, and troubleshooting Aragora in production.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Server Management](#server-management)
3. [Monitoring & Observability](#monitoring--observability)
4. [Admin Console & Developer Portal](#admin-console--developer-portal)
5. [Common Issues & Debugging](#common-issues--debugging)
6. [Scaling Guide](#scaling-guide)
7. [Incident Response](#incident-response)
8. [Database Operations](#database-operations)
9. [Backup & Recovery](#backup--recovery)

---

## Quick Start

### Starting the Server

```bash
# Production mode
python -m aragora.server.unified_server --port 8080

# Development mode with hot reload
python -m aragora.server.unified_server --port 8080 --dev

# With custom database paths
python -m aragora.server.unified_server \
  --port 8080 \
  --db-path /data/aragora_memory.db \
  --elo-path /data/aragora_elo.db
```

### Verifying Server Health

```bash
# HTTP health check
curl http://localhost:8080/api/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", "uptime": 3600}

# WebSocket connectivity test
wscat -c ws://localhost:8765
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | - | Anthropic Claude API key |
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `OPENROUTER_API_KEY` | No | - | Fallback provider (auto-used on 429) |
| `ARAGORA_API_TOKEN` | No | - | API authentication token |
| `ARAGORA_ALLOWED_ORIGINS` | No | `*` | CORS allowed origins |
| `SUPABASE_URL` | No | - | Supabase project URL |
| `SUPABASE_KEY` | No | - | Supabase anon key |

*At least one AI provider key is required.

---

## Server Management

### Process Supervision

Use systemd for production deployments:

```ini
# /etc/systemd/system/aragora.service
[Unit]
Description=Aragora Multi-Agent Debate Server
After=network.target

[Service]
Type=simple
User=aragora
WorkingDirectory=/opt/aragora
Environment=PYTHONPATH=/opt/aragora
EnvironmentFile=/opt/aragora/.env
ExecStart=/opt/aragora/venv/bin/python -m aragora.server.unified_server --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable aragora
sudo systemctl start aragora

# Check status
sudo systemctl status aragora

# View logs
journalctl -u aragora -f
```

### Graceful Shutdown

The server handles SIGTERM gracefully:

```bash
# Graceful shutdown (waits for active debates to complete)
kill -TERM $(pgrep -f "aragora.server")

# Force shutdown (immediate)
kill -9 $(pgrep -f "aragora.server")
```

---

## Monitoring & Observability

### Prometheus Metrics

Metrics are exposed at `/api/metrics`:

```bash
curl http://localhost:8080/api/metrics
```

Key metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `aragora_debates_total` | Counter | Total debates started |
| `aragora_debates_completed` | Counter | Completed debates |
| `aragora_agent_response_seconds` | Histogram | Agent response latency |
| `aragora_tokens_used_total` | Counter | Total tokens consumed |
| `aragora_ws_connections_active` | Gauge | Active WebSocket connections |
| `aragora_circuit_breaker_state` | Gauge | Circuit breaker status per agent |

### Grafana Dashboard

Import the provided dashboard:

```bash
# Dashboard JSON at:
docs/grafana/aragora-dashboard.json
```

### Log Levels

Set log level via environment:

```bash
export ARAGORA_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

Log output includes:
- Request correlation IDs (`X-Request-ID`)
- Debate IDs in all related log lines
- Agent response times and token counts

### Alerting Rules

Recommended Prometheus alerting rules:

```yaml
groups:
  - name: aragora
    rules:
      - alert: HighAgentLatency
        expr: histogram_quantile(0.95, aragora_agent_response_seconds) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent response time > 30s at p95"

      - alert: CircuitBreakerOpen
        expr: aragora_circuit_breaker_state == 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker open for {{ $labels.agent }}"

      - alert: HighErrorRate
        expr: rate(aragora_debates_failed[5m]) / rate(aragora_debates_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Debate failure rate > 10%"
```

---

## Admin Console & Developer Portal

### Admin Console (`/admin`)

The admin console surfaces system health, circuit breaker state, recent errors, and rate-limit status.
It is intended for on-call and operations use.

Operational dependencies:
- Auth must be enabled (JWT) and the user role must be `admin`.
- The console reads from these endpoints:
  - `GET /api/health`
  - `GET /api/system/circuit-breakers`
  - `GET /api/system/errors?limit=20` (optional)
  - `GET /api/system/rate-limits` (optional)

Notes:
- If optional endpoints are not available, the UI degrades gracefully.
- For production, restrict `/admin` behind SSO, an allowlist, or an internal network boundary.

### Developer Portal (`/developer`)

The developer portal provides API key management and usage telemetry for authenticated users.

Operational dependencies:
- Auth must be enabled (JWT). Standard users can access their own portal.
- The portal reads from these endpoints:
  - `GET /api/auth/me`
  - `POST /api/auth/api-key`
  - `DELETE /api/auth/api-key`
  - `GET /api/billing/usage`

Notes:
- API keys are bearer credentials; display them once and store only client-side.
- Encourage users to rotate keys on compromise or after team changes.

---

## Common Issues & Debugging

### Agent Timeouts

**Symptoms:** Debates hang, agents stop responding

**Diagnosis:**
```bash
# Check circuit breaker state
curl http://localhost:8080/api/agents/health

# Check agent-specific logs
grep "agent=openai-api" /var/log/aragora/server.log | tail -100
```

**Resolution:**
1. Check API key validity
2. Verify rate limits not exceeded
3. Check network connectivity to provider
4. Circuit breaker will auto-recover after 60s

### WebSocket Disconnections

**Symptoms:** Real-time updates stop, clients disconnect

**Diagnosis:**
```bash
# Check active connections
curl http://localhost:8080/api/ws/stats

# Check for connection errors
grep "WebSocket" /var/log/aragora/server.log | grep -i error
```

**Resolution:**
1. Check nginx/proxy timeout settings (increase to 300s)
2. Verify client heartbeat interval matches server
3. Check for network issues (firewall, NAT timeout)

### Database Locks

**Symptoms:** Slow queries, write failures

**Diagnosis:**
```bash
# Check for WAL bloat
ls -la /data/*.db*

# Check active locks
sqlite3 /data/aragora_memory.db ".shell fuser /data/aragora_memory.db"
```

**Resolution:**
```bash
# Force WAL checkpoint
sqlite3 /data/aragora_memory.db "PRAGMA wal_checkpoint(TRUNCATE);"

# Vacuum database (offline)
sqlite3 /data/aragora_memory.db "VACUUM;"
```

### Memory Leaks

**Symptoms:** Increasing memory usage over time

**Diagnosis:**
```bash
# Check process memory
ps aux | grep aragora

# Profile memory (dev mode)
python -m aragora.server.unified_server --profile-memory
```

**Resolution:**
1. Check for unclosed database connections
2. Review event buffer sizes
3. Restart server (gracefully) during low-traffic periods

---

## Scaling Guide

### Horizontal Scaling

Aragora supports horizontal scaling with shared state:

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │ Node 1  │        │ Node 2  │        │ Node 3  │
    │ :8080   │        │ :8080   │        │ :8080   │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Redis/Postgres │
                    │  (shared state) │
                    └─────────────────┘
```

**Requirements for horizontal scaling:**
1. Shared database (PostgreSQL or Redis for session state)
2. Sticky sessions for WebSocket connections
3. Shared file storage for replays/checkpoints

### Vertical Scaling

Single-node optimization:

```bash
# Increase worker threads
export ARAGORA_WORKERS=4

# Increase connection pool
export ARAGORA_DB_POOL_SIZE=20

# Increase event buffer
export ARAGORA_EVENT_BUFFER_SIZE=10000
```

### Load Balancer Configuration

nginx example for WebSocket support:

```nginx
upstream aragora {
    ip_hash;  # Sticky sessions for WebSocket
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
}

server {
    listen 443 ssl http2;
    server_name api.aragora.ai;

    location / {
        proxy_pass http://aragora;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 | Service down | 15 min | Server crash, all agents failing |
| P2 | Major degradation | 1 hour | 50%+ debates failing, high latency |
| P3 | Minor issue | 4 hours | Single agent failing, UI bugs |
| P4 | Low priority | 1 day | Documentation issues, minor UX |

### Response Playbook

#### P1: Service Down

1. **Verify outage scope**
   ```bash
   curl -I http://localhost:8080/api/health
   ```

2. **Check server process**
   ```bash
   systemctl status aragora
   journalctl -u aragora --since "5 minutes ago"
   ```

3. **Attempt restart**
   ```bash
   sudo systemctl restart aragora
   ```

4. **Check database integrity**
   ```bash
   sqlite3 /data/aragora_memory.db "PRAGMA integrity_check;"
   ```

5. **Rollback if needed**
   ```bash
   cd /opt/aragora && git checkout v1.x.x
   sudo systemctl restart aragora
   ```

#### P2: High Error Rate

1. **Identify failing component**
   ```bash
   curl http://localhost:8080/api/agents/health
   ```

2. **Check rate limits**
   ```bash
   grep "rate_limit\|429" /var/log/aragora/server.log
   ```

3. **Check API provider status**
   - Anthropic: https://status.anthropic.com
   - OpenAI: https://status.openai.com

4. **Enable fallback providers**
   ```bash
   export OPENROUTER_API_KEY="..."
   sudo systemctl restart aragora
   ```

### Post-Incident Review

Document in `.nomic/incidents/`:

```markdown
# Incident: YYYY-MM-DD-title

## Summary
Brief description of what happened

## Timeline
- HH:MM - First alert
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Service restored

## Root Cause
Technical description

## Action Items
- [ ] Preventive measure 1
- [ ] Preventive measure 2
```

---

## Database Operations

### Routine Maintenance

```bash
# Daily: WAL checkpoint
sqlite3 /data/aragora_memory.db "PRAGMA wal_checkpoint(PASSIVE);"

# Weekly: Analyze for query optimization
sqlite3 /data/aragora_memory.db "ANALYZE;"

# Monthly: Vacuum (during maintenance window)
sqlite3 /data/aragora_memory.db "VACUUM;"
```

### Schema Migrations

Migrations are in `aragora/migrations/`:

```bash
# Apply pending migrations
python -m aragora.migrations.apply

# Check migration status
python -m aragora.migrations.status

# Rollback last migration
python -m aragora.migrations.rollback
```

### Backup Procedures

```bash
# Hot backup (while server running)
sqlite3 /data/aragora_memory.db ".backup /backups/aragora_memory_$(date +%Y%m%d).db"

# Full backup script
#!/bin/bash
BACKUP_DIR=/backups/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

for db in /data/*.db; do
    sqlite3 "$db" ".backup $BACKUP_DIR/$(basename $db)"
done

# Compress and upload
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
aws s3 cp $BACKUP_DIR.tar.gz s3://aragora-backups/
```

---

## Backup & Recovery

### Backup Schedule

| Type | Frequency | Retention | Storage |
|------|-----------|-----------|---------|
| WAL checkpoint | Hourly | 24 hours | Local |
| Hot backup | Daily | 7 days | S3 |
| Full backup | Weekly | 30 days | S3 + Glacier |
| Debate traces | On completion | 90 days | S3 |

### Recovery Procedures

#### Point-in-Time Recovery

```bash
# Stop server
sudo systemctl stop aragora

# Restore from backup
cp /backups/20240115/aragora_memory.db /data/aragora_memory.db

# Replay WAL if available
sqlite3 /data/aragora_memory.db "PRAGMA wal_checkpoint(RESTART);"

# Verify integrity
sqlite3 /data/aragora_memory.db "PRAGMA integrity_check;"

# Start server
sudo systemctl start aragora
```

#### Disaster Recovery

1. Provision new server
2. Install Aragora from git
3. Restore latest backup from S3
4. Update DNS to point to new server
5. Verify functionality

---

## Appendix

### Useful Commands

```bash
# Watch server logs
tail -f /var/log/aragora/server.log | jq .

# Count active debates
curl -s http://localhost:8080/api/debates | jq '.debates | length'

# List agents and their ELO
curl -s http://localhost:8080/api/leaderboard/rankings | jq '.agents[] | {name, elo}'

# View circuit breaker status
curl -s http://localhost:8080/api/circuit-breakers | jq .

# Export debate history
curl -s http://localhost:8080/api/debates/export?format=json > debates.json
```

### Configuration Reference

Full configuration options in `aragora/config.py`:

```python
# Server settings
HTTP_PORT = 8080
WS_PORT = 8765
WORKERS = 4

# Database settings
DB_POOL_SIZE = 10
DB_TIMEOUT = 30

# Agent settings
AGENT_TIMEOUT = 60
MAX_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60
```

### Support Contacts

- GitHub Issues: https://github.com/aragora/aragora/issues
- Documentation: https://docs.aragora.ai
- Status Page: https://status.aragora.ai
