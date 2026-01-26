# Production Deployment Checklist

> Version: 1.3.0 | Last Updated: January 2026

This checklist ensures Aragora is properly configured for production deployment.

---

## Pre-Deployment

### Environment Configuration

- [ ] **Required API Keys**
  - [ ] `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (at least one)
  - [ ] `OPENROUTER_API_KEY` (recommended for fallback)
  - [ ] `JWT_SECRET` (32+ characters, cryptographically random)

- [ ] **Database Configuration**
  - [ ] `ARAGORA_POSTGRES_DSN` or `DATABASE_URL` set
  - [ ] Database migrations applied (`python -m aragora.db.migrate`)
  - [ ] Connection pooling configured for production load

- [ ] **Distributed State (Multi-Instance Deployments)**
  - [ ] `REDIS_URL` set for session store, control plane, and caching
  - [ ] `ARAGORA_MULTI_INSTANCE=true` or `ARAGORA_ENV=production` set
  - [ ] Redis connectivity verified (`python -m aragora.cli validate-env`)

- [ ] **Encryption Requirements**
  - [ ] `ARAGORA_ENCRYPTION_KEY` set (32-byte hex string for AES-256)
  - [ ] Key stored in secrets manager (AWS KMS, HashiCorp Vault, etc.)
  - [ ] Key rotation schedule documented

- [ ] **Security Settings**
  - [ ] `ARAGORA_ALLOWED_ORIGINS` set (no wildcards in production)
  - [ ] HTTPS enforced (SSL/TLS certificates installed)
  - [ ] Rate limiting enabled (default: 120 req/min per IP)

### Environment Validation

```bash
# Validate configuration and backend connectivity
python -m aragora.cli validate-env

# Example output:
# ✓ Environment: production
# ✓ Encryption key: configured (32 bytes)
# ✓ Redis: connected (version 7.2.3)
# ✓ PostgreSQL: connected (PostgreSQL 15.4)
# ✓ AI provider: anthropic configured
# ✓ All production requirements met
```

**Startup Validation**: The server automatically validates backend connectivity at startup:
- **Redis connectivity** (when `ARAGORA_MULTI_INSTANCE=true` or `ARAGORA_ENV=production`)
- **PostgreSQL connectivity** (when `ARAGORA_REQUIRE_DATABASE=true`)
- **Configuration integrity** (encryption keys, JWT secrets)

If validation fails, the server refuses to start with a clear error message.

### Code Quality Gates

```bash
# All must pass before deployment
pytest tests/ --tb=short -q           # 22,908 tests
python -m mypy aragora/ --config-file pyproject.toml  # 0 errors (core)
python -m bandit -r aragora/ -ll      # 0 HIGH severity
cd aragora/live && npm run build      # Build succeeds
cd aragora/live && npm run lint       # 0 warnings/errors
```

### Security Audit

- [ ] **Bandit Scan**: 0 HIGH severity issues
- [ ] **Dependency Audit**: `safety check` passes (or known CVEs documented)
- [ ] **SQL Injection**: All `# nosec B608` comments reviewed and justified
- [ ] **Secrets**: No hardcoded credentials in codebase

---

## Infrastructure

### Minimum Requirements

| Resource | Development | Production |
|----------|-------------|------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 10 GB | 50+ GB (for debate history) |
| Database | SQLite | PostgreSQL 14+ |

### Recommended Architecture

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │    (HTTPS)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────┴──────┐ ┌─────┴─────┐ ┌─────┴─────┐
       │  Aragora    │ │  Aragora  │ │  Aragora  │
       │  Server 1   │ │  Server 2 │ │  Server 3 │
       └──────┬──────┘ └─────┬─────┘ └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────┴────────┐
                    │   PostgreSQL    │
                    │   (Primary)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │     Redis       │
                    │  (Optional)     │
                    └─────────────────┘
```

### Scaling Considerations

| Metric | Single Instance | Clustered |
|--------|-----------------|-----------|
| Concurrent Debates | 10-20 | 100+ |
| Active WebSocket Connections | 100 | 1000+ |
| API Requests/sec | 50 | 500+ |
| Data Retention | Local SQLite | PostgreSQL + Archival |

---

## Performance Targets

### API Response Times (p95)

| Endpoint | Target | Alert Threshold |
|----------|--------|-----------------|
| `GET /api/health` | < 50ms | > 200ms |
| `GET /api/debates` | < 200ms | > 500ms |
| `POST /api/debates` | < 500ms | > 2s |
| `GET /api/leaderboard` | < 100ms | > 300ms |

### WebSocket Latency

| Event | Target | Alert Threshold |
|-------|--------|-----------------|
| Message Delivery | < 100ms | > 500ms |
| Connection Handshake | < 1s | > 3s |

### Agent Response Times

| Provider | Target | Circuit Breaker |
|----------|--------|-----------------|
| Anthropic | < 30s | 3 failures in 60s |
| OpenAI | < 30s | 3 failures in 60s |
| OpenRouter (fallback) | < 45s | 5 failures in 120s |

---

## Monitoring & Alerting

### Required Metrics

- [ ] **Application**
  - [ ] Request rate (RPM)
  - [ ] Error rate (5xx responses)
  - [ ] Latency (p50, p95, p99)
  - [ ] WebSocket connection count

- [ ] **Infrastructure**
  - [ ] CPU utilization
  - [ ] Memory usage
  - [ ] Disk I/O
  - [ ] Network throughput

- [ ] **Business**
  - [ ] Active debates
  - [ ] Debates completed/hour
  - [ ] Consensus rate
  - [ ] Agent error rate

### Health Endpoints

```bash
# Basic health check
curl https://your-domain.com/api/health
# Expected: {"status": "healthy", "version": "1.3.0"}

# Readiness check (includes dependencies)
curl https://your-domain.com/api/ready
# Expected: {"status": "ready", "database": "connected", "agents": "available"}
```

### Alerting Rules

| Condition | Severity | Action |
|-----------|----------|--------|
| Error rate > 5% | WARNING | Investigate |
| Error rate > 10% | CRITICAL | Page on-call |
| p95 latency > 2s | WARNING | Scale up |
| Database connection failures | CRITICAL | Failover |
| Agent circuit breaker open | WARNING | Check API quotas |

---

## Backup & Recovery

### Backup Schedule

| Data | Frequency | Retention |
|------|-----------|-----------|
| Database (full) | Daily | 30 days |
| Database (WAL) | Continuous | 7 days |
| Configuration | On change | Version controlled |
| Debate artifacts | Weekly | 90 days |

### Recovery Procedures

**Database Recovery**
```bash
# Point-in-time recovery
pg_restore -d aragora backup_20260113.dump

# Verify data integrity
python -m aragora.db.verify
```

**Rollback Procedure**
```bash
# Revert to previous version
git checkout v1.2.0
pip install -e .
systemctl restart aragora

# Verify rollback
curl https://your-domain.com/api/health
```

### RTO/RPO Targets

| Metric | Target | Notes |
|--------|--------|-------|
| RTO (Recovery Time Objective) | < 1 hour | Time to restore service |
| RPO (Recovery Point Objective) | < 5 minutes | Maximum data loss |

---

## Security Hardening

> **Related documentation**:
> - [COMPLIANCE.md](COMPLIANCE.md) - SOC 2 controls and evidence
> - [DATA_CLASSIFICATION.md](DATA_CLASSIFICATION.md) - Data handling policies
> - [PRIVACY_POLICY.md](PRIVACY_POLICY.md) - Privacy requirements

### Network Security

- [ ] Firewall configured (only 80/443 exposed)
- [ ] DDoS protection enabled
- [ ] WAF rules configured
- [ ] Internal services on private network

### Application Security

- [ ] CORS origins explicitly listed
- [ ] CSP headers configured (no unsafe-eval)
- [ ] Rate limiting per IP enabled
- [ ] JWT tokens expire appropriately (access: 24h, refresh: 30d)
- [ ] Token revocation mechanism active

### Access Control

- [ ] Admin endpoints require authentication
- [ ] API keys hashed in database
- [ ] MFA available for admin accounts
- [ ] Audit logging enabled

---

## Deployment Checklist

### Pre-Deployment

- [ ] All tests pass (`pytest tests/`)
- [ ] Security scan clean (`bandit -r aragora/`)
- [ ] Frontend builds (`npm run build`)
- [ ] Environment variables set
- [ ] Database migrations applied
- [ ] SSL certificates valid

### Deployment

- [ ] Deploy to staging first
- [ ] Run smoke tests on staging
- [ ] Deploy to production (rolling update)
- [ ] Verify health endpoints
- [ ] Monitor error rates for 30 minutes

### Post-Deployment

- [ ] Verify all services healthy
- [ ] Check WebSocket connections working
- [ ] Run one test debate
- [ ] Notify stakeholders of successful deployment
- [ ] Update status page if applicable

---

## Incident Response

> **Detailed documentation**: See [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) for complete playbooks.

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| P1 | Service down | < 15 minutes |
| P2 | Major feature broken | < 1 hour |
| P3 | Minor issue | < 4 hours |
| P4 | Cosmetic/low impact | Next business day |

### Escalation Path

1. On-call engineer
2. Team lead
3. Engineering manager
4. CTO (P1 only)

### Incident Template

```markdown
## Incident Report

**Title:** [Brief description]
**Severity:** P1/P2/P3/P4
**Duration:** [Start] - [End]
**Impact:** [Users affected, functionality impaired]

### Timeline
- HH:MM - Incident detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Incident resolved

### Root Cause
[Description]

### Resolution
[What was done to fix it]

### Action Items
- [ ] [Preventive measure 1]
- [ ] [Preventive measure 2]
```

---

## Contacts

| Role | Contact |
|------|---------|
| On-call | [Your PagerDuty/OpsGenie] |
| Security | security@aragora.ai |
| Support | support@aragora.ai |

---

## Self-Hosted Deployment

### Docker Deployment

**Quick Start:**
```bash
# Pull the official image
docker pull aragora/aragora:latest

# Run with required environment variables
docker run -d \
  -p 8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-xxx \
  -e JWT_SECRET=$(openssl rand -hex 32) \
  -e ARAGORA_ENCRYPTION_KEY=$(openssl rand -hex 32) \
  -v aragora-data:/app/data \
  --name aragora \
  aragora/aragora:latest
```

**Docker Compose (Recommended):**
```yaml
# docker-compose.yml
version: '3.8'

services:
  aragora:
    image: aragora/aragora:latest
    ports:
      - "8080:8080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - ARAGORA_ENCRYPTION_KEY=${ARAGORA_ENCRYPTION_KEY}
      - DATABASE_URL=postgresql://aragora:password@postgres:5432/aragora
      - REDIS_URL=redis://redis:6379
      - ARAGORA_ENV=production
    depends_on:
      - postgres
      - redis
    volumes:
      - aragora-data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=aragora
      - POSTGRES_USER=aragora
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  aragora-data:
  postgres-data:
  redis-data:
```

### Kubernetes Deployment

**Deployment Manifest:**
```yaml
# aragora-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora
  labels:
    app: aragora
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aragora
  template:
    metadata:
      labels:
        app: aragora
    spec:
      containers:
      - name: aragora
        image: aragora/aragora:latest
        ports:
        - containerPort: 8080
        envFrom:
        - secretRef:
            name: aragora-secrets
        - configMapRef:
            name: aragora-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: aragora
spec:
  selector:
    app: aragora
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aragora
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - aragora.yourdomain.com
    secretName: aragora-tls
  rules:
  - host: aragora.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aragora
            port:
              number: 80
```

**Secrets and ConfigMap:**
```yaml
# aragora-secrets.yaml (apply with kubectl create secret)
apiVersion: v1
kind: Secret
metadata:
  name: aragora-secrets
type: Opaque
stringData:
  ANTHROPIC_API_KEY: "sk-ant-xxx"
  JWT_SECRET: "your-32-char-secret"
  ARAGORA_ENCRYPTION_KEY: "your-32-byte-hex-key"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: aragora-config
data:
  ARAGORA_ENV: "production"
  ARAGORA_MULTI_INSTANCE: "true"
  DATABASE_URL: "postgresql://aragora:password@postgres:5432/aragora"
  REDIS_URL: "redis://redis:6379"
```

### SSL/TLS Certificates

**Let's Encrypt with Certbot:**
```bash
# Install certbot
apt install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d aragora.yourdomain.com

# Auto-renewal (added automatically)
# Verify: certbot renew --dry-run
```

**Self-Signed (Development Only):**
```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/aragora.key \
  -out /etc/ssl/certs/aragora.crt \
  -subj "/CN=aragora.local"
```

### Load Balancer Configuration

**Nginx Reverse Proxy:**
```nginx
# /etc/nginx/sites-available/aragora
upstream aragora {
    least_conn;
    server 127.0.0.1:8080;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    listen 443 ssl http2;
    server_name aragora.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/aragora.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aragora.yourdomain.com/privkey.pem;

    # WebSocket support
    location /ws {
        proxy_pass http://aragora;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    location / {
        proxy_pass http://aragora;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Observability Stack Setup

**Prometheus Scrape Config:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Grafana Dashboard:**
Import dashboard ID `aragora-debates` or configure:
- Debate throughput (debates/min)
- Agent response latency (p50, p95, p99)
- Consensus rate
- Active WebSocket connections
- Budget utilization

### Maintenance & Upgrades

**Zero-Downtime Upgrade (Kubernetes):**
```bash
# Rolling update
kubectl set image deployment/aragora aragora=aragora/aragora:v1.4.0

# Verify rollout
kubectl rollout status deployment/aragora

# Rollback if needed
kubectl rollout undo deployment/aragora
```

**Database Migrations:**
```bash
# Before upgrade, backup database
pg_dump -Fc aragora > backup_$(date +%Y%m%d).dump

# Apply migrations
python -m aragora.db.migrate

# Verify
python -m aragora.db.verify
```

### Disaster Recovery Drills

**Quarterly DR Test Checklist:**
- [ ] Restore database from backup to test environment
- [ ] Verify debate history intact
- [ ] Verify user sessions and preferences
- [ ] Run smoke tests on restored instance
- [ ] Document restoration time (target: <1 hour)

**Backup Scripts:**
```bash
# Use provided scripts
python scripts/backup_databases.py --output /backups
python scripts/restore_databases.py --input /backups/latest --verify
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.4.0 | Jan 2026 | Added self-hosted deployment section |
| 1.3.0 | Jan 2026 | Initial GA checklist |
