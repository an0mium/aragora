# Aragora Deployment Checklist

Use this checklist before deploying Aragora to production environments.

## Pre-Deployment Checklist

### 1. Infrastructure Requirements

- [ ] **Compute Resources**
  - [ ] Minimum 4GB RAM (8GB recommended)
  - [ ] Minimum 2 CPU cores (4 recommended)
  - [ ] 50GB disk space (SSD recommended)

- [ ] **Software Prerequisites**
  - [ ] Docker 20.10+ installed
  - [ ] Docker Compose v2 installed
  - [ ] Git installed
  - [ ] curl/wget available for testing

- [ ] **Network Requirements**
  - [ ] Outbound HTTPS (443) to AI providers
  - [ ] Inbound ports configured (8080, 8765 for WebSocket)
  - [ ] DNS configured (if using domain)
  - [ ] SSL certificates ready (production)

### 2. Configuration Validation

- [ ] **Environment Variables**
  - [ ] `.env` file created from `.env.example`
  - [ ] `POSTGRES_PASSWORD` is strong (16+ chars, mixed)
  - [ ] `ARAGORA_JWT_SECRET` generated (`openssl rand -base64 32`)
  - [ ] `REDIS_PASSWORD` is strong
  - [ ] At least one AI provider API key configured:
    - [ ] `ANTHROPIC_API_KEY` (recommended)
    - [ ] `OPENAI_API_KEY` (optional)
    - [ ] `OPENROUTER_API_KEY` (fallback)

- [ ] **Security Settings**
  - [ ] Default passwords changed
  - [ ] Debug mode disabled (`ARAGORA_DEBUG=false`)
  - [ ] Production environment set (`ARAGORA_ENV=production`)
  - [ ] CORS origins restricted (`ARAGORA_ALLOWED_ORIGINS`)
  - [ ] Rate limiting configured

- [ ] **Database Configuration**
  - [ ] PostgreSQL connection string valid
  - [ ] Database user has appropriate permissions
  - [ ] Connection pool size configured
  - [ ] Backup schedule configured

### 3. Security Audit

- [ ] **Secrets Management**
  - [ ] No secrets in code/commits
  - [ ] `.env` file not committed
  - [ ] Secrets rotated from any defaults
  - [ ] API keys valid and not expired

- [ ] **Network Security**
  - [ ] Unnecessary ports closed
  - [ ] Database not exposed publicly
  - [ ] Redis not exposed publicly
  - [ ] Firewall rules configured

- [ ] **TLS/SSL**
  - [ ] SSL certificate valid
  - [ ] Certificate not expiring within 30 days
  - [ ] HTTPS enforced (redirect HTTP)
  - [ ] HSTS configured (production)

### 4. Monitoring Setup

- [ ] **Health Endpoints**
  - [ ] `/healthz` returns 200
  - [ ] `/readyz` returns 200
  - [ ] Database health check passing
  - [ ] Redis health check passing

- [ ] **Logging**
  - [ ] Log level appropriate (INFO for production)
  - [ ] Log rotation configured
  - [ ] Log aggregation configured (optional)

- [ ] **Metrics (if enabled)**
  - [ ] Prometheus scraping configured
  - [ ] Grafana dashboards imported
  - [ ] Alerting rules configured

### 5. Backup Verification

- [ ] **Backup Configuration**
  - [ ] Automated backups enabled
  - [ ] Backup location configured
  - [ ] Retention policy set
  - [ ] Off-site backup configured (production)

- [ ] **Backup Testing**
  - [ ] Manual backup successful
  - [ ] Backup restore tested
  - [ ] Backup integrity verified

---

## Deployment Steps

### Step 1: Pull Latest Changes

```bash
cd /path/to/aragora
git fetch origin
git checkout main
git pull origin main
```

- [ ] Latest code pulled
- [ ] No uncommitted changes
- [ ] Version tag noted: `v________`

### Step 2: Build Images

```bash
docker compose build --no-cache
```

- [ ] Build completed successfully
- [ ] No build errors
- [ ] Image size reasonable (<2GB)

### Step 3: Database Migration (if needed)

```bash
# Check if migrations needed
docker compose run --rm aragora python -m alembic current
docker compose run --rm aragora python -m alembic heads

# Run migrations
docker compose run --rm aragora python -m alembic upgrade head
```

- [ ] Current migration version noted
- [ ] Migrations applied successfully
- [ ] No migration errors

### Step 4: Deploy Services

```bash
# Stop old services
docker compose down

# Start new services
docker compose up -d

# Watch logs for startup issues
docker compose logs -f aragora
```

- [ ] Old services stopped cleanly
- [ ] New services started
- [ ] No startup errors in logs

### Step 5: Verification

```bash
# Run smoke tests
./smoke_test.sh --verbose
```

- [ ] All health checks passing
- [ ] API responding correctly
- [ ] WebSocket endpoint accessible
- [ ] Database connectivity verified
- [ ] Redis connectivity verified

---

## Post-Deployment Checklist

### Immediate (0-15 minutes)

- [ ] Health endpoints returning 200
- [ ] No error spikes in logs
- [ ] API response times normal
- [ ] No 5xx errors

### Short-term (15-60 minutes)

- [ ] Monitor error rates
- [ ] Check memory usage
- [ ] Verify debate execution works
- [ ] Test user authentication (if applicable)

### Extended (1-24 hours)

- [ ] No memory leaks
- [ ] Database connections stable
- [ ] No unexpected restarts
- [ ] Performance metrics normal

---

## Rollback Procedure

If deployment fails, rollback immediately:

```bash
# Stop failed deployment
docker compose down

# Restore previous version
git checkout <previous-tag>

# Rebuild and redeploy
docker compose build
docker compose up -d

# Restore database if needed
gunzip -c ./backups/postgres/pre-deploy.sql.gz | \
  docker exec -i aragora-postgres psql -U aragora aragora

# Verify rollback
./smoke_test.sh
```

---

## Deployment Sign-off

| Check | Verified By | Date | Notes |
|-------|-------------|------|-------|
| Pre-deployment checklist complete | | | |
| Deployment successful | | | |
| Post-deployment verification | | | |
| Monitoring confirmed | | | |
| Rollback plan tested | | | |

**Deployment Approved:** [ ] Yes / [ ] No

**Deployer:** _________________ **Date:** _________________

---

## Quick Reference

### Essential Commands

```bash
# Status
docker compose ps
docker compose logs -f aragora

# Health checks
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz

# Restart
docker compose restart aragora

# Full redeploy
docker compose down && docker compose up -d

# View metrics
curl http://localhost:8080/metrics
```

### Emergency Contacts

| Role | Contact |
|------|---------|
| On-Call | [PagerDuty/Opsgenie] |
| Database | database-team@company.com |
| Platform | platform-team@company.com |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01-27 | Initial checklist |
