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

- [ ] **Security Settings**
  - [ ] `ARAGORA_ALLOWED_ORIGINS` set (no wildcards in production)
  - [ ] HTTPS enforced (SSL/TLS certificates installed)
  - [ ] Rate limiting enabled (default: 120 req/min per IP)

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

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.3.0 | Jan 2026 | Initial GA checklist |
