# Aragora Launch Checklist

Comprehensive checklist for preparing Aragora for production launch.

## Pre-Launch (T-7 days)

### Infrastructure
- [ ] Production environment provisioned
- [ ] Database backups configured and tested
- [ ] Redis cluster deployed (if high-availability)
- [ ] CDN configured for static assets
- [ ] SSL/TLS certificates installed and auto-renewal configured
- [ ] DNS records configured with appropriate TTLs

### Security
- [ ] Security audit completed (see `.github/workflows/security.yml`)
- [ ] All HIGH/CRITICAL vulnerabilities resolved
- [ ] API keys rotated from development
- [ ] RBAC permissions reviewed
- [ ] Rate limiting configured
- [ ] CORS origins restricted to production domains

### Monitoring
- [ ] Grafana dashboards deployed (`deploy/grafana/dashboards/`)
- [ ] Prometheus alerts configured (`deploy/alerting/`)
- [ ] Log aggregation configured (Loki recommended)
- [ ] Uptime monitoring enabled
- [ ] Error tracking configured (Sentry optional)

### Documentation
- [ ] API documentation up-to-date (`docs/api/`)
- [ ] SDK documentation complete (`docs/SDK_GUIDE.md`)
- [ ] Self-hosted guide reviewed (`deploy/self-hosted/README.md`)
- [ ] Changelog updated for release

## Launch Day (T-0)

### Technical
- [ ] Run smoke tests (`scripts/smoke_test.sh`)
- [ ] Verify all health endpoints responding
- [ ] Confirm database migrations applied
- [ ] Test critical user flows:
  - [ ] User registration/login
  - [ ] Create debate
  - [ ] View results
  - [ ] Export receipts

### Operational
- [ ] On-call schedule confirmed
- [ ] Runbooks accessible (`docs/runbooks/`)
- [ ] Support channels ready
- [ ] Status page live

### Communication
- [ ] Release notes published
- [ ] Announcement prepared
- [ ] Support team briefed

## Post-Launch (T+1 to T+7)

### Monitoring
- [ ] Review error rates (target: <1%)
- [ ] Check latency SLOs (p95 <500ms)
- [ ] Monitor resource utilization
- [ ] Review user feedback

### Iteration
- [ ] Address critical issues immediately
- [ ] Collect feature requests
- [ ] Schedule post-launch retrospective

## Rollback Plan

If critical issues arise:

1. **Immediate**: Route traffic to maintenance page
2. **Assess**: Check logs, identify root cause
3. **Rollback** (if needed):
   ```bash
   # Revert to previous version
   docker compose down
   docker compose pull aragora:previous-tag
   docker compose up -d
   ```
4. **Communicate**: Update status page, notify users
5. **Fix Forward**: Patch and redeploy when ready

## Contacts

| Role | Name | Contact |
|------|------|---------|
| On-call Primary | TBD | TBD |
| On-call Secondary | TBD | TBD |
| Engineering Lead | TBD | TBD |
| Support Lead | TBD | TBD |

## Sign-off

| Area | Owner | Date | Status |
|------|-------|------|--------|
| Infrastructure | | | |
| Security | | | |
| Documentation | | | |
| Support | | | |
| Final Approval | | | |
