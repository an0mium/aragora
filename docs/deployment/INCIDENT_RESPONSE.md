# Incident Response Playbooks

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** Security Team

---

## Overview

This document provides step-by-step playbooks for responding to security incidents. All team members should be familiar with these procedures.

**SOC 2 Control:** CC7-02 - Incident response procedures

---

## Incident Classification

### Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| P1 - Critical | Service down, data breach, active attack | Immediate | Production down, credential leak, ransomware |
| P2 - High | Major feature unavailable, potential breach | 1 hour | Auth broken, database unreachable, suspicious activity |
| P3 - Medium | Feature degraded, minor security issue | 4 hours | Slow performance, single customer affected |
| P4 - Low | Cosmetic issues, potential vulnerability | 24 hours | Minor bug, low-severity vuln report |

### Escalation Contacts

| Role | Contact | Availability |
|------|---------|--------------|
| On-Call Engineer | PagerDuty | 24/7 |
| Security Lead | security@aragora.ai | Business hours |
| VP Engineering | [phone] | 24/7 for P1 |
| CEO/CTO | [phone] | 24/7 for P1 |
| Legal | legal@aragora.ai | P1 incidents |

---

## Playbook 1: Service Outage

### Detection

- Automated: Uptime Kuma alerts, health check failures
- Customer: Support tickets reporting unavailability
- Monitoring: Prometheus alerts, error rate spikes

### Response Steps

```
[ ] 1. ACKNOWLEDGE
    - Acknowledge alert in PagerDuty
    - Join incident Slack channel (#incident-response)
    - Time: <2 minutes

[ ] 2. ASSESS
    - Check health endpoints: curl https://api.aragora.ai/api/health
    - Check detailed health: curl https://api.aragora.ai/api/health/detailed
    - Check error logs: kubectl logs -l app=aragora --tail=100
    - Identify affected services
    - Time: <5 minutes

[ ] 3. COMMUNICATE
    - Update status page (status.aragora.ai)
    - Post in #ops-alerts: "Investigating API unavailability"
    - If P1: Notify VP Engineering
    - Time: <10 minutes

[ ] 4. MITIGATE
    - If deployment-related: Rollback
      kubectl rollout undo deployment/aragora
    - If resource exhaustion: Scale up
      kubectl scale deployment/aragora --replicas=5
    - If database: Check connection pool, restart if needed
    - If external dependency: Enable degraded mode
    - Time: <30 minutes

[ ] 5. RESOLVE
    - Confirm service restored
    - Update status page: "Service restored"
    - Monitor for 30 minutes
    - Time: <1 hour

[ ] 6. POST-INCIDENT
    - Create post-mortem document
    - Schedule review meeting
    - Update runbooks if needed
    - Time: Within 48 hours
```

### Rollback Commands

```bash
# Kubernetes rollback
kubectl -n aragora rollout undo deployment/aragora

# Verify rollback
kubectl -n aragora rollout status deployment/aragora

# Database migration rollback (if needed)
alembic downgrade -1

# Restore from backup (extreme cases)
pg_restore -d aragora backup_YYYYMMDD.dump
```

---

## Playbook 2: Data Breach

### Detection

- Security monitoring: Anomalous data access patterns
- Customer report: Unauthorized access to account
- External: Security researcher disclosure
- Audit logs: Unusual admin activity

### Response Steps

```
[ ] 1. CONTAIN (Immediate)
    - Disable compromised credentials
    - Revoke affected API keys
    - Block suspicious IPs
    - If widespread: Enable emergency maintenance mode
    - Time: <15 minutes

[ ] 2. PRESERVE EVIDENCE
    - Snapshot affected systems
    - Export relevant audit logs
    - Document timeline of events
    - Do NOT modify or delete logs
    - Time: <30 minutes

[ ] 3. ASSESS SCOPE
    - What data was accessed?
    - How many users affected?
    - What was the attack vector?
    - Is the vulnerability still present?
    - Time: <2 hours

[ ] 4. ESCALATE
    - Notify Security Lead
    - Notify VP Engineering
    - If PII involved: Notify Legal
    - If >500 users: Prepare regulatory notification
    - Time: <4 hours

[ ] 5. REMEDIATE
    - Patch vulnerability
    - Force password reset for affected users
    - Rotate compromised credentials
    - Deploy additional monitoring
    - Time: <24 hours

[ ] 6. NOTIFY
    - Affected users (within 72 hours)
    - Regulatory bodies (as required)
    - Board of directors (for significant breaches)
    - Time: As required by law (max 72 hours for GDPR)

[ ] 7. DOCUMENT
    - Complete incident report
    - Update threat model
    - Implement preventive measures
    - Time: Within 1 week
```

### Containment Commands

```bash
# Disable user
curl -X POST https://api.aragora.ai/admin/users/{id}/disable \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Revoke all tokens for user
curl -X POST https://api.aragora.ai/admin/users/{id}/revoke-all \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Block IP (if using WAF)
# Add to WAF blocklist

# Enable maintenance mode
kubectl -n aragora set env deployment/aragora MAINTENANCE_MODE=true
```

---

## Playbook 3: Credential Compromise

### Types

- **User credential compromise**: Customer's account taken over
- **Service credential compromise**: API keys, database credentials exposed
- **Internal credential compromise**: Employee credentials leaked

### Response Steps - User Credential

```
[ ] 1. VERIFY
    - Confirm unauthorized access
    - Check audit logs for suspicious activity
    - Time: <15 minutes

[ ] 2. CONTAIN
    - Force logout (revoke all tokens)
    - Disable account temporarily
    - Block suspicious IPs
    - Time: <30 minutes

[ ] 3. NOTIFY USER
    - Email user about compromise
    - Provide password reset link
    - Document any data accessed
    - Time: <1 hour

[ ] 4. INVESTIGATE
    - How was credential obtained? (phishing, reuse, breach)
    - Was any data exfiltrated?
    - Are other accounts affected?
    - Time: <4 hours

[ ] 5. RESTORE
    - User resets password
    - Enable MFA (recommend strongly)
    - Restore account access
    - Monitor for 30 days
    - Time: <24 hours
```

### Response Steps - Service Credential

```
[ ] 1. IDENTIFY
    - Which credentials were exposed?
    - Where were they exposed? (code, logs, public repo)
    - Time: <15 minutes

[ ] 2. ROTATE IMMEDIATELY
    - Generate new credentials
    - Update in secrets manager
    - Deploy configuration update
    - Time: <30 minutes

[ ] 3. REVOKE OLD CREDENTIALS
    - Disable old API keys
    - Rotate database passwords
    - Invalidate old JWTs
    - Time: <1 hour

[ ] 4. ASSESS DAMAGE
    - Were credentials used maliciously?
    - Check API provider logs
    - Review database access logs
    - Time: <4 hours

[ ] 5. PREVENT RECURRENCE
    - Add to secret scanning blocklist
    - Review CI/CD pipelines
    - Update handling procedures
    - Time: Within 1 week
```

### Credential Rotation Commands

```bash
# Rotate JWT secret
# 1. Generate new secret
openssl rand -base64 64 > new_jwt_secret.txt

# 2. Update in secrets manager
aws secretsmanager update-secret --secret-id aragora/jwt \
  --secret-string "$(cat new_jwt_secret.txt)"

# 3. Rolling restart to pick up new secret
kubectl -n aragora rollout restart deployment/aragora

# Rotate database password
# 1. Generate new password
openssl rand -base64 32 > new_db_password.txt

# 2. Update in database
psql -c "ALTER USER aragora_app WITH PASSWORD '$(cat new_db_password.txt)'"

# 3. Update in secrets manager
aws secretsmanager update-secret --secret-id aragora/database \
  --secret-string "$(cat new_db_password.txt)"

# 4. Rolling restart
kubectl -n aragora rollout restart deployment/aragora
```

---

## Playbook 4: DDoS Attack

### Detection

- Monitoring: Unusual traffic spike, high latency
- Alerts: Rate limit exceeded alerts
- Customer: Slow or unavailable service

### Response Steps

```
[ ] 1. CONFIRM ATTACK
    - Check traffic patterns
    - Verify not legitimate traffic spike
    - Identify attack type (volumetric, application layer)
    - Time: <10 minutes

[ ] 2. ACTIVATE MITIGATION
    - Enable CDN DDoS protection (Cloudflare, AWS Shield)
    - Increase rate limiting thresholds strategically
    - Enable geo-blocking if attack is localized
    - Time: <30 minutes

[ ] 3. SCALE DEFENSES
    - Scale up application instances
    - Increase database connection limits
    - Enable caching aggressively
    - Time: <1 hour

[ ] 4. COORDINATE WITH PROVIDER
    - Contact CDN provider support
    - Request traffic analysis
    - Implement custom rules
    - Time: <2 hours

[ ] 5. MONITOR AND ADJUST
    - Monitor attack patterns
    - Adjust rules as needed
    - Document effective mitigations
    - Time: Duration of attack

[ ] 6. POST-ATTACK
    - Review effectiveness of response
    - Update DDoS playbook
    - Consider additional protection
    - Time: Within 1 week
```

### Mitigation Commands

```bash
# Enable aggressive rate limiting
kubectl -n aragora set env deployment/aragora \
  RATE_LIMIT_REQUESTS_PER_MINUTE=10

# Scale up
kubectl -n aragora scale deployment/aragora --replicas=10

# Enable maintenance page for non-essential endpoints
kubectl -n aragora set env deployment/aragora \
  DDOS_MODE=true

# Block suspicious IP ranges (example with iptables)
# iptables -A INPUT -s 192.168.1.0/24 -j DROP
```

---

## Playbook 5: Dependency Vulnerability

### Detection

- Automated: Dependabot alerts, safety scans
- External: CVE announcement, security advisory
- Vendor: Upstream vulnerability disclosure

### Response Steps

```
[ ] 1. ASSESS SEVERITY
    - What is the CVSS score?
    - Is Aragora affected? (check if vulnerable code path is used)
    - Is exploit publicly available?
    - Time: <1 hour

[ ] 2. CLASSIFY URGENCY
    | CVSS | Exploit Available | Response |
    |------|-------------------|----------|
    | 9.0+ | Yes | Immediate patch |
    | 9.0+ | No | Patch within 24h |
    | 7.0-8.9 | Yes | Patch within 24h |
    | 7.0-8.9 | No | Patch within 1 week |
    | <7.0 | Any | Patch in next release |

[ ] 3. DEVELOP FIX
    - Update dependency to patched version
    - If no patch: implement workaround
    - If workaround not possible: disable feature
    - Time: Varies

[ ] 4. TEST
    - Run full test suite
    - Verify vulnerability is fixed
    - Check for regressions
    - Time: <4 hours

[ ] 5. DEPLOY
    - Deploy to staging first
    - Verify in staging
    - Deploy to production
    - Monitor for issues
    - Time: <2 hours

[ ] 6. DOCUMENT
    - Update vulnerability tracking
    - Close security advisory
    - Update dependencies manifest
    - Time: <24 hours
```

### Patching Commands

```bash
# Check for vulnerable dependencies
pip-audit
safety check

# Update specific package
pip install --upgrade package_name

# Update all dependencies (careful)
pip install --upgrade -r requirements.txt

# Test
pytest tests/ -v

# Deploy
git add . && git commit -m "fix: patch CVE-XXXX-XXXX"
git push && # wait for CI
# deploy via normal process
```

---

## Communication Templates

### Status Page Update

```
[INVESTIGATING]
We are investigating reports of [issue]. Our team is actively working
to identify the cause. Updates will be posted as we learn more.

[IDENTIFIED]
We have identified the cause of [issue] and are implementing a fix.
Affected services: [list]
Expected resolution: [time]

[MONITORING]
A fix has been implemented for [issue]. We are monitoring to ensure
service stability. Some users may still experience [residual effects].

[RESOLVED]
[Issue] has been resolved. All services are operating normally.
We apologize for any inconvenience. A post-incident report will be
published within 48 hours.
```

### Customer Notification (Breach)

```
Subject: Security Notice - Action Required

Dear [Customer],

We are writing to inform you of a security incident that may have
affected your account.

What happened:
[Brief description of incident]

What information was involved:
[Types of data potentially accessed]

What we are doing:
[Steps taken to address the incident]

What you can do:
- Reset your password immediately
- Enable multi-factor authentication
- Review your account activity
- Contact us with any concerns

We take the security of your data seriously and are committed to
preventing similar incidents in the future.

If you have questions, please contact security@aragora.ai.

Sincerely,
Aragora Security Team
```

---

## Post-Incident Review

### Required Documentation

1. **Timeline**: Minute-by-minute account of incident
2. **Impact**: Users affected, data exposed, downtime
3. **Root Cause**: What caused the incident
4. **Response**: Actions taken and their effectiveness
5. **Lessons Learned**: What could be improved
6. **Action Items**: Preventive measures with owners and dates

### Post-Mortem Template

```markdown
# Post-Incident Review: [Incident Name]

## Summary
- **Date**: YYYY-MM-DD
- **Duration**: HH:MM
- **Severity**: P1/P2/P3/P4
- **Impact**: [brief description]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| HH:MM | First alert |
| HH:MM | Incident acknowledged |
| ... | ... |
| HH:MM | Incident resolved |

## Root Cause
[Description of what caused the incident]

## Impact
- Users affected: X
- Data exposed: None / [description]
- Revenue impact: $X (estimated)
- Reputation impact: [description]

## Response Assessment
- What went well
- What could be improved

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| ... | ... | ... | ... |

## Lessons Learned
[Key takeaways for preventing similar incidents]
```

---

## Training Requirements

- All engineers: Annual incident response training
- On-call engineers: Quarterly tabletop exercises
- Security team: Monthly incident simulations

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |
