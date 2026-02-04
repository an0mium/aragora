# Incident Response Runbook

This runbook defines procedures for handling production incidents in Aragora.

## Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| SEV-1 | Critical - Complete service outage | 15 min | API down, database unavailable, data loss |
| SEV-2 | High - Major functionality impaired | 30 min | Debate engine failures, auth broken |
| SEV-3 | Medium - Partial impact | 2 hours | Single connector down, slow responses |
| SEV-4 | Low - Minor issue | 24 hours | UI glitches, non-critical warnings |

## Incident Commander Checklist

### 1. Initial Assessment (0-5 minutes)
- [ ] Identify severity level
- [ ] Acknowledge in #incidents Slack channel
- [ ] Start incident timer
- [ ] Assign roles: IC, Communications, Technical Lead

### 2. Triage (5-15 minutes)
```bash
# Check system health
curl https://api.aragora.ai/api/health | jq .

# Check service status
kubectl get pods -n aragora
kubectl logs -l app=aragora-api --tail=100

# Check database connectivity
psql $DATABASE_URL -c "SELECT 1"

# Check recent deployments
git log --oneline -10
```

### 3. Communication Template

**Initial Update (post within 15 min):**
```
[INCIDENT] SEV-X: {Brief description}
- Status: Investigating
- Impact: {User-facing impact}
- ETA: Under investigation
- IC: @{name}
```

**Follow-up (every 30 min for SEV-1/2):**
```
[UPDATE] SEV-X: {Description}
- Status: {Investigating|Identified|Mitigating|Resolved}
- Root cause: {If known}
- ETA: {If known}
- Next update: {time}
```

### 4. Escalation Matrix

| Condition | Escalate To |
|-----------|-------------|
| SEV-1 > 30 min | Engineering Lead, CTO |
| Data breach suspected | Security Team, Legal |
| Customer data affected | Customer Success, Legal |
| SLA violation imminent | Account Manager |

### 5. Resolution Steps

```bash
# Rollback deployment if recent change caused issue
kubectl rollout undo deployment/aragora-api -n aragora

# Scale up if capacity issue
kubectl scale deployment/aragora-api --replicas=5 -n aragora

# Restart pods if stuck
kubectl delete pod -l app=aragora-api -n aragora

# Failover database if primary down
# (See DATABASE_RECOVERY.md)
```

### 6. Post-Incident

- [ ] Mark incident resolved in #incidents
- [ ] Schedule postmortem within 48 hours
- [ ] Create JIRA ticket for follow-up actions
- [ ] Update status page

## Postmortem Template

```markdown
## Incident: {Title}
**Date:** {Date}
**Duration:** {Minutes}
**Severity:** SEV-{X}
**IC:** {Name}

### Summary
{One paragraph description}

### Timeline
| Time | Event |
|------|-------|
| HH:MM | {Event} |

### Root Cause
{Technical explanation}

### Impact
- Users affected: {number}
- Revenue impact: {if applicable}
- SLA impact: {yes/no}

### Action Items
| Item | Owner | Due Date |
|------|-------|----------|
| {Action} | @{name} | {date} |

### Lessons Learned
- What went well
- What could improve
```

## Quick Reference Commands

```bash
# Emergency scale up
kubectl scale deployment/aragora-api --replicas=10 -n aragora

# Check error rate
curl -s https://prometheus.aragora.ai/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])

# Force restart all pods
kubectl rollout restart deployment/aragora-api -n aragora

# Check database connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity"

# Check Redis health
redis-cli ping

# Drain node
kubectl drain {node-name} --ignore-daemonsets --delete-emptydir-data
```

## Contact List

| Role | Primary | Backup |
|------|---------|--------|
| Engineering Lead | - | - |
| DevOps | - | - |
| Database Admin | - | - |
| Security | - | - |

---
*Last updated: 2026-02-03*
