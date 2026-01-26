# Incident Runbooks

Operational runbooks for responding to Aragora alerts and incidents.

## Alert Reference

| Alert | Severity | Runbook |
|-------|----------|---------|
| ServiceDown | critical | [service-down.md](./service-down.md) |
| HighErrorRate | critical | [high-error-rate.md](./high-error-rate.md) |
| HighAPILatency | warning | [high-latency.md](./high-latency.md) |
| DatabaseConnectionExhausted | critical | [database-issues.md](./database-issues.md) |
| RedisUnavailable | critical | [redis-issues.md](./redis-issues.md) |
| StartupSLOExceeded | warning | [RUNBOOK_STARTUP_ISSUES.md](./RUNBOOK_STARTUP_ISSUES.md) |
| StartupFailed | critical | [RUNBOOK_STARTUP_ISSUES.md](./RUNBOOK_STARTUP_ISSUES.md) |
| DecaySchedulerStopped | warning | [RUNBOOK_KNOWLEDGE_DECAY.md](./RUNBOOK_KNOWLEDGE_DECAY.md) |
| StaleWorkspace | warning | [RUNBOOK_KNOWLEDGE_DECAY.md](./RUNBOOK_KNOWLEDGE_DECAY.md) |

## Operational Runbooks

| Topic | Runbook | Description |
|-------|---------|-------------|
| Server Startup | [RUNBOOK_STARTUP_ISSUES.md](./RUNBOOK_STARTUP_ISSUES.md) | Startup failures, SLO violations |
| Database Migration | [RUNBOOK_DATABASE_CONSOLIDATION.md](./RUNBOOK_DATABASE_CONSOLIDATION.md) | Consolidating legacy databases |
| Knowledge Decay | [RUNBOOK_KNOWLEDGE_DECAY.md](./RUNBOOK_KNOWLEDGE_DECAY.md) | Confidence decay monitoring |
| Deployment | [RUNBOOK_DEPLOYMENT.md](./RUNBOOK_DEPLOYMENT.md) | Deployment procedures |
| Security | [RUNBOOK_SECURITY.md](./RUNBOOK_SECURITY.md) | Security incident response |
| Key Rotation | [RUNBOOK_KEY_ROTATION.md](./RUNBOOK_KEY_ROTATION.md) | Encryption key rotation |
| PostgreSQL | [RUNBOOK_POSTGRESQL_MIGRATION.md](./RUNBOOK_POSTGRESQL_MIGRATION.md) | PostgreSQL migration |

## Incident Severity Levels

| Level | Response Time | Description |
|-------|---------------|-------------|
| SEV1 | 15 min | Complete outage, data loss risk |
| SEV2 | 1 hour | Major feature degraded |
| SEV3 | 4 hours | Minor feature impacted |
| SEV4 | Next business day | Cosmetic/low-impact issues |

## On-Call Contacts

Configure in environment:
- `ONCALL_SLACK_CHANNEL` - Slack channel for alerts
- `ONCALL_PAGERDUTY_KEY` - PagerDuty integration key
- `ONCALL_EMAIL` - Fallback email address
