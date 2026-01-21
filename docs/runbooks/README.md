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
