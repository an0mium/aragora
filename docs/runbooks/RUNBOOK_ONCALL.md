# On-Call Procedures Runbook

Procedures and checklists for on-call engineers.

## On-Call Schedule

| Rotation | Hours (UTC) | Primary | Backup |
|----------|-------------|---------|--------|
| Weekday Day | 08:00-18:00 | Engineering | DevOps |
| Weekday Night | 18:00-08:00 | DevOps | Engineering |
| Weekend | 24/7 | Rotating | Engineering Lead |

## Shift Handoff Checklist

### Starting Your Shift

```markdown
[ ] Review active incidents (if any)
[ ] Check #incidents channel for recent discussions
[ ] Review PagerDuty/alerts dashboard
[ ] Verify access to all systems:
    - [ ] SSH to production servers
    - [ ] Grafana dashboards
    - [ ] Provider dashboards (Anthropic, OpenAI)
    - [ ] Source code repository
[ ] Check system health:
```

```bash
# Quick health check script
for host in "localhost:8080"; do
  echo "=== Checking $host ==="
  curl -sf "http://$host/api/health" | jq -r '.status // "UNREACHABLE"'
done
```

### During Your Shift

**Every 4 hours:**
```bash
# System health snapshot
echo "=== Health Check $(date) ===" >> /tmp/oncall_log.txt
curl -s http://localhost:8080/api/health/detailed >> /tmp/oncall_log.txt

# Check error rates
curl -s 'http://localhost:9090/api/v1/query?query=rate(aragora_http_requests_total{status=~"5.."}[5m])' | jq .

# Check active debates
curl -s http://localhost:8080/api/debates/active | jq '.count // 0'
```

### Ending Your Shift

```markdown
[ ] Update shift log with notable events
[ ] Document any ongoing issues
[ ] Brief incoming on-call engineer
[ ] Transfer PagerDuty responsibility
[ ] Ensure no unacknowledged alerts
```

**Handoff Template:**
```
## Shift Handoff: [Date] [Shift]

**Outgoing:** [Name]
**Incoming:** [Name]

### Status
- Overall: [Green/Yellow/Red]
- Active Incidents: [None / List]

### Notable Events
- [Event 1]
- [Event 2]

### Ongoing Issues
- [Issue 1 - Status]
- [Issue 2 - Status]

### Action Items for Incoming
- [ ] [Item 1]
- [ ] [Item 2]

### Provider Status
- Anthropic: [OK / Degraded]
- OpenAI: [OK / Degraded]
- OpenRouter: [OK / Degraded]
```

---

## First Response Checklist

When an alert fires:

### Step 1: Acknowledge (< 2 min)
```bash
# 1. Acknowledge the alert in PagerDuty
# 2. Note the start time
START_TIME=$(date +%s)

# 3. Quick triage
curl -sf http://localhost:8080/api/health | jq .
```

### Step 2: Assess Severity (< 5 min)

| Check | Command | SEV1 Indicator |
|-------|---------|----------------|
| Service up? | `curl -sf localhost:8080/api/health` | Not reachable |
| All providers down? | `curl localhost:8080/api/health \| jq .providers` | All failed |
| Database ok? | `curl localhost:8080/api/health \| jq .checks.database` | status != "healthy" |
| Error rate? | Check Grafana | > 10% |

### Step 3: Initial Mitigation (< 10 min)

**Service Down:**
```bash
sudo systemctl restart aragora
# Wait 30 seconds, then verify
curl -sf http://localhost:8080/api/health && echo "Service recovered"
```

**High Error Rate:**
```bash
# Check recent logs
sudo journalctl -u aragora -n 50 --no-pager | grep -i error

# Check if specific endpoint
curl -s 'localhost:9090/api/v1/query?query=topk(5,rate(aragora_http_requests_total{status="500"}[5m]))'
```

**Provider Issues:**
```bash
# Check circuit breakers
curl -s http://localhost:8080/api/circuit-breakers | jq .

# Verify fallback is active
echo "OpenRouter key set: $([ -n \"$OPENROUTER_API_KEY\" ] && echo 'Yes' || echo 'No')"
```

### Step 4: Escalate if Needed

| Condition | Escalation |
|-----------|------------|
| SEV1 not resolved in 15 min | Engineering Lead |
| SEV2 not resolved in 30 min | Engineering Lead |
| Security incident suspected | Security + Engineering Lead |
| Data loss possible | Engineering Lead + CTO |

---

## Common Scenarios Quick Reference

### Debate Timeout
```bash
# Check debate status
curl -s http://localhost:8080/api/debates/active | jq '.debates[] | {id, status, started_at}'

# Check for stuck debates (> 10 min old)
curl -s http://localhost:8080/api/debates/active | jq '.debates[] | select(.started_at < (now - 600))'
```

### Memory Pressure
```bash
# Check memory
free -h
# Check process memory
ps aux --sort=-%mem | head -10
# If needed, restart service
sudo systemctl restart aragora
```

### Database Slow Queries
```bash
# Check DB metrics
curl -s http://localhost:8080/api/health | jq '.checks.database'
# Check Grafana DB dashboard for slow queries
```

### WebSocket Disconnections
```bash
# Check connection count
curl -s http://localhost:8080/api/ws/stats | jq .
# Check for connection errors
sudo journalctl -u aragora -n 100 | grep -i websocket
```

---

## Monitoring Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| Overview | /grafana/d/overview | System health at a glance |
| API Performance | /grafana/d/api-latency | Request latency & errors |
| Debate Metrics | /grafana/d/debate-metrics | Debate success/failure |
| Agent Performance | /grafana/d/agent-performance | Agent latency & failures |
| SLO Tracking | /grafana/d/slo-tracking | Error budget burn rates |

---

## Useful Prometheus Queries

```promql
# Error rate (last 5 min)
rate(aragora_http_requests_total{status=~"5.."}[5m])
  / rate(aragora_http_requests_total[5m]) * 100

# P99 latency
histogram_quantile(0.99, rate(aragora_http_request_duration_seconds_bucket[5m]))

# Active WebSocket connections
aragora_websocket_connections_active

# Circuit breaker open
aragora_agent_circuit_breaker_state == 1

# Debate failure rate
rate(aragora_debates_total{outcome="error"}[1h])
  / rate(aragora_debates_total[1h]) * 100
```

---

## Contact List

| Role | Contact | Reach Via |
|------|---------|-----------|
| Engineering Lead | [Name] | PagerDuty / Phone |
| DevOps Lead | [Name] | PagerDuty / Phone |
| Security | security@aragora.ai | Email + PagerDuty |
| External: Anthropic | support@anthropic.com | Email |
| External: OpenAI | support@openai.com | Email |

---

## Post-Incident Tasks

After resolving any SEV1/SEV2:

```markdown
[ ] Update status page (if public)
[ ] Send resolution notification
[ ] Create post-mortem document
[ ] Schedule post-mortem meeting (within 48h)
[ ] File tickets for follow-up work
[ ] Update runbooks if needed
```

---

## Related Runbooks

- [Main Runbook](../RUNBOOK.md) - Detailed operational procedures
- [Incident Response](RUNBOOK_INCIDENT.md) - Full incident process
- [Provider Failure](RUNBOOK_PROVIDER_FAILURE.md) - AI provider issues
- [Database Issues](RUNBOOK_DATABASE_ISSUES.md) - Database troubleshooting
- [Deployment](RUNBOOK_DEPLOYMENT.md) - Deployment procedures
