---
title: Incident Response Runbook
description: Incident Response Runbook
---

# Incident Response Runbook

Standard procedures for handling incidents affecting Aragora services.

## Severity Levels

| Level | Definition | Response Time | Examples |
|-------|------------|---------------|----------|
| **SEV1** | Service down, all users affected | Immediate | Complete outage, data loss |
| **SEV2** | Major feature broken, many users affected | < 30 min | Debates not starting, auth broken |
| **SEV3** | Minor feature broken, some users affected | < 2 hours | Single provider down, slow queries |
| **SEV4** | Cosmetic issues, workaround available | Next business day | UI glitch, minor bug |

## Incident Response Process

### 1. Detection & Triage (0-5 minutes)

```bash
# Quick health check all services
curl -s http://EC2_HOST:8765/api/health | jq .status
curl -s http://LIGHTSAIL_HOST:8080/api/health | jq .status
curl -s https://aragora.live/api/health 2>/dev/null || echo "Frontend may be down"

# Check detailed health
curl -s http://localhost:8080/api/health/detailed | jq .
```

**Determine severity based on:**
- Number of users affected
- Core functionality impacted
- Data integrity at risk
- Revenue impact

### 2. Communication (5-10 minutes)

**SEV1/SEV2:**
- Update status page (if available)
- Notify stakeholders
- Create incident channel/thread

**Template:**
```
INCIDENT: [Brief description]
SEVERITY: SEV[1-4]
IMPACT: [What's broken, who's affected]
STATUS: Investigating
NEXT UPDATE: [Time]
```

### 3. Investigation (10-30 minutes)

```bash
# Check recent logs for errors
sudo journalctl -u aragora -n 200 --no-pager | grep -i "error\|exception\|failed"

# Check service status
sudo systemctl status aragora

# Check resource usage
free -h
df -h
top -bn1 | head -20

# Check recent deployments
git log --oneline -5

# Check provider status
curl -s http://localhost:8080/api/health | jq .providers
```

### 4. Mitigation

**Quick Mitigations:**

| Issue | Mitigation |
|-------|------------|
| Service crashed | `sudo systemctl restart aragora` |
| Memory exhaustion | Restart + reduce `MAX_CONCURRENT_DEBATES` |
| Provider down | Verify OpenRouter fallback active |
| Database locked | Restart service |
| Port conflict | `sudo fuser -k 8765/tcp && sudo systemctl start aragora` |

**Rollback if recent deployment caused issue:**
```bash
# Find last good commit
git log --oneline -10

# Rollback to specific commit
git checkout <commit-hash>
pip install -e . --quiet
sudo systemctl restart aragora
```

### 5. Resolution

1. **Verify fix:**
   ```bash
   curl -sf http://localhost:8080/api/health && echo "Service healthy"
   ```

2. **Monitor for recurrence** (30 minutes)

3. **Update communication:**
   ```
   INCIDENT: [Brief description]
   STATUS: RESOLVED
   RESOLUTION: [What fixed it]
   DURATION: [How long it lasted]
   ```

### 6. Post-Incident

**Within 24 hours:**
- Document timeline
- Identify root cause
- Create follow-up tasks
- Update runbooks if needed

**Post-mortem template:**
```markdown
## Incident: [Title]
**Date:** YYYY-MM-DD
**Duration:** HH:MM
**Severity:** SEV[1-4]
**Author:** [Name]

### Summary
[Brief description of what happened]

### Timeline
- HH:MM - [Event]
- HH:MM - [Event]

### Root Cause
[Technical explanation]

### Resolution
[What fixed it]

### Action Items
- [ ] [Task 1]
- [ ] [Task 2]

### Lessons Learned
- [Lesson 1]
- [Lesson 2]
```

## Common Incident Scenarios

### Complete Service Outage (SEV1)

```bash
# 1. Check if service is running
sudo systemctl status aragora

# 2. Check logs for crash reason
sudo journalctl -u aragora -n 100 --no-pager

# 3. Restart service
sudo systemctl restart aragora

# 4. If fails, check port conflicts
sudo lsof -i :8080
sudo lsof -i :8765

# 5. Force kill and restart
sudo fuser -k 8080/tcp 8765/tcp
sleep 2
sudo systemctl start aragora
```

### All Debates Failing (SEV2)

```bash
# 1. Check provider health
curl -s http://localhost:8080/api/health | jq .providers

# 2. Check circuit breakers
curl -s http://localhost:8080/api/circuit-breakers | jq .

# 3. Test specific provider
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages -d '{"model":"claude-sonnet-4-20250514","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'

# 4. If all providers down, check OpenRouter
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models | head
```

### Database Unavailable (SEV1)

See [Database Issues Runbook](../RUNBOOK_DATABASE_ISSUES.md)

### Authentication Broken (SEV2)

```bash
# 1. Check JWT configuration
echo "ARAGORA_JWT_SECRET length: ${#ARAGORA_JWT_SECRET}"

# 2. Check auth stats
curl -s http://localhost:8080/api/auth/stats | jq .

# 3. Test token generation
curl -X POST http://localhost:8080/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'

# 4. If ARAGORA_JWT_SECRET changed, existing tokens are invalid
# Users need to re-authenticate
```

### High Latency (SEV3)

```bash
# 1. Check system resources
top -bn1 | head -20
free -h

# 2. Check debate queue
curl -s http://localhost:8080/api/queue/status | jq .

# 3. Check database performance
curl -s "http://localhost:8080/api/system/maintenance?task=status" | jq .

# 4. If queue backed up, scale or rate limit
```

## Escalation Matrix

| Severity | Escalate After | Escalate To |
|----------|----------------|-------------|
| SEV1 | Immediately | All hands |
| SEV2 | 30 minutes | Engineering lead |
| SEV3 | 2 hours | On-call engineer |
| SEV4 | Next standup | Backlog |

## Useful Commands Cheat Sheet

```bash
# Service management
sudo systemctl status aragora
sudo systemctl restart aragora
sudo journalctl -u aragora -f

# Health checks
curl -s http://localhost:8080/api/health | jq .
curl -s http://localhost:8080/api/health/detailed | jq .

# Resource monitoring
htop
df -h
free -h

# Network
sudo lsof -i :8080
sudo netstat -tlnp | grep 8080

# Process management
ps aux | grep aragora
sudo fuser -k 8080/tcp

# Git operations
git log --oneline -10
git status
git checkout <commit>
```

## Related Runbooks

- [Deployment](../RUNBOOK_DEPLOYMENT.md)
- [Database Issues](../RUNBOOK_DATABASE_ISSUES.md)
- [Provider Failure](../RUNBOOK_PROVIDER_FAILURE.md)
