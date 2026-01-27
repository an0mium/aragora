# Alert Response Procedures Runbook

**Purpose:** Standard operating procedures for responding to production alerts.
**Audience:** On-Call Engineers, SRE, DevOps
**Last Updated:** January 2026

---

## Overview

This runbook provides:
- Triage procedures for common alerts
- Escalation paths and timelines
- Communication templates
- Root cause investigation guides
- Recovery procedures

---

## Alert Severity Levels

| Severity | Response Time | Examples | Escalation |
|----------|---------------|----------|------------|
| **P1 - Critical** | 15 min | Service down, data loss risk | Immediate page, exec notify |
| **P2 - High** | 30 min | Degraded performance, partial outage | Page on-call |
| **P3 - Medium** | 2 hours | Non-critical feature broken | Slack alert |
| **P4 - Low** | 24 hours | Cosmetic issues, minor bugs | Ticket |

---

## Alert Categories

### Infrastructure Alerts

- [Server Down](#server-down)
- [High CPU Usage](#high-cpu-usage)
- [High Memory Usage](#high-memory-usage)
- [Disk Space Low](#disk-space-low)
- [Network Latency](#network-latency)

### Database Alerts

- [PostgreSQL Connection Exhausted](#postgresql-connection-exhausted)
- [Replication Lag](#replication-lag)
- [Database Deadlock](#database-deadlock)

### Redis Alerts

- [Redis Memory Full](#redis-memory-full)
- [Redis Sentinel Failover](#redis-sentinel-failover)

### Application Alerts

- [High Error Rate](#high-error-rate)
- [API Latency High](#api-latency-high)
- [Debate Timeout](#debate-timeout)
- [Agent Quota Exceeded](#agent-quota-exceeded)

### Security Alerts

- [Authentication Failures Spike](#authentication-failures-spike)
- [Rate Limit Triggered](#rate-limit-triggered)
- [Certificate Expiring](#certificate-expiring)

---

## Infrastructure Alerts

### Server Down

**Severity:** P1 - Critical

**Alert:** `InstanceDown` - Target instance not responding to health checks

**Symptoms:**
- Health endpoint returning non-200
- No metrics from instance
- User reports of connection failures

**Immediate Actions:**

```bash
# 1. Check instance status
docker ps -a | grep aragora
kubectl get pods -n production

# 2. Check recent logs
docker logs aragora --tail 100
kubectl logs -n production deployment/aragora --tail 100

# 3. Check system resources
docker stats aragora
kubectl top pods -n production

# 4. Attempt restart
docker restart aragora
kubectl rollout restart deployment/aragora -n production
```

**Escalation:**
- If not recovered in 5 min → Page secondary on-call
- If not recovered in 15 min → Page team lead
- Update status page immediately

**Communication Template:**
```
[INCIDENT] Aragora service unavailable
Status: Investigating
Impact: Users cannot access the platform
ETA: Investigating, update in 15 minutes
```

---

### High CPU Usage

**Severity:** P2 - High

**Alert:** `HighCPUUsage` - CPU > 80% for 5 minutes

**Symptoms:**
- Slow response times
- Request timeouts
- Increased latency metrics

**Investigation:**

```bash
# 1. Identify hot processes
docker exec aragora top -bn1 | head -20
kubectl exec -it <pod> -- top -bn1 | head -20

# 2. Check for runaway debates
curl -s http://localhost:8080/api/v1/debates/active | jq '.count'

# 3. Check agent activity
curl -s http://localhost:8080/api/v1/metrics | grep agent_active

# 4. Profile if needed
docker exec aragora py-spy top --pid 1
```

**Resolution:**

```bash
# Scale horizontally if load is legitimate
kubectl scale deployment/aragora --replicas=5 -n production

# Or identify and terminate expensive operations
curl -X DELETE http://localhost:8080/api/v1/debates/<debate_id>
```

---

### High Memory Usage

**Severity:** P2 - High

**Alert:** `HighMemoryUsage` - Memory > 85% for 5 minutes

**Investigation:**

```bash
# 1. Check memory breakdown
docker exec aragora cat /proc/meminfo
kubectl exec -it <pod> -- cat /proc/meminfo

# 2. Check for memory leaks
curl -s http://localhost:8080/api/v1/debug/memory | jq

# 3. Check debate session count
curl -s http://localhost:8080/api/v1/debates/stats | jq '.active_sessions'
```

**Resolution:**

```bash
# Clear caches
curl -X POST http://localhost:8080/api/v1/admin/cache/clear

# Graceful restart (with rolling update)
kubectl rollout restart deployment/aragora -n production

# If OOM imminent, scale up
kubectl scale deployment/aragora --replicas=5 -n production
```

---

### Disk Space Low

**Severity:** P2 - High

**Alert:** `DiskSpaceLow` - Disk usage > 85%

**Investigation:**

```bash
# 1. Check disk usage
df -h
du -sh /var/lib/docker/*
du -sh /var/log/*

# 2. Find large files
find / -type f -size +100M 2>/dev/null | head -20

# 3. Check Docker disk usage
docker system df
```

**Resolution:**

```bash
# Clean Docker resources
docker system prune -af --volumes

# Rotate logs
journalctl --vacuum-size=500M
truncate -s 0 /var/log/aragora/*.log

# Clear old backups
find /backup -mtime +7 -delete

# Archive and compress old data
tar -czf /backup/archive/old_logs.tar.gz /var/log/aragora/old/
rm -rf /var/log/aragora/old/
```

---

### Network Latency

**Severity:** P3 - Medium

**Alert:** `NetworkLatencyHigh` - Latency > 100ms to database

**Investigation:**

```bash
# 1. Check connectivity
ping -c 10 postgres.internal
traceroute postgres.internal

# 2. Check DNS resolution
dig postgres.internal

# 3. Check for packet loss
mtr -c 100 postgres.internal

# 4. Check network interface
ip -s link show
netstat -i
```

**Resolution:**
- Contact network team if persistent
- Consider failing over to different AZ
- Check for infrastructure maintenance notices

---

## Database Alerts

### PostgreSQL Connection Exhausted

**Severity:** P1 - Critical

**Alert:** `PostgreSQLConnectionsExhausted` - Connections > 90% of max

**Symptoms:**
- "too many connections" errors
- New requests failing
- Database timeout errors

**Investigation:**

```bash
# 1. Check current connections
psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# 2. Identify connection hogs
psql -U postgres -c "
SELECT usename, application_name, client_addr, count(*)
FROM pg_stat_activity
GROUP BY 1,2,3
ORDER BY 4 DESC
LIMIT 10;"

# 3. Find idle connections
psql -U postgres -c "
SELECT pid, usename, state, query_start, query
FROM pg_stat_activity
WHERE state = 'idle'
AND query_start < now() - interval '10 minutes';"
```

**Resolution:**

```bash
# 1. Kill idle connections
psql -U postgres -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND query_start < now() - interval '10 minutes'
AND usename != 'postgres';"

# 2. If PgBouncer is used, check pool
psql -p 6432 -U pgbouncer pgbouncer -c "SHOW POOLS;"

# 3. Restart problematic application
kubectl rollout restart deployment/aragora -n production
```

**Prevention:**
- Ensure connection pooling is enabled
- Review max_connections setting
- Check for connection leaks in application code

---

### Replication Lag

**Severity:** P2 - High

**Alert:** `PostgreSQLReplicationLag` - Lag > 30 seconds

**Investigation:**

```bash
# 1. Check lag on primary
psql -U postgres -c "
SELECT client_addr, state,
       pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as lag_bytes,
       replay_lag
FROM pg_stat_replication;"

# 2. Check lag on replica
psql -U postgres -c "
SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds;"

# 3. Check WAL sender status
psql -U postgres -c "SELECT * FROM pg_stat_wal_receiver;"

# 4. Check network between primary and replica
ping -c 10 replica.internal
```

**Resolution:**

```bash
# If replica is too far behind, rebuild it
sudo systemctl stop postgresql
sudo rm -rf /var/lib/postgresql/15/main/*
pg_basebackup -h primary.internal -U replicator -D /var/lib/postgresql/15/main -Fp -Xs -P -R
sudo systemctl start postgresql
```

---

### Database Deadlock

**Severity:** P2 - High

**Alert:** `PostgreSQLDeadlocks` - Deadlock detected

**Investigation:**

```sql
-- Check for current locks
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

**Resolution:**

```sql
-- Kill blocking query (use with caution)
SELECT pg_terminate_backend(<blocking_pid>);

-- Or cancel the query without killing connection
SELECT pg_cancel_backend(<blocking_pid>);
```

---

## Redis Alerts

### Redis Memory Full

**Severity:** P1 - Critical

**Alert:** `RedisMemoryFull` - Memory > 95% of maxmemory

**Investigation:**

```bash
# 1. Check memory usage
redis-cli -a "${REDIS_PASSWORD}" INFO memory

# 2. Find large keys
redis-cli -a "${REDIS_PASSWORD}" --bigkeys

# 3. Check key count by prefix
redis-cli -a "${REDIS_PASSWORD}" --scan --pattern "debate:*" | wc -l
redis-cli -a "${REDIS_PASSWORD}" --scan --pattern "session:*" | wc -l
```

**Resolution:**

```bash
# 1. Clear expired sessions
redis-cli -a "${REDIS_PASSWORD}" --scan --pattern "session:*" | \
    xargs -L 100 redis-cli -a "${REDIS_PASSWORD}" DEL

# 2. Clear old debate cache
redis-cli -a "${REDIS_PASSWORD}" --scan --pattern "debate:cache:*" | \
    xargs -L 100 redis-cli -a "${REDIS_PASSWORD}" DEL

# 3. If still critical, flush non-essential data
redis-cli -a "${REDIS_PASSWORD}" FLUSHDB  # Caution: clears entire database
```

---

### Redis Sentinel Failover

**Severity:** P2 - High

**Alert:** `RedisSentinelFailover` - Master changed

**Investigation:**

```bash
# 1. Check new master
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# 2. Check what happened
docker logs sentinel-1 --since 10m | grep -i failover

# 3. Check all node status
redis-cli -p 26379 sentinel master aragora-master
redis-cli -p 26379 sentinel replicas aragora-master
```

**Resolution:**

```bash
# 1. Verify new master is healthy
redis-cli -h <new-master> -a "${REDIS_PASSWORD}" ping

# 2. Verify replicas are syncing
redis-cli -h <new-master> -a "${REDIS_PASSWORD}" info replication

# 3. Check old master (if recoverable)
docker logs redis-master --since 30m

# 4. If old master needs to rejoin as replica
docker restart redis-master
```

---

## Application Alerts

### High Error Rate

**Severity:** P1 - Critical

**Alert:** `HighErrorRate` - Error rate > 5% for 5 minutes

**Investigation:**

```bash
# 1. Check error logs
docker logs aragora --since 10m 2>&1 | grep -i error | tail -50

# 2. Check error metrics
curl -s http://localhost:8080/api/v1/metrics | grep http_requests_total | grep status=\"5

# 3. Check recent deployments
kubectl rollout history deployment/aragora -n production

# 4. Check downstream services
curl -s http://localhost:8080/api/v1/health/dependencies
```

**Resolution:**

```bash
# If recent deployment caused it, rollback
kubectl rollout undo deployment/aragora -n production

# If external dependency, enable circuit breaker
curl -X POST http://localhost:8080/api/v1/admin/circuit-breaker/enable

# Scale up to handle load
kubectl scale deployment/aragora --replicas=5 -n production
```

---

### API Latency High

**Severity:** P2 - High

**Alert:** `APILatencyHigh` - p95 latency > 2 seconds

**Investigation:**

```bash
# 1. Check slow endpoints
curl -s http://localhost:8080/api/v1/metrics | grep http_request_duration | sort -t'=' -k2 -rn | head -20

# 2. Check database query times
psql -U postgres -d aragora -c "
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;"

# 3. Check agent response times
curl -s http://localhost:8080/api/v1/metrics | grep agent_response_time
```

**Resolution:**

```bash
# Enable query caching
curl -X POST http://localhost:8080/api/v1/admin/cache/enable

# Scale read replicas
kubectl scale deployment/aragora-reader --replicas=3 -n production

# Enable CDN caching for static content
```

---

### Debate Timeout

**Severity:** P3 - Medium

**Alert:** `DebateTimeout` - Debate exceeded time limit

**Investigation:**

```bash
# 1. Check active debates
curl -s http://localhost:8080/api/v1/debates/active | jq

# 2. Check debate details
curl -s http://localhost:8080/api/v1/debates/<debate_id> | jq

# 3. Check agent status
curl -s http://localhost:8080/api/v1/debates/<debate_id>/agents | jq
```

**Resolution:**

```bash
# Gracefully terminate stuck debate
curl -X POST http://localhost:8080/api/v1/debates/<debate_id>/terminate

# If agent is stuck, restart agent pool
curl -X POST http://localhost:8080/api/v1/admin/agents/restart
```

---

### Agent Quota Exceeded

**Severity:** P3 - Medium

**Alert:** `AgentQuotaExceeded` - API provider quota limit reached

**Investigation:**

```bash
# 1. Check quota status
curl -s http://localhost:8080/api/v1/metrics | grep agent_quota

# 2. Check which provider is exhausted
curl -s http://localhost:8080/api/v1/admin/agents/status | jq '.providers'

# 3. Check usage over time
curl -s http://localhost:8080/api/v1/metrics | grep agent_api_calls
```

**Resolution:**

```bash
# Enable fallback to OpenRouter
curl -X POST http://localhost:8080/api/v1/admin/agents/fallback/enable

# Reduce concurrency
curl -X PATCH http://localhost:8080/api/v1/admin/config \
    -H "Content-Type: application/json" \
    -d '{"max_concurrent_agents": 3}'

# Wait for quota reset (usually hourly/daily)
```

---

## Security Alerts

### Authentication Failures Spike

**Severity:** P2 - High

**Alert:** `AuthFailuresSpike` - > 100 failures in 5 minutes

**Investigation:**

```bash
# 1. Check failure sources
grep "authentication failed" /var/log/aragora/auth.log | \
    awk '{print $NF}' | sort | uniq -c | sort -rn | head -10

# 2. Check for patterns
grep "authentication failed" /var/log/aragora/auth.log | \
    awk '{print $1, $2}' | uniq -c | sort -rn | head -20

# 3. Check specific IPs
grep "authentication failed.*<suspicious_ip>" /var/log/aragora/auth.log | wc -l
```

**Resolution:**

```bash
# Block suspicious IPs
iptables -A INPUT -s <suspicious_ip> -j DROP

# Or via application
curl -X POST http://localhost:8080/api/v1/admin/security/block \
    -H "Content-Type: application/json" \
    -d '{"ip": "<suspicious_ip>", "duration": "24h"}'

# Enable stricter rate limiting
curl -X PATCH http://localhost:8080/api/v1/admin/config \
    -H "Content-Type: application/json" \
    -d '{"auth_rate_limit": "10/minute"}'
```

---

### Rate Limit Triggered

**Severity:** P3 - Medium

**Alert:** `RateLimitTriggered` - Rate limit exceeded

**Investigation:**

```bash
# 1. Check which endpoints are being hit
grep "rate limit" /var/log/aragora/access.log | \
    awk '{print $7}' | sort | uniq -c | sort -rn | head -10

# 2. Check source IPs
grep "rate limit" /var/log/aragora/access.log | \
    awk '{print $1}' | sort | uniq -c | sort -rn | head -10

# 3. Check if legitimate traffic spike
curl -s http://localhost:8080/api/v1/metrics | grep http_requests_total
```

**Resolution:**

```bash
# If legitimate, increase limits
curl -X PATCH http://localhost:8080/api/v1/admin/config \
    -H "Content-Type: application/json" \
    -d '{"rate_limit": "1000/minute"}'

# If abuse, block source
curl -X POST http://localhost:8080/api/v1/admin/security/block \
    -H "Content-Type: application/json" \
    -d '{"ip": "<abusive_ip>", "duration": "1h"}'
```

---

### Certificate Expiring

**Severity:** P2 - High

**Alert:** `CertificateExpiringSoon` - Certificate expires in < 14 days

**Investigation:**

```bash
# 1. Check certificate expiry
echo | openssl s_client -servername aragora.example.com \
    -connect aragora.example.com:443 2>/dev/null | \
    openssl x509 -noout -dates

# 2. Check certbot status
docker compose run --rm certbot certificates
```

**Resolution:**

```bash
# Force renewal
docker compose run --rm certbot renew --force-renewal

# Reload nginx
docker compose exec nginx nginx -s reload

# Verify new certificate
echo | openssl s_client -servername aragora.example.com \
    -connect aragora.example.com:443 2>/dev/null | \
    openssl x509 -noout -dates
```

See [RUNBOOK_CERTIFICATE_ROTATION_DOCKER.md](./RUNBOOK_CERTIFICATE_ROTATION_DOCKER.md) for detailed procedures.

---

## Escalation Matrix

| Time Since Alert | Action |
|------------------|--------|
| 0 min | Primary on-call acknowledges |
| 15 min | If unresolved, page secondary on-call |
| 30 min | If unresolved, page team lead |
| 1 hour | If P1 unresolved, notify engineering manager |
| 2 hours | If P1 unresolved, notify VP Engineering |

---

## Communication Templates

### Initial Incident Report
```
[INCIDENT] <Brief Description>
Severity: P<1-4>
Status: Investigating / Identified / Monitoring / Resolved
Impact: <User-facing impact>
Start Time: <HH:MM UTC>
Next Update: <HH:MM UTC>
```

### Status Update
```
[UPDATE] <Brief Description>
Status: <Current Status>
Progress: <What's been done>
Next Steps: <What's being done>
ETA: <If known>
Next Update: <HH:MM UTC>
```

### Resolution Report
```
[RESOLVED] <Brief Description>
Duration: <X hours Y minutes>
Impact: <Summary of user impact>
Root Cause: <Brief explanation>
Resolution: <What fixed it>
Follow-up: <Postmortem ticket/link>
```

---

## Post-Incident

1. **Within 24 hours:** Create incident ticket with timeline
2. **Within 48 hours:** Complete postmortem document
3. **Within 1 week:** Action items assigned and tracked
4. **Monthly:** Review incidents for patterns

---

**Document Owner:** SRE Team
**Review Cycle:** Monthly
