# Runbook: Database Issues

**Alerts:** `DatabaseConnectionExhausted`, `DatabaseHighLatency`, `DatabaseDiskFull`
**Severity:** Critical/Warning
**Dependencies:** PostgreSQL

## Symptoms

- Connection errors in logs
- Queries timing out
- "too many connections" errors
- Slow API responses

## Diagnosis

### 1. Check connectivity

```bash
# Basic connection test
psql $DATABASE_URL -c "SELECT 1"

# Check connection count
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"

# Check max connections
psql $DATABASE_URL -c "SHOW max_connections"
```

### 2. Check active queries

```bash
# Long-running queries
psql $DATABASE_URL -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '30 seconds'
AND state != 'idle'
ORDER BY duration DESC;"
```

### 3. Check disk space

```bash
# Database size
psql $DATABASE_URL -c "SELECT pg_size_pretty(pg_database_size(current_database()))"

# Table sizes
psql $DATABASE_URL -c "
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;"
```

## Common Causes

| Cause | Indicators | Fix |
|-------|------------|-----|
| Connection pool exhausted | Max connections reached | Restart app, tune pool |
| Long-running query | Query blocking others | Kill query |
| Disk full | Write errors | Clean up, expand storage |
| Missing index | Slow queries | Add index |
| Lock contention | Queries waiting | Identify and kill blocker |

## Resolution Steps

### Connection pool exhausted

```bash
# Check Aragora pool settings
grep -i pool /etc/aragora/config.yaml

# Restart to reset connections
systemctl restart aragora

# If persistent, reduce pool size temporarily
export ARAGORA_DB_POOL_SIZE=5
```

### Kill long-running query

```bash
# Find the PID
psql $DATABASE_URL -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC
LIMIT 5;"

# Kill it (replace PID)
psql $DATABASE_URL -c "SELECT pg_terminate_backend(12345)"
```

### Disk space cleanup

```bash
# Vacuum to reclaim space
psql $DATABASE_URL -c "VACUUM FULL VERBOSE"

# Delete old audit logs (if safe)
psql $DATABASE_URL -c "DELETE FROM audit_log WHERE timestamp < NOW() - INTERVAL '90 days'"
```

### Add missing index

```bash
# Identify slow queries
psql $DATABASE_URL -c "
SELECT query, calls, total_time/calls as avg_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;"

# Add index (example)
psql $DATABASE_URL -c "CREATE INDEX CONCURRENTLY idx_debates_org_created ON debates(org_id, created_at DESC)"
```

## Escalation

If database is unresponsive:
1. Contact DBA/platform team
2. Consider failover to replica (if available)
3. Prepare for potential data loss communication

## Prevention

- Monitor connection pool utilization
- Set up disk space alerts at 80%
- Regular VACUUM ANALYZE via cron
- Query performance monitoring
