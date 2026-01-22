---
title: Database Issues Runbook
description: Database Issues Runbook
---

# Database Issues Runbook

Procedures for handling database-related issues in Aragora.

## Quick Diagnosis

```bash
# Check database status via API
curl -s http://localhost:8080/api/health/detailed | jq .components

# Check disk space
df -h

# Check database file size (SQLite)
ls -lh /app/data/*.db
```

## Common Issues

### 1. Database Locked

**Symptoms:**
- "database is locked" errors
- Timeouts on write operations
- Slow queries

**Resolution:**

```bash
# 1. Check for long-running queries (if using PostgreSQL)
# For SQLite, check for other processes holding locks:
sudo lsof /app/data/aragora.db

# 2. Restart the service to release locks
sudo systemctl restart aragora

# 3. If persists, check for zombie processes
ps aux | grep aragora
sudo kill -9 <zombie_pid>
```

### 2. Database Corruption

**Symptoms:**
- "malformed database schema" errors
- "database disk image is malformed"
- Query results are incorrect or partial

**Resolution:**

```bash
# 1. Stop the service
sudo systemctl stop aragora

# 2. Backup the corrupted database
cp /app/data/aragora.db /app/data/aragora.db.corrupted.$(date +%Y%m%d)

# 3. Try integrity check
sqlite3 /app/data/aragora.db "PRAGMA integrity_check;"

# 4. Attempt recovery
sqlite3 /app/data/aragora.db ".recover" | sqlite3 /app/data/aragora_recovered.db

# 5. If recovery succeeds, replace original
mv /app/data/aragora_recovered.db /app/data/aragora.db

# 6. Restart service
sudo systemctl start aragora
```

### 3. Disk Space Full

**Symptoms:**
- "disk full" or "no space left on device" errors
- Service crashes on write
- Database operations fail silently

**Resolution:**

```bash
# 1. Check disk usage
df -h
du -sh /app/data/*

# 2. Clean old logs
sudo journalctl --vacuum-time=7d

# 3. Run database maintenance (vacuum)
curl -X GET "http://localhost:8080/api/system/maintenance?task=vacuum"

# 4. Remove old backups
ls -la /app/data/backups/
rm /app/data/backups/aragora.db.backup.*  # Keep recent ones

# 5. If using WAL mode, checkpoint
sqlite3 /app/data/aragora.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

### 4. Connection Pool Exhaustion

**Symptoms:**
- "too many connections" errors
- Requests timing out
- Memory usage climbing

**Resolution:**

```bash
# 1. Check current connections
curl -s http://localhost:8080/api/health/detailed | jq .database

# 2. Restart to reset pool
sudo systemctl restart aragora

# 3. If persistent, increase pool size in config:
# Set DATABASE_POOL_SIZE environment variable
```

### 5. Supabase Connection Issues

**Symptoms:**
- "connection refused" to Supabase
- "authentication failed"
- Timeout errors on persistence operations

**Resolution:**

```bash
# 1. Check Supabase status
curl -s https://status.supabase.com/api/v2/status.json | jq .status

# 2. Test connection
curl -H "apikey: $SUPABASE_KEY" \
  -H "Authorization: Bearer $SUPABASE_KEY" \
  "$SUPABASE_URL/rest/v1/"

# 3. Verify credentials
echo "URL: $SUPABASE_URL"
echo "Key length: ${#SUPABASE_KEY}"

# 4. Check network connectivity
nc -zv ${SUPABASE_URL#https://} 443

# 5. If credentials rotated, update:
# - Update .env file
# - Update GitHub secrets
# - Update systemd environment
sudo systemctl daemon-reload
sudo systemctl restart aragora
```

## Database Maintenance

### Scheduled Maintenance

```bash
# Run full maintenance via API
curl -X GET "http://localhost:8080/api/system/maintenance?task=full"

# Individual tasks:
curl -X GET "http://localhost:8080/api/system/maintenance?task=vacuum"
curl -X GET "http://localhost:8080/api/system/maintenance?task=analyze"
curl -X GET "http://localhost:8080/api/system/maintenance?task=checkpoint"
```

### Manual SQLite Maintenance

```bash
# Vacuum (reclaim space)
sqlite3 /app/data/aragora.db "VACUUM;"

# Analyze (update statistics)
sqlite3 /app/data/aragora.db "ANALYZE;"

# Integrity check
sqlite3 /app/data/aragora.db "PRAGMA integrity_check;"

# WAL checkpoint
sqlite3 /app/data/aragora.db "PRAGMA wal_checkpoint(TRUNCATE);"
```

## Backup and Restore

### Create Backup

```bash
# Hot backup (safe while service running)
sqlite3 /app/data/aragora.db ".backup /app/data/backups/aragora.$(date +%Y%m%d_%H%M%S).db"

# Or via API
curl -X POST http://localhost:8080/api/admin/backup
```

### Restore from Backup

```bash
# 1. Stop service
sudo systemctl stop aragora

# 2. Backup current (corrupted) database
mv /app/data/aragora.db /app/data/aragora.db.bad

# 3. Restore from backup
cp /app/data/backups/aragora.YYYYMMDD_HHMMSS.db /app/data/aragora.db

# 4. Set permissions
chown aragora:aragora /app/data/aragora.db

# 5. Start service
sudo systemctl start aragora

# 6. Verify
curl http://localhost:8080/api/health
```

## Monitoring Queries

### Check Table Sizes

```sql
SELECT
    name,
    (SELECT COUNT(*) FROM debates) as count
FROM sqlite_master
WHERE type='table' AND name='debates';
```

### Check Recent Activity

```sql
-- Recent debates
SELECT id, task, created_at
FROM debates
ORDER BY created_at DESC
LIMIT 10;

-- ELO updates
SELECT agent, rating, updated_at
FROM agent_ratings
ORDER BY updated_at DESC
LIMIT 10;
```

## Escalation

If database issues persist after following this runbook:

1. **Check logs** for detailed error messages:
   ```bash
   sudo journalctl -u aragora -n 500 --no-pager | grep -i "database\|sqlite\|supabase"
   ```

2. **Create backup** before attempting risky operations

3. **Consider failover** to read-only mode if write corruption suspected

4. **Contact support** with:
   - Error messages
   - Database file size
   - Disk usage stats
   - Recent changes to configuration

## Related Runbooks

- [Deployment](../RUNBOOK_DEPLOYMENT.md)
- [Incident Response](../RUNBOOK_INCIDENT.md)
- [Provider Failure](../RUNBOOK_PROVIDER_FAILURE.md)
