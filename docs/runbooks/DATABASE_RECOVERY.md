# Database Recovery Runbook

This runbook covers database backup, restore, and disaster recovery procedures for Aragora.

## Backup Architecture

| Component | Backup Type | Frequency | Retention |
|-----------|-------------|-----------|-----------|
| PostgreSQL | Full + WAL | Continuous | 30 days |
| Redis | RDB snapshot | Hourly | 7 days |
| S3/Storage | Cross-region | Continuous | 90 days |

## Backup Verification

### Daily Health Check
```bash
# Verify PostgreSQL backups
aws s3 ls s3://aragora-backups/postgres/ --recursive | tail -5

# Check backup age
LATEST=$(aws s3 ls s3://aragora-backups/postgres/ | sort | tail -1 | awk '{print $4}')
echo "Latest backup: $LATEST"

# Verify backup integrity
pg_restore --list s3://aragora-backups/postgres/$LATEST

# Check WAL archiving
psql $DATABASE_URL -c "SELECT * FROM pg_stat_archiver"
```

## Recovery Procedures

### Scenario 1: Point-in-Time Recovery (PITR)

Use when: Data corruption, accidental deletion, need to recover to specific moment.

```bash
# 1. Stop application traffic
kubectl scale deployment/aragora-api --replicas=0

# 2. Note the target recovery time
TARGET_TIME="2026-02-03 10:30:00 UTC"

# 3. Create recovery cluster
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier aragora-prod \
  --target-db-instance-identifier aragora-recovery \
  --restore-time "$TARGET_TIME"

# 4. Wait for recovery instance
aws rds wait db-instance-available \
  --db-instance-identifier aragora-recovery

# 5. Verify data integrity
psql $RECOVERY_URL -c "SELECT COUNT(*) FROM debates"

# 6. Swap connection strings (update secrets)
kubectl patch secret aragora-db-credentials \
  -p '{"data":{"DATABASE_URL":"'$(echo -n $RECOVERY_URL | base64)'"}}'

# 7. Restart application
kubectl scale deployment/aragora-api --replicas=3
```

### Scenario 2: Full Restore from Backup

Use when: Complete database loss, regional outage.

```bash
# 1. Download latest backup
aws s3 cp s3://aragora-backups/postgres/latest.dump /tmp/

# 2. Create new database instance
aws rds create-db-instance \
  --db-instance-identifier aragora-new \
  --db-instance-class db.r5.large \
  --engine postgres \
  --allocated-storage 100

# 3. Wait for instance
aws rds wait db-instance-available \
  --db-instance-identifier aragora-new

# 4. Restore backup
pg_restore -h $NEW_HOST -U postgres -d aragora /tmp/latest.dump

# 5. Replay WAL logs if available
# (Handled automatically by RDS if WAL archiving enabled)

# 6. Verify restore
psql $NEW_URL -c "SELECT MAX(created_at) FROM debates"
```

### Scenario 3: Redis Recovery

```bash
# 1. Check Redis status
redis-cli info replication

# 2. If Redis is down, restart from RDB
# Option A: Restore from snapshot
aws elasticache restore-replica-group-from-snapshot \
  --replication-group-id aragora-redis-new \
  --snapshot-name aragora-redis-latest

# Option B: Restart with persistence disabled temporarily
kubectl delete pod redis-0
kubectl apply -f k8s/redis-recovery.yaml

# 3. Warm cache after recovery
python scripts/warm_cache.py --all
```

## Data Verification Queries

```sql
-- Check debate counts by date
SELECT DATE(created_at), COUNT(*)
FROM debates
GROUP BY DATE(created_at)
ORDER BY 1 DESC
LIMIT 10;

-- Check for data gaps
SELECT
  d1.created_at as gap_start,
  d2.created_at as gap_end,
  d2.created_at - d1.created_at as gap_duration
FROM debates d1
JOIN debates d2 ON d2.id = (
  SELECT MIN(id) FROM debates WHERE id > d1.id
)
WHERE d2.created_at - d1.created_at > INTERVAL '1 hour';

-- Verify foreign key integrity
SELECT COUNT(*)
FROM debate_messages
WHERE debate_id NOT IN (SELECT id FROM debates);

-- Check knowledge mound integrity
SELECT COUNT(*), node_type
FROM knowledge_nodes
GROUP BY node_type;
```

## Emergency Procedures

### Database Failover (RDS Multi-AZ)

```bash
# Force failover to standby
aws rds reboot-db-instance \
  --db-instance-identifier aragora-prod \
  --force-failover

# Monitor failover
aws rds describe-events \
  --source-identifier aragora-prod \
  --source-type db-instance \
  --duration 60
```

### Connection Pool Exhaustion

```bash
# Check active connections
psql $DATABASE_URL -c "
SELECT
  state,
  COUNT(*)
FROM pg_stat_activity
GROUP BY state;"

# Kill idle connections
psql $DATABASE_URL -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
  AND query_start < NOW() - INTERVAL '10 minutes';"

# Restart connection pooler (PgBouncer)
kubectl rollout restart deployment/pgbouncer
```

## Backup Testing Schedule

| Test | Frequency | Owner |
|------|-----------|-------|
| Restore to staging | Weekly | DevOps |
| Full DR drill | Quarterly | Engineering |
| WAL replay test | Monthly | DBA |

## Recovery Time Objectives

| Scenario | RTO | RPO |
|----------|-----|-----|
| Primary failover | 5 min | 0 (sync replication) |
| PITR | 30 min | 5 min |
| Full restore | 2 hours | 1 hour |
| Cross-region DR | 4 hours | 1 hour |

---
*Last updated: 2026-02-03*
