# Backup Automation & Validation Runbook

**Purpose:** Automated backup scheduling, validation, and recovery testing.
**Audience:** DevOps, SRE, Platform Engineers
**Last Updated:** January 2026

---

## Overview

This runbook covers:
- Automated backup scheduling for PostgreSQL and Redis
- Backup validation and integrity testing
- Offsite replication to S3/GCS
- Recovery testing procedures
- Retention policies for compliance (SOC 2, GDPR)

---

## Backup Strategy

### Backup Types

| Type | Frequency | Retention | RTO | RPO |
|------|-----------|-----------|-----|-----|
| Full PostgreSQL | Daily 2 AM | 30 days | 4 hours | 24 hours |
| Incremental WAL | Continuous | 7 days | 15 min | 5 min |
| Redis RDB | Every 6 hours | 7 days | 30 min | 6 hours |
| Redis AOF | Continuous | 24 hours | 5 min | ~1 sec |
| Application Config | On change | Forever | 5 min | 0 |

### Backup Locations

```
/backup/
├── postgresql/
│   ├── daily/           # pg_dump full backups
│   ├── wal/             # WAL archive
│   └── basebackup/      # pg_basebackup for PITR
├── redis/
│   ├── rdb/             # RDB snapshots
│   └── aof/             # AOF files
└── config/
    └── snapshots/       # Application configuration
```

---

## Phase 1: PostgreSQL Backup Automation

### 1.1 Configure WAL Archiving

```bash
# /etc/postgresql/15/main/postgresql.conf

archive_mode = on
archive_command = 'test ! -f /backup/postgresql/wal/%f && cp %p /backup/postgresql/wal/%f'
archive_timeout = 300  # 5 minutes max between archives

# For S3 archiving (using wal-g)
# archive_command = 'wal-g wal-push %p'
```

### 1.2 Daily Backup Script

```bash
#!/bin/bash
# /usr/local/bin/backup_postgresql.sh

set -euo pipefail

# Configuration
BACKUP_DIR="/backup/postgresql/daily"
RETENTION_DAYS=30
DATABASE="aragora"
S3_BUCKET="aragora-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/${DATABASE}_${DATE}.dump"

# Logging
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Starting PostgreSQL backup..."

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Full backup with compression
pg_dump -Fc -Z 9 -d "${DATABASE}" -f "${BACKUP_FILE}"

# Verify backup
pg_restore -l "${BACKUP_FILE}" > /dev/null
if [ $? -eq 0 ]; then
    log "Backup verified successfully: ${BACKUP_FILE}"
else
    log "ERROR: Backup verification failed"
    exit 1
fi

# Get backup size
SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
log "Backup size: ${SIZE}"

# Upload to S3
if [ -n "${S3_BUCKET}" ]; then
    aws s3 cp "${BACKUP_FILE}" "s3://${S3_BUCKET}/postgresql/daily/"
    log "Uploaded to S3: s3://${S3_BUCKET}/postgresql/daily/"
fi

# Cleanup old backups (local)
find "${BACKUP_DIR}" -name "*.dump" -mtime +${RETENTION_DAYS} -delete
log "Cleaned up backups older than ${RETENTION_DAYS} days"

# Report metrics
echo "backup_postgresql_success{database=\"${DATABASE}\"} 1" | curl --data-binary @- http://pushgateway:9091/metrics/job/backup

log "Backup completed successfully"
```

### 1.3 Cron Schedule

```bash
# /etc/cron.d/aragora-backup

# Daily full backup at 2 AM
0 2 * * * postgres /usr/local/bin/backup_postgresql.sh >> /var/log/aragora/backup.log 2>&1

# WAL archive cleanup (keep 7 days)
0 4 * * * postgres find /backup/postgresql/wal -mtime +7 -delete

# Base backup for PITR (weekly)
0 3 * * 0 postgres pg_basebackup -D /backup/postgresql/basebackup -Ft -z -P
```

### 1.4 Point-in-Time Recovery Setup

```bash
#!/bin/bash
# /usr/local/bin/setup_pitr.sh

# Create base backup
pg_basebackup \
    -D /backup/postgresql/basebackup \
    -Ft \
    -z \
    -P \
    --checkpoint=fast \
    --wal-method=stream

# Create recovery.conf for PITR
cat > /backup/postgresql/recovery.conf << EOF
restore_command = 'cp /backup/postgresql/wal/%f %p'
recovery_target_time = '2026-01-27 10:00:00'
recovery_target_action = 'promote'
EOF
```

---

## Phase 2: Redis Backup Automation

### 2.1 Configure Redis Persistence

```bash
# /etc/redis/redis.conf

# RDB snapshots
save 900 1      # Save if 1 key changed in 15 min
save 300 10     # Save if 10 keys changed in 5 min
save 60 10000   # Save if 10000 keys changed in 1 min

dbfilename dump.rdb
dir /var/lib/redis

# AOF persistence
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### 2.2 Redis Backup Script

```bash
#!/bin/bash
# /usr/local/bin/backup_redis.sh

set -euo pipefail

BACKUP_DIR="/backup/redis/rdb"
RETENTION_DAYS=7
S3_BUCKET="aragora-backups"
DATE=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Starting Redis backup..."

mkdir -p "${BACKUP_DIR}"

# Trigger RDB save
redis-cli BGSAVE
sleep 5

# Wait for save to complete
while [ "$(redis-cli LASTSAVE)" == "$(redis-cli LASTSAVE)" ]; do
    sleep 1
done

# Copy RDB file
cp /var/lib/redis/dump.rdb "${BACKUP_DIR}/dump_${DATE}.rdb"

# Verify backup
redis-check-rdb "${BACKUP_DIR}/dump_${DATE}.rdb"
if [ $? -eq 0 ]; then
    log "RDB backup verified"
else
    log "ERROR: RDB verification failed"
    exit 1
fi

# Upload to S3
aws s3 cp "${BACKUP_DIR}/dump_${DATE}.rdb" "s3://${S3_BUCKET}/redis/"

# Cleanup old backups
find "${BACKUP_DIR}" -name "*.rdb" -mtime +${RETENTION_DAYS} -delete

log "Redis backup completed"
```

### 2.3 AOF Backup (for minimal data loss)

```bash
#!/bin/bash
# /usr/local/bin/backup_redis_aof.sh

# Rewrite AOF to compact it
redis-cli BGREWRITEAOF

# Wait for rewrite
while redis-cli INFO persistence | grep -q "aof_rewrite_in_progress:1"; do
    sleep 1
done

# Copy AOF file
cp /var/lib/redis/appendonly.aof "/backup/redis/aof/appendonly_$(date +%Y%m%d_%H%M%S).aof"
```

---

## Phase 3: Backup Validation

### 3.1 Automated Validation Script

```bash
#!/bin/bash
# /usr/local/bin/validate_backups.sh

set -euo pipefail

VALIDATION_DB="aragora_validation"
VALIDATION_DIR="/tmp/backup_validation"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
alert() {
    log "ALERT: $1"
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Backup Validation Alert: $1\"}" \
            "${SLACK_WEBHOOK}"
    fi
}

log "Starting backup validation..."

# Get latest backup
LATEST_BACKUP=$(ls -t /backup/postgresql/daily/*.dump | head -1)
log "Validating: ${LATEST_BACKUP}"

# Create validation directory
rm -rf "${VALIDATION_DIR}"
mkdir -p "${VALIDATION_DIR}"

# Drop and recreate validation database
psql -U postgres -c "DROP DATABASE IF EXISTS ${VALIDATION_DB};"
psql -U postgres -c "CREATE DATABASE ${VALIDATION_DB};"

# Restore backup to validation database
pg_restore -d "${VALIDATION_DB}" "${LATEST_BACKUP}"
if [ $? -ne 0 ]; then
    alert "PostgreSQL backup restore failed: ${LATEST_BACKUP}"
    exit 1
fi

# Validate data integrity
DEBATE_COUNT=$(psql -U postgres -d "${VALIDATION_DB}" -t -c "SELECT COUNT(*) FROM debates;")
USER_COUNT=$(psql -U postgres -d "${VALIDATION_DB}" -t -c "SELECT COUNT(*) FROM users;")
AUDIT_COUNT=$(psql -U postgres -d "${VALIDATION_DB}" -t -c "SELECT COUNT(*) FROM audit_logs;")

log "Restored counts: debates=${DEBATE_COUNT}, users=${USER_COUNT}, audit_logs=${AUDIT_COUNT}"

# Compare with production (optional)
PROD_DEBATES=$(psql -U postgres -d aragora -t -c "SELECT COUNT(*) FROM debates;")
DIFF=$((PROD_DEBATES - DEBATE_COUNT))

if [ $DIFF -gt 100 ]; then
    alert "Backup may be stale: ${DIFF} debates missing"
fi

# Cleanup
psql -U postgres -c "DROP DATABASE ${VALIDATION_DB};"

# Redis validation
LATEST_RDB=$(ls -t /backup/redis/rdb/*.rdb | head -1)
redis-check-rdb "${LATEST_RDB}"
if [ $? -ne 0 ]; then
    alert "Redis RDB validation failed: ${LATEST_RDB}"
    exit 1
fi

log "Backup validation completed successfully"

# Report metrics
cat << EOF | curl --data-binary @- http://pushgateway:9091/metrics/job/backup_validation
backup_validation_success 1
backup_validation_debates ${DEBATE_COUNT}
backup_validation_users ${USER_COUNT}
EOF
```

### 3.2 Validation Schedule

```bash
# /etc/cron.d/aragora-backup-validation

# Daily validation at 6 AM (after 2 AM backup)
0 6 * * * postgres /usr/local/bin/validate_backups.sh >> /var/log/aragora/backup_validation.log 2>&1

# Weekly full restore test (Sunday 4 AM)
0 4 * * 0 postgres /usr/local/bin/full_restore_test.sh >> /var/log/aragora/restore_test.log 2>&1
```

---

## Phase 4: Offsite Replication

### 4.1 S3 Sync Configuration

```bash
# /etc/aragora/backup-s3.conf

S3_BUCKET="aragora-backups"
S3_REGION="us-east-1"
S3_PREFIX="production"
ENCRYPTION="AES256"
STORAGE_CLASS="STANDARD_IA"  # Infrequent access for cost savings
```

### 4.2 S3 Sync Script

```bash
#!/bin/bash
# /usr/local/bin/sync_backups_s3.sh

source /etc/aragora/backup-s3.conf

# Sync PostgreSQL backups
aws s3 sync /backup/postgresql/ "s3://${S3_BUCKET}/${S3_PREFIX}/postgresql/" \
    --storage-class "${STORAGE_CLASS}" \
    --sse "${ENCRYPTION}" \
    --delete

# Sync Redis backups
aws s3 sync /backup/redis/ "s3://${S3_BUCKET}/${S3_PREFIX}/redis/" \
    --storage-class "${STORAGE_CLASS}" \
    --sse "${ENCRYPTION}" \
    --delete

# Verify sync
aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --recursive --summarize
```

### 4.3 Cross-Region Replication

```bash
# AWS S3 bucket replication rule (via Terraform)
resource "aws_s3_bucket_replication_configuration" "backup" {
  bucket = aws_s3_bucket.backups.id
  role   = aws_iam_role.replication.arn

  rule {
    id     = "cross-region"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.backups_dr.arn
      storage_class = "GLACIER"
    }
  }
}
```

---

## Phase 5: Retention Policies

### 5.1 Retention Schedule

| Backup Type | Daily | Weekly | Monthly | Yearly |
|-------------|-------|--------|---------|--------|
| PostgreSQL Full | 7 days | 4 weeks | 12 months | 7 years |
| PostgreSQL WAL | 7 days | - | - | - |
| Redis RDB | 7 days | 4 weeks | - | - |
| Config Snapshots | 30 days | Forever | - | - |

### 5.2 Lifecycle Policy (S3)

```json
{
  "Rules": [
    {
      "ID": "PostgreSQLRetention",
      "Prefix": "postgresql/daily/",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 2555
      }
    },
    {
      "ID": "RedisRetention",
      "Prefix": "redis/",
      "Status": "Enabled",
      "Expiration": {
        "Days": 30
      }
    }
  ]
}
```

### 5.3 Compliance Requirements

| Regulation | Requirement | Implementation |
|------------|-------------|----------------|
| SOC 2 | Audit trail retention | 7 year PostgreSQL retention |
| GDPR | Data deletion capability | Backup rotation with right-to-erasure |
| HIPAA | Encrypted backups | S3 SSE-KMS encryption |

---

## Recovery Procedures

### Quick Recovery (< 15 min)

```bash
# Restore from latest backup
pg_restore -d aragora /backup/postgresql/daily/latest.dump

# Or point-in-time recovery
pg_restore -d aragora /backup/postgresql/basebackup/base.tar.gz
# Then apply WAL logs
```

### Full Recovery Procedure

See [DISASTER_RECOVERY.md](../DISASTER_RECOVERY.md) for complete procedures.

---

## Monitoring & Alerting

### Prometheus Metrics

```yaml
# Backup metrics to track
backup_last_success_timestamp{type="postgresql"}
backup_last_success_timestamp{type="redis"}
backup_size_bytes{type="postgresql"}
backup_validation_success
backup_s3_sync_success
```

### Alert Rules

```yaml
groups:
  - name: backup_alerts
    rules:
      - alert: BackupFailed
        expr: time() - backup_last_success_timestamp > 90000  # 25 hours
        labels:
          severity: critical
        annotations:
          summary: "Backup has not completed in 25 hours"

      - alert: BackupValidationFailed
        expr: backup_validation_success == 0
        labels:
          severity: warning
        annotations:
          summary: "Backup validation failed"
```

---

## Troubleshooting

### Backup Too Slow

```bash
# Check I/O utilization
iostat -x 1

# Use parallel dump
pg_dump -Fd -j 4 -d aragora -f /backup/postgresql/parallel/

# Compress with faster algorithm
pg_dump -Fc -Z 1 -d aragora  # Level 1 compression
```

### Backup Too Large

```bash
# Check table sizes
psql -d aragora -c "
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;"

# Exclude large tables if acceptable
pg_dump -T large_table -d aragora
```

### S3 Upload Failures

```bash
# Check AWS credentials
aws sts get-caller-identity

# Test upload
aws s3 cp test.txt s3://aragora-backups/test/

# Check bucket policy
aws s3api get-bucket-policy --bucket aragora-backups
```

---

**Document Owner:** Platform Team
**Review Cycle:** Quarterly
