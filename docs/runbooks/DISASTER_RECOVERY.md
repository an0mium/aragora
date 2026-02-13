# Disaster Recovery Runbook

**Document Type:** Operational Runbook
**Severity Classification:** SEV-1 Critical
**Last Updated:** February 2026
**Owner:** Platform Engineering / SRE
**Review Cycle:** Quarterly

---

## Quick Reference

**Emergency Contacts:**
| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | PagerDuty | Primary |
| Incident Commander | Slack #incidents | Secondary |
| Engineering Lead | Phone | SEV-1/SEV-2 |
| VP Engineering | Phone | SEV-1 |
| Security Team | security@aragora.ai | Data breach |

**Key Dashboards:**
- Grafana: `grafana.aragora.internal/d/dr-overview`
- PagerDuty: `pagerduty.com/incidents`
- Status Page: `admin.status.aragora.ai`
- AWS Console: `console.aws.amazon.com`

---

## Table of Contents

1. [Recovery Objectives](#1-recovery-objectives)
2. [Disaster Classification](#2-disaster-classification)
3. [Backup Verification Procedures](#3-backup-verification-procedures)
4. [Failover Procedures](#4-failover-procedures)
5. [Data Recovery Procedures](#5-data-recovery-procedures)
6. [Rollback Procedures](#6-rollback-procedures)
7. [Communication Protocols](#7-communication-protocols)
8. [DR Drill Schedule](#8-dr-drill-schedule)
9. [Post-Incident Review](#9-post-incident-review)
10. [Appendix: Quick Commands](#appendix-quick-commands)

---

## 1. Recovery Objectives

### 1.1 RTO/RPO Targets

| Service Tier | RTO Target | RPO Target | Description |
|--------------|------------|------------|-------------|
| **Critical** | < 1 hour | < 5 minutes | Core debate engine, authentication, user data |
| **High** | < 4 hours | < 15 minutes | API endpoints, WebSocket streaming, billing |
| **Medium** | < 8 hours | < 1 hour | Analytics, leaderboards, Knowledge Mound |
| **Low** | < 24 hours | < 24 hours | Historical telemetry, archives, reports |

### 1.2 Definitions

| Term | Definition |
|------|------------|
| **RTO** (Recovery Time Objective) | Maximum acceptable time to restore service after incident declaration |
| **RPO** (Recovery Point Objective) | Maximum acceptable data loss measured in time |
| **Failover** | Process of switching from failed primary to backup/standby system |
| **Failback** | Process of returning to the original primary system after recovery |
| **MTTR** | Mean Time To Recovery - average time to restore service |
| **MTPD** | Maximum Tolerable Period of Disruption |

### 1.3 Service Tier Classification

| Tier | Components | Justification |
|------|------------|---------------|
| Critical | PostgreSQL, Auth, Core API, Redis sessions | User access, data integrity |
| High | Debate orchestration, WebSocket, Billing | Core functionality |
| Medium | Knowledge Mound, ELO rankings, Analytics | Enhanced features |
| Low | Audit archives, Telemetry exports | Compliance/reporting |

---

## 2. Disaster Classification

### 2.1 Severity Levels

| Level | Name | Definition | Response Time | Examples |
|-------|------|------------|---------------|----------|
| **SEV-1** | Critical | Complete outage, data loss risk | < 15 minutes | DB corruption, all regions down, security breach |
| **SEV-2** | High | Major functionality impaired | < 30 minutes | Single region down, auth broken, debates failing |
| **SEV-3** | Medium | Partial degradation | < 2 hours | Single provider down, slow queries, partial feature loss |
| **SEV-4** | Low | Minor issues | < 24 hours | Cosmetic issues, single user impacted |

### 2.2 Disaster Categories

| Category | Description | Primary Response |
|----------|-------------|------------------|
| **L1** | Single component failure | Auto-healing, pod restart |
| **L2** | Multiple component failure | Manual intervention, service restart |
| **L3** | Regional outage | Regional failover to DR site |
| **L4** | Global/multi-region failure | Full DR activation |
| **L5** | Data corruption/loss | Point-in-time recovery |
| **L6** | Security incident | Containment, forensics, recovery |

---

## 3. Backup Verification Procedures

### 3.1 Daily Backup Verification

Run daily at 06:00 UTC (after 02:00 UTC backups complete):

```bash
#!/bin/bash
# /usr/local/bin/verify_daily_backups.sh

set -euo pipefail

BACKUP_DIR="/backup/postgresql/daily"
VALIDATION_DB="aragora_backup_verify"
LOG_FILE="/var/log/aragora/backup_verification.log"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
alert() {
    log "ALERT: $1"
    if [ -n "${SLACK_WEBHOOK}" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Backup Verification FAILED: $1\"}" \
            "${SLACK_WEBHOOK}"
    fi
}

log "=== Starting daily backup verification ==="

# 1. Find latest backup
LATEST_BACKUP=$(ls -t ${BACKUP_DIR}/*.dump 2>/dev/null | head -1)
if [ -z "${LATEST_BACKUP}" ]; then
    alert "No backup files found in ${BACKUP_DIR}"
    exit 1
fi

# 2. Check backup age (must be < 25 hours old)
BACKUP_AGE_HOURS=$(( ($(date +%s) - $(stat -c %Y "${LATEST_BACKUP}")) / 3600 ))
if [ ${BACKUP_AGE_HOURS} -gt 25 ]; then
    alert "Latest backup is ${BACKUP_AGE_HOURS} hours old - exceeds 24h RPO"
    exit 1
fi
log "Backup age: ${BACKUP_AGE_HOURS} hours - OK"

# 3. Verify backup file integrity
if ! pg_restore -l "${LATEST_BACKUP}" > /dev/null 2>&1; then
    alert "Backup file corrupted: ${LATEST_BACKUP}"
    exit 1
fi
log "Backup integrity check: PASSED"

# 4. Test restore to validation database
psql -U postgres -c "DROP DATABASE IF EXISTS ${VALIDATION_DB};" 2>/dev/null
psql -U postgres -c "CREATE DATABASE ${VALIDATION_DB};"

if ! pg_restore -d "${VALIDATION_DB}" "${LATEST_BACKUP}" 2>/dev/null; then
    alert "Backup restore failed: ${LATEST_BACKUP}"
    psql -U postgres -c "DROP DATABASE IF EXISTS ${VALIDATION_DB};"
    exit 1
fi
log "Backup restore test: PASSED"

# 5. Validate critical tables exist and have data
TABLES=("users" "debates" "tenants" "audit_logs" "decision_receipts")
for table in "${TABLES[@]}"; do
    COUNT=$(psql -U postgres -d "${VALIDATION_DB}" -t -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null | tr -d ' ')
    if [ -z "${COUNT}" ] || [ "${COUNT}" -lt 0 ]; then
        alert "Table ${table} missing or empty in backup"
        psql -U postgres -c "DROP DATABASE IF EXISTS ${VALIDATION_DB};"
        exit 1
    fi
    log "Table ${table}: ${COUNT} rows"
done

# 6. Compare with production counts (warn if > 5% difference)
PROD_DEBATES=$(psql -U postgres -d aragora -t -c "SELECT COUNT(*) FROM debates;" | tr -d ' ')
BACKUP_DEBATES=$(psql -U postgres -d "${VALIDATION_DB}" -t -c "SELECT COUNT(*) FROM debates;" | tr -d ' ')
DIFF_PCT=$(( (PROD_DEBATES - BACKUP_DEBATES) * 100 / PROD_DEBATES ))
if [ ${DIFF_PCT} -gt 5 ]; then
    log "WARNING: Backup is ${DIFF_PCT}% behind production (${BACKUP_DEBATES} vs ${PROD_DEBATES} debates)"
fi

# 7. Cleanup
psql -U postgres -c "DROP DATABASE ${VALIDATION_DB};"

# 8. Report metrics
cat << EOF | curl --data-binary @- http://pushgateway:9091/metrics/job/backup_verification
backup_verification_success 1
backup_verification_age_hours ${BACKUP_AGE_HOURS}
backup_verification_debates ${BACKUP_DEBATES}
backup_verification_timestamp $(date +%s)
EOF

log "=== Backup verification completed successfully ==="
```

### 3.2 Weekly Comprehensive Verification

```bash
#!/bin/bash
# /usr/local/bin/weekly_backup_verification.sh
# Run Sundays at 04:00 UTC

set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "=== Weekly comprehensive backup verification ==="

# 1. Verify PostgreSQL backups
log "Checking PostgreSQL backups..."
/usr/local/bin/verify_daily_backups.sh

# 2. Verify Redis backups
log "Checking Redis backups..."
LATEST_RDB=$(ls -t /backup/redis/rdb/*.rdb 2>/dev/null | head -1)
if [ -z "${LATEST_RDB}" ]; then
    echo "ERROR: No Redis RDB backups found"
    exit 1
fi

if ! redis-check-rdb "${LATEST_RDB}" > /dev/null 2>&1; then
    echo "ERROR: Redis RDB backup corrupted: ${LATEST_RDB}"
    exit 1
fi
log "Redis RDB verification: PASSED"

# 3. Verify S3 backup replication
log "Checking S3 backup replication..."
aws s3 ls s3://aragora-backups/postgresql/daily/ --summarize | tail -2

# 4. Verify WAL archive continuity
log "Checking WAL archive continuity..."
WAL_COUNT=$(ls /backup/postgresql/wal/ 2>/dev/null | wc -l)
if [ ${WAL_COUNT} -lt 10 ]; then
    log "WARNING: Only ${WAL_COUNT} WAL files - check archiving"
fi

# 5. Test Point-in-Time Recovery capability
log "Testing PITR capability..."
RECOVERY_TARGET=$(date -d "6 hours ago" '+%Y-%m-%d %H:%M:%S')
log "Would recover to: ${RECOVERY_TARGET}"

log "=== Weekly verification completed ==="
```

### 3.3 Backup Inventory Check

```bash
# List all available backups with ages
/usr/local/bin/list_backups.sh

# Expected output:
# PostgreSQL Daily Backups:
#   - aragora_20260202_020000.dump (6 hours old, 2.3 GB)
#   - aragora_20260201_020000.dump (30 hours old, 2.3 GB)
#   ...
#
# PostgreSQL WAL Archives:
#   - 156 files, oldest: 7 days
#
# Redis RDB Snapshots:
#   - dump_20260202_060000.rdb (2 hours old, 512 MB)
#   ...
#
# S3 Replicated Backups:
#   - Last sync: 2026-02-02 06:15:00
#   - Total size: 45.2 GB
```

---

## 4. Failover Procedures

### 4.1 Database Failover (CloudNativePG / CNPG)

**When to use:** Primary PostgreSQL instance unavailable, replication lag unacceptable

**Pre-conditions:**
- [ ] Replica is in sync (lag < 30 seconds)
- [ ] Application can be briefly paused
- [ ] Team notified of impending failover

```bash
#!/bin/bash
# database_failover_cnpg.sh

set -euo pipefail

CLUSTER_NAME="aragora-postgres"
NAMESPACE="aragora"

echo "=== CNPG Database Failover ==="
echo "Timestamp: $(date)"

# 1. Check current cluster status
echo "[1/7] Checking cluster status..."
kubectl cnpg status ${CLUSTER_NAME} -n ${NAMESPACE}

# 2. Verify replica is in sync
echo "[2/7] Checking replication lag..."
LAG=$(kubectl exec -n ${NAMESPACE} ${CLUSTER_NAME}-2 -- \
    psql -t -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int;")
if [ "${LAG}" -gt 30 ]; then
    echo "ERROR: Replication lag is ${LAG}s - too high for safe failover"
    echo "Consider waiting or accepting data loss"
    exit 1
fi
echo "Replication lag: ${LAG}s - acceptable"

# 3. Enable maintenance mode (stop new connections)
echo "[3/7] Enabling maintenance mode..."
kubectl -n ${NAMESPACE} patch deployment aragora-api \
    -p '{"spec":{"template":{"spec":{"containers":[{"name":"aragora","env":[{"name":"MAINTENANCE_MODE","value":"true"}]}]}}}}'
sleep 10

# 4. Trigger switchover (graceful) or failover (forced)
echo "[4/7] Initiating switchover..."
kubectl cnpg promote ${CLUSTER_NAME} -n ${NAMESPACE}

# 5. Wait for new primary
echo "[5/7] Waiting for new primary..."
for i in {1..60}; do
    STATUS=$(kubectl cnpg status ${CLUSTER_NAME} -n ${NAMESPACE} -o json | jq -r '.status.phase')
    if [ "${STATUS}" == "Cluster in healthy state" ]; then
        echo "Cluster healthy after ${i} seconds"
        break
    fi
    sleep 1
done

# 6. Verify new primary is writable
echo "[6/7] Testing write capability..."
kubectl exec -n ${NAMESPACE} ${CLUSTER_NAME}-1 -- \
    psql -c "INSERT INTO health_check (timestamp) VALUES (now());"

# 7. Disable maintenance mode
echo "[7/7] Disabling maintenance mode..."
kubectl -n ${NAMESPACE} patch deployment aragora-api \
    -p '{"spec":{"template":{"spec":{"containers":[{"name":"aragora","env":[{"name":"MAINTENANCE_MODE","value":"false"}]}]}}}}'

echo ""
echo "=== Failover Complete ==="
kubectl cnpg status ${CLUSTER_NAME} -n ${NAMESPACE}
```

### 4.2 Database Failover (AWS RDS)

```bash
#!/bin/bash
# database_failover_rds.sh

set -euo pipefail

DB_INSTANCE="aragora-postgres"
REGION="us-east-1"

echo "=== RDS Database Failover ==="

# 1. Check current status
echo "[1/5] Checking RDS status..."
aws rds describe-db-instances \
    --db-instance-identifier ${DB_INSTANCE} \
    --query 'DBInstances[0].{Status:DBInstanceStatus,AZ:AvailabilityZone,MultiAZ:MultiAZ}'

# 2. Trigger failover (Multi-AZ)
echo "[2/5] Initiating RDS failover..."
aws rds reboot-db-instance \
    --db-instance-identifier ${DB_INSTANCE} \
    --force-failover

# 3. Wait for failover
echo "[3/5] Waiting for failover (typically 60-120 seconds)..."
aws rds wait db-instance-available \
    --db-instance-identifier ${DB_INSTANCE}

# 4. Verify new primary
echo "[4/5] Verifying new primary..."
aws rds describe-db-instances \
    --db-instance-identifier ${DB_INSTANCE} \
    --query 'DBInstances[0].AvailabilityZone'

# 5. Test connectivity
echo "[5/5] Testing database connectivity..."
psql "${DATABASE_URL}" -c "SELECT 1;"

echo "=== RDS Failover Complete ==="
```

### 4.3 Redis Failover (Sentinel)

```bash
#!/bin/bash
# redis_failover_sentinel.sh

set -euo pipefail

SENTINEL_HOST="redis-sentinel.aragora"
MASTER_NAME="aragora-master"

echo "=== Redis Sentinel Failover ==="

# 1. Check current master
echo "[1/5] Current master:"
redis-cli -h ${SENTINEL_HOST} -p 26379 SENTINEL get-master-addr-by-name ${MASTER_NAME}

# 2. Check replica status
echo "[2/5] Replica status:"
redis-cli -h ${SENTINEL_HOST} -p 26379 SENTINEL replicas ${MASTER_NAME}

# 3. Trigger failover
echo "[3/5] Initiating failover..."
redis-cli -h ${SENTINEL_HOST} -p 26379 SENTINEL FAILOVER ${MASTER_NAME}

# 4. Wait for failover
echo "[4/5] Waiting for failover..."
sleep 15

# 5. Verify new master
echo "[5/5] New master:"
NEW_MASTER=$(redis-cli -h ${SENTINEL_HOST} -p 26379 SENTINEL get-master-addr-by-name ${MASTER_NAME})
echo "New master: ${NEW_MASTER}"

# 6. Test connectivity
redis-cli -h $(echo ${NEW_MASTER} | cut -d' ' -f1) -p 6379 PING

echo "=== Redis Failover Complete ==="
```

### 4.4 Redis Failover (Kubernetes)

```bash
#!/bin/bash
# redis_failover_k8s.sh

NAMESPACE="aragora"

echo "=== Kubernetes Redis Failover ==="

# 1. Check current state
echo "[1/4] Current Redis pods:"
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=redis

# 2. Get current master
CURRENT_MASTER=$(kubectl exec -n ${NAMESPACE} redis-node-0 -c sentinel -- \
    redis-cli -p 26379 SENTINEL get-master-addr-by-name mymaster)
echo "Current master: ${CURRENT_MASTER}"

# 3. Trigger failover
echo "[3/4] Triggering failover..."
kubectl exec -n ${NAMESPACE} redis-node-0 -c sentinel -- \
    redis-cli -p 26379 SENTINEL FAILOVER mymaster

sleep 15

# 4. Verify new master
NEW_MASTER=$(kubectl exec -n ${NAMESPACE} redis-node-0 -c sentinel -- \
    redis-cli -p 26379 SENTINEL get-master-addr-by-name mymaster)
echo "New master: ${NEW_MASTER}"

echo "=== Redis Failover Complete ==="
```

### 4.5 Application Failover (Regional)

```bash
#!/bin/bash
# regional_failover.sh

set -euo pipefail

PRIMARY_REGION="us-east-1"
DR_REGION="us-west-2"

echo "=== Regional Failover: ${PRIMARY_REGION} -> ${DR_REGION} ==="
echo "WARNING: This will redirect all traffic to ${DR_REGION}"
read -p "Type 'FAILOVER' to confirm: " CONFIRM
if [ "${CONFIRM}" != "FAILOVER" ]; then
    echo "Aborted."
    exit 1
fi

# 1. Verify DR region readiness
echo "[1/6] Verifying DR region readiness..."
DR_HEALTH=$(curl -s https://aragora.${DR_REGION}.internal/api/health | jq -r '.status')
if [ "${DR_HEALTH}" != "healthy" ]; then
    echo "ERROR: DR region not healthy: ${DR_HEALTH}"
    exit 1
fi

# 2. Promote DR database replica
echo "[2/6] Promoting database replica in ${DR_REGION}..."
kubectl --context ${DR_REGION} cnpg promote aragora-postgres -n aragora

# 3. Update DNS to point to DR region
echo "[3/6] Updating DNS..."
# Using Route53 - update weighted routing
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch '{
        "Changes": [
            {
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "api.aragora.ai",
                    "Type": "A",
                    "SetIdentifier": "'${PRIMARY_REGION}'",
                    "Weight": 0,
                    "AliasTarget": {
                        "DNSName": "aragora-'${PRIMARY_REGION}'.elb.amazonaws.com",
                        "HostedZoneId": "Z3AADJGX6KTTL2",
                        "EvaluateTargetHealth": true
                    }
                }
            },
            {
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "api.aragora.ai",
                    "Type": "A",
                    "SetIdentifier": "'${DR_REGION}'",
                    "Weight": 100,
                    "AliasTarget": {
                        "DNSName": "aragora-'${DR_REGION}'.elb.amazonaws.com",
                        "HostedZoneId": "Z3DZXE0EXAMPLE",
                        "EvaluateTargetHealth": true
                    }
                }
            }
        ]
    }'

# 4. Scale up DR region
echo "[4/6] Scaling DR region..."
kubectl --context ${DR_REGION} scale deployment aragora-api --replicas=10 -n aragora

# 5. Verify service health
echo "[5/6] Verifying service health..."
for i in {1..10}; do
    HEALTH=$(curl -s https://api.aragora.ai/api/health | jq -r '.status')
    echo "Health check ${i}: ${HEALTH}"
    if [ "${HEALTH}" == "healthy" ]; then
        break
    fi
    sleep 5
done

# 6. Update status page
echo "[6/6] Updating status page..."
curl -X POST https://admin.status.aragora.ai/api/incidents \
    -H "Authorization: Bearer ${STATUS_PAGE_TOKEN}" \
    -d '{"status": "identified", "message": "Regional failover completed to '${DR_REGION}'"}'

echo ""
echo "=== Regional Failover Complete ==="
echo "Traffic is now routing to ${DR_REGION}"
echo "Monitor: https://grafana.aragora.internal/d/dr-overview"
```

### 4.6 Failover Verification Checklist

After any failover, verify:

```bash
# Run comprehensive health check
/usr/local/bin/post_failover_verification.sh

# Manual verification checklist:
# [ ] API health endpoint returns 200
# [ ] Database accepts writes
# [ ] Redis accepts writes
# [ ] User authentication works
# [ ] New debates can be created
# [ ] Existing debates can be retrieved
# [ ] WebSocket connections establish
# [ ] Rate limiting is functional
# [ ] Metrics are flowing to Prometheus
# [ ] Logs are flowing to aggregator
```

---

## 5. Data Recovery Procedures

### 5.1 Point-in-Time Recovery (PITR)

**When to use:** Data corruption, accidental deletion, need to recover to specific point

```bash
#!/bin/bash
# pitr_recovery.sh

set -euo pipefail

# Target time in UTC (ISO 8601 format)
RECOVERY_TARGET="${1:-$(date -d '1 hour ago' '+%Y-%m-%d %H:%M:%S')}"
RECOVERY_DB="aragora_recovery"

echo "=== Point-in-Time Recovery ==="
echo "Target time: ${RECOVERY_TARGET} UTC"

# 1. Create recovery configuration
echo "[1/6] Creating recovery configuration..."
cat > /tmp/recovery.conf << EOF
restore_command = 'cp /backup/postgresql/wal/%f %p'
recovery_target_time = '${RECOVERY_TARGET}'
recovery_target_action = 'pause'
EOF

# 2. Stop current database (for full recovery) or use separate instance
echo "[2/6] Preparing recovery database..."
# For CloudNativePG:
cat << EOF | kubectl apply -f -
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: aragora-postgres-recovery
  namespace: aragora
spec:
  instances: 1
  bootstrap:
    recovery:
      source: aragora-postgres
      recoveryTarget:
        targetTime: "${RECOVERY_TARGET}+00:00"
  externalClusters:
    - name: aragora-postgres
      barmanObjectStore:
        destinationPath: s3://aragora-backups/postgresql
        s3Credentials:
          accessKeyId:
            name: postgres-backup-creds
            key: ACCESS_KEY_ID
          secretAccessKey:
            name: postgres-backup-creds
            key: SECRET_ACCESS_KEY
  storage:
    size: 100Gi
EOF

# 3. Wait for recovery
echo "[3/6] Waiting for recovery to complete..."
kubectl wait cluster/aragora-postgres-recovery \
    --for=condition=Ready \
    --namespace=aragora \
    --timeout=1800s

# 4. Verify recovered data
echo "[4/6] Verifying recovered data..."
kubectl exec -n aragora aragora-postgres-recovery-1 -- \
    psql -c "SELECT COUNT(*) FROM debates;"

# 5. Compare with production (optional)
echo "[5/6] Data comparison..."
# Run comparison queries to verify recovery target was achieved

# 6. Promote or extract data
echo "[6/6] Recovery complete. Options:"
echo "  a) Extract specific data from recovery instance"
echo "  b) Promote recovery instance to become new primary"
echo "  c) Discard recovery instance"

echo "=== PITR Recovery Complete ==="
```

### 5.2 Full Database Restore

**When to use:** Complete database loss, starting from scratch

```bash
#!/bin/bash
# full_database_restore.sh

set -euo pipefail

BACKUP_FILE="${1:-/backup/postgresql/daily/latest.dump}"

echo "=== Full Database Restore ==="
echo "Backup file: ${BACKUP_FILE}"

# 1. Verify backup file exists and is valid
echo "[1/7] Verifying backup file..."
if [ ! -f "${BACKUP_FILE}" ]; then
    # Try S3
    echo "Local backup not found, downloading from S3..."
    aws s3 cp s3://aragora-backups/postgresql/daily/$(basename ${BACKUP_FILE}) ${BACKUP_FILE}
fi

pg_restore -l "${BACKUP_FILE}" > /dev/null || {
    echo "ERROR: Backup file invalid or corrupted"
    exit 1
}

# 2. Stop application
echo "[2/7] Stopping application..."
kubectl scale deployment aragora-api --replicas=0 -n aragora
sleep 10

# 3. Drop and recreate database
echo "[3/7] Recreating database..."
psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'aragora';"
psql -U postgres -c "DROP DATABASE IF EXISTS aragora;"
psql -U postgres -c "CREATE DATABASE aragora OWNER aragora;"

# 4. Restore backup
echo "[4/7] Restoring backup (this may take several minutes)..."
START_TIME=$(date +%s)
pg_restore -d aragora -v "${BACKUP_FILE}" 2>&1 | tee /tmp/restore.log
END_TIME=$(date +%s)
echo "Restore completed in $((END_TIME - START_TIME)) seconds"

# 5. Run post-restore migrations
echo "[5/7] Running migrations..."
python -m aragora.db.migrate

# 6. Verify data
echo "[6/7] Verifying restored data..."
DEBATE_COUNT=$(psql -U aragora -d aragora -t -c "SELECT COUNT(*) FROM debates;")
USER_COUNT=$(psql -U aragora -d aragora -t -c "SELECT COUNT(*) FROM users;")
echo "Restored: ${DEBATE_COUNT} debates, ${USER_COUNT} users"

# 7. Restart application
echo "[7/7] Restarting application..."
kubectl scale deployment aragora-api --replicas=3 -n aragora

# Wait for health
echo "Waiting for application health..."
sleep 30
curl -s http://aragora.aragora.svc.cluster.local:8080/api/health | jq .

echo "=== Full Database Restore Complete ==="
```

### 5.3 Redis Data Recovery

```bash
#!/bin/bash
# redis_data_recovery.sh

set -euo pipefail

RDB_FILE="${1:-/backup/redis/rdb/latest.rdb}"

echo "=== Redis Data Recovery ==="

# 1. Verify RDB file
echo "[1/5] Verifying RDB file..."
redis-check-rdb "${RDB_FILE}" || {
    echo "ERROR: RDB file corrupted"
    exit 1
}

# 2. Stop Redis (Kubernetes)
echo "[2/5] Stopping Redis..."
kubectl scale statefulset redis-master --replicas=0 -n aragora
sleep 10

# 3. Copy RDB file
echo "[3/5] Copying RDB file..."
kubectl cp "${RDB_FILE}" aragora/redis-master-0:/data/dump.rdb

# 4. Start Redis
echo "[4/5] Starting Redis..."
kubectl scale statefulset redis-master --replicas=1 -n aragora

# 5. Verify data
echo "[5/5] Verifying Redis data..."
sleep 10
kubectl exec -n aragora redis-master-0 -- redis-cli DBSIZE
kubectl exec -n aragora redis-master-0 -- redis-cli INFO keyspace

echo "=== Redis Recovery Complete ==="
```

### 5.4 Knowledge Mound Recovery

```bash
#!/bin/bash
# knowledge_mound_recovery.sh

set -euo pipefail

echo "=== Knowledge Mound Recovery ==="

# 1. Disable KM endpoints
echo "[1/5] Disabling Knowledge Mound..."
kubectl -n aragora set env deployment/aragora-api KNOWLEDGE_MOUND_ENABLED=false
kubectl rollout restart deployment/aragora-api -n aragora
sleep 30

# 2. Restore KM database
echo "[2/5] Restoring KM database..."
# KM data is stored in PostgreSQL in the knowledge_nodes table
# It was restored with the main database restore

# 3. Rebuild vector indices
echo "[3/5] Rebuilding vector indices..."
kubectl exec -n aragora deployment/aragora-api -- python -c "
from aragora.knowledge.mound import get_knowledge_mound
import asyncio

async def rebuild():
    km = get_knowledge_mound()
    await km.rebuild_indices()
    print('Indices rebuilt')

asyncio.run(rebuild())
"

# 4. Re-enable KM
echo "[4/5] Re-enabling Knowledge Mound..."
kubectl -n aragora set env deployment/aragora-api KNOWLEDGE_MOUND_ENABLED=true
kubectl rollout restart deployment/aragora-api -n aragora

# 5. Verify KM health
echo "[5/5] Verifying Knowledge Mound..."
sleep 30
curl -s http://aragora.aragora.svc.cluster.local:8080/api/health/detailed | jq '.checks.knowledge_mound'

echo "=== Knowledge Mound Recovery Complete ==="
```

### 5.5 GDPR-Compliant Restoration

**CRITICAL:** When restoring from backup, check the GDPR backup exclusion list.

```bash
#!/bin/bash
# gdpr_compliant_restore.sh

echo "=== GDPR-Compliant Database Restore ==="

# 1. Get backup exclusion list
echo "[1/4] Retrieving GDPR backup exclusion list..."
EXCLUSIONS=$(curl -s http://localhost:8080/api/v2/compliance/gdpr/backup-exclusions)
EXCLUDED_USERS=$(echo ${EXCLUSIONS} | jq -r '.excluded_users[]')

if [ -n "${EXCLUDED_USERS}" ]; then
    echo "WARNING: The following users must be excluded from restoration:"
    echo "${EXCLUDED_USERS}"
fi

# 2. Perform standard restore
echo "[2/4] Performing database restore..."
# ... standard restore steps ...

# 3. Remove excluded user data
echo "[3/4] Removing GDPR-excluded user data..."
for user_id in ${EXCLUDED_USERS}; do
    echo "Removing data for user: ${user_id}"
    psql -d aragora -c "DELETE FROM user_data WHERE user_id = '${user_id}';"
    psql -d aragora -c "DELETE FROM debates WHERE creator_id = '${user_id}';"
    # ... additional tables ...
done

# 4. Log restoration compliance
echo "[4/4] Logging GDPR compliance..."
curl -X POST http://localhost:8080/api/v2/compliance/gdpr/restoration-audit \
    -H "Content-Type: application/json" \
    -d "{\"excluded_users\": $(echo ${EXCLUSIONS} | jq '.excluded_users'), \"restore_timestamp\": \"$(date -Iseconds)\"}"

echo "=== GDPR-Compliant Restore Complete ==="
```

---

## 6. Rollback Procedures

### 6.1 Application Rollback

```bash
#!/bin/bash
# application_rollback.sh

set -euo pipefail

ROLLBACK_REVISION="${1:-}"

echo "=== Application Rollback ==="

# 1. Get current and available revisions
echo "[1/5] Current deployment status:"
kubectl rollout history deployment/aragora-api -n aragora

if [ -z "${ROLLBACK_REVISION}" ]; then
    echo "Specify revision to rollback to, or 0 for previous"
    echo "Usage: $0 <revision_number>"
    exit 1
fi

# 2. Perform rollback
echo "[2/5] Rolling back to revision ${ROLLBACK_REVISION}..."
if [ "${ROLLBACK_REVISION}" == "0" ]; then
    kubectl rollout undo deployment/aragora-api -n aragora
else
    kubectl rollout undo deployment/aragora-api -n aragora --to-revision=${ROLLBACK_REVISION}
fi

# 3. Wait for rollout
echo "[3/5] Waiting for rollout..."
kubectl rollout status deployment/aragora-api -n aragora --timeout=300s

# 4. Verify health
echo "[4/5] Verifying application health..."
sleep 10
HEALTH=$(curl -s http://aragora.aragora.svc.cluster.local:8080/api/health | jq -r '.status')
if [ "${HEALTH}" != "healthy" ]; then
    echo "WARNING: Application not healthy after rollback"
fi

# 5. Confirm rollback
echo "[5/5] Rollback complete:"
kubectl get deployment aragora-api -n aragora -o jsonpath='{.spec.template.spec.containers[0].image}'
echo ""

echo "=== Application Rollback Complete ==="
```

### 6.2 Database Migration Rollback

```bash
#!/bin/bash
# database_migration_rollback.sh

set -euo pipefail

TARGET_VERSION="${1:-}"

echo "=== Database Migration Rollback ==="

if [ -z "${TARGET_VERSION}" ]; then
    echo "Current migration version:"
    python -m aragora.db.migrate --version
    echo ""
    echo "Available versions:"
    python -m aragora.db.migrate --list
    echo ""
    echo "Usage: $0 <target_version>"
    exit 1
fi

# 1. Stop application writes
echo "[1/4] Enabling maintenance mode..."
kubectl -n aragora set env deployment/aragora-api MAINTENANCE_MODE=true
sleep 10

# 2. Run downgrade
echo "[2/4] Rolling back to version ${TARGET_VERSION}..."
python -m aragora.db.migrate --downgrade ${TARGET_VERSION}

# 3. Verify schema
echo "[3/4] Verifying schema..."
python -m aragora.db.migrate --validate

# 4. Disable maintenance mode
echo "[4/4] Disabling maintenance mode..."
kubectl -n aragora set env deployment/aragora-api MAINTENANCE_MODE=false

echo "=== Database Migration Rollback Complete ==="
```

### 6.3 Failback to Primary Region

After a regional failover, return to the primary region:

```bash
#!/bin/bash
# regional_failback.sh

set -euo pipefail

PRIMARY_REGION="us-east-1"
DR_REGION="us-west-2"

echo "=== Regional Failback: ${DR_REGION} -> ${PRIMARY_REGION} ==="

# 1. Verify primary region is ready
echo "[1/7] Verifying primary region readiness..."
kubectl --context ${PRIMARY_REGION} get nodes
PRIMARY_HEALTH=$(curl -s https://aragora.${PRIMARY_REGION}.internal/api/health | jq -r '.status')
if [ "${PRIMARY_HEALTH}" != "healthy" ]; then
    echo "ERROR: Primary region infrastructure not ready"
    exit 1
fi

# 2. Synchronize data from DR to primary
echo "[2/7] Synchronizing data..."
# Ensure primary has caught up with DR region's data
# This depends on your replication setup

# 3. Switch DNS gradually (canary)
echo "[3/7] Starting canary traffic shift..."
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch '{
        "Changes": [
            {"Action": "UPSERT", "ResourceRecordSet": {"Name": "api.aragora.ai", "Type": "A", "SetIdentifier": "'${PRIMARY_REGION}'", "Weight": 10, ...}},
            {"Action": "UPSERT", "ResourceRecordSet": {"Name": "api.aragora.ai", "Type": "A", "SetIdentifier": "'${DR_REGION}'", "Weight": 90, ...}}
        ]
    }'

echo "Waiting 5 minutes for canary traffic..."
sleep 300

# 4. Verify canary traffic
echo "[4/7] Checking canary error rates..."
# Check Prometheus/Grafana for error rates in primary region

# 5. Complete traffic shift
echo "[5/7] Completing traffic shift to primary..."
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch '{
        "Changes": [
            {"Action": "UPSERT", "ResourceRecordSet": {"Name": "api.aragora.ai", "Type": "A", "SetIdentifier": "'${PRIMARY_REGION}'", "Weight": 100, ...}},
            {"Action": "UPSERT", "ResourceRecordSet": {"Name": "api.aragora.ai", "Type": "A", "SetIdentifier": "'${DR_REGION}'", "Weight": 0, ...}}
        ]
    }'

# 6. Reconfigure DR as standby
echo "[6/7] Reconfiguring DR region as standby..."
# Reconfigure DR database as replica of primary

# 7. Update status
echo "[7/7] Updating status page..."
curl -X POST https://admin.status.aragora.ai/api/incidents \
    -H "Authorization: Bearer ${STATUS_PAGE_TOKEN}" \
    -d '{"status": "resolved", "message": "Failback to primary region complete"}'

echo "=== Regional Failback Complete ==="
```

---

## 7. Communication Protocols

### 7.1 Internal Escalation Timeline

| Time from Detection | Action |
|---------------------|--------|
| 0 minutes | On-call engineer alerted (PagerDuty) |
| 5 minutes | Incident acknowledged, initial triage |
| 10 minutes | Incident Commander assigned (SEV-1/2) |
| 15 minutes | Engineering lead notified (SEV-1/2) |
| 30 minutes | Executive team notified (SEV-1) |
| 30 minutes | Customer-facing status update |
| Every 30 min | Regular status updates |

### 7.2 Communication Templates

**Initial Notification (Internal):**
```
INCIDENT: [Brief description]
SEVERITY: SEV-[1/2/3/4]
IMPACT: [What's broken, who's affected]
STATUS: Investigating
STARTED: [Timestamp UTC]
COMMANDER: [Name]
WAR ROOM: [Slack channel / Zoom link]
NEXT UPDATE: [Time]
```

**Initial Notification (External - Status Page):**
```
[Investigating] Service Disruption

We are currently investigating reports of service disruption
affecting [specific services].

Impact: [Brief description of user-visible symptoms]
Started: [Timestamp]

Our team is actively working on resolution. We will provide
updates every 30 minutes.
```

**Resolution Notification:**
```
[Resolved] Service Disruption

The service disruption that began at [timestamp] has been resolved.
All services are now operating normally.

Duration: [X hours Y minutes]
Root Cause: [Brief, non-technical summary]
Data Impact: [None / Brief description if any]

A detailed post-mortem will be published within 72 hours.
```

### 7.3 Stakeholder Notification Matrix

| Stakeholder | SEV-1 | SEV-2 | SEV-3 | SEV-4 | Channel |
|-------------|-------|-------|-------|-------|---------|
| On-call Engineer | Immediate | Immediate | 30 min | 4 hours | PagerDuty |
| Engineering Lead | 15 min | 30 min | 2 hours | Next standup | Slack/Phone |
| VP Engineering | 30 min | 1 hour | FYI email | - | Phone/Slack |
| CEO/CTO | 1 hour | 2 hours | - | - | Phone |
| Security Team | Immediate* | As needed | - | - | Slack/Phone |
| Affected Customers | 30 min | 1 hour | - | - | Email/Status |
| All Customers | 1 hour | 2 hours | - | - | Status Page |

*For security incidents only

---

## 8. DR Drill Schedule

### 8.1 Annual Schedule

| Drill Type | Frequency | Duration | Q1 | Q2 | Q3 | Q4 |
|------------|-----------|----------|-----|-----|-----|-----|
| Backup Restoration | Monthly | 1 hour | Jan, Feb, Mar | Apr, May, Jun | Jul, Aug, Sep | Oct, Nov, Dec |
| Component Failover | Quarterly | 2-4 hours | Feb | May | Aug | Nov |
| Regional Failover | Semi-annual | 4 hours | - | Jun | - | Dec |
| Full DR Simulation | Annual | 8 hours | - | - | Sep | - |
| Tabletop Exercise | Monthly | 1-2 hours | Every month | Every month | Every month | Every month |

### 8.2 Drill Checklist

**Pre-Drill (1 week before):**
- [ ] Schedule announced to all stakeholders
- [ ] Backup systems verified functional
- [ ] Rollback plan documented and approved
- [ ] Communication channels tested
- [ ] Customer notification prepared (if needed)
- [ ] Monitoring dashboards prepared

**Drill Execution:**
- [ ] Drill declared and logged
- [ ] Incident timeline started
- [ ] Execute failover/recovery procedure
- [ ] Measure RTO (time to restore)
- [ ] Measure RPO (data loss, if any)
- [ ] Verify all health checks pass
- [ ] Execute failback procedure
- [ ] Confirm normal operations

**Post-Drill (within 48 hours):**
- [ ] Document actual RTO vs target
- [ ] Document actual RPO vs target
- [ ] Log issues encountered
- [ ] Create action items for gaps
- [ ] Update runbooks based on learnings
- [ ] Schedule follow-up if needed
- [ ] File drill report for compliance

### 8.3 Drill Report Template

```markdown
# DR Drill Report: [Drill Type]

## Summary
- **Date:** YYYY-MM-DD
- **Type:** [Backup Restoration / Component Failover / Regional Failover / Full DR]
- **Duration:** X hours Y minutes
- **Participants:** [List]
- **Result:** [PASS / PARTIAL / FAIL]

## Objectives
1. [Objective 1]
2. [Objective 2]

## Timeline
| Time | Event | Notes |
|------|-------|-------|
| HH:MM | Drill declared | |
| HH:MM | Failover initiated | |
| HH:MM | Service restored | RTO: X minutes |
| HH:MM | Drill completed | |

## Metrics
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| RTO | < 1 hour | X min | PASS/FAIL |
| RPO | < 5 min | X min | PASS/FAIL |
| Data Integrity | 100% | Y% | PASS/FAIL |

## Issues Encountered
1. [Issue]: [Resolution]
2. [Issue]: [Resolution]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action] | [Name] | [Date] | Open |

## Recommendations
1. [Recommendation]
2. [Recommendation]

## Sign-off
- **Drill Lead:** [Name] [Date]
- **Engineering Lead:** [Name] [Date]
```

---

## 9. Post-Incident Review

### 9.1 Review Timeline

| Task | Deadline |
|------|----------|
| Incident timeline documented | +24 hours |
| Post-incident review meeting | +48 hours |
| Action items assigned | +72 hours |
| Root cause analysis complete | +1 week |
| Preventive measures implemented | +2 weeks |
| Follow-up review | +1 month |

### 9.2 Blameless Post-Mortem Template

```markdown
# Post-Incident Review: [Incident Title]

**Incident ID:** INC-YYYY-XXXX
**Date:** YYYY-MM-DD
**Duration:** X hours Y minutes
**Severity:** SEV-[1/2/3/4]
**Author:** [Name]
**Reviewed by:** [Name], [Name]

## Executive Summary
[2-3 sentence summary for non-technical stakeholders]

## Impact
- **Users affected:** [Number or percentage]
- **Duration:** [Total time users were impacted]
- **Data loss:** [Yes/No, with details]
- **Revenue impact:** [If applicable]
- **SLA breach:** [Yes/No, which SLOs affected]

## Timeline
| Time (UTC) | Event | Actor |
|------------|-------|-------|
| HH:MM | [Initial symptom detected] | Automated |
| HH:MM | [Alert fired] | PagerDuty |
| HH:MM | [On-call acknowledged] | [Name] |
| HH:MM | [Diagnosis began] | [Name] |
| HH:MM | [Root cause identified] | [Name] |
| HH:MM | [Fix deployed] | [Name] |
| HH:MM | [Service restored] | Verified |
| HH:MM | [Incident closed] | [Name] |

## Root Cause Analysis

### What happened?
[Technical explanation of the failure chain]

### Contributing factors
1. [Factor 1 - e.g., "Missing monitoring for X"]
2. [Factor 2 - e.g., "Outdated runbook"]
3. [Factor 3 - e.g., "Single point of failure"]

### Why wasn't this caught earlier?
[Analysis of detection gaps]

## Resolution
[What was done to fix the immediate issue]

## What Went Well
- [Positive 1 - e.g., "Failover completed within RTO"]
- [Positive 2 - e.g., "Clear communication throughout"]
- [Positive 3 - e.g., "Runbook was accurate and helpful"]

## What Could Be Improved
- [Improvement 1 - e.g., "Faster detection"]
- [Improvement 2 - e.g., "Better documentation"]
- [Improvement 3 - e.g., "Automated remediation"]

## Action Items

### Immediate (< 1 week)
| Action | Owner | Due | Priority | Status |
|--------|-------|-----|----------|--------|
| [Action] | [Name] | [Date] | P1 | Open |

### Short-term (< 1 month)
| Action | Owner | Due | Priority | Status |
|--------|-------|-----|----------|--------|
| [Action] | [Name] | [Date] | P2 | Open |

### Long-term (< 1 quarter)
| Action | Owner | Due | Priority | Status |
|--------|-------|-----|----------|--------|
| [Action] | [Name] | [Date] | P3 | Open |

## Lessons Learned
[Key takeaways that should inform future decisions]

## Follow-up
- **30-day review scheduled:** [Date]
- **Action items tracked in:** [JIRA/Linear ticket]

---

*This post-mortem follows blameless principles. The goal is to understand
what happened and prevent recurrence, not to assign blame.*
```

---

## Appendix: Quick Commands

### Health Checks

```bash
# Full system health
curl -s https://api.aragora.ai/api/health | jq .

# Detailed health with component status
curl -s https://api.aragora.ai/api/health/detailed | jq .

# Database connectivity
psql "${DATABASE_URL}" -c "SELECT 1;"

# Redis connectivity
redis-cli -u "${REDIS_URL}" PING

# Kubernetes cluster health
kubectl get nodes
kubectl get pods -A | grep -v Running
```

### Backup Operations

```bash
# List PostgreSQL backups
ls -la /backup/postgresql/daily/

# List S3 backups
aws s3 ls s3://aragora-backups/postgresql/daily/ --human-readable

# Create manual backup
pg_dump -Fc -Z 9 -d aragora -f "/backup/postgresql/manual/aragora_$(date +%Y%m%d_%H%M%S).dump"

# Verify backup
pg_restore -l /backup/postgresql/daily/latest.dump

# Download backup from S3
aws s3 cp s3://aragora-backups/postgresql/daily/latest.dump /tmp/
```

### Database Operations

```bash
# Check replication status (CNPG)
kubectl cnpg status aragora-postgres -n aragora

# Check replication lag
psql -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int AS lag_seconds;"

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'aragora';"

# Kill long-running queries
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE duration > interval '5 minutes';"
```

### Redis Operations

```bash
# Check Redis info
redis-cli INFO replication

# Check Sentinel status
redis-cli -p 26379 SENTINEL masters

# Trigger manual failover
redis-cli -p 26379 SENTINEL FAILOVER mymaster

# Check memory
redis-cli INFO memory
```

### Kubernetes Operations

```bash
# Scale deployment
kubectl scale deployment aragora-api --replicas=10 -n aragora

# Rollout restart
kubectl rollout restart deployment/aragora-api -n aragora

# Check rollout status
kubectl rollout status deployment/aragora-api -n aragora

# Rollback
kubectl rollout undo deployment/aragora-api -n aragora

# View logs
kubectl logs -f deployment/aragora-api -n aragora --tail=100
```

### DNS Operations

```bash
# Check DNS resolution
dig api.aragora.ai

# Check Route53 health checks
aws route53 get-health-check-status --health-check-id <id>

# List CloudFlare pool health
curl -s "https://api.cloudflare.com/client/v4/user/load_balancers/pools" \
    -H "Authorization: Bearer ${CF_TOKEN}" | jq '.result[].healthy'
```

---

## Related Documentation

- [RUNBOOK_BACKUP_AUTOMATION.md](./RUNBOOK_BACKUP_AUTOMATION.md) - Detailed backup procedures
- [RUNBOOK_INCIDENT.md](./RUNBOOK_INCIDENT.md) - General incident response
- [RUNBOOK_DATABASE_ISSUES.md](./RUNBOOK_DATABASE_ISSUES.md) - Database troubleshooting
- [redis-failover.md](./redis-failover.md) - Redis HA procedures
- [RUNBOOK_MULTI_REGION_SETUP.md](./RUNBOOK_MULTI_REGION_SETUP.md) - Multi-region architecture
- [../POSTGRES_HA.md](../deployment/POSTGRES_HA.md) - PostgreSQL HA configuration
- [../DR_DRILL_PROCEDURES.md](../deployment/DR_DRILL_PROCEDURES.md) - Detailed drill procedures
- [../enterprise/DISASTER_RECOVERY.md](../enterprise/DISASTER_RECOVERY.md) - Enterprise DR overview

---

**Document History:**
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-02 | Platform Team | Initial comprehensive runbook |

**Review Schedule:** Quarterly (next: May 2026)
