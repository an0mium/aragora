# Disaster Recovery Runbook

Comprehensive disaster recovery procedures for Aragora.

## Table of Contents

- [Recovery Objectives](#recovery-objectives)
- [Incident Classification](#incident-classification)
- [Backup Strategy](#backup-strategy)
- [Recovery Procedures](#recovery-procedures)
- [Service Recovery Checklist](#service-recovery-checklist)
- [Communication Templates](#communication-templates)
- [Post-Incident Review](#post-incident-review)

---

## Recovery Objectives

### Recovery Time Objective (RTO)

| Service Tier | RTO Target | Description |
|--------------|------------|-------------|
| Critical | < 1 hour | Core debate engine, authentication |
| High | < 4 hours | API endpoints, WebSocket streaming |
| Medium | < 8 hours | Analytics, leaderboards, telemetry |
| Low | < 24 hours | Historical data, archives |

### Recovery Point Objective (RPO)

| Data Type | RPO Target | Backup Frequency |
|-----------|------------|------------------|
| User accounts | < 1 hour | Continuous replication |
| Active debates | < 15 minutes | Transaction log shipping |
| Debate history | < 1 hour | Hourly snapshots |
| Agent ratings | < 1 hour | Hourly snapshots |
| Audit logs | < 24 hours | Daily backups |
| Telemetry | < 24 hours | Daily exports |

---

## Incident Classification

### Severity Levels

| Level | Name | Definition | Response Time |
|-------|------|------------|---------------|
| SEV-1 | Critical | Complete service outage, data loss risk | < 15 minutes |
| SEV-2 | High | Major functionality impaired | < 1 hour |
| SEV-3 | Medium | Partial service degradation | < 4 hours |
| SEV-4 | Low | Minor issues, workarounds exist | < 24 hours |

### Common Incident Types

| Type | Severity | Description |
|------|----------|-------------|
| Database corruption | SEV-1 | Data integrity compromised |
| Complete outage | SEV-1 | All services unavailable |
| Authentication failure | SEV-1 | Users cannot log in |
| Data breach | SEV-1 | Unauthorized data access |
| API degradation | SEV-2 | Slow or failing API calls |
| Agent failures | SEV-2 | LLM providers unavailable |
| Memory exhaustion | SEV-2 | Server OOM conditions |
| Partial outage | SEV-3 | Some features unavailable |
| Performance issues | SEV-3 | Slow response times |
| Scheduled maintenance | SEV-4 | Planned downtime |

---

## Backup Strategy

### Automated Backups

#### Database Backups

```bash
# SQLite backup (default)
# Location: .nomic/backups/
# Frequency: Hourly
# Retention: 14 days

# Verify backup schedule
ls -la .nomic/backups/

# Manual backup
python scripts/migrate_databases.py --backup
```

#### PostgreSQL Backups (Production)

```bash
# Continuous archiving (WAL)
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'

# Full backup (daily)
pg_dump -Fc aragora > /backup/aragora_$(date +%Y%m%d).dump

# Point-in-time recovery enabled
restore_command = 'cp /backup/wal/%f %p'
```

#### Redis Backups

```bash
# RDB snapshots (default every 15 minutes)
save 900 1
save 300 10
save 60 10000

# AOF persistence
appendonly yes
appendfsync everysec

# Manual backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backup/redis_$(date +%Y%m%d).rdb
```

### Backup Verification

#### Daily Verification Script

```bash
#!/bin/bash
# scripts/verify_backups.sh

set -e

echo "=== Backup Verification $(date) ==="

# 1. Check backup files exist
BACKUP_DIR=".nomic/backups"
LATEST_BACKUP=$(ls -t $BACKUP_DIR | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No backups found!"
    exit 1
fi

echo "Latest backup: $LATEST_BACKUP"

# 2. Verify backup age (should be < 24 hours old)
BACKUP_AGE=$(( ($(date +%s) - $(stat -f %m "$BACKUP_DIR/$LATEST_BACKUP")) / 3600 ))
if [ $BACKUP_AGE -gt 24 ]; then
    echo "WARNING: Backup is $BACKUP_AGE hours old!"
fi

# 3. Validate database integrity
echo "Validating database integrity..."
python scripts/migrate_databases.py --validate

# 4. Test restore to temporary location
TEMP_DIR=$(mktemp -d)
echo "Testing restore to $TEMP_DIR..."
cp -r "$BACKUP_DIR/$LATEST_BACKUP"/* "$TEMP_DIR/"

# 5. Verify restored data
sqlite3 "$TEMP_DIR/aragora_users.db" "SELECT COUNT(*) FROM users" > /dev/null
sqlite3 "$TEMP_DIR/aragora_debates.db" "SELECT COUNT(*) FROM debates" > /dev/null

echo "Restore verification: PASSED"

# Cleanup
rm -rf "$TEMP_DIR"

echo "=== Verification Complete ==="
```

#### Weekly Full Recovery Test

```bash
#!/bin/bash
# scripts/test_full_recovery.sh

# Run in isolated environment (Docker)
docker-compose -f docker-compose.recovery-test.yml up -d

# Restore from backup
docker exec aragora-recovery python scripts/migrate_databases.py --restore /backup/latest

# Run health checks
docker exec aragora-recovery curl -f http://localhost:8080/api/health

# Run integration tests
docker exec aragora-recovery pytest tests/integration/ -v --timeout=300

# Cleanup
docker-compose -f docker-compose.recovery-test.yml down -v
```

---

## Recovery Procedures

### Procedure 1: Database Recovery (SQLite)

**Scenario:** Database corruption or data loss

**Steps:**

1. **Stop the service**
   ```bash
   # Kubernetes
   kubectl scale deployment aragora --replicas=0

   # Systemd
   sudo systemctl stop aragora
   ```

2. **Assess damage**
   ```bash
   # Check database integrity
   sqlite3 .nomic/aragora_users.db "PRAGMA integrity_check"
   sqlite3 .nomic/aragora_debates.db "PRAGMA integrity_check"
   ```

3. **List available backups**
   ```bash
   ls -la .nomic/backups/
   # Output shows: backup_YYYYMMDD_HHMMSS/
   ```

4. **Restore from backup**
   ```bash
   # Move corrupted files
   mv .nomic/aragora_users.db .nomic/aragora_users.db.corrupted
   mv .nomic/aragora_debates.db .nomic/aragora_debates.db.corrupted

   # Restore from backup
   BACKUP="backup_20260113_100000"
   cp ".nomic/backups/$BACKUP/aragora_users.db" .nomic/
   cp ".nomic/backups/$BACKUP/aragora_debates.db" .nomic/
   ```

5. **Verify restoration**
   ```bash
   python scripts/migrate_databases.py --validate
   ```

6. **Restart service**
   ```bash
   kubectl scale deployment aragora --replicas=3
   # OR
   sudo systemctl start aragora
   ```

7. **Verify service health**
   ```bash
   curl http://localhost:8080/api/health
   ```

### Procedure 2: Database Recovery (PostgreSQL)

**Scenario:** PostgreSQL database failure

**Steps:**

1. **Stop application connections**
   ```bash
   kubectl scale deployment aragora --replicas=0
   ```

2. **Connect to PostgreSQL**
   ```bash
   psql -h $DB_HOST -U postgres
   ```

3. **Drop and recreate database** (if necessary)
   ```sql
   DROP DATABASE IF EXISTS aragora;
   CREATE DATABASE aragora;
   ```

4. **Restore from backup**
   ```bash
   # From pg_dump
   pg_restore -d aragora /backup/aragora_YYYYMMDD.dump

   # OR point-in-time recovery
   # Edit recovery.conf:
   restore_command = 'cp /backup/wal/%f %p'
   recovery_target_time = '2026-01-13 10:00:00'
   ```

5. **Run migrations**
   ```bash
   python scripts/migrate_databases.py --migrate
   ```

6. **Restart application**
   ```bash
   kubectl scale deployment aragora --replicas=3
   ```

### Procedure 3: Redis Recovery

**Scenario:** Redis data loss or corruption

**Steps:**

1. **Stop Redis**
   ```bash
   redis-cli SHUTDOWN NOSAVE
   ```

2. **Restore RDB snapshot**
   ```bash
   cp /backup/redis_YYYYMMDD.rdb /var/lib/redis/dump.rdb
   chown redis:redis /var/lib/redis/dump.rdb
   ```

3. **Start Redis**
   ```bash
   systemctl start redis
   ```

4. **Verify data**
   ```bash
   redis-cli INFO keyspace
   redis-cli DBSIZE
   ```

**Note:** Aragora automatically falls back to in-memory storage if Redis is unavailable. Rate limiting and session data may be temporarily reset.

### Procedure 4: Complete Service Recovery

**Scenario:** Total infrastructure failure

**Steps:**

1. **Provision infrastructure**
   ```bash
   # Terraform (if using IaC)
   cd terraform/
   terraform apply -auto-approve

   # OR Kubernetes
   kubectl apply -f k8s/
   ```

2. **Deploy application**
   ```bash
   # Build and push image
   docker build -t aragora:recovery .
   docker push registry/aragora:recovery

   # Deploy
   kubectl set image deployment/aragora aragora=registry/aragora:recovery
   ```

3. **Restore databases**
   ```bash
   # Follow database recovery procedures above
   ```

4. **Restore secrets**
   ```bash
   # From backup or secret manager
     kubectl create secret generic aragora-secrets \
       --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
       --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
       --from-literal=ARAGORA_JWT_SECRET=$ARAGORA_JWT_SECRET
   ```

5. **Verify all services**
   ```bash
   # Health check
   curl http://aragora.example.com/api/health

   # Run smoke tests
   pytest tests/smoke/ -v
   ```

### Procedure 5: Security Incident Response

**Scenario:** Suspected data breach or security incident

**Steps:**

1. **Immediate containment**
   ```bash
   # Revoke all active sessions
   redis-cli FLUSHDB

   # Rotate JWT secret
   kubectl patch secret aragora-secrets -p '{"data":{"ARAGORA_JWT_SECRET":"'$(openssl rand -base64 32 | base64)'"}}'

   # Restart application (forces re-authentication)
   kubectl rollout restart deployment/aragora
   ```

2. **Preserve evidence**
   ```bash
   # Snapshot current state
   kubectl logs deployment/aragora > /evidence/logs_$(date +%Y%m%d_%H%M%S).log

   # Backup databases before any changes
   python scripts/migrate_databases.py --backup

   # Export audit logs
   sqlite3 .nomic/aragora_audit.db ".dump" > /evidence/audit_$(date +%Y%m%d).sql
   ```

3. **Investigate**
   ```bash
   # Review audit logs
   sqlite3 .nomic/aragora_audit.db "SELECT * FROM audit_log WHERE timestamp > datetime('now', '-24 hours')"

   # Check for suspicious activity
   grep -E "(failed|unauthorized|suspicious)" /evidence/logs_*.log
   ```

4. **Notify stakeholders**
   - Use communication templates below
   - Contact security@aragora.ai

5. **Remediate**
   - Patch vulnerability if identified
   - Reset affected user passwords
   - Update firewall rules if necessary

6. **Document**
   - Complete post-incident review
   - Update runbooks if needed

---

## Service Recovery Checklist

### Pre-Recovery

- [ ] Incident classified and severity assigned
- [ ] On-call personnel notified
- [ ] Communication sent to stakeholders
- [ ] Backup availability confirmed
- [ ] Recovery environment prepared

### During Recovery

- [ ] Services stopped gracefully
- [ ] Corrupted data preserved for analysis
- [ ] Backup restored successfully
- [ ] Data integrity verified
- [ ] Services restarted in correct order:
  1. [ ] Database (PostgreSQL/SQLite)
  2. [ ] Cache (Redis)
  3. [ ] Application servers
  4. [ ] Load balancer health checks passing

### Post-Recovery

- [ ] All health checks passing
- [ ] User authentication working
- [ ] Debate creation/retrieval functional
- [ ] WebSocket connections established
- [ ] Agent API calls successful
- [ ] Rate limiting operational
- [ ] Metrics and logging flowing
- [ ] Users notified of recovery
- [ ] Post-incident review scheduled

---

## Communication Templates

### Initial Incident Notification

```
Subject: [ARAGORA] Service Incident - [SEVERITY]

Team,

We are currently investigating an incident affecting Aragora services.

**Status:** Investigating
**Severity:** [SEV-1/SEV-2/SEV-3/SEV-4]
**Impact:** [Brief description of user impact]
**Started:** [Timestamp]

We will provide updates every [15/30/60] minutes.

Current actions:
- [Action 1]
- [Action 2]

Next update: [Timestamp]

---
Aragora Incident Response Team
```

### Status Update

```
Subject: [ARAGORA] Incident Update - [STATUS]

**Incident ID:** [ID]
**Status:** [Investigating/Identified/Monitoring/Resolved]
**Duration:** [X hours Y minutes]

**Update:**
[Description of current status and actions taken]

**Next Steps:**
- [Step 1]
- [Step 2]

**ETA to Resolution:** [Estimate or "TBD"]

Next update: [Timestamp]
```

### Resolution Notification

```
Subject: [ARAGORA] Incident Resolved - [BRIEF SUMMARY]

Team,

The incident affecting Aragora services has been resolved.

**Resolution Time:** [Timestamp]
**Total Duration:** [X hours Y minutes]
**Root Cause:** [Brief summary]

**Actions Taken:**
- [Action 1]
- [Action 2]

**Data Impact:**
- [Any data loss or recovery details]
- [Time range affected]

A post-incident review will be conducted within 48 hours.

---
Aragora Incident Response Team
```

### User-Facing Status Page

```
**Current Status: [Operational/Degraded/Outage]**

[Date] [Time] - [Update Message]

We are currently experiencing [issues with X / degraded performance / an outage].

Affected services:
- [Service 1]
- [Service 2]

What you may experience:
- [Symptom 1]
- [Symptom 2]

We are actively working to resolve this issue. Updates will be posted here.

Last updated: [Timestamp]
```

---

## Post-Incident Review

### Review Timeline

| Task | Deadline |
|------|----------|
| Incident timeline documented | 24 hours |
| Post-incident review meeting | 48 hours |
| Action items assigned | 72 hours |
| Root cause analysis complete | 1 week |
| Preventive measures implemented | 2 weeks |

### Review Template

```markdown
# Post-Incident Review: [Incident Title]

**Date:** [Date]
**Duration:** [X hours Y minutes]
**Severity:** [SEV-X]
**Author:** [Name]

## Summary
[1-2 paragraph summary of what happened]

## Timeline
| Time | Event |
|------|-------|
| HH:MM | [Event description] |
| HH:MM | [Event description] |

## Root Cause
[Detailed explanation of what caused the incident]

## Impact
- **Users affected:** [Number/percentage]
- **Data loss:** [Yes/No, details]
- **Revenue impact:** [If applicable]

## What Went Well
- [Item 1]
- [Item 2]

## What Could Be Improved
- [Item 1]
- [Item 2]

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| [Action] | [Name] | [Date] | [Open/Done] |

## Lessons Learned
[Key takeaways and recommendations]
```

---

## Emergency Contacts

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-call Engineer | [PagerDuty/Slack] | Immediate |
| Engineering Lead | [Contact] | 15 minutes |
| Security Team | security@aragora.ai | SEV-1 only |
| Infrastructure | [Contact] | As needed |

---

## Related Documentation

- [SECURITY.md](../SECURITY.md) - Security policies and incident response
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [RUNBOOK.md](RUNBOOK.md) - Operational procedures
- [DATABASE.md](DATABASE.md) - Database operations and encryption
