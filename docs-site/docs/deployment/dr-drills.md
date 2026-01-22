---
title: Disaster Recovery Drill Procedures
description: Disaster Recovery Drill Procedures
---

# Disaster Recovery Drill Procedures

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** DevOps Team

---

## Overview

This document defines procedures for conducting disaster recovery (DR) drills to validate Aragora's business continuity capabilities. Regular DR drills ensure that recovery procedures work as expected and that the team is prepared for actual incidents.

**SOC 2 Control:** CC9-01 - Disaster recovery procedures and testing

---

## Drill Schedule

| Drill Type | Frequency | Duration | Participants |
|------------|-----------|----------|--------------|
| Tabletop Exercise | Monthly | 1-2 hours | All engineers |
| Component Failover | Quarterly | 2-4 hours | DevOps + On-call |
| Full DR Simulation | Annually | 4-8 hours | All teams |
| Backup Restoration | Monthly | 1 hour | DevOps |

---

## Drill Types

### 1. Tabletop Exercise (Monthly)

A discussion-based exercise where team members walk through incident scenarios without actually executing recovery procedures.

**Objectives:**
- Validate understanding of procedures
- Identify gaps in documentation
- Train new team members
- Update runbooks based on learnings

**Process:**

```
[ ] 1. PREPARATION (30 min before)
    - Select scenario from drill library
    - Prepare scenario materials
    - Invite participants
    - Set up conference call/meeting room

[ ] 2. EXECUTION (1-2 hours)
    - Present scenario
    - Walk through response steps
    - Discuss decision points
    - Identify blockers and gaps
    - Document action items

[ ] 3. DEBRIEF (30 min)
    - Review what went well
    - Identify improvements
    - Assign action items
    - Schedule follow-up if needed
```

**Sample Scenarios:**

| Scenario | Focus Area | Key Questions |
|----------|------------|---------------|
| Database corruption | Data recovery | How do we restore? What's the data loss? |
| API provider outage | Failover | How do we switch to fallback? |
| Credential compromise | Security | How do we rotate and contain? |
| Complete region failure | Infrastructure | How do we failover regions? |
| Ransomware attack | Security + DR | How do we isolate and recover? |

---

### 2. Component Failover Drill (Quarterly)

Controlled tests of individual component failover capabilities.

**Components to Test:**

| Component | Test Method | Success Criteria |
|-----------|-------------|------------------|
| Database | Promote replica | &lt;5 min failover, zero data loss |
| Redis | Sentinel failover | &lt;30 sec failover |
| API Server | Rolling restart | Zero downtime |
| Load Balancer | Health check failure | Automatic rerouting |
| AI Provider | Quota exhaustion | OpenRouter fallback |

**Database Failover Drill:**

```bash
#!/bin/bash
# dr_drill_database_failover.sh

echo "=== Database Failover Drill ==="
echo "Start time: $(date)"
echo ""

# 1. Record baseline metrics
echo "[1/6] Recording baseline metrics..."
BASELINE_LATENCY=$(curl -s http://localhost:8080/api/health | jq '.checks.database.latency_ms')
echo "Current DB latency: $\{BASELINE_LATENCY\}ms"

# 2. Verify replica is in sync
echo "[2/6] Verifying replica sync status..."
psql -h replica -c "SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();"

# 3. Stop writes (put in maintenance mode)
echo "[3/6] Enabling maintenance mode..."
kubectl -n aragora set env deployment/aragora MAINTENANCE_MODE=true
sleep 10

# 4. Promote replica
echo "[4/6] Promoting replica to primary..."
# For cloud providers:
# aws rds promote-read-replica --db-instance-identifier aragora-replica
# For manual PostgreSQL:
# psql -h replica -c "SELECT pg_promote();"

# 5. Update connection string
echo "[5/6] Updating connection string..."
kubectl -n aragora set env deployment/aragora \
  DATABASE_URL="postgresql://user:pass@new-primary:5432/aragora"

# 6. Disable maintenance mode
echo "[6/6] Disabling maintenance mode..."
kubectl -n aragora set env deployment/aragora MAINTENANCE_MODE=false

# 7. Verify
echo ""
echo "=== Verification ==="
sleep 30
NEW_LATENCY=$(curl -s http://localhost:8080/api/health | jq '.checks.database.latency_ms')
echo "New DB latency: $\{NEW_LATENCY\}ms"
curl -s http://localhost:8080/api/health | jq '.checks.database'

echo ""
echo "End time: $(date)"
echo "=== Drill Complete ==="
```

**Redis Failover Drill:**

```bash
#!/bin/bash
# dr_drill_redis_failover.sh

echo "=== Redis Failover Drill ==="

# 1. Check current master
echo "[1/4] Current master:"
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# 2. Force failover
echo "[2/4] Initiating failover..."
redis-cli -p 26379 sentinel failover aragora-master

# 3. Wait for failover
echo "[3/4] Waiting for failover..."
sleep 10

# 4. Verify new master
echo "[4/4] New master:"
redis-cli -p 26379 sentinel get-master-addr-by-name aragora-master

# 5. Verify application connectivity
echo ""
echo "=== Application Verification ==="
curl -s http://localhost:8080/api/health | jq '.checks.redis'
```

---

### 3. Full DR Simulation (Annually)

Complete simulation of a disaster scenario requiring full infrastructure recovery.

**Pre-Drill Checklist:**

```
[ ] Schedule approved by management
[ ] All stakeholders notified
[ ] Backup systems verified
[ ] Rollback plan documented
[ ] Communication channels established
[ ] Customer notification prepared (if needed)
```

**Drill Execution:**

```
PHASE 1: PREPARATION (T-24h to T-0)
[ ] Verify all backups are current
[ ] Test backup restoration in isolated environment
[ ] Confirm DR site readiness
[ ] Brief all participants
[ ] Set up war room / communication channel

PHASE 2: DECLARATION (T-0)
[ ] Declare drill start
[ ] Begin incident timeline
[ ] Activate DR team
[ ] Start communication log

PHASE 3: FAILOVER (T+0 to T+2h)
[ ] Execute database failover
[ ] Execute application failover
[ ] Update DNS/routing
[ ] Verify service health
[ ] Begin customer notification (simulated)

PHASE 4: VALIDATION (T+2h to T+4h)
[ ] Run health checks
[ ] Execute functional tests
[ ] Verify data integrity
[ ] Test critical workflows
[ ] Measure performance metrics

PHASE 5: RECOVERY (T+4h to T+6h)
[ ] Plan failback to primary
[ ] Execute failback
[ ] Verify primary restoration
[ ] Confirm normal operations

PHASE 6: DEBRIEF (T+6h to T+8h)
[ ] Document timeline
[ ] Calculate metrics (RTO, RPO)
[ ] Identify issues
[ ] Create action items
[ ] Schedule follow-up
```

**Success Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| RTO (Recovery Time Objective) | &lt;4 hours | Time from declaration to service restoration |
| RPO (Recovery Point Objective) | &lt;1 hour | Data loss measured in time |
| Failover Success | 100% | All components successfully failed over |
| Data Integrity | 100% | No data corruption or loss |
| Test Coverage | >80% | Critical workflows verified |

---

### 4. Backup Restoration Drill (Monthly)

Regular testing of backup restoration capabilities.

**PostgreSQL Backup Restoration:**

```bash
#!/bin/bash
# dr_drill_backup_restore.sh

DATE=$(date +%Y%m%d)
RESTORE_DB="aragora_restore_test_$\{DATE\}"
BACKUP_FILE="/backups/postgres/aragora_latest.dump"

echo "=== Backup Restoration Drill ==="
echo "Date: $(date)"
echo "Backup file: $\{BACKUP_FILE\}"
echo ""

# 1. Verify backup exists
echo "[1/6] Verifying backup file..."
ls -la $\{BACKUP_FILE\}
if [ ! -f "$\{BACKUP_FILE\}" ]; then
    echo "ERROR: Backup file not found!"
    exit 1
fi

# 2. Create test database
echo "[2/6] Creating test database..."
psql -c "DROP DATABASE IF EXISTS $\{RESTORE_DB\};"
psql -c "CREATE DATABASE $\{RESTORE_DB\};"

# 3. Restore backup
echo "[3/6] Restoring backup..."
START_TIME=$(date +%s)
pg_restore -d $\{RESTORE_DB\} $\{BACKUP_FILE\}
END_TIME=$(date +%s)
RESTORE_TIME=$((END_TIME - START_TIME))
echo "Restore completed in $\{RESTORE_TIME\} seconds"

# 4. Verify data integrity
echo "[4/6] Verifying data integrity..."
psql -d $\{RESTORE_DB\} -c "SELECT COUNT(*) as users FROM users;"
psql -d $\{RESTORE_DB\} -c "SELECT COUNT(*) as debates FROM debates;"
psql -d $\{RESTORE_DB\} -c "SELECT COUNT(*) as audit_events FROM audit_events;"

# 5. Verify recent data
echo "[5/6] Checking data freshness..."
psql -d $\{RESTORE_DB\} -c "SELECT MAX(created_at) as latest_user FROM users;"
psql -d $\{RESTORE_DB\} -c "SELECT MAX(created_at) as latest_debate FROM debates;"

# 6. Cleanup
echo "[6/6] Cleaning up test database..."
psql -c "DROP DATABASE $\{RESTORE_DB\};"

echo ""
echo "=== Drill Results ==="
echo "Restore time: $\{RESTORE_TIME\} seconds"
echo "Status: SUCCESS"
echo ""
```

**S3/Object Storage Backup Verification:**

```bash
#!/bin/bash
# verify_s3_backups.sh

echo "=== S3 Backup Verification ==="

# List recent backups
echo "Recent backups:"
aws s3 ls s3://aragora-backups/postgres/ --recursive | tail -5

# Verify backup integrity
echo ""
echo "Verifying latest backup integrity..."
LATEST=$(aws s3 ls s3://aragora-backups/postgres/ --recursive | tail -1 | awk '{print $4}')
aws s3api head-object --bucket aragora-backups --key $\{LATEST\}

# Download and verify checksum
echo ""
echo "Downloading for checksum verification..."
aws s3 cp s3://aragora-backups/$\{LATEST\} /tmp/backup_verify.dump
md5sum /tmp/backup_verify.dump
rm /tmp/backup_verify.dump
```

---

## Drill Documentation

### Drill Report Template

```markdown
# DR Drill Report

## Summary
- **Date:** YYYY-MM-DD
- **Type:** [Tabletop/Component/Full/Backup]
- **Duration:** X hours
- **Participants:** [list]
- **Result:** [Pass/Partial/Fail]

## Objectives
- Objective 1
- Objective 2

## Timeline
| Time | Event | Notes |
|------|-------|-------|
| HH:MM | Event 1 | Notes |
| HH:MM | Event 2 | Notes |

## Metrics
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| RTO | 4h | Xh | Pass/Fail |
| RPO | 1h | Xh | Pass/Fail |

## Issues Encountered
1. Issue 1 - Resolution
2. Issue 2 - Resolution

## Action Items
| Item | Owner | Due Date | Status |
|------|-------|----------|--------|
| Item 1 | Owner | Date | Open |

## Lessons Learned
- Lesson 1
- Lesson 2

## Recommendations
- Recommendation 1
- Recommendation 2
```

---

## Drill Library

### Scenario 1: Complete Database Failure

**Trigger:** Primary database becomes unavailable

**Expected Response:**
1. Alert fires (PagerDuty)
2. On-call acknowledges
3. Assess scope (primary vs replica)
4. If primary: promote replica
5. Update DNS/connection strings
6. Verify application health
7. Notify stakeholders
8. Plan data reconciliation

**Recovery Steps:**
- Follow RUNBOOK.md PostgreSQL Recovery section
- RTO target: 30 minutes
- RPO target: 5 minutes (with streaming replication)

### Scenario 2: AI Provider Mass Outage

**Trigger:** All primary AI providers (Anthropic, OpenAI) unavailable

**Expected Response:**
1. Circuit breakers open
2. Automatic failover to OpenRouter
3. If OpenRouter down: degrade gracefully
4. Enable queue mode for non-urgent debates
5. Notify customers of degraded service

**Recovery Steps:**
- Monitor provider status pages
- Adjust circuit breaker thresholds
- Consider enabling additional fallback providers

### Scenario 3: Credential Compromise

**Trigger:** API keys or secrets exposed

**Expected Response:**
1. Immediately revoke compromised credentials
2. Rotate all related secrets
3. Review audit logs for unauthorized access
4. Assess data exposure scope
5. Notify affected parties if required
6. Conduct forensic investigation

**Recovery Steps:**
- Follow RUNBOOK.md Security Incident Response
- Execute credential rotation scripts
- Update secrets manager

### Scenario 4: Ransomware Attack

**Trigger:** Encryption malware detected

**Expected Response:**
1. Isolate affected systems immediately
2. Preserve evidence (do not wipe)
3. Assess scope of encryption
4. Activate clean backup environment
5. Restore from known-good backups
6. Forensic investigation
7. Report to authorities if required

**Recovery Steps:**
- Never pay ransom
- Restore from offline/immutable backups
- Verify backup integrity before restoration
- Full security audit before returning to production

---

## Compliance Checklist

### Annual DR Testing Requirements (SOC 2)

- [ ] Full DR simulation conducted
- [ ] Recovery objectives (RTO/RPO) validated
- [ ] Backup restoration verified
- [ ] Documentation updated
- [ ] Training completed for all team members
- [ ] Third-party dependencies tested
- [ ] Communication procedures tested
- [ ] Results documented and reviewed by management

### Quarterly Requirements

- [ ] Component failover tests completed
- [ ] Backup integrity verified
- [ ] Runbooks reviewed and updated
- [ ] Contact lists verified

### Monthly Requirements

- [ ] Tabletop exercise conducted
- [ ] Backup restoration test completed
- [ ] Alert system tested
- [ ] On-call procedures reviewed

---

## Contact Information

| Role | Contact | Responsibility |
|------|---------|----------------|
| DR Coordinator | devops@aragora.ai | Drill planning and execution |
| On-Call Engineer | PagerDuty | Incident response |
| Security Lead | security@aragora.ai | Security-related drills |
| VP Engineering | [phone] | Escalation for full DR |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |
