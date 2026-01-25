# Aragora Disaster Recovery Procedures

**Version:** 1.0
**Last Updated:** January 2026
**Classification:** Internal / Enterprise Customers

---

## 1. Overview

This document outlines Aragora's disaster recovery (DR) procedures to ensure business continuity and minimize data loss during catastrophic events.

### 1.1 Recovery Objectives

| Metric | Target | Description |
|--------|--------|-------------|
| **RTO** | 4 hours | Maximum time to restore service |
| **RPO** | 1 hour | Maximum acceptable data loss |
| **Failover** | 15 minutes | Time to switch regions |

### 1.2 Disaster Classifications

| Level | Description | Examples | Response |
|-------|-------------|----------|----------|
| **L1** | Partial degradation | Single service down | Auto-healing |
| **L2** | Major degradation | Multiple services affected | Manual intervention |
| **L3** | Regional outage | Primary region unavailable | Regional failover |
| **L4** | Global incident | All regions affected | Full DR activation |

---

## 2. Infrastructure Architecture

### 2.1 Multi-Region Deployment

```
Primary Region (us-east-1)          Backup Region (us-west-2)
┌─────────────────────────┐        ┌─────────────────────────┐
│  Load Balancer (ALB)    │        │  Load Balancer (ALB)    │
│         │               │        │         │               │
│    ┌────┴────┐          │        │    ┌────┴────┐          │
│    │   K8s   │          │◄──────►│    │   K8s   │          │
│    │ Cluster │          │  Sync  │    │ Cluster │          │
│    └────┬────┘          │        │    └────┬────┘          │
│         │               │        │         │               │
│  ┌──────┴──────┐        │        │  ┌──────┴──────┐        │
│  │ PostgreSQL  │────────┼───────►│  │ PostgreSQL  │        │
│  │  Primary    │  WAL   │        │  │  Replica    │        │
│  └─────────────┘        │        │  └─────────────┘        │
│                         │        │                         │
│  ┌─────────────┐        │        │  ┌─────────────┐        │
│  │    Redis    │────────┼───────►│  │    Redis    │        │
│  │   Primary   │  Sync  │        │  │   Replica   │        │
│  └─────────────┘        │        │  └─────────────┘        │
└─────────────────────────┘        └─────────────────────────┘
```

### 2.2 Data Replication

| Component | Replication Type | Lag Target | Consistency |
|-----------|------------------|------------|-------------|
| PostgreSQL | Streaming (async) | < 1 second | Eventual |
| Redis | Sentinel + replica | < 100ms | Eventual |
| Object Storage | Cross-region sync | < 1 minute | Eventual |
| Secrets | Vault replication | Real-time | Strong |

---

## 3. Backup Strategy

### 3.1 Backup Schedule

| Data Type | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| Database (full) | Daily | 30 days | S3 + cross-region |
| Database (WAL) | Continuous | 7 days | S3 |
| Redis (RDB) | Every 6 hours | 7 days | S3 |
| Configuration | On change | 90 days | Git + S3 |
| Secrets | On change | 30 versions | Vault |

### 3.2 Backup Verification

- **Automated testing**: Daily restore to test environment
- **Integrity checks**: SHA-256 checksums on all backups
- **Encryption**: AES-256-GCM with KMS-managed keys
- **Access logs**: All backup access logged and audited

---

## 4. Failover Procedures

### 4.1 Automatic Failover (L1-L2)

**Trigger Conditions:**
- Health check failures > 3 consecutive
- Error rate > 10% for 5 minutes
- Latency P99 > 10 seconds for 5 minutes

**Automatic Actions:**
1. Kubernetes auto-scales affected pods
2. Failed pods terminated and replaced
3. Circuit breakers activated
4. Alert sent to on-call engineer

### 4.2 Regional Failover (L3)

**Trigger Conditions:**
- Primary region health checks fail for 5 minutes
- AWS regional service disruption
- Manual decision by incident commander

**Failover Steps:**

```bash
# 1. Verify backup region readiness
./scripts/dr/verify-backup-region.sh

# 2. Promote PostgreSQL replica to primary
./scripts/dr/promote-db-replica.sh

# 3. Update DNS to point to backup region
./scripts/dr/update-dns.sh --region us-west-2

# 4. Scale up backup region capacity
kubectl --context backup scale deployment/aragora-api --replicas=10

# 5. Verify service health
./scripts/dr/health-check.sh --region us-west-2

# 6. Notify customers via status page
./scripts/dr/update-status-page.sh --incident "Regional failover activated"
```

**Estimated Duration:** 15-30 minutes

### 4.3 Full DR Activation (L4)

For global incidents affecting all regions:

1. Activate DR war room
2. Assess damage scope
3. Restore from most recent backup
4. Bring up services in order: DB → Cache → API → Workers
5. Verify data integrity
6. Resume customer traffic
7. Post-incident review

**Estimated Duration:** 2-4 hours

---

## 5. Recovery Procedures

### 5.1 Database Recovery

**Point-in-Time Recovery:**
```bash
# Restore to specific timestamp
pg_restore \
  --target-time="2026-01-24 14:30:00 UTC" \
  --target-action=promote \
  --dbname=aragora_prod

# Verify data integrity
./scripts/db/verify-integrity.sh
```

**Full Restore from Backup:**
```bash
# Download latest backup
aws s3 cp s3://aragora-backups/db/latest.dump.gpg /tmp/

# Decrypt and restore
gpg --decrypt /tmp/latest.dump.gpg | pg_restore -d aragora_prod

# Apply WAL logs
pg_wal_replay --start-lsn=0/XXXXXXXX
```

### 5.2 Application Recovery

```bash
# 1. Deploy infrastructure (Terraform)
cd infrastructure/
terraform apply -auto-approve

# 2. Deploy Kubernetes resources
kubectl apply -f k8s/production/

# 3. Verify pod health
kubectl get pods -n aragora --watch

# 4. Run smoke tests
./scripts/smoke-test.sh --env production
```

### 5.3 Data Verification

After recovery, verify:
- [ ] Database row counts match pre-incident
- [ ] Recent transactions are present
- [ ] User authentication works
- [ ] API endpoints respond correctly
- [ ] WebSocket connections establish
- [ ] Scheduled jobs resume

---

## 6. Communication Plan

### 6.1 Internal Escalation

| Time | Action |
|------|--------|
| 0 min | On-call engineer alerted |
| 5 min | Incident commander assigned |
| 15 min | Engineering lead notified |
| 30 min | Executive team notified (L3+) |
| 1 hour | All-hands if L4 |

### 6.2 Customer Communication

| Time | Action |
|------|--------|
| 15 min | Status page updated |
| 30 min | Email to affected customers |
| Ongoing | Status page updates every 30 min |
| Resolution | All-clear notification |
| +72 hours | Post-mortem shared (L2+) |

### 6.3 Communication Templates

**Initial Notification:**
```
Subject: [Aragora] Service Disruption - Investigating

We are currently investigating reports of service disruption
affecting [specific services]. Our team is actively working
on resolution.

Current Status: Investigating
Impact: [description]
Started: [timestamp]

We will provide updates every 30 minutes.
```

**Resolution Notification:**
```
Subject: [Aragora] Service Restored

The service disruption that began at [timestamp] has been
resolved. All services are now operating normally.

Duration: [X hours Y minutes]
Root Cause: [brief description]
Prevention: [actions taken]

A detailed post-mortem will be shared within 72 hours.
```

---

## 7. Testing & Drills

### 7.1 DR Test Schedule

| Test Type | Frequency | Scope |
|-----------|-----------|-------|
| Backup restore | Weekly | Single database |
| Failover simulation | Monthly | Single service |
| Regional failover | Quarterly | Full region |
| Full DR drill | Annually | Complete activation |

### 7.2 Game Day Scenarios

1. **Database failure**: Primary PostgreSQL becomes unavailable
2. **Region outage**: Simulate us-east-1 failure
3. **Data corruption**: Restore from backup after corruption
4. **Security incident**: Credential rotation under pressure

### 7.3 Test Documentation

Each DR test must document:
- Test date and participants
- Scenario description
- Actual vs. expected recovery time
- Issues encountered
- Remediation actions
- Sign-off by engineering lead

---

## 8. Runbook Quick Reference

### 8.1 Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | PagerDuty | Primary |
| Incident Commander | Slack #incidents | Secondary |
| Engineering Lead | Phone | L3+ |
| CEO/CTO | Phone | L4 |

### 8.2 Critical Commands

```bash
# Check cluster health
kubectl get nodes && kubectl get pods -A | grep -v Running

# View recent errors
kubectl logs -n aragora -l app=aragora-api --since=10m | grep ERROR

# Database connection check
psql $DATABASE_URL -c "SELECT 1;"

# Redis health check
redis-cli -u $REDIS_URL PING

# Force pod restart
kubectl rollout restart deployment/aragora-api -n aragora

# Scale for capacity
kubectl scale deployment/aragora-api --replicas=20 -n aragora
```

### 8.3 Key Dashboards

- **Grafana**: grafana.aragora.internal/d/main
- **PagerDuty**: pagerduty.com/incidents
- **Status Page**: admin.status.aragora.ai
- **AWS Console**: console.aws.amazon.com

---

## 9. Post-Incident Process

### 9.1 Immediate Actions (0-24 hours)

1. Confirm full service restoration
2. Verify data integrity
3. Document timeline of events
4. Identify affected customers
5. Send resolution notification

### 9.2 Post-Mortem (24-72 hours)

**Template:**
```markdown
## Incident Post-Mortem: [Title]

**Date:** [date]
**Duration:** [X hours Y minutes]
**Severity:** L[1-4]
**Author:** [name]

### Summary
[2-3 sentence description]

### Timeline
- HH:MM - [event]
- HH:MM - [event]

### Root Cause
[Technical explanation]

### Impact
- Customers affected: [number]
- Requests failed: [number]
- Data loss: [yes/no, amount]

### Resolution
[How it was fixed]

### Prevention
- [ ] Action item 1 - Owner - Due date
- [ ] Action item 2 - Owner - Due date

### Lessons Learned
[What we learned]
```

### 9.3 Follow-Up Actions

- Update runbooks based on learnings
- Implement prevention measures
- Schedule follow-up review (2 weeks)
- Update DR test scenarios

---

*Document maintained by: Platform Engineering*
*Review frequency: Quarterly*
