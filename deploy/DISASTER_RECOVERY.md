# Aragora Disaster Recovery Runbook

This document provides procedures for recovering Aragora deployments from various failure scenarios.

## Service Level Objectives

| Metric | Target | Description |
|--------|--------|-------------|
| **RTO** (Recovery Time Objective) | 4 hours | Maximum time to restore service |
| **RPO** (Recovery Point Objective) | 1 hour | Maximum data loss window |
| **Availability** | 99.9% | Annual uptime target |
| **MTTR** (Mean Time to Recovery) | 30 minutes | Average recovery time |

## Backup Overview

### Automated Backups

| Component | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| PostgreSQL | Hourly | 7 days | `./backups/postgres/` |
| Redis | Every 6 hours | 3 days | `./backups/redis/` |
| Configuration | Daily | 30 days | `./backups/config/` |
| Full System | Daily | 14 days | Off-site S3/GCS |

### Backup Verification

Backups are automatically verified:
- Integrity check (checksums)
- Restore test (weekly)
- Alert on verification failure

---

## Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | PagerDuty/Opsgenie | Immediate |
| Database Admin | database-team@company.com | 15 min |
| Platform Lead | platform-lead@company.com | 30 min |
| Executive | CTO/VP Eng | 1 hour |

---

## Incident Classification

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **P1 - Critical** | Total service outage | Immediate | Database down, API unresponsive |
| **P2 - Major** | Partial outage | 15 minutes | High error rates, degraded performance |
| **P3 - Minor** | Limited impact | 1 hour | Single feature broken, non-critical errors |
| **P4 - Low** | Minimal impact | 4 hours | Cosmetic issues, documentation bugs |

---

## Recovery Procedures

### 1. Complete Database Failure

**Symptoms:**
- API returns 500 errors
- `pg_isready` fails
- Connection refused errors

**Recovery Steps:**

```bash
# Step 1: Assess the situation
docker compose logs postgres 2>&1 | tail -100
docker exec aragora-postgres pg_isready -U aragora

# Step 2: Attempt restart
docker compose restart postgres
sleep 30
docker exec aragora-postgres pg_isready -U aragora

# Step 3: If restart fails, restore from backup
# Find latest backup
ls -la ./backups/postgres/ | head -10

# Stop services
docker compose down

# Remove corrupted data
docker volume rm aragora_postgres_data

# Recreate container and restore
docker compose up -d postgres
sleep 10

# Restore from backup
gunzip -c ./backups/postgres/latest.sql.gz | \
  docker exec -i aragora-postgres psql -U aragora aragora

# Restart all services
docker compose up -d

# Verify
./smoke_test.sh
```

**Estimated Recovery Time:** 15-45 minutes

---

### 2. Redis Cache Failure

**Symptoms:**
- High latency
- Session issues
- Rate limiting failures

**Recovery Steps:**

```bash
# Step 1: Check Redis status
docker exec aragora-redis redis-cli -a $REDIS_PASSWORD ping
docker exec aragora-redis redis-cli -a $REDIS_PASSWORD info

# Step 2: Attempt restart
docker compose restart redis
sleep 10

# Step 3: If restart fails, recreate
docker compose stop redis
docker volume rm aragora_redis_data
docker compose up -d redis

# Step 4: Warm cache (optional but recommended)
curl -X POST http://localhost:8080/api/v1/admin/cache/warm

# Verify
./smoke_test.sh --quick
```

**Note:** Redis data loss is acceptable (cache-only). Service will rebuild cache.

**Estimated Recovery Time:** 5-15 minutes

---

### 3. Application Container Crash

**Symptoms:**
- Container exits/restarts
- Health checks failing
- API unresponsive

**Recovery Steps:**

```bash
# Step 1: Check logs for error
docker compose logs aragora 2>&1 | tail -200

# Step 2: Check container status
docker compose ps
docker inspect aragora-api | jq '.[0].State'

# Step 3: Restart with fresh state
docker compose restart aragora

# Step 4: If persistent, rebuild image
docker compose build aragora
docker compose up -d aragora

# Step 5: If still failing, check resources
docker stats --no-stream
free -h
df -h

# Step 6: Increase resources if needed
# Edit docker-compose.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 8G

docker compose up -d

# Verify
./smoke_test.sh
```

**Estimated Recovery Time:** 5-30 minutes

---

### 4. Full System Recovery (New Host)

**Scenario:** Original host is unrecoverable; restore to new infrastructure.

**Prerequisites:**
- Access to backup storage (S3/GCS or local backup)
- Docker and Docker Compose installed
- Network access to AI provider APIs

**Recovery Steps:**

```bash
# Step 1: Clone repository
git clone https://github.com/an0mium/aragora.git
cd aragora/deploy/self-hosted

# Step 2: Download latest backup
aws s3 cp s3://aragora-backups/latest/ ./backups/ --recursive
# Or: gsutil cp -r gs://aragora-backups/latest/ ./backups/

# Step 3: Restore configuration
cp ./backups/config/.env .env
# Update any host-specific settings (IPs, domains)
nano .env

# Step 4: Start infrastructure services
docker compose up -d postgres redis
sleep 30

# Step 5: Restore PostgreSQL
gunzip -c ./backups/postgres/latest.sql.gz | \
  docker exec -i aragora-postgres psql -U aragora aragora

# Step 6: Restore Redis (if AOF backup exists)
docker exec aragora-redis redis-cli -a $REDIS_PASSWORD BGREWRITEAOF
# Or restore RDB: copy ./backups/redis/dump.rdb to volume

# Step 7: Start application
docker compose up -d

# Step 8: Run full verification
./smoke_test.sh --verbose

# Step 9: Update DNS/Load Balancer
# Point domain to new host IP

# Step 10: Monitor for 30 minutes
docker compose logs -f aragora
```

**Estimated Recovery Time:** 2-4 hours

---

### 5. Data Corruption Recovery

**Symptoms:**
- Invalid data in API responses
- Integrity constraint violations
- Checksum mismatches

**Recovery Steps:**

```bash
# Step 1: Identify corruption scope
docker exec -i aragora-postgres psql -U aragora aragora <<EOF
SELECT relname, n_dead_tup, last_autovacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000;
EOF

# Step 2: Check for constraint violations
docker exec -i aragora-postgres psql -U aragora aragora <<EOF
SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint
WHERE contype = 'f'
AND NOT EXISTS (
  SELECT 1 FROM information_schema.referential_constraints
  WHERE constraint_name = conname
);
EOF

# Step 3: If localized corruption, repair specific tables
docker exec -i aragora-postgres psql -U aragora aragora <<EOF
REINDEX TABLE affected_table;
VACUUM FULL affected_table;
ANALYZE affected_table;
EOF

# Step 4: If widespread corruption, point-in-time recovery
# Find corruption timestamp in logs
docker compose logs aragora 2>&1 | grep -i "error\|corrupt" | head -20

# Restore to point before corruption
# (Requires WAL archiving enabled)
docker exec -i aragora-postgres pg_restore \
  --target-time="2024-01-15 14:30:00" \
  --dbname=aragora \
  ./backups/postgres/base_backup.tar

# Step 5: Verify data integrity
curl http://localhost:8080/api/v1/admin/integrity-check

# Verify
./smoke_test.sh
```

**Estimated Recovery Time:** 30 minutes - 2 hours

---

### 6. Network/Connectivity Issues

**Symptoms:**
- Timeouts to external services
- DNS resolution failures
- AI provider API unreachable

**Recovery Steps:**

```bash
# Step 1: Diagnose network
docker exec aragora-api ping -c 3 8.8.8.8
docker exec aragora-api nslookup api.anthropic.com
docker exec aragora-api curl -v https://api.anthropic.com/v1/messages

# Step 2: Check Docker network
docker network ls
docker network inspect aragora_default

# Step 3: Recreate network if corrupted
docker compose down
docker network prune -f
docker compose up -d

# Step 4: Check firewall rules
sudo iptables -L -n
sudo ufw status

# Step 5: Verify DNS configuration
cat /etc/resolv.conf
# In .env, add: ARAGORA_DNS_SERVERS=8.8.8.8,1.1.1.1

# Step 6: Check for rate limiting
curl -I https://api.anthropic.com/v1/messages
# Check for 429 status codes

# Step 7: Fallback to alternative provider
# Edit .env to use backup API key or different provider
nano .env
docker compose restart aragora

# Verify
./smoke_test.sh
```

**Estimated Recovery Time:** 15-60 minutes

---

### 7. Kubernetes Cluster Recovery

**For Kubernetes deployments only.**

**Symptoms:**
- Pods in CrashLoopBackOff
- Services unreachable
- PVC issues

**Recovery Steps:**

```bash
# Step 1: Check cluster health
kubectl get nodes
kubectl get pods -n aragora
kubectl describe pod -n aragora -l app=aragora

# Step 2: Check events
kubectl get events -n aragora --sort-by='.lastTimestamp' | tail -20

# Step 3: Force pod restart
kubectl rollout restart deployment/aragora -n aragora

# Step 4: If PVC issues, check storage
kubectl get pvc -n aragora
kubectl describe pvc -n aragora

# Step 5: For StatefulSet issues (PostgreSQL)
kubectl rollout status statefulset/postgres -n aragora
kubectl delete pod postgres-0 -n aragora  # Forces recreate

# Step 6: Restore from etcd backup (cluster-wide failure)
ETCDCTL_API=3 etcdctl snapshot restore /backup/etcd-snapshot.db \
  --data-dir=/var/lib/etcd-restore

# Step 7: Apply manifests
kubectl apply -k deploy/kubernetes/

# Verify
kubectl exec -it deploy/aragora -n aragora -- /app/smoke_test.sh
```

**Estimated Recovery Time:** 15 minutes - 2 hours

---

## Preventive Measures

### Daily Checks

```bash
# Run daily at 9 AM
0 9 * * * /opt/aragora/scripts/health_check.sh

# Check backup integrity
0 3 * * * /opt/aragora/scripts/verify_backups.sh

# Clean old logs
0 4 * * * find /var/log/aragora -mtime +7 -delete
```

### Weekly Checks

- [ ] Verify backup restore works (test restore to staging)
- [ ] Check disk space trends
- [ ] Review error rate dashboards
- [ ] Update dependencies if security patches available
- [ ] Test failover procedures

### Monthly Checks

- [ ] Full disaster recovery drill
- [ ] Security audit
- [ ] Capacity planning review
- [ ] Documentation updates
- [ ] Runbook review and updates

---

## Post-Incident Procedures

### Incident Report Template

```markdown
## Incident Report: [TITLE]

**Date:** YYYY-MM-DD
**Duration:** HH:MM - HH:MM (X hours)
**Severity:** P1/P2/P3/P4
**Impact:** [Number of users affected, features impacted]

### Timeline
- HH:MM - Issue detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Service restored

### Root Cause
[Detailed explanation]

### Resolution
[Steps taken to resolve]

### Action Items
- [ ] Prevent recurrence
- [ ] Improve detection
- [ ] Update runbooks
- [ ] Add monitoring

### Lessons Learned
[What we learned and how to prevent similar issues]
```

### Post-Incident Checklist

- [ ] All services verified operational
- [ ] Data integrity confirmed
- [ ] Customer communication sent (if applicable)
- [ ] Incident report drafted
- [ ] Timeline documented
- [ ] Root cause analysis completed
- [ ] Action items assigned
- [ ] Runbooks updated

---

## Quick Reference Commands

```bash
# Service status
docker compose ps
docker compose logs -f aragora

# Database
docker exec aragora-postgres pg_isready -U aragora
docker exec -it aragora-postgres psql -U aragora aragora

# Redis
docker exec aragora-redis redis-cli -a $REDIS_PASSWORD ping
docker exec aragora-redis redis-cli -a $REDIS_PASSWORD info memory

# Backup
./backup.sh
ls -la ./backups/

# Restore
gunzip -c ./backups/postgres/latest.sql.gz | docker exec -i aragora-postgres psql -U aragora aragora

# Health check
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz

# Full verification
./smoke_test.sh --verbose
```

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-27 | Claude | Initial runbook |

---

## References

- [Self-Hosted Deployment Guide](./self-hosted/README.md)
- [Kubernetes Deployment Guide](./kubernetes/README.md)
- [Database Migration Guide](./DATABASE_MIGRATION.md)
- [Monitoring Setup](./observability/README.md)
