# PostgreSQL High Availability Guide

This document covers PostgreSQL high availability configurations for Aragora production deployments.

## Table of Contents

- [Overview](#overview)
- [Why PostgreSQL HA?](#why-postgresql-ha)
- [Option 1: Managed PostgreSQL (Recommended)](#option-1-managed-postgresql-recommended)
- [Option 2: CloudNativePG Operator](#option-2-cloudnativepg-operator)
- [Option 3: Manual Replication Setup](#option-3-manual-replication-setup)
- [Failover Testing](#failover-testing)
- [Aragora Configuration](#aragora-configuration)
- [Backup and Recovery](#backup-and-recovery)
- [Monitoring](#monitoring)

---

## Overview

### Current State

The default Kubernetes deployment (`deploy/k8s/postgres-statefulset.yaml`) uses a single-replica PostgreSQL StatefulSet. This is **for development/testing only** and is NOT suitable for production.

**Critical Issues with Single-Replica:**
- Single point of failure - any Pod failure causes complete data unavailability
- No automatic failover
- Data loss risk during unplanned restarts
- No read scaling

### Choosing an HA Strategy

| Strategy | Complexity | Failover Time | RTO | Best For |
|----------|------------|---------------|-----|----------|
| AWS RDS Multi-AZ | Low | 60-120 sec | 2 min | AWS deployments |
| GCP Cloud SQL HA | Low | ~60 sec | 1-2 min | GCP deployments |
| Supabase | Low | Provider-managed | < 1 min | Serverless deployments |
| CloudNativePG | Medium | 10-30 sec | 30 sec | Self-managed K8s |
| Manual Streaming | High | Manual | Varies | Legacy/specialized |

---

## Why PostgreSQL HA?

Aragora uses PostgreSQL for:

- **User and tenant data** - Authentication, authorization, org membership
- **Debate persistence** - All debate history, messages, and outcomes
- **Decision receipts** - 7-year compliance retention (critical for audit)
- **Knowledge Mound** - Persistent knowledge store for cross-debate learning
- **Usage metering** - Billing-relevant usage tracking
- **Audit logs** - Security and compliance audit trails

**Impact of PostgreSQL failure:**

| Component | Impact | Recovery Difficulty |
|-----------|--------|---------------------|
| Auth data | Users cannot authenticate | Medium - sessions may survive in Redis |
| Debates | New debates fail, history unavailable | High - complete feature loss |
| Receipts | Compliance gap, audit trail broken | Critical - regulatory risk |
| Knowledge Mound | Learning unavailable, stale context | Medium - degrades debate quality |
| Billing | Usage not tracked accurately | High - revenue impact |

---

## Option 1: Managed PostgreSQL (Recommended)

Managed services provide HA with minimal operational overhead. This is the **recommended approach** for production.

### AWS RDS Multi-AZ

```bash
# Create Multi-AZ RDS instance using AWS CLI
aws rds create-db-instance \
  --db-instance-identifier aragora-postgres \
  --db-instance-class db.r6g.large \
  --engine postgres \
  --engine-version 16.1 \
  --master-username aragora \
  --master-user-password "<STRONG_PASSWORD>" \
  --allocated-storage 100 \
  --storage-type gp3 \
  --multi-az \
  --vpc-security-group-ids sg-xxxx \
  --db-subnet-group-name aragora-subnet-group \
  --backup-retention-period 35 \
  --preferred-backup-window "03:00-04:00" \
  --deletion-protection \
  --storage-encrypted \
  --enable-cloudwatch-logs-exports '["postgresql", "upgrade"]' \
  --tags Key=Environment,Value=production Key=Application,Value=aragora
```

**Key Settings:**
- `--multi-az`: Enables synchronous replication and automatic failover
- `--backup-retention-period 35`: 35-day point-in-time recovery (max is 35)
- `--deletion-protection`: Prevents accidental deletion
- `--storage-encrypted`: Encryption at rest

### GCP Cloud SQL

```bash
# Create HA Cloud SQL instance
gcloud sql instances create aragora-postgres \
  --database-version=POSTGRES_16 \
  --tier=db-custom-2-7680 \
  --region=us-central1 \
  --availability-type=REGIONAL \
  --storage-auto-increase \
  --storage-size=100GB \
  --backup \
  --backup-start-time=03:00 \
  --retained-backups-count=35 \
  --enable-bin-log \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=04

# Set root password
gcloud sql users set-password postgres \
  --instance=aragora-postgres \
  --password="<STRONG_PASSWORD>"
```

### Supabase

Supabase provides fully managed PostgreSQL with:
- Automatic backups
- Point-in-time recovery
- Connection pooling via PgBouncer
- Built-in monitoring

```python
# Aragora already supports Supabase - set environment variables:
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your-service-role-key
# SUPABASE_DB_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
```

---

## Option 2: CloudNativePG Operator

For self-managed Kubernetes deployments, CloudNativePG provides production-grade PostgreSQL with:
- Automatic failover (< 30 seconds)
- Continuous WAL archiving
- Declarative backups to S3/GCS/Azure
- Rolling updates with no downtime

### Installation

```bash
# Install CloudNativePG operator
kubectl apply -f \
  https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/main/releases/cnpg-1.22.0.yaml

# Wait for operator to be ready
kubectl wait --for=condition=Available deployment/cnpg-controller-manager \
  -n cnpg-system --timeout=120s
```

### Cluster Configuration

Create `deploy/k8s/cnpg-cluster.yaml`:

```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: aragora-postgres
  namespace: aragora
spec:
  # HA Configuration
  instances: 3

  # Primary update strategy
  primaryUpdateStrategy: unsupervised

  # PostgreSQL configuration
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "768MB"
      maintenance_work_mem: "128MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      min_wal_size: "1GB"
      max_wal_size: "4GB"
      max_worker_processes: "4"
      max_parallel_workers_per_gather: "2"
      max_parallel_workers: "4"
      max_parallel_maintenance_workers: "2"
      log_statement: "ddl"
      log_min_duration_statement: "1000"  # Log queries > 1s

  # Storage
  storage:
    size: 100Gi
    storageClass: premium-rwo  # Adjust for your cluster

  # Resources
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2"

  # Backup configuration (S3 example)
  backup:
    barmanObjectStore:
      destinationPath: s3://aragora-backups/postgres
      s3Credentials:
        accessKeyId:
          name: postgres-backup-creds
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: postgres-backup-creds
          key: SECRET_ACCESS_KEY
      wal:
        compression: gzip
        maxParallel: 4
    retentionPolicy: "35d"

  # Bootstrap from backup (for disaster recovery)
  # bootstrap:
  #   recovery:
  #     source: clusterBackup

  # Affinity - spread across zones
  affinity:
    topologyKey: topology.kubernetes.io/zone
    nodeSelector:
      node-type: database

  # Monitoring
  monitoring:
    enablePodMonitor: true

---
# Scheduled backup
apiVersion: postgresql.cnpg.io/v1
kind: ScheduledBackup
metadata:
  name: aragora-postgres-backup
  namespace: aragora
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  cluster:
    name: aragora-postgres
  backupOwnerReference: self
```

### Apply Configuration

```bash
# Create backup credentials secret
kubectl create secret generic postgres-backup-creds \
  --namespace=aragora \
  --from-literal=ACCESS_KEY_ID=<your-access-key> \
  --from-literal=SECRET_ACCESS_KEY=<your-secret-key>

# Apply cluster configuration
kubectl apply -f deploy/k8s/cnpg-cluster.yaml

# Wait for cluster to be ready
kubectl wait cluster/aragora-postgres \
  --for=condition=Ready \
  --namespace=aragora \
  --timeout=300s

# Check cluster status
kubectl get cluster aragora-postgres -n aragora
kubectl get pods -l cnpg.io/cluster=aragora-postgres -n aragora
```

---

## Option 3: Manual Replication Setup

For environments where managed services and operators aren't available, you can set up streaming replication manually. This is **NOT recommended** for production due to complexity.

See the [PostgreSQL Streaming Replication Documentation](https://www.postgresql.org/docs/16/warm-standby.html) for details.

---

## Failover Testing

### CloudNativePG Failover Test

```bash
# Check current primary
kubectl cnpg status aragora-postgres -n aragora

# Simulate primary failure (kills primary pod)
kubectl delete pod aragora-postgres-1 -n aragora

# Monitor failover (should complete in < 30s)
kubectl get pods -l cnpg.io/cluster=aragora-postgres -n aragora -w

# Verify new primary
kubectl cnpg status aragora-postgres -n aragora
```

### Application-Level Testing

```python
# Test connection during failover
import asyncio
from aragora.db import get_db_connection

async def test_failover():
    """Test database connectivity during failover."""
    errors = 0
    successes = 0

    for i in range(60):  # 60 second test window
        try:
            async with get_db_connection() as conn:
                await conn.execute("SELECT 1")
                successes += 1
        except Exception as e:
            errors += 1
            print(f"Error at {i}s: {e}")
        await asyncio.sleep(1)

    print(f"Successes: {successes}, Errors: {errors}")
    print(f"Availability: {successes / (successes + errors) * 100:.1f}%")
```

---

## Aragora Configuration

### Environment Variables

```bash
# Connection string format for HA deployments
# For CloudNativePG, use the -rw service for writes
DATABASE_URL=postgresql://aragora:password@aragora-postgres-rw.aragora:5432/aragora

# Connection pool settings
DATABASE_POOL_MIN=5
DATABASE_POOL_MAX=20
DATABASE_POOL_RECYCLE=300

# Enable read replicas for read-heavy operations (optional)
DATABASE_READ_REPLICA_URL=postgresql://aragora:password@aragora-postgres-ro.aragora:5432/aragora
```

### Connection Pooling

For production, use PgBouncer or the built-in connection pooler:

```yaml
# PgBouncer sidecar for CloudNativePG
apiVersion: postgresql.cnpg.io/v1
kind: Pooler
metadata:
  name: aragora-pooler-rw
  namespace: aragora
spec:
  cluster:
    name: aragora-postgres
  instances: 3
  type: rw
  pgbouncer:
    poolMode: transaction
    parameters:
      max_client_conn: "1000"
      default_pool_size: "20"
```

---

## Backup and Recovery

### Backup Verification

```bash
# List available backups (CloudNativePG)
kubectl cnpg backup list aragora-postgres -n aragora

# Verify backup integrity
kubectl cnpg backup verify aragora-postgres-backup-XXXXXX -n aragora
```

### Point-in-Time Recovery

```yaml
# Recover to specific timestamp
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: aragora-postgres-recovery
spec:
  instances: 1
  bootstrap:
    recovery:
      source: aragora-postgres
      recoveryTarget:
        targetTime: "2025-01-15 10:30:00.00000+00"
  externalClusters:
    - name: aragora-postgres
      barmanObjectStore:
        destinationPath: s3://aragora-backups/postgres
        s3Credentials:
          # ...
```

---

## Monitoring

### Key Metrics

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `pg_up` | 0 | Database availability |
| `pg_replication_lag` | > 30s | Replication delay (seconds) |
| `pg_stat_activity_count` | > 180 (90% max) | Active connections |
| `pg_database_size_bytes` | > 80% storage | Database size |
| `pg_stat_statements_calls` | Trend | Query frequency |

### Prometheus Rules

```yaml
groups:
  - name: postgres
    rules:
      - alert: PostgresDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL instance is down"

      - alert: PostgresReplicationLag
        expr: pg_replication_lag_seconds > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL replication lag exceeds 30 seconds"

      - alert: PostgresHighConnections
        expr: pg_stat_activity_count > 180
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL connection count approaching limit"
```

### Grafana Dashboard

Import the CloudNativePG dashboard: [Dashboard ID: 20417](https://grafana.com/grafana/dashboards/20417)

---

## Migration Checklist

Before migrating to HA PostgreSQL:

- [ ] Test backup/restore procedure in staging
- [ ] Verify application reconnection handling
- [ ] Update connection strings in secrets
- [ ] Configure monitoring and alerting
- [ ] Document runbook for failover scenarios
- [ ] Schedule failover drill for the team
- [ ] Verify receipt data migration (compliance critical)
- [ ] Update `RUNBOOK_DATABASE_ISSUES.md` with HA-specific procedures

---

## Related Documentation

- [Redis High Availability Guide](./REDIS_HA.md)
- [Database Issues Runbook](./runbooks/RUNBOOK_DATABASE_ISSUES.md)
- [Production Readiness Checklist](./PRODUCTION_READINESS.md)
- [CloudNativePG Documentation](https://cloudnative-pg.io/documentation/)
