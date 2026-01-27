# PostgreSQL Replication Setup Runbook

**Purpose:** Set up PostgreSQL streaming replication for high availability.
**Audience:** DevOps, SRE, Platform Engineers
**Last Updated:** January 2026

---

## Overview

This runbook covers setting up PostgreSQL streaming replication with:
- Primary-replica architecture
- Automatic failover with Patroni (optional)
- Manual failover procedures
- Monitoring replication lag

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Load Balancer                              │
│                    (HAProxy/PgBouncer)                          │
└─────────────────────┬───────────────────┬───────────────────────┘
                      │                   │
                      ▼                   ▼
         ┌────────────────────┐ ┌────────────────────┐
         │     Primary        │ │     Replica        │
         │   (Read/Write)     │ │   (Read Only)      │
         │                    │ │                    │
         │  PostgreSQL 15     │◀│  PostgreSQL 15     │
         │  WAL Streaming     │ │  Hot Standby       │
         └────────────────────┘ └────────────────────┘
                      │                   ▲
                      │    WAL Segments   │
                      └───────────────────┘
```

---

## Prerequisites

| Requirement | Primary | Replica |
|-------------|---------|---------|
| PostgreSQL | 14+ (15 recommended) | Same version |
| Network | Reachable from replica | Reachable from primary |
| Disk Space | Production data + WAL | 1.5x primary size |
| RAM | 8GB minimum | Same as primary |

---

## Phase 1: Primary Server Setup

### 1.1 Configure postgresql.conf

```bash
# /etc/postgresql/15/main/postgresql.conf

# Replication settings
wal_level = replica
max_wal_senders = 10
wal_keep_size = 1GB
max_replication_slots = 10

# Synchronous replication (optional, for zero data loss)
# synchronous_commit = on
# synchronous_standby_names = 'replica1'

# Performance tuning
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 256MB
maintenance_work_mem = 1GB

# WAL settings
wal_buffers = 64MB
checkpoint_completion_target = 0.9
```

### 1.2 Configure pg_hba.conf

```bash
# /etc/postgresql/15/main/pg_hba.conf

# Allow replication connections from replica
host    replication     replicator      10.0.0.0/24     scram-sha-256
host    replication     replicator      replica_ip/32   scram-sha-256

# Allow application connections
host    aragora         aragora_app     10.0.0.0/24     scram-sha-256
```

### 1.3 Create Replication User

```sql
-- Connect as superuser
psql -U postgres

-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'secure_replication_password';

-- Create replication slot (prevents WAL removal)
SELECT pg_create_physical_replication_slot('replica1_slot');

-- Verify slot
SELECT slot_name, active FROM pg_replication_slots;
```

### 1.4 Restart Primary

```bash
sudo systemctl restart postgresql

# Verify configuration
psql -U postgres -c "SHOW wal_level;"
# Expected: replica

psql -U postgres -c "SHOW max_wal_senders;"
# Expected: 10
```

---

## Phase 2: Replica Server Setup

### 2.1 Stop PostgreSQL on Replica

```bash
sudo systemctl stop postgresql
```

### 2.2 Clear Existing Data

```bash
# Backup existing data if needed
sudo mv /var/lib/postgresql/15/main /var/lib/postgresql/15/main.backup

# Create empty data directory
sudo mkdir /var/lib/postgresql/15/main
sudo chown postgres:postgres /var/lib/postgresql/15/main
sudo chmod 700 /var/lib/postgresql/15/main
```

### 2.3 Base Backup from Primary

```bash
# Run as postgres user
sudo -u postgres pg_basebackup \
    -h primary_ip \
    -U replicator \
    -D /var/lib/postgresql/15/main \
    -Fp \
    -Xs \
    -P \
    -R \
    -S replica1_slot

# -Fp: Plain format
# -Xs: Stream WAL during backup
# -P: Show progress
# -R: Create standby.signal and postgresql.auto.conf
# -S: Use replication slot
```

### 2.4 Configure postgresql.auto.conf

The `pg_basebackup -R` creates this automatically. Verify contents:

```bash
cat /var/lib/postgresql/15/main/postgresql.auto.conf

# Should contain:
# primary_conninfo = 'host=primary_ip port=5432 user=replicator password=xxx'
# primary_slot_name = 'replica1_slot'
```

### 2.5 Configure Replica postgresql.conf

```bash
# /etc/postgresql/15/main/postgresql.conf

# Enable hot standby (read-only queries on replica)
hot_standby = on

# Feedback to primary (prevents query conflicts)
hot_standby_feedback = on

# Recovery settings
recovery_min_apply_delay = 0
# Set to '5min' for delayed replica (disaster recovery)
```

### 2.6 Start Replica

```bash
sudo systemctl start postgresql

# Check logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log
# Should show: "entering standby mode"
# "started streaming WAL from primary"
```

---

## Phase 3: Verify Replication

### 3.1 Check Primary Status

```sql
-- On primary
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    sync_state
FROM pg_stat_replication;

-- Expected output:
-- client_addr | state     | sync_state
-- 10.0.0.2    | streaming | async
```

### 3.2 Check Replica Status

```sql
-- On replica
SELECT
    pg_is_in_recovery(),
    pg_last_wal_receive_lsn(),
    pg_last_wal_replay_lsn(),
    pg_last_xact_replay_timestamp();

-- pg_is_in_recovery should be 't' (true)
```

### 3.3 Check Replication Lag

```sql
-- On replica
SELECT
    CASE
        WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn()
        THEN 0
        ELSE EXTRACT(EPOCH FROM now() - pg_last_xact_replay_timestamp())
    END AS replication_lag_seconds;

-- Should be < 1 second for healthy replication
```

### 3.4 Test Data Sync

```sql
-- On primary
INSERT INTO test_replication (id, data) VALUES (1, 'test');

-- On replica (within seconds)
SELECT * FROM test_replication WHERE id = 1;
```

---

## Phase 4: Connection Pooling with PgBouncer

### 4.1 Install PgBouncer

```bash
sudo apt install pgbouncer
```

### 4.2 Configure PgBouncer

```ini
# /etc/pgbouncer/pgbouncer.ini

[databases]
# Primary for writes
aragora = host=primary_ip port=5432 dbname=aragora

# Replica for reads
aragora_ro = host=replica_ip port=5432 dbname=aragora

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5

# Health checks
server_check_query = SELECT 1
server_check_delay = 30
```

### 4.3 Configure Authentication

```bash
# /etc/pgbouncer/userlist.txt
"aragora_app" "SCRAM-SHA-256$4096:salt$storedkey:serverkey"

# Generate password hash:
psql -U postgres -c "SELECT rolname, rolpassword FROM pg_authid WHERE rolname='aragora_app';"
```

### 4.4 Application Connection Strings

```bash
# Read/Write (primary)
DATABASE_URL="postgresql://aragora_app:password@pgbouncer:6432/aragora"

# Read-only (replica)
DATABASE_URL_RO="postgresql://aragora_app:password@pgbouncer:6432/aragora_ro"
```

---

## Phase 5: Manual Failover

### 5.1 Planned Failover

```bash
# 1. Stop writes on primary (application-level)
# 2. Wait for replica to catch up
psql -h replica_ip -U postgres -c "SELECT pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn();"

# 3. Promote replica
sudo -u postgres pg_ctl promote -D /var/lib/postgresql/15/main

# Or via SQL (PostgreSQL 12+)
psql -U postgres -c "SELECT pg_promote();"

# 4. Verify promotion
psql -U postgres -c "SELECT pg_is_in_recovery();"
# Should be 'f' (false)

# 5. Update DNS/load balancer to point to new primary

# 6. Reconfigure old primary as replica (see Phase 2)
```

### 5.2 Emergency Failover

```bash
# 1. Promote replica immediately
sudo -u postgres pg_ctl promote -D /var/lib/postgresql/15/main

# 2. Update application connection strings
export DATABASE_URL="postgresql://aragora_app:password@new_primary:5432/aragora"

# 3. Restart application
kubectl rollout restart deployment/aragora

# 4. Investigate old primary
# DO NOT restart old primary until properly reconfigured as replica
```

### 5.3 Failback Procedure

```bash
# After recovering old primary, convert it to replica:

# 1. Stop old primary
sudo systemctl stop postgresql

# 2. Clear data
sudo rm -rf /var/lib/postgresql/15/main/*

# 3. Base backup from new primary
sudo -u postgres pg_basebackup -h new_primary_ip -U replicator \
    -D /var/lib/postgresql/15/main -Fp -Xs -P -R -S replica1_slot

# 4. Start as replica
sudo systemctl start postgresql
```

---

## Monitoring

### Prometheus Metrics

```yaml
# prometheus-postgres-exporter queries

pg_replication_lag:
  query: |
    SELECT
      CASE
        WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn() THEN 0
        ELSE EXTRACT(EPOCH FROM now() - pg_last_xact_replay_timestamp())
      END as lag_seconds
  metrics:
    - lag_seconds:
        usage: "GAUGE"
        description: "Replication lag in seconds"
```

### Alert Rules

```yaml
# alertmanager rules

groups:
  - name: postgresql_replication
    rules:
      - alert: PostgreSQLReplicationLag
        expr: pg_replication_lag_seconds > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL replication lag > 30s"

      - alert: PostgreSQLReplicationDown
        expr: pg_replication_is_replica == 1 and pg_replication_lag_seconds < 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL replication stream disconnected"
```

### Health Check Script

```bash
#!/bin/bash
# /usr/local/bin/check_replication.sh

LAG=$(psql -U postgres -t -c "
SELECT COALESCE(
    EXTRACT(EPOCH FROM now() - pg_last_xact_replay_timestamp()),
    -1
)::int;")

if [ "$LAG" -lt 0 ]; then
    echo "CRITICAL: Replication not running"
    exit 2
elif [ "$LAG" -gt 60 ]; then
    echo "WARNING: Replication lag ${LAG}s"
    exit 1
else
    echo "OK: Replication lag ${LAG}s"
    exit 0
fi
```

---

## Troubleshooting

### Replication Not Starting

```bash
# Check logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Common issues:
# - Wrong password in primary_conninfo
# - pg_hba.conf not allowing replication
# - Firewall blocking port 5432
# - Replication slot doesn't exist

# Test connectivity
psql -h primary_ip -U replicator -c "IDENTIFY_SYSTEM" replication=1
```

### Replication Slot Full

```sql
-- Check slot status
SELECT slot_name, active, pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) as retained_wal
FROM pg_replication_slots;

-- If replica is permanently down, drop the slot
SELECT pg_drop_replication_slot('replica1_slot');
```

### Split Brain Prevention

```bash
# If both servers think they're primary:

# 1. Stop BOTH servers
sudo systemctl stop postgresql

# 2. Determine which has more recent data
psql -U postgres -c "SELECT pg_last_wal_replay_lsn();"

# 3. Keep the one with more recent LSN as primary
# 4. Reconfigure the other as replica
```

### Query Conflicts on Replica

```sql
-- On replica, check for query conflicts
SELECT * FROM pg_stat_database_conflicts;

-- Increase max_standby_streaming_delay if reads are cancelled
-- /etc/postgresql/15/main/postgresql.conf
max_standby_streaming_delay = 30s
```

---

## Reference

### Key Files

| File | Purpose |
|------|---------|
| `/etc/postgresql/15/main/postgresql.conf` | Main configuration |
| `/etc/postgresql/15/main/pg_hba.conf` | Authentication rules |
| `/var/lib/postgresql/15/main/standby.signal` | Indicates replica mode |
| `/var/lib/postgresql/15/main/postgresql.auto.conf` | Replication connection |

### Useful Commands

```bash
# Check if server is primary or replica
psql -U postgres -c "SELECT pg_is_in_recovery();"

# Show connected replicas (on primary)
psql -U postgres -c "SELECT * FROM pg_stat_replication;"

# Show replication status (on replica)
psql -U postgres -c "SELECT * FROM pg_stat_wal_receiver;"

# Show replication slots
psql -U postgres -c "SELECT * FROM pg_replication_slots;"
```

### Related Documentation

- [RUNBOOK_POSTGRESQL_MIGRATION.md](./RUNBOOK_POSTGRESQL_MIGRATION.md) - Initial setup
- [DISASTER_RECOVERY.md](../DISASTER_RECOVERY.md) - Recovery procedures
- [monitoring-setup.md](./monitoring-setup.md) - Monitoring configuration

---

**Document Owner:** Platform Team
**Review Cycle:** Quarterly
