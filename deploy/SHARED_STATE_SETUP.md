# Shared State Setup: PostgreSQL + Redis for Multi-Instance Aragora

Configures two EC2 instances behind an ALB to share a single PostgreSQL database
and a single Redis instance, replacing the current per-instance SQLite and
local Redis.

## 1. Environment Variables (set on BOTH instances)

Add to `/etc/aragora/env` or a systemd drop-in at
`/etc/systemd/system/aragora.service.d/shared-state.conf`:

```ini
[Service]
# --- PostgreSQL ---
# The code checks ARAGORA_POSTGRES_DSN -> DATABASE_URL -> AWS Secrets Manager
# (see aragora/storage/postgres_store.py lines 291-301)
Environment="DATABASE_URL=postgresql://aragora:<PASSWORD>@<RDS_ENDPOINT>:5432/aragora?sslmode=require"
Environment="ARAGORA_DB_BACKEND=postgresql"

# Pool tuning (defaults shown; adjust per instance)
# Total connections across both instances = 2 * (pool_size + pool_overflow)
# RDS default max_connections for db.r6g.large is 1600, so 70 per instance is safe.
Environment="ARAGORA_DB_POOL_SIZE=20"
Environment="ARAGORA_DB_POOL_OVERFLOW=15"
Environment="ARAGORA_DB_COMMAND_TIMEOUT=60"
Environment="ARAGORA_DB_STATEMENT_TIMEOUT=60"
Environment="ARAGORA_DB_POOL_RECYCLE=1800"

# --- Redis ---
# The code checks ARAGORA_REDIS_URL -> REDIS_URL
# (see aragora/config/redis.py line 227, aragora/server/redis_config.py line 43)
Environment="ARAGORA_REDIS_URL=rediss://<ELASTICACHE_ENDPOINT>:6379"

# --- Distributed mode flags ---
# Prevents silent fallback to SQLite (see deploy/systemd/distributed.conf)
Environment="ARAGORA_REQUIRE_DISTRIBUTED=true"
Environment="ARAGORA_MULTI_INSTANCE=true"
Environment="ARAGORA_SINGLE_INSTANCE=false"
Environment="ARAGORA_ENV=production"
```

After editing, reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart aragora
```

Alternatively, if using `/etc/aragora/env` as an `EnvironmentFile`:

```bash
# /etc/aragora/env
DATABASE_URL=postgresql://aragora:<PASSWORD>@<RDS_ENDPOINT>:5432/aragora?sslmode=require
ARAGORA_DB_BACKEND=postgresql
ARAGORA_DB_POOL_SIZE=20
ARAGORA_DB_POOL_OVERFLOW=15
ARAGORA_REDIS_URL=rediss://<ELASTICACHE_ENDPOINT>:6379
ARAGORA_REQUIRE_DISTRIBUTED=true
ARAGORA_MULTI_INSTANCE=true
ARAGORA_SINGLE_INSTANCE=false
ARAGORA_ENV=production
```

For secrets, set `ARAGORA_USE_SECRETS_MANAGER=true` and store `DATABASE_URL`
and `ARAGORA_REDIS_URL` in AWS Secrets Manager instead of plaintext files
(the code falls back to Secrets Manager automatically -- see
`aragora/config/secrets.py`).

## 2. AWS RDS PostgreSQL Setup

| Setting | Recommended Value |
|---------|-------------------|
| Engine | PostgreSQL 16.x |
| Instance type | `db.r6g.large` (2 vCPU, 16 GB) for production; `db.t4g.medium` for staging |
| Storage | 100 GB gp3, autoscaling enabled |
| Multi-AZ | Yes (automatic failover) |
| Subnet group | Same VPC as EC2 instances, private subnets only |
| Security group | Inbound TCP 5432 from EC2 security group only |
| Parameter group | Custom, based on `deploy/postgres/postgresql.conf` |
| Encryption | At rest (KMS) + in transit (require SSL via `sslmode=require` in DSN) |
| Backups | Automated, 7-day retention, preferred window outside business hours |
| Database name | `aragora` |
| Master user | `aragora` (use Secrets Manager for the password) |

Key parameter group overrides (from `deploy/postgres/postgresql.conf`):

```
shared_buffers           = {DBInstanceClassMemory/4}   # 25% of instance RAM
work_mem                 = 16MB
maintenance_work_mem     = 256MB
max_connections          = 200
log_min_duration_statement = 1000
checkpoint_completion_target = 0.9
random_page_cost         = 1.1
effective_io_concurrency = 200
```

Schema initialization happens automatically on first connection -- the
`PostgresStore.initialize()` method creates tables via `INITIAL_SCHEMA` with
`CREATE TABLE IF NOT EXISTS` and tracks versions in `_schema_versions`.

## 3. AWS ElastiCache Redis Setup

| Setting | Recommended Value |
|---------|-------------------|
| Engine | Redis OSS 7.x |
| Node type | `cache.r6g.large` for production; `cache.t4g.medium` for staging |
| Cluster mode | Disabled (single shard, single primary) |
| Replicas | 1 read replica (automatic failover) |
| Multi-AZ | Yes |
| Subnet group | Same VPC as EC2, private subnets |
| Security group | Inbound TCP 6379 from EC2 security group only |
| Encryption | In transit (TLS) + at rest; use `rediss://` scheme in URL |
| Auth | AUTH token (store in Secrets Manager, append to URL) |
| Parameter group | `maxmemory-policy = allkeys-lru` |
| Backup | Daily snapshots, 3-day retention |

The URL format with auth:

```
rediss://:<AUTH_TOKEN>@<primary-endpoint>:6379
```

## 4. Verification

### 4a. Health endpoint checks

After restarting both instances, hit these endpoints through the ALB:

```bash
# Fast readiness probe -- should return 200 with db_pool and redis_pool checks
curl -s https://<ALB>/readyz | jq .
# Expected:
# {
#   "status": "ready",
#   "checks": {
#     "degraded_mode": true,
#     "storage_initialized": true,
#     "elo_initialized": true,
#     "redis_pool": true,
#     "db_pool": true
#   },
#   ...
# }

# Full dependency validation -- validates actual Redis and PostgreSQL connectivity
curl -s https://<ALB>/readyz/dependencies | jq .
# Expected: "status": "ready", redis.connected: true

# Liveness probe
curl -s https://<ALB>/healthz | jq .
# Expected: {"status": "ok"}
```

### 4b. Verify both instances share state

```bash
# Run a debate on instance A (via ALB or direct)
curl -X POST https://<ALB>/api/v1/debates -d '{"task":"test shared state"}' -H 'Content-Type: application/json'

# Fetch it from instance B (pin to B via direct IP or header)
curl https://<INSTANCE_B_IP>:8080/api/v1/debates?limit=1
# Should return the same debate
```

### 4c. Verify PostgreSQL connection pool metrics

```bash
# Pool metrics endpoint (if exposed)
curl -s https://<ALB>/api/health | jq '.checks'
```

Or connect to RDS directly and check active connections:

```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'aragora';
-- Should show connections from both instance IPs
```

## 5. Data Migration (SQLite to PostgreSQL)

If debate data currently exists in local SQLite and must be preserved:

### 5a. Identify existing SQLite files

```bash
# On each EC2 instance, find SQLite databases:
find /home/aragora -name "*.db" -type f 2>/dev/null
# Common locations (from aragora/config/settings.py DatabaseSettings):
#   agent_elo.db, continuum.db, aragora_insights.db,
#   consensus_memory.db, agent_personas.db, grounded_positions.db, genesis.db
```

### 5b. Export and import approach

There is no built-in SQLite-to-PostgreSQL migration tool in the codebase. The
recommended approach:

1. **Stop traffic** -- remove one instance from the ALB target group.
2. **Dump SQLite data** on that instance:
   ```bash
   sqlite3 /home/aragora/aragora/agent_elo.db ".mode csv" ".headers on" ".output elo_export.csv" "SELECT * FROM elo_ratings;"
   ```
3. **Import into RDS** using `psql` or `\copy`:
   ```bash
   psql "$DATABASE_URL" -c "\copy elo_ratings FROM 'elo_export.csv' CSV HEADER"
   ```
4. Repeat for each table that has meaningful data.
5. Re-add the instance to the ALB target group.

For most deployments, the local SQLite data is ephemeral (session-level debate
state, ELO seeds). If the deployment is new or data loss is acceptable, skip
migration entirely -- the `PostgresStore.initialize()` method creates fresh
schemas on first use.

### 5c. Post-migration cleanup

After confirming PostgreSQL is working and the ALB routes traffic correctly:

```bash
# On each instance, remove or archive local SQLite files
sudo -u aragora mv /home/aragora/aragora/*.db /home/aragora/backup/
```

## Quick Checklist

- [ ] RDS instance created with Multi-AZ, encryption, and correct security group
- [ ] ElastiCache Redis created with TLS, Multi-AZ, and correct security group
- [ ] `DATABASE_URL` set identically on both EC2 instances
- [ ] `ARAGORA_REDIS_URL` set identically on both EC2 instances
- [ ] `ARAGORA_DB_BACKEND=postgresql` set on both instances
- [ ] `ARAGORA_REQUIRE_DISTRIBUTED=true` set on both instances
- [ ] Both instances restarted with `systemctl restart aragora`
- [ ] `/readyz` returns `"status": "ready"` on both instances
- [ ] `/readyz/dependencies` shows Redis and PostgreSQL connected
- [ ] ALB health check target set to `/healthz` (HTTP 200 = healthy)
- [ ] Cross-instance data visibility confirmed (debate created on A visible on B)
