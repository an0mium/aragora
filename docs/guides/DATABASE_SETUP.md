# Database Setup Guide

Aragora supports both SQLite (development) and PostgreSQL (production). For production deployments with multiple replicas, PostgreSQL is required.

## Quick Start

### Development (SQLite - Default)

No setup required. SQLite databases are created automatically in the `.nomic/` directory.

### Production (Managed PostgreSQL)

1. Create a managed PostgreSQL instance (see provider options below)
2. Set `DATABASE_URL` environment variable
3. Run migrations
4. Deploy

```bash
# Set connection string
export DATABASE_URL="postgresql://user:pass@host:5432/aragora?sslmode=require"

# Run migrations (if you have existing SQLite data)
python scripts/migrate_sqlite_to_postgres.py

# Start server (auto-detects PostgreSQL from DATABASE_URL)
aragora serve
```

## Managed PostgreSQL Options

### Supabase (Recommended)

Supabase provides a generous free tier and easy setup.

1. Create account at [supabase.com](https://supabase.com)
2. Create new project
3. Go to **Settings > Database**
4. Copy the **Connection string** (URI format)

```bash
# Example Supabase connection string
DATABASE_URL="postgresql://postgres.[project-ref]:[password]@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
```

**Connection Pooling**: Use the "Transaction pooler" endpoint (port 6543) for serverless/Kubernetes deployments.

### AWS RDS

1. Create RDS PostgreSQL instance in AWS Console
2. Configure security group for your cluster's IP range
3. Enable SSL (recommended)

```bash
DATABASE_URL="postgresql://admin:password@mydb.xxxx.us-west-2.rds.amazonaws.com:5432/aragora?sslmode=require"
```

### Neon

Serverless PostgreSQL with auto-scaling:

```bash
DATABASE_URL="postgresql://user:pass@ep-xxx.us-west-2.aws.neon.tech/neondb?sslmode=require"
```

### Railway

Simple deployment with PostgreSQL:

```bash
# Railway provides DATABASE_URL automatically when you add PostgreSQL plugin
```

### Self-Hosted PostgreSQL

For Kubernetes deployments, you can use the provided Redis StatefulSet as a template:

```yaml
# deploy/k8s/postgres/statefulset.yaml (example)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  template:
    spec:
      containers:
        - name: postgres
          image: postgres:16
          env:
            - name: POSTGRES_DB
              value: aragora
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: aragora-secrets
                  key: postgres-user
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: aragora-secrets
                  key: postgres-password
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

## Configuration

### Environment Variables

```bash
# Option 1: Connection URL (recommended for managed services)
DATABASE_URL=postgresql://user:pass@host:5432/dbname?sslmode=require

# Option 2: Individual settings (for fine-grained control)
ARAGORA_DB_BACKEND=postgres
ARAGORA_PG_HOST=localhost
ARAGORA_PG_PORT=5432
ARAGORA_PG_DATABASE=aragora
ARAGORA_PG_USER=aragora
ARAGORA_PG_PASSWORD=secret
ARAGORA_PG_SSL_MODE=require

# Connection pool settings
ARAGORA_DB_POOL_SIZE=10          # Max connections
ARAGORA_DB_POOL_MAX_OVERFLOW=5   # Extra connections during burst
ARAGORA_DB_POOL_TIMEOUT=30.0     # Connection timeout (seconds)
```

### SSL Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `disable` | No SSL | Local development only |
| `prefer` | Try SSL, fall back to unencrypted | Testing |
| `require` | Require SSL, don't verify cert | **Recommended for managed** |
| `verify-ca` | Require SSL, verify CA | High security |
| `verify-full` | Require SSL, verify CA + hostname | Maximum security |

## Migrating from SQLite

### Using Migration Script

```bash
# Dry run (see what would be migrated)
python scripts/migrate_sqlite_to_postgres.py --dry-run

# Migrate all databases
python scripts/migrate_sqlite_to_postgres.py

# Migrate specific database
python scripts/migrate_sqlite_to_postgres.py --database debates

# With explicit paths
python scripts/migrate_sqlite_to_postgres.py \
  --source-dir .nomic \
  --target-url postgresql://...
```

### Manual Migration

For complex schemas or selective migration:

```bash
# 1. Export SQLite data
sqlite3 .nomic/aragora.db ".dump" > dump.sql

# 2. Convert syntax (manual adjustments may be needed)
# - Replace INTEGER PRIMARY KEY AUTOINCREMENT with BIGSERIAL PRIMARY KEY
# - Replace BOOLEAN defaults (1 -> TRUE, 0 -> FALSE)
# - Convert datetime formats

# 3. Import to PostgreSQL
psql $DATABASE_URL < dump.sql
```

## Kubernetes Deployment

### Using External Secrets

For managed PostgreSQL in Kubernetes, use External Secrets Operator:

```yaml
# deploy/k8s/external-secrets/database-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: aragora-database
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager  # or vault, gcp, etc.
    kind: ClusterSecretStore
  target:
    name: aragora-database-secret
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: aragora/production/database
        property: url
```

### ConfigMap for Non-Secret Settings

```yaml
# deploy/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aragora-config
data:
  ARAGORA_DB_POOL_SIZE: "20"
  ARAGORA_DB_POOL_MAX_OVERFLOW: "10"
  ARAGORA_DB_POOL_TIMEOUT: "30"
```

## Health Checks

### Connection Test

```bash
# Using psql
psql $DATABASE_URL -c "SELECT 1"

# Using Python
python -c "
from aragora.db import get_database
db = get_database()
print('Connected:', db.health_check())
"
```

### From Kubernetes

```bash
# Check database health via API
kubectl exec -it deploy/aragora -- curl localhost:8080/api/health/db

# Or directly test connection
kubectl exec -it deploy/aragora -- python -c "
import os
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Connected!')
conn.close()
"
```

## Troubleshooting

### Connection Refused

1. Check security group/firewall rules allow your IP
2. Verify the host and port are correct
3. For managed services, ensure SSL is enabled

### SSL Certificate Error

```bash
# Use sslmode=require for managed services (trusts provider's cert)
DATABASE_URL="postgresql://...?sslmode=require"
```

### Connection Pool Exhausted

Increase pool settings:

```bash
ARAGORA_DB_POOL_SIZE=30
ARAGORA_DB_POOL_MAX_OVERFLOW=20
```

### Slow Queries

1. Enable query logging in PostgreSQL
2. Add indexes for frequently filtered columns
3. Consider connection pooling with PgBouncer

### Migration Failures

```bash
# Check migration logs
python scripts/migrate_sqlite_to_postgres.py --dry-run

# Common issues:
# - Type mismatches (JSON stored as string in SQLite)
# - Datetime format differences
# - Foreign key constraints
```

## Backup and Recovery

### Managed Services

Most managed PostgreSQL services provide automated backups. Check your provider's documentation.

### Manual Backup

```bash
# Full backup
pg_dump $DATABASE_URL > backup.sql

# Compressed backup
pg_dump $DATABASE_URL | gzip > backup.sql.gz

# Restore
psql $DATABASE_URL < backup.sql
```

### Point-in-Time Recovery

For production, enable WAL archiving in your managed service for point-in-time recovery capabilities.

---

## Schema Migrations

Aragora uses a forward-only migration strategy for schema changes.

### Migration Runner

Schema migrations are located in `aragora/persistence/migrations/` and run via:

```bash
# Run pending migrations
python -m aragora.persistence.migrations.runner migrate

# Check migration status
python -m aragora.persistence.migrations.runner status

# Run specific migration
python -m aragora.persistence.migrations.runner migrate --version 20260115_add_workspace_id
```

### Migration Best Practices

1. **Always backup first**: `pg_dump $DATABASE_URL > pre_migration_backup.sql`
2. **Test in staging**: Run migrations against a staging database first
3. **Deploy during low-traffic**: Schedule migrations during maintenance windows
4. **Use transactions**: All migrations run in transactions for atomicity

### Rollback Strategy

Aragora uses **forward-only migrations** with compensating changes. Instead of traditional rollback:

1. **For failed migrations**: The transaction is automatically rolled back
2. **For reverting changes**: Create a new migration that undoes the previous one
3. **For emergencies**: Restore from backup (see Backup and Recovery section)

**Why forward-only?**
- Simpler mental model
- Rollback scripts often untested
- Compensating migrations are explicit and reviewable
- Matches modern deployment practices (blue-green, canary)

**Example compensating migration:**

```python
# migrations/20260120_remove_deprecated_column.py
def upgrade(conn):
    conn.execute("ALTER TABLE users DROP COLUMN IF EXISTS deprecated_field")

# To "rollback", create a new migration:
# migrations/20260121_restore_deprecated_column.py
def upgrade(conn):
    conn.execute("ALTER TABLE users ADD COLUMN deprecated_field TEXT")
```

### Emergency Recovery

If a migration causes production issues:

1. **Immediate**: Restore from latest backup
2. **After analysis**: Create compensating migration
3. **Deploy**: Roll forward with the fix
