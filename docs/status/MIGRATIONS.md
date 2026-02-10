# Database Migrations

Aragora uses a lightweight migration system that supports both PostgreSQL and SQLite backends. Migrations are versioned Python modules located in `aragora/migrations/versions/`.

## Overview

The migration system provides:

- **Dual-tier execution** -- PostgreSQL (primary) and SQLite (local/dev) migrations run independently via `aragora.migrations.runner` and `aragora.persistence.migrations.runner` respectively.
- **SQL and Python migrations** -- Each migration defines `up_sql`/`down_sql` strings or `up_fn`/`down_fn` callables.
- **Advisory locking** -- PostgreSQL uses `pg_try_advisory_lock` to prevent concurrent migration runs across pods.
- **Checksum verification** -- SHA-256 checksums detect modified migration files after they have been applied.
- **Rollback history** -- All rollbacks are recorded with timestamp, operator, and reason for audit.
- **Zero-downtime patterns** -- `aragora.migrations.patterns` provides helpers for expand/contract migrations, safe column operations, and batched backfills.

## Running Migrations

### Auto-migration on startup

Set the environment variable before starting the server:

```bash
ARAGORA_AUTO_MIGRATE_ON_STARTUP=true python -m aragora.server.unified_server
```

This runs pending migrations for both PostgreSQL and SQLite backends. Failures are logged but do not block server startup.

### Manual migration commands

```bash
# Apply all pending migrations
python -m aragora.migrations upgrade

# Apply up to a specific version
python -m aragora.migrations upgrade --target 20260120000000

# Check current status
python -m aragora.migrations status

# Create a new migration file
python -m aragora.migrations create "Add users table"
```

### Specifying a database

```bash
# PostgreSQL (via URL or DATABASE_URL env var)
python -m aragora.migrations upgrade --database-url postgresql://user:pass@host/db

# SQLite (default: aragora.db)
python -m aragora.migrations upgrade --db-path /path/to/aragora.db
```

## Rollback Procedures

All rollback commands support `--dry-run` to preview changes and `--reason` to record why the rollback was performed.

### Single step rollback

Rolls back the most recently applied migration:

```bash
python -m aragora.migrations downgrade --reason "broke user signups"
```

### N-step rollback

Rolls back a specific number of recent migrations:

```bash
python -m aragora.migrations downgrade --steps 3 --reason "bad batch deploy"
```

### Target version rollback

Rolls back everything above a given version:

```bash
python -m aragora.migrations downgrade --target 20260119000000 --reason "reverting to stable"
```

### Dry-run preview

```bash
python -m aragora.migrations downgrade --steps 2 --dry-run
```

### Using stored rollback SQL

If migration files have been modified or lost, use the rollback SQL that was stored in the database at apply time:

```bash
python -m aragora.migrations downgrade --use-stored-rollback
```

### Viewing rollback history

```bash
python -m aragora.migrations rollback-history
```

## Safety Features

### Checksum verification

When `verify_checksums=True` (default), the runner computes SHA-256 checksums of each migration's content and compares them to the checksums stored at apply time. If a migration file was modified after being applied, upgrade will raise a `ValueError` and refuse to proceed.

### Advisory locking (PostgreSQL)

The runner calls `pg_try_advisory_lock(2089872453)` before applying or rolling back migrations. This prevents concurrent runs across multiple server instances. The lock times out after 30 seconds by default. SQLite relies on its inherent file-level locking.

### Rollback validation

Before executing a rollback, the runner performs pre-flight validation (`validate_rollback`). It checks that:

- Migrations to roll back actually exist and are applied.
- Each migration has a `down_sql` or `down_fn` defined.
- The target version is valid.

Warnings are printed for non-blocking issues; errors block the rollback entirely.

### Atomic transactions

Each migration runs within a transaction. If a migration fails, the transaction is rolled back and no partial state is committed. The migration lock is always released in a `finally` block.

## Writing Migrations

Create a new migration:

```bash
python -m aragora.migrations create "Add email_verified column"
```

This generates a timestamped file in `aragora/migrations/versions/`. Edit the file to add your SQL or Python logic:

```python
from aragora.migrations.runner import Migration

migration = Migration(
    version=20260201120000,
    name="Add email_verified column",
    up_sql="ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE;",
    down_sql="ALTER TABLE users DROP COLUMN email_verified;",
)
```

For complex migrations, use Python functions and the zero-downtime patterns:

```python
from aragora.migrations.runner import Migration
from aragora.migrations.patterns import safe_add_nullable_column, backfill_column

def up(backend):
    safe_add_nullable_column(backend, "users", "email_verified", "BOOLEAN")
    backfill_column(backend, "users", "email_verified", "FALSE", batch_size=1000)

def down(backend):
    backend.execute_write("ALTER TABLE users DROP COLUMN email_verified")

migration = Migration(version=20260201120000, name="Add email_verified", up_fn=up, down_fn=down)
```

## Best Practices

1. **Test locally first.** Run `upgrade` and `downgrade` against a local database before deploying.
2. **Keep migrations idempotent.** Use `IF NOT EXISTS` / `IF EXISTS` guards where possible.
3. **Always define `down_sql` or `down_fn`.** Migrations without rollback logic cannot be safely reversed.
4. **Never modify an applied migration.** Checksum verification will flag the mismatch. Create a new migration instead.
5. **Back up data before migrating.** The auto-migration system creates a backup when the backup manager is available. For manual runs, back up explicitly.
6. **Use the expand/contract pattern** for zero-downtime column changes on large tables (see `aragora/migrations/patterns.py`).
7. **Preview with `--dry-run`** before running rollbacks in production.

## Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `Migration lock acquisition timeout` | Another migration process holds the advisory lock. | Wait for it to finish, or check for stuck connections: `SELECT * FROM pg_locks WHERE objid = 2089872453;` then terminate if stale: `SELECT pg_terminate_backend(pid);` |
| `Migration checksum mismatch` | A migration file was edited after it was applied. | Do not modify applied migrations. Create a new corrective migration instead. If intentional, manually update the checksum in the `_aragora_migrations` table. |
| `No pending migrations` but schema is wrong | Migration was recorded but SQL partially failed (rare). | Check the `_aragora_migrations` table for the version in question. Remove the record and re-run, or apply a corrective migration. |
| Rollback validation fails with missing `down_sql` | Migration was written without rollback logic. | Write the rollback SQL manually and use `--use-stored-rollback` or add `down_sql`/`down_fn` to the migration file. |
| Auto-migration not running on startup | `ARAGORA_AUTO_MIGRATE_ON_STARTUP` is not set to `true`. | Export the variable: `export ARAGORA_AUTO_MIGRATE_ON_STARTUP=true` |
