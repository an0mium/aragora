# Runbook: Database Consolidation Migration

## Overview

This runbook covers the migration from 22 legacy SQLite databases to 4 consolidated databases:
- `core.db` - debates, traces, tournaments, embeddings, positions
- `memory.db` - continuum, consensus, critiques, patterns
- `analytics.db` - elo, calibration, insights, prompt evolution
- `agents.db` - personas, relationships, genesis, experiments

## Pre-Migration Checklist

- [ ] Backup current databases
- [ ] Verify disk space (2x current database size)
- [ ] Schedule maintenance window
- [ ] Notify stakeholders
- [ ] Prepare rollback plan

## Migration Steps

### 1. Create Backup

```bash
# Create timestamped backup
BACKUP_DIR=".nomic/backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp .nomic/*.db "$BACKUP_DIR/"
echo "Backup created in: $BACKUP_DIR"
```

### 2. Dry Run

```bash
# Preview what will be migrated
python -m aragora.persistence.migrations.consolidate --dry-run --verbose

# Expected output shows:
# - Source databases found
# - Tables to migrate
# - Row counts
```

### 3. Execute Migration

```bash
# Run the consolidation
python -m aragora.persistence.migrations.consolidate --migrate

# Expected output:
# Migration completed!
# Tables migrated: 45
# Total rows: 12345
# Duration: 23.45s
```

### 4. Verify Migration

```bash
# Verify the consolidated databases
python -m aragora.persistence.migrations.consolidate --verify

# Check database health endpoint
curl http://localhost:8080/api/v1/health/database | jq
```

### 5. Switch to Consolidated Mode

```bash
# Update environment to use consolidated mode (already default)
export ARAGORA_DB_MODE=consolidated

# Restart the server
systemctl restart aragora

# Verify startup is successful
curl http://localhost:8080/readyz
```

## Verification

### Schema Validation

```bash
# Check required tables exist
curl http://localhost:8080/api/v1/health/database | jq '.validation'

# Should show:
{
  "success": true,
  "errors": [],
  "warnings": []
}
```

### Data Integrity

```bash
# Verify row counts
sqlite3 .nomic/core.db "SELECT COUNT(*) FROM debates;"
sqlite3 .nomic/memory.db "SELECT COUNT(*) FROM continuum_memory;"
sqlite3 .nomic/analytics.db "SELECT COUNT(*) FROM ratings;"
sqlite3 .nomic/agents.db "SELECT COUNT(*) FROM personas;"
```

### Application Tests

```bash
# Run integration tests with consolidated mode
ARAGORA_DB_MODE=consolidated pytest tests/integration/ -v
```

## Rollback Procedure

If issues are encountered:

### 1. Stop Server
```bash
systemctl stop aragora
```

### 2. Restore Backup
```bash
# Identify latest backup
ls -lt .nomic/backup/

# Restore from backup
BACKUP_DIR=".nomic/backup/YYYYMMDD_HHMMSS"
cp "$BACKUP_DIR"/*.db .nomic/
```

### 3. Switch to Legacy Mode
```bash
export ARAGORA_DB_MODE=legacy
```

### 4. Restart Server
```bash
systemctl start aragora
curl http://localhost:8080/readyz
```

## Troubleshooting

### Migration Fails Mid-Way

```bash
# Check for partial migration
ls -la .nomic/{core,memory,analytics,agents}.db

# If partial, remove and retry
rm .nomic/{core,memory,analytics,agents}.db
python -m aragora.persistence.migrations.consolidate --migrate
```

### Schema Validation Errors

```bash
# Check for missing tables
curl http://localhost:8080/api/v1/health/database | jq '.databases'

# If tables are missing, apply schema manually
sqlite3 .nomic/core.db < aragora/persistence/schemas/core.sql
```

### Startup Fails After Migration

```bash
# Check startup health
curl http://localhost:8080/api/v1/health/startup | jq

# Skip schema validation temporarily
ARAGORA_SKIP_SCHEMA_VALIDATION=1 python -m aragora.server.unified_server

# Fix schema issues and remove skip flag
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_DB_MODE` | `consolidated` | Database mode: `legacy` or `consolidated` |
| `ARAGORA_DATA_DIR` | `.nomic` | Directory for database files |
| `ARAGORA_REQUIRE_VALID_SCHEMA` | `false` | Fail startup on schema errors |
| `ARAGORA_SKIP_SCHEMA_VALIDATION` | `false` | Bypass schema validation |

## Related Runbooks

- [RUNBOOK_DATABASE_ISSUES.md](./RUNBOOK_DATABASE_ISSUES.md)
- [database-migration.md](./database-migration.md)
- [RUNBOOK_POSTGRESQL_MIGRATION.md](./RUNBOOK_POSTGRESQL_MIGRATION.md)
