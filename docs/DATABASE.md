# Database Architecture

> **Last Updated:** 2026-01-18

Aragora uses a dual-storage architecture: SQLite for local development and persistent state, Supabase for cloud deployment. PostgreSQL is supported for production deployments.

## Related Documentation

| Document | Purpose |
|----------|---------|
| **DATABASE.md** (this) | Architecture and configuration |
| [DATABASE_SETUP.md](./DATABASE_SETUP.md) | Installation and setup guide |
| [ENVIRONMENT.md](./ENVIRONMENT.md) | Environment variables reference |

## Quick Reference

| Aspect | Value |
|--------|-------|
| **Default Mode** | `legacy` (multiple SQLite files) |
| **Consolidated Mode** | `consolidated` (4 databases, optional) |
| **PostgreSQL** | Full support via `psycopg2` |
| **Connection Pooling** | Built-in (10-20 per DB) |

### Consolidated Databases (Default)

| Database | Purpose | Key Tables |
|----------|---------|------------|
| **core.db** | Debates, traces, tournaments | debates, traces, positions, embeddings |
| **memory.db** | Agent memory systems | continuum_memory, consensus, patterns |
| **analytics.db** | ELO ratings, insights | ratings, matches, elo_history, insights |
| **agents.db** | Agent personas, genesis | personas, genomes, relationships |

### Legacy Databases (multiple individual files)

| Database | Purpose | Default Path |
|----------|---------|--------------|
| `agent_elo.db` | Agent ELO rankings | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `continuum.db` | Continuum memory tiers | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `consensus_memory.db` | Consensus history | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `agora_memory.db` | CritiqueStore patterns (CLI) | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `agent_calibration.db` | Calibration tracking | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `aragora_insights.db` | Insights and analytics | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `agent_personas.db` | Personas | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `grounded_positions.db` | Truth-grounded positions | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `genesis.db` | Genesis ledger / genomes | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `token_blacklist.db` | JWT revocation store | `ARAGORA_DATA_DIR` (default `.nomic/`) |
| `users.db` | User/org accounts | `ARAGORA_DATA_DIR` (default `.nomic/`) |

Other legacy files are created on demand (e.g., `persona_lab.db`, `agent_relationships.db`).
Use `ARAGORA_DB_*` overrides or `aragora.config.legacy.DB_NAMES` for the full list.

---

## Database Mode Configuration

```bash
# Use consolidated mode (optional)
export ARAGORA_DB_MODE=consolidated

# Use legacy mode (default)
export ARAGORA_DB_MODE=legacy
```

### Programmatic Access

```python
from aragora.persistence.db_config import (
    get_db_path,
    DatabaseType,
    get_db_mode,
    DatabaseMode,
)

# Get path for a specific database type
elo_path = get_db_path(DatabaseType.ELO)  # -> analytics.db in consolidated mode

# Check current mode
mode = get_db_mode()
if mode == DatabaseMode.CONSOLIDATED:
    print("Using consolidated databases")
```

---

## Migration: Legacy to Consolidated

### Running the Migration

```bash
# 1. Dry run - see what would be migrated
python -m aragora.persistence.migrations.consolidate --dry-run --source .nomic

# 2. Execute migration (creates backup automatically)
python -m aragora.persistence.migrations.consolidate --migrate --source .nomic

# 3. Verify migration
python -m aragora.persistence.migrations.consolidate --verify --source .nomic
```

### Migration Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be migrated |
| `--migrate` | Execute the consolidation |
| `--verify` | Verify consolidated databases |
| `--source DIR` | Source directory (default: .nomic) |
| `--target DIR` | Target directory (default: same as source) |
| `--no-backup` | Skip backup (not recommended) |

Backups are created at `.nomic/backup/YYYYMMDD_HHMMSS/`

---

## PostgreSQL Production Setup

For high-concurrency or distributed deployments.

### Prerequisites

```bash
pip install psycopg2-binary  # or psycopg2 for production
```

### Environment Configuration

```bash
# Required
export DATABASE_URL="postgresql://user:password@host:5432/aragora"

# Optional (pool tuning)
export ARAGORA_DB_POOL_SIZE=10        # Pool size
export ARAGORA_DB_POOL_MAX_OVERFLOW=5 # Extra connections
export ARAGORA_DB_POOL_TIMEOUT=30     # Connection timeout
```

`ARAGORA_DATABASE_URL` is accepted as a legacy alias for `DATABASE_URL`.

### Connection String Format

```
postgresql://[user]:[password]@[host]:[port]/[database]

# Examples:
postgresql://aragora:secret@localhost:5432/aragora_prod
postgresql://user:pass@db.example.com:5432/aragora?sslmode=require
```

### Database Setup

```sql
-- Create database
CREATE DATABASE aragora_prod;
CREATE USER aragora WITH ENCRYPTED PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE aragora_prod TO aragora;
```

### Apply Schemas

```bash
psql -U aragora -d aragora_prod -f aragora/persistence/schemas/core.sql
psql -U aragora -d aragora_prod -f aragora/persistence/schemas/memory.sql
psql -U aragora -d aragora_prod -f aragora/persistence/schemas/analytics.sql
psql -U aragora -d aragora_prod -f aragora/persistence/schemas/agents.sql
```

---

## Encryption

### Encryption at Rest

Data at rest encryption protects database files from unauthorized access if storage media is compromised.

#### SQLite Encryption

**Option 1: SQLCipher (Recommended for sensitive data)**

SQLCipher provides transparent 256-bit AES encryption for SQLite databases.

```bash
# Install SQLCipher
pip install sqlcipher3

# Or use system SQLCipher
brew install sqlcipher  # macOS
apt-get install sqlcipher  # Debian/Ubuntu
```

Configuration:
```bash
export ARAGORA_SQLITE_ENCRYPTION=1
export ARAGORA_SQLITE_KEY="your-32-character-encryption-key"
```

Usage:
```python
from aragora.storage.encrypted import get_encrypted_connection

# Opens database with encryption
conn = get_encrypted_connection("/path/to/secure.db")
```

**Option 2: OS-Level Encryption**

For less sensitive deployments, use filesystem-level encryption:
- **macOS**: FileVault (enabled by default)
- **Linux**: LUKS dm-crypt for volume encryption
- **Windows**: BitLocker

```bash
# Linux LUKS example
cryptsetup luksFormat /dev/sdb1
cryptsetup luksOpen /dev/sdb1 aragora_data
mkfs.ext4 /dev/mapper/aragora_data
mount /dev/mapper/aragora_data /var/lib/aragora
```

#### PostgreSQL Encryption

**Transparent Data Encryption (TDE)**

For PostgreSQL 16+, use TDE for full database encryption:

```sql
-- Enable TDE (requires PostgreSQL Enterprise or contrib)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt sensitive columns
ALTER TABLE users
  ALTER COLUMN password_hash
  SET DATA TYPE bytea
  USING pgp_sym_encrypt(password_hash::text, current_setting('app.encryption_key'))::bytea;
```

**AWS RDS Encryption**

For AWS-hosted PostgreSQL:
```bash
# Enable encryption when creating RDS instance
aws rds create-db-instance \
  --db-instance-identifier aragora-prod \
  --storage-encrypted \
  --kms-key-id alias/aws/rds
```

**Column-Level Encryption**

For specific sensitive fields:
```python
from cryptography.fernet import Fernet

# Generate key (store securely)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt before storage
encrypted_data = cipher.encrypt(sensitive_data.encode())

# Decrypt when reading
decrypted_data = cipher.decrypt(encrypted_data).decode()
```

#### Redis Encryption

**TLS for Redis Connections**

```bash
# Use rediss:// scheme for TLS
export REDIS_URL="rediss://user:password@redis-host:6379/0"
```

Redis configuration (`redis.conf`):
```
tls-port 6379
port 0
tls-cert-file /path/to/redis.crt
tls-key-file /path/to/redis.key
tls-ca-cert-file /path/to/ca.crt
```

**AWS ElastiCache Encryption**

```bash
# Enable encryption in transit and at rest
aws elasticache create-replication-group \
  --replication-group-id aragora-cache \
  --transit-encryption-enabled \
  --at-rest-encryption-enabled
```

### Encryption in Transit

All database connections should use encrypted transport:

| Database | Protocol | Configuration |
|----------|----------|---------------|
| PostgreSQL | TLS/SSL | `?sslmode=require` in connection string |
| Redis | TLS | `rediss://` scheme |
| Supabase | HTTPS | Automatic (always encrypted) |

PostgreSQL SSL configuration:
```bash
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=verify-full&sslrootcert=/path/to/ca.crt"
```

### Key Management

**Environment Variables (Development)**
```bash
export ARAGORA_ENCRYPTION_KEY="base64-encoded-32-byte-key"
export ARAGORA_SQLITE_KEY="another-encryption-key"
```

**Kubernetes Secrets (Production)**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aragora-encryption
type: Opaque
stringData:
  encryption-key: "your-base64-encoded-key"
  sqlite-key: "your-sqlite-encryption-key"
```

**External Secret Managers**

AWS Secrets Manager:
```python
import boto3

def get_encryption_key():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='aragora/encryption-key')
    return response['SecretString']
```

HashiCorp Vault:
```python
import hvac

def get_encryption_key():
    client = hvac.Client(url='https://vault.example.com')
    secret = client.secrets.kv.read_secret_version(path='aragora/encryption')
    return secret['data']['data']['key']
```

### Key Rotation

**Rotating SQLCipher Keys**
```bash
# Export data with old key
sqlite3 encrypted.db "PRAGMA key = 'old-key'; .dump" > backup.sql

# Create new database with new key
sqlite3 new_encrypted.db "PRAGMA key = 'new-key';"
sqlite3 new_encrypted.db < backup.sql

# Replace old database
mv new_encrypted.db encrypted.db
```

**Rotating PostgreSQL Keys**

For column-level encryption, re-encrypt data:
```python
from cryptography.fernet import Fernet

old_cipher = Fernet(old_key)
new_cipher = Fernet(new_key)

# Re-encrypt all sensitive data
for row in get_all_encrypted_rows():
    decrypted = old_cipher.decrypt(row.encrypted_data)
    new_encrypted = new_cipher.encrypt(decrypted)
    update_row(row.id, new_encrypted)
```

---

## Local Development (SQLite)

### Database Location

By default, databases are stored in `ARAGORA_DATA_DIR` (default `.nomic/`). Override with:

```bash
export ARAGORA_DATA_DIR=/path/to/data
```

Legacy aliases:
```bash
export ARAGORA_NOMIC_DIR=/path/to/data  # used by some migration tooling
export NOMIC_DIR=/path/to/data          # older env var
```

If you ran Aragora in the repo root and created stray `.db` files, move them under `ARAGORA_DATA_DIR` with:

```bash
scripts/cleanup_runtime_artifacts.sh
```

### Connection Configuration

```python
from aragora.storage.schema import get_wal_connection

# Default connection with WAL mode
conn = get_wal_connection("/path/to/database.db")

# With custom timeout
conn = get_wal_connection("/path/to/database.db", timeout=60.0)
```

WAL (Write-Ahead Logging) mode is enabled by default for:
- Better concurrent read/write performance
- Improved crash recovery
- Non-blocking reads during writes

### Database Stores

| Store | Module | Purpose |
|-------|--------|---------|
| `UserStore` | `aragora.storage.user_store` | User accounts, auth |
| `OrganizationStore` | `aragora.storage.organization_store` | Multi-tenant orgs |
| `AuditStore` | `aragora.storage.audit_store` | Security audit log |
| `ShareLinkStore` | `aragora.storage.share_store` | Shared debate links |
| `WebhookStore` | `aragora.storage.webhook_store` | Webhook configs |

```python
from aragora.storage.user_store import UserStore

store = UserStore(db_path="/path/to/users.db")
user = store.get_by_email("user@example.com")
```

## Cloud Deployment (Supabase)

### Setup

1. Create a [Supabase project](https://supabase.com)

2. Set environment variables:
   ```bash
   export SUPABASE_URL=https://yourproject.supabase.co
   export SUPABASE_KEY=your-service-role-key
   ```

3. Run schema migrations (see SQL files in `supabase/migrations/`)

### Configuration Check

```python
from aragora.persistence.supabase_client import SupabaseClient

client = SupabaseClient()
print(f"Configured: {client.is_configured}")
```

### Supabase Tables

| Table | Purpose |
|-------|---------|
| `nomic_cycles` | Nomic loop cycle state |
| `debate_artifacts` | Debate transcripts |
| `stream_events` | Real-time events |
| `agent_metrics` | Agent performance |
| `nomic_rollbacks` | Rollback history |
| `cycle_evolution` | Codebase evolution |
| `cycle_file_changes` | File change tracking |

## Schema

### Users Table

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    name TEXT DEFAULT '',
    org_id TEXT,
    role TEXT DEFAULT 'member',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_login_at TEXT,
    is_active INTEGER DEFAULT 1,
    email_verified INTEGER DEFAULT 0,
    avatar_url TEXT,
    preferences TEXT DEFAULT '{}',
    -- Added in migration 002
    locked_until TEXT,
    failed_login_count INTEGER DEFAULT 0,
    lockout_reason TEXT,
    last_activity_at TEXT,
    last_debate_at TEXT
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_org ON users(org_id);
```

### Organizations Table

```sql
CREATE TABLE organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    tier TEXT DEFAULT 'free',
    owner_id TEXT REFERENCES users(id),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    settings TEXT DEFAULT '{}'
);

CREATE INDEX idx_orgs_slug ON organizations(slug);
CREATE INDEX idx_orgs_stripe ON organizations(stripe_customer_id);
```

### Audit Log Table

```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user_id TEXT,
    org_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    details TEXT DEFAULT '{}',
    ip_address TEXT,
    user_agent TEXT
);

CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_org ON audit_log(org_id);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
```

### Usage Events Table

```sql
CREATE TABLE usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL REFERENCES organizations(id),
    event_type TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_org ON usage_events(org_id);
CREATE INDEX idx_usage_timestamp ON usage_events(timestamp);
```

## Migrations

### Running Migrations

```bash
# Check migration status
python -m aragora.persistence.migrations.runner --status

# Dry-run (show what would be done)
python -m aragora.persistence.migrations.runner --dry-run

# Run all pending migrations
python -m aragora.persistence.migrations.runner --migrate

# Run migrations for specific database
python -m aragora.persistence.migrations.runner --migrate --db users
```

### Creating Migrations

```bash
# Create new migration
python -m aragora.persistence.migrations.runner --create "Add user lockout fields" --db users
```

This creates a file like `aragora/persistence/migrations/users/00x_add_user_lockout_fields.py`:

```python
"""
Migration 00x: Add user lockout fields

Created: 2024-01-03T12:00:00
"""

import sqlite3

def upgrade(conn: sqlite3.Connection) -> None:
    """Apply this migration."""
    conn.execute("""
        ALTER TABLE users ADD COLUMN locked_until TEXT
    """)

def downgrade(conn: sqlite3.Connection) -> None:
    """Reverse this migration (optional)."""
    # SQLite has limited ALTER TABLE support
    pass
```

### Migration File Naming

Migration files follow the pattern: `NNN_description.py`

- `001_initial.py` - Initial schema
- `002_add_lockout.py` - Add lockout fields
- `003_add_analytics.py` - Add analytics

### Safe Column Addition

Use `safe_add_column` to handle existing columns:

```python
from aragora.storage.schema import safe_add_column

def upgrade(conn: sqlite3.Connection) -> None:
    # Won't fail if column already exists
    safe_add_column(conn, "users", "new_field", "TEXT", "NULL")
```

## Backup and Restore

### SQLite Backup

```bash
# Manual backup
cp "${ARAGORA_DATA_DIR:-.nomic}/users.db" \
  "${ARAGORA_DATA_DIR:-.nomic}/backups/users_$(date +%Y%m%d).db"

# Using sqlite3 backup command
sqlite3 "${ARAGORA_DATA_DIR:-.nomic}/users.db" \
  ".backup '${ARAGORA_DATA_DIR:-.nomic}/backups/users.db'"
```

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup_dbs.sh

BACKUP_DIR="${ARAGORA_DATA_DIR:-.nomic}/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

for db in agent_elo.db continuum.db consensus_memory.db agent_calibration.db \
         aragora_insights.db agent_personas.db grounded_positions.db genesis.db \
         token_blacklist.db users.db agora_memory.db; do
    src="${ARAGORA_DATA_DIR:-.nomic}/${db}"
    if [ -f "$src" ]; then
        sqlite3 "$src" ".backup '$BACKUP_DIR/${db%.db}_${DATE}.db'"
        echo "Backed up: $db"
    fi
done

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.db" -mtime +7 -delete
```

### Restore from Backup

```bash
# Stop the server first
cp "${ARAGORA_DATA_DIR:-.nomic}/backups/users_20240101.db" \
  "${ARAGORA_DATA_DIR:-.nomic}/users.db"
```

## Performance Monitoring

### Slow Query Logging

Set threshold in milliseconds:

```bash
export ARAGORA_SLOW_QUERY_MS=500
```

Queries exceeding this threshold are logged:

```
WARNING: Slow query (0.523s): save_cycle [cycle=5] (threshold: 0.500s)
```

### Connection Pooling

For high-concurrency scenarios, use connection pools:

```python
from aragora.storage.schema import DatabaseManager

# Singleton manager handles connection pooling
manager = DatabaseManager.get_instance("users", "/path/to/users.db")
conn = manager.get_connection()
```

### Query Optimization

1. **Use indexes** - All foreign keys and commonly queried columns should be indexed
2. **Batch operations** - Use `executemany()` for bulk inserts
3. **Prepared statements** - Use parameterized queries
4. **WAL mode** - Enabled by default for concurrent access

## Data Retention

### Audit Log Retention

Configure retention in `aragora/storage/audit_store.py`:

```python
# Default: 90 days
AUDIT_RETENTION_DAYS = int(os.getenv("ARAGORA_AUDIT_RETENTION_DAYS", "90"))
```

Cleanup old audit entries:

```python
from aragora.storage.audit_store import AuditStore

store = AuditStore(db_path)
store.cleanup_old_entries()  # Uses AUDIT_RETENTION_DAYS
```

### Usage Event Aggregation

Usage events are aggregated monthly and raw events older than 90 days are pruned.

## Troubleshooting

### Database Locked

If you see "database is locked" errors:

1. Check for long-running queries
2. Ensure WAL mode is enabled
3. Increase timeout: `DB_TIMEOUT=60`

```python
# Check WAL mode
conn = get_wal_connection("/path/to/db.db")
cursor = conn.execute("PRAGMA journal_mode")
print(cursor.fetchone())  # Should be ('wal',)
```

### Schema Version Mismatch

If migration fails with version mismatch:

```bash
# Check current version
python -m aragora.persistence.migrations.runner --status

# View schema_version table directly
sqlite3 "${ARAGORA_DATA_DIR:-.nomic}/users.db" "SELECT * FROM schema_version"
```

### Corrupted Database

If database is corrupted:

1. Restore from backup
2. Or try SQLite recovery:
   ```bash
   sqlite3 corrupted.db ".dump" | sqlite3 recovered.db
   ```

### Connection Issues (Supabase)

Verify configuration:

```python
import os
print(f"URL: {os.getenv('SUPABASE_URL')}")
print(f"Key set: {bool(os.getenv('SUPABASE_KEY'))}")

from aragora.persistence.supabase_client import SupabaseClient
client = SupabaseClient()
print(f"Configured: {client.is_configured}")
```

## Data Models

### NomicCycle

```python
@dataclass
class NomicCycle:
    loop_id: str
    cycle_number: int
    phase: str  # debate, design, implement, verify, commit
    stage: str  # proposing, critiquing, voting, executing
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None
    git_commit: Optional[str] = None
    task_description: Optional[str] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    error_message: Optional[str] = None
```

### DebateArtifact

```python
@dataclass
class DebateArtifact:
    loop_id: str
    cycle_number: int
    phase: str
    task: str
    agents: list[str]
    transcript: list[dict]  # Full message history
    consensus_reached: bool
    confidence: float
    winning_proposal: Optional[str] = None
    vote_tally: Optional[dict] = None
```

### StreamEvent

```python
@dataclass
class StreamEvent:
    loop_id: str
    cycle: int
    event_type: str  # cycle_start, phase_start, task_complete, error
    event_data: dict
    agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_DATA_DIR` | `.nomic` | Database directory |
| `ARAGORA_NOMIC_DIR` | `.nomic` | Legacy alias for some migration tooling |
| `SUPABASE_URL` | - | Supabase project URL |
| `SUPABASE_KEY` | - | Supabase service key |
| `ARAGORA_SLOW_QUERY_MS` | `500` | Slow query threshold |
| `ARAGORA_AUDIT_RETENTION_DAYS` | `90` | Audit log retention |
| `DB_TIMEOUT` | `30` | SQLite connection timeout |
