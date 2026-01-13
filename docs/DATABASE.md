# Database Architecture

Aragora uses a dual-storage architecture: SQLite for local development and persistent state, Supabase for cloud deployment.

## Quick Reference

| Database | Purpose | Default Path |
|----------|---------|--------------|
| `aragora_elo.db` | Agent ELO rankings | `~/.aragora/` |
| `aragora_memory.db` | Memory tiers | `~/.aragora/` |
| `aragora_users.db` | User accounts | `~/.aragora/` |
| `aragora_positions.db` | Agent positions | `~/.aragora/` |
| `aragora_replay.db` | Debate replays | `~/.aragora/` |
| `aragora_tokens.db` | Token blacklist | `~/.aragora/` |

## Local Development (SQLite)

### Database Location

By default, databases are stored in `~/.aragora/`. Override with:

```bash
export NOMIC_DIR=/path/to/data
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

This creates a file like `aragora/persistence/migrations/users/003_add_user_lockout_fields.py`:

```python
"""
Migration 003: Add user lockout fields

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
cp ~/.aragora/aragora_users.db ~/.aragora/backups/users_$(date +%Y%m%d).db

# Using sqlite3 backup command
sqlite3 ~/.aragora/aragora_users.db ".backup '~/.aragora/backups/users.db'"
```

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup_dbs.sh

BACKUP_DIR="${NOMIC_DIR:-$HOME/.aragora}/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

for db in elo memory users positions replay tokens; do
    src="${NOMIC_DIR:-$HOME/.aragora}/aragora_${db}.db"
    if [ -f "$src" ]; then
        sqlite3 "$src" ".backup '$BACKUP_DIR/${db}_${DATE}.db'"
        echo "Backed up: $db"
    fi
done

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.db" -mtime +7 -delete
```

### Restore from Backup

```bash
# Stop the server first
cp ~/.aragora/backups/users_20240101.db ~/.aragora/aragora_users.db
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
sqlite3 ~/.aragora/aragora_users.db "SELECT * FROM schema_version"
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
| `NOMIC_DIR` | `~/.aragora` | Database directory |
| `SUPABASE_URL` | - | Supabase project URL |
| `SUPABASE_KEY` | - | Supabase service key |
| `ARAGORA_SLOW_QUERY_MS` | `500` | Slow query threshold |
| `ARAGORA_AUDIT_RETENTION_DAYS` | `90` | Audit log retention |
| `DB_TIMEOUT` | `30` | SQLite connection timeout |
