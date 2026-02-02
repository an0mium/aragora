# Aragora Persistence Layer

The persistence layer provides unified database access for the Aragora system, supporting both SQLite (local development) and PostgreSQL (production) backends with connection pooling, transaction management, and real-time synchronization.

## Overview

The persistence module handles:

- **Multi-database support**: SQLite for local/single-instance, PostgreSQL for production
- **Connection pooling**: Read replica routing and connection management
- **Transactions**: Savepoints, deadlock detection, isolation levels
- **Migrations**: Version-controlled schema changes with rollback support
- **Sync services**: Background replication to Supabase
- **Repository pattern**: Type-safe data access layer

## Architecture

```
aragora/persistence/
├── __init__.py           # Public API exports
├── db_backend.py         # Unified backend adapter (SQLite/PostgreSQL)
├── db_config.py          # Centralized database configuration
├── postgres_pool.py      # Connection pooling with read replicas
├── transaction.py        # Transaction management with savepoints
├── supabase_client.py    # Supabase persistence client
├── sync_service.py       # Background sync to cloud
├── models.py             # Data models (NomicCycle, DebateArtifact, etc.)
├── evolution.py          # Nomic loop history tracking
├── query_utils.py        # Batch queries and performance utilities
├── validator.py          # Schema validation for health checks
├── schema.sql            # Core SQL schema definitions
├── schemas/              # Consolidated database schemas
│   ├── core.sql          # Core tables (debates, traces, tournaments)
│   ├── memory.sql        # Memory tables (continuum, consensus, critiques)
│   ├── analytics.sql     # Analytics tables (ELO, calibration, insights)
│   └── agents.sql        # Agent tables (personas, genomes, genesis)
├── repositories/         # Data access repositories
│   ├── base.py           # BaseRepository with CRUD operations
│   ├── debate.py         # Debate persistence with slug management
│   ├── elo.py            # ELO ratings repository
│   ├── memory.py         # Memory tier repository
│   └── phase2.py         # Phase 2 feature repositories
└── migrations/           # Database migrations
    ├── runner.py         # Migration CLI and runner
    ├── consolidate.py    # Database consolidation utilities
    ├── postgres/         # PostgreSQL-specific migrations
    └── users/            # User database migrations
```

## Store Abstractions

### UnifiedBackend

The `UnifiedBackend` class provides a single interface for both sync and async database operations, abstracting away backend differences.

```python
from aragora.persistence.db_backend import get_unified_backend

backend = get_unified_backend()

# Check backend type
if backend.is_postgres:
    print("Using PostgreSQL")
elif backend.is_sqlite:
    print("Using SQLite")

# Sync operations (migrations, admin)
row = backend.sync.fetch_one("SELECT count(*) FROM debates")

# Check capabilities before using features
if backend.capabilities.supports_advisory_locks:
    # Use PostgreSQL advisory locks for coordination
    pass

if backend.capabilities.supports_listen_notify:
    # Use LISTEN/NOTIFY for real-time events
    pass

# Get async pool for high-throughput operations
pool = await backend.get_async_pool()
if pool:
    async with pool.acquire() as conn:
        await conn.fetch("SELECT * FROM debates LIMIT 10")
```

### BackendCapabilities

Describes what the current backend supports for runtime feature detection:

| Capability | SQLite | PostgreSQL | Description |
|------------|--------|------------|-------------|
| `supports_concurrent_writes` | No | Yes | Multiple concurrent writers |
| `supports_advisory_locks` | No | Yes | Distributed coordination locks |
| `supports_listen_notify` | No | Yes | Real-time event notification |
| `supports_jsonb` | No | Yes | JSONB columns with operators |
| `supports_full_text_search` | Yes (FTS5) | Yes (tsvector) | Full-text search |
| `supports_read_replicas` | No | Yes | Load distribution |
| `supports_savepoints` | Yes | Yes | Nested transactions |
| `supports_returning` | No | Yes | RETURNING clause |
| `has_async_pool` | No | Yes | Async connection pool |

### BackendType

```python
from aragora.persistence.db_backend import BackendType

BackendType.SQLITE   # Local development
BackendType.POSTGRES # Production deployment
```

## Repository Pattern

### BaseRepository

Abstract base class providing common CRUD operations with SQL injection protection:

```python
from aragora.persistence.repositories.base import BaseRepository

class MyEntityRepository(BaseRepository[MyEntity]):
    @property
    def _table_name(self) -> str:
        return "my_entities"

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS my_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _to_entity(self, row) -> MyEntity:
        return MyEntity(id=row["id"], name=row["name"])

    def _from_entity(self, entity: MyEntity) -> dict:
        return {"id": entity.id, "name": entity.name}

# Usage
repo = MyEntityRepository("my_db.db")
entity = repo.get("entity-123")
repo.save(entity)
repo.delete("entity-123")
entities = repo.list_all(limit=100, offset=0)
```

### DebateRepository

Specialized repository for debate persistence with slug-based permalinks:

```python
from aragora.persistence.repositories.debate import DebateRepository, DebateEntity

repo = DebateRepository()

# Save with auto-generated slug
slug = repo.save_with_slug(artifact_dict)
# Returns: "rate-limiter-2024-01-15"

# Get by slug
debate = repo.get_by_slug("rate-limiter-2024-01-15")

# Search debates
results = repo.search("rate limiter", limit=20)

# List recent debates
recent = repo.list_recent(limit=20)

# Track views
new_count = repo.increment_view_count(slug)
```

## Transaction Patterns

### TransactionManager

Explicit transaction management for PostgreSQL with savepoints and deadlock handling:

```python
from aragora.persistence.transaction import (
    TransactionManager,
    TransactionIsolation,
    TransactionConfig,
)

# Basic usage
manager = TransactionManager(pool)

async with manager.transaction() as conn:
    await conn.execute("INSERT INTO users ...")
    await conn.execute("UPDATE accounts ...")
    # Auto-commit on success, rollback on exception

# With isolation level
async with manager.transaction(
    isolation=TransactionIsolation.SERIALIZABLE,
    timeout=30.0,
) as conn:
    await conn.execute("SELECT ... FOR UPDATE")
    await conn.execute("UPDATE ...")

# Nested transactions with savepoints
async with manager.transaction() as conn:
    await conn.execute("INSERT INTO orders ...")

    async with manager.savepoint(conn, "inventory_update"):
        await conn.execute("UPDATE inventory ...")
        # If this fails, only inventory update is rolled back
        # Outer transaction continues
```

### TransactionIsolation

Available isolation levels:

| Level | Description |
|-------|-------------|
| `READ_COMMITTED` | Default. Each statement sees committed data |
| `REPEATABLE_READ` | Snapshot from start of first statement |
| `SERIALIZABLE` | Strictest. Transactions appear serial |

### TransactionConfig

```python
config = TransactionConfig(
    isolation=TransactionIsolation.READ_COMMITTED,
    timeout_seconds=30.0,      # Transaction timeout
    deadlock_retries=3,        # Retry on deadlock
    deadlock_base_delay=0.1,   # Base retry delay (exponential backoff)
    deadlock_max_delay=2.0,    # Maximum retry delay
    savepoint_on_nested=True,  # Use savepoints for nesting
    validate_connection_state=True,  # Check connection health
)
```

### NestedTransactionManager

Automatic savepoint management for nested transaction calls:

```python
from aragora.persistence.transaction import create_transaction_manager

manager = create_transaction_manager(
    pool,
    nested_support=True,  # Enable automatic nesting
)

# Nested calls automatically use savepoints
async with manager.transaction() as conn:
    await conn.execute("INSERT ...")

    # This automatically uses a savepoint
    async with manager.transaction() as nested_conn:
        await nested_conn.execute("UPDATE ...")
```

## Connection Management

### ReplicaAwarePool

Connection pool with automatic read replica routing:

```python
from aragora.persistence.postgres_pool import (
    ReplicaAwarePool,
    configure_pool,
    get_pool,
)

# Configure globally
configure_pool(
    primary_dsn="postgresql://primary:5432/db",
    replica_dsns=["postgresql://replica1:5432/db"],
    min_size=2,
    max_size=10,
)

# Get connection (auto-routes reads to replicas)
pool = get_pool()
async with pool.acquire(readonly=True) as conn:
    result = await conn.fetch("SELECT * FROM table")

# Write operations always go to primary
async with pool.acquire(readonly=False) as conn:
    await conn.execute("INSERT INTO ...")

# Health status
status = pool.get_health_status()
# {
#     "initialized": True,
#     "primary_healthy": True,
#     "replica_count": 2,
#     "healthy_replicas": 2,
#     "stats": {...}
# }
```

### PoolStats

```python
stats = pool.stats
# PoolStats(
#     total_connections=10,
#     active_connections=3,
#     idle_connections=7,
#     wait_count=150,
#     total_queries=10000,
#     read_queries=8000,
#     write_queries=2000,
# )
```

## Configuration

### Environment Variables

#### Backend Selection

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_DB_BACKEND` | Force backend: "sqlite", "postgres", "auto" | "auto" |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `ARAGORA_POSTGRES_DSN` | Alternative PostgreSQL DSN | - |
| `SUPABASE_POSTGRES_DSN` | Supabase PostgreSQL DSN | - |

#### Database Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_DATA_DIR` | Base directory for databases | ".nomic" or "data" |
| `ARAGORA_NOMIC_DIR` | Legacy alias for data directory | ".nomic" |
| `ARAGORA_DB_MODE` | "legacy" or "consolidated" | "consolidated" |

#### Connection Pool

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_POSTGRES_PRIMARY` | Primary PostgreSQL DSN | - |
| `ARAGORA_POSTGRES_REPLICAS` | Comma-separated replica DSNs | - |
| `ARAGORA_POSTGRES_POOL_MIN` | Minimum connections per pool | 2 |
| `ARAGORA_POSTGRES_POOL_MAX` | Maximum connections per pool | 10 |
| `ARAGORA_POOL_MAX_SIZE` | Async pool max size | 20 |

#### Supabase Sync

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | - |
| `SUPABASE_KEY` | Supabase service role key | - |
| `SUPABASE_SYNC_ENABLED` | Enable background sync | "false" |
| `SUPABASE_SYNC_BATCH_SIZE` | Items per sync batch | 10 |
| `SUPABASE_SYNC_INTERVAL_SECONDS` | Sync interval | 30 |
| `SUPABASE_SYNC_MAX_RETRIES` | Max retries per item | 3 |

#### Performance

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_SLOW_QUERY_MS` | Slow query threshold (ms) | 500 |
| `ARAGORA_SKIP_SCHEMA_VALIDATION` | Skip startup validation | "false" |

### DatabaseType Enum

All supported database types for path resolution:

```python
from aragora.persistence.db_config import DatabaseType, get_db_path

# Core databases
get_db_path(DatabaseType.DEBATES)
get_db_path(DatabaseType.TRACES)
get_db_path(DatabaseType.TOURNAMENTS)

# Memory databases
get_db_path(DatabaseType.CONTINUUM_MEMORY)
get_db_path(DatabaseType.CONSENSUS_MEMORY)

# Analytics databases
get_db_path(DatabaseType.ELO)
get_db_path(DatabaseType.CALIBRATION)
get_db_path(DatabaseType.INSIGHTS)

# Agent databases
get_db_path(DatabaseType.PERSONAS)
get_db_path(DatabaseType.GENESIS)
```

### Database Modes

| Mode | Description |
|------|-------------|
| `LEGACY` | Individual database files (e.g., `agent_elo.db`, `continuum.db`) |
| `CONSOLIDATED` | Four consolidated databases: `core.db`, `memory.db`, `analytics.db`, `agents.db` |

## Migrations

### CLI Usage

```bash
# Check migration status
python -m aragora.persistence.migrations.runner --status

# Dry-run migrations
python -m aragora.persistence.migrations.runner --dry-run

# Run migrations (creates automatic backup first)
python -m aragora.persistence.migrations.runner --migrate

# Run migrations without backup (not recommended)
python -m aragora.persistence.migrations.runner --migrate --no-backup

# Create a new migration
python -m aragora.persistence.migrations.runner --create "Add user lockout fields" --db users

# Rollback last migration (development only)
python -m aragora.persistence.migrations.runner --rollback --db users

# Rollback to specific version
python -m aragora.persistence.migrations.runner --rollback-to 5 --db users --dry-run
```

### Programmatic Usage

```python
from aragora.persistence.migrations.runner import MigrationRunner

runner = MigrationRunner(backup_before_migrate=True)

# Get status
status = runner.get_status("users")
print(f"Current: {status.current_version}, Target: {status.target_version}")
print(f"Pending: {status.pending_migrations}")

# Run migrations
result = runner.migrate("users", dry_run=False)
if result["status"] == "completed":
    print(f"Applied: {result['applied']}")

# Get migration history
history = runner.get_migration_history("users")
for migration in history:
    print(f"v{migration.version}: {migration.name} ({migration.applied_at})")
```

### Migration File Structure

```python
"""
Migration 002: Add user lockout fields
"""

import sqlite3

def upgrade(conn: sqlite3.Connection) -> None:
    """Apply this migration."""
    conn.execute("""
        ALTER TABLE users ADD COLUMN locked_until TIMESTAMP
    """)
    conn.execute("""
        ALTER TABLE users ADD COLUMN failed_attempts INTEGER DEFAULT 0
    """)

def downgrade(conn: sqlite3.Connection) -> None:
    """Reverse this migration (optional)."""
    # SQLite table recreation pattern for column removal
    conn.executescript("""
        CREATE TABLE users_backup AS SELECT id, email, created_at FROM users;
        DROP TABLE users;
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO users SELECT * FROM users_backup;
        DROP TABLE users_backup;
    """)
```

## Data Models

### NomicCycle

Represents a single nomic loop cycle:

```python
from aragora.persistence.models import NomicCycle

cycle = NomicCycle(
    loop_id="loop-123",
    cycle_number=5,
    phase="verify",            # debate, design, implement, verify, commit
    stage="executing",         # proposing, critiquing, voting, executing
    started_at=datetime.now(),
    completed_at=None,
    success=None,
    git_commit="abc123",
    task_description="Implement rate limiter",
    total_tasks=10,
    completed_tasks=7,
    error_message=None,
)
```

### DebateArtifact

Stores debate transcripts and results:

```python
from aragora.persistence.models import DebateArtifact

artifact = DebateArtifact(
    loop_id="loop-123",
    cycle_number=5,
    phase="debate",
    task="Design a rate limiter",
    agents=["claude", "gpt-4", "gemini"],
    transcript=[{"role": "agent", "content": "..."}],
    consensus_reached=True,
    confidence=0.85,
    winning_proposal="Token bucket algorithm...",
    vote_tally={"claude": 2, "gpt-4": 1},
)
```

### StreamEvent

Real-time events from the nomic loop:

```python
from aragora.persistence.models import StreamEvent

event = StreamEvent(
    loop_id="loop-123",
    cycle=5,
    event_type="phase_start",  # cycle_start, task_complete, error, etc.
    event_data={"phase": "verify", "task_count": 10},
    agent="claude",
)
```

### NomicRollback

Records rollback events for debugging:

```python
from aragora.persistence.models import NomicRollback

rollback = NomicRollback(
    id="rb-123",
    loop_id="loop-123",
    cycle_number=5,
    phase="verify",
    reason="verify_failure",  # verify_failure, manual_intervention, timeout
    severity="high",          # low, medium, high, critical
    rolled_back_commit="abc123",
    preserved_branch="failed/cycle-5",
    files_affected=["src/api.py", "tests/test_api.py"],
    diff_summary="2 files changed, 50 insertions(+), 10 deletions(-)",
    error_message="Tests failed: 3 assertions",
)
```

## Sync Service

Background replication from SQLite to Supabase:

```python
from aragora.persistence.sync_service import get_sync_service

sync = get_sync_service()

# Queue items for sync (non-blocking)
sync.queue_debate(debate_dict)
sync.queue_cycle(cycle_dict)
sync.queue_event(event_dict)
sync.queue_metrics(metrics_dict)

# Check status
status = sync.get_status()
print(f"Queue size: {status.queue_size}")
print(f"Synced: {status.synced_count}")
print(f"Failed: {status.failed_count}")

# Flush before shutdown
sync.flush(timeout=30.0)
```

## Query Utilities

### Batch Operations

```python
from aragora.persistence.query_utils import batch_select, batch_exists, chunked

# Batch select to avoid SQLite expression tree depth limits
rows = batch_select(
    conn,
    table="debates",
    ids=["id1", "id2", ...],  # Can be hundreds of IDs
    columns=["id", "task", "consensus_reached"],
    batch_size=100,
)

# Check which IDs exist
existing = batch_exists(conn, "debates", ids, batch_size=100)

# Split large lists into chunks
for chunk in chunked(large_list, size=100):
    process(chunk)
```

### Timed Queries

```python
from aragora.persistence.query_utils import timed_query

# Execute with timing and slow query logging
cursor = timed_query(
    conn,
    "SELECT * FROM debates WHERE task LIKE ?",
    params=("%rate%",),
    operation_name="search_debates",
    threshold_ms=500.0,
)
# Logs warning if query exceeds threshold
```

## Schema Validation

Startup health checks for database schema:

```python
from aragora.persistence.validator import (
    validate_consolidated_schema,
    get_database_health,
)

# Validate schema exists
result = validate_consolidated_schema()
if not result.success:
    for error in result.errors:
        print(f"Schema error: {error}")

# Get comprehensive health info
health = get_database_health()
# {
#     "status": "healthy",
#     "mode": "consolidated",
#     "validation": {"success": True, "errors": [], "warnings": []},
#     "databases": {
#         "core.db": {"exists": True, "table_count": 8, ...},
#         ...
#     }
# }
```

## API Reference

### Public Exports

From `aragora.persistence`:

```python
from aragora.persistence import (
    # Client
    SupabaseClient,

    # Models
    NomicCycle,
    DebateArtifact,
    StreamEvent,
    AgentMetrics,

    # Configuration
    DatabaseType,
    DatabaseMode,
    get_db_path,
    get_db_path_str,
    get_db_mode,
    get_nomic_dir,
    get_elo_db_path,
    get_memory_db_path,
    get_positions_db_path,
    get_personas_db_path,
    get_insights_db_path,
    get_genesis_db_path,
)
```

### Backend Module

From `aragora.persistence.db_backend`:

```python
from aragora.persistence.db_backend import (
    BackendType,
    BackendCapabilities,
    UnifiedBackend,
    SchemaInfo,
    ColumnInfo,
    get_unified_backend,
    reset_unified_backend,
    normalize_sqlite_type,
    SQLITE_TO_PG_TYPE_MAP,
)
```

### Transaction Module

From `aragora.persistence.transaction`:

```python
from aragora.persistence.transaction import (
    TransactionManager,
    NestedTransactionManager,
    TransactionConfig,
    TransactionStats,
    TransactionContext,
    TransactionState,
    TransactionIsolation,
    TransactionError,
    TransactionStateError,
    SavepointError,
    DeadlockError,
    create_transaction_manager,
)
```

### Pool Module

From `aragora.persistence.postgres_pool`:

```python
from aragora.persistence.postgres_pool import (
    ReplicaAwarePool,
    PoolStats,
    ReplicaHealth,
    ConnectionWrapper,
    configure_pool,
    get_pool,
    close_pool,
)
```

### Repository Module

From `aragora.persistence.repositories.base`:

```python
from aragora.persistence.repositories.base import (
    BaseRepository,
    RepositoryError,
    EntityNotFoundError,
)
```
