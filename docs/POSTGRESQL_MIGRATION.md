# PostgreSQL Migration Guide

This guide covers migrating Aragora from SQLite to PostgreSQL for production deployments.

## Overview

Aragora supports both SQLite (default) and PostgreSQL backends:

- **SQLite**: Default for development and single-instance deployments
- **PostgreSQL**: Recommended for production with horizontal scaling and concurrent writes

## Prerequisites

1. **PostgreSQL 13+** installed and running
2. **asyncpg** Python package installed:
   ```bash
   pip install asyncpg
   # or
   pip install aragora[postgres]
   ```

## Configuration

### Environment Variables

Set the PostgreSQL connection string:

```bash
# Primary connection string
export DATABASE_URL="postgresql://user:password@host:5432/aragora"

# Or use Aragora-specific variable
export ARAGORA_POSTGRES_DSN="postgresql://user:password@host:5432/aragora"
```

### Connection Pool Settings

Optional environment variables for tuning:

```bash
export ARAGORA_DB_POOL_SIZE=20      # Max connections (default: 20)
export ARAGORA_DB_BACKEND=postgresql # Explicitly set backend
```

## Available PostgreSQL Stores

### PostgresConsensusMemory

Async PostgreSQL-backed consensus memory with connection pooling.

**Tables:**
- `consensus` - Consensus records with JSONB data
- `dissent` - Dissenting opinions with acknowledgment tracking
- `verified_proofs` - Formal verification records

**Usage:**
```python
from aragora.memory.postgres_consensus import get_postgres_consensus_memory

# Initialize (uses DATABASE_URL or ARAGORA_POSTGRES_DSN)
memory = await get_postgres_consensus_memory()

# Or with explicit DSN
memory = await get_postgres_consensus_memory(
    dsn="postgresql://user:pass@host/db"
)

# Store consensus
await memory.store_consensus(
    topic="Rate limiting approach",
    conclusion="Use token bucket algorithm",
    strength="strong",
    confidence=0.85,
    participating_agents=["claude", "gpt4"],
    agreeing_agents=["claude", "gpt4"],
)

# Find similar consensus
results = await memory.find_similar("rate limiting", limit=5)
```

### PostgresCritiqueStore

Async PostgreSQL-backed critique pattern storage.

**Tables:**
- `debates` - Debate results
- `critiques` - Individual critique records
- `patterns` - Successful critique patterns
- `agent_reputation` - Per-agent reputation tracking
- `patterns_archive` - Archived patterns (adaptive forgetting)

**Usage:**
```python
from aragora.memory.postgres_critique import get_postgres_critique_store

# Initialize
store = await get_postgres_critique_store()

# Store debate result
await store.store_debate(
    debate_id="debate-123",
    task="Design a rate limiter",
    final_answer="Use token bucket",
    consensus_reached=True,
    confidence=0.9,
)

# Get statistics
stats = await store.get_stats()
```

## Data Migration

### Using the Memory Migrator

The `MemoryMigrator` tool migrates data from SQLite to PostgreSQL:

```bash
# Migrate all memory stores
python -m aragora.persistence.migrations.postgres.memory_migrator \
    --sqlite-path .nomic/agora_memory.db \
    --postgres-dsn "postgresql://user:pass@host/db"

# Migrate only ConsensusMemory
python -m aragora.persistence.migrations.postgres.memory_migrator \
    --sqlite-path .nomic/agora_memory.db \
    --postgres-dsn "postgresql://..." \
    --store consensus

# Migrate only CritiqueStore
python -m aragora.persistence.migrations.postgres.memory_migrator \
    --sqlite-path .nomic/agora_memory.db \
    --postgres-dsn "postgresql://..." \
    --store critique
```

### Migration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sqlite-path` | Path to SQLite database | Required |
| `--postgres-dsn` | PostgreSQL connection string | Required |
| `--store` | Which store to migrate (all/consensus/critique) | all |
| `--batch-size` | Rows per insert batch | 1000 |
| `--no-skip-existing` | Error on duplicates instead of skipping | Skip |
| `-v, --verbose` | Verbose output | Off |

### Programmatic Migration

```python
from aragora.persistence.migrations.postgres import MemoryMigrator

migrator = MemoryMigrator(
    sqlite_path=".nomic/agora_memory.db",
    postgres_dsn="postgresql://user:pass@host/db",
    batch_size=1000,
    skip_existing=True,
)

# Migrate everything
report = await migrator.migrate_all()

# Or selectively
consensus_stats = await migrator.migrate_consensus_memory()
critique_stats = await migrator.migrate_critique_store()

# Check results
print(f"Migrated: {report.total_migrated}")
print(f"Skipped: {report.total_skipped}")
print(f"Errors: {report.total_errors}")

await migrator.close()
```

## Schema Details

### ConsensusMemory Schema

```sql
CREATE TABLE consensus (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    topic_hash TEXT NOT NULL,
    conclusion TEXT NOT NULL,
    strength TEXT DEFAULT 'moderate',
    confidence REAL DEFAULT 0.5,
    participating_agents JSONB DEFAULT '[]'::jsonb,
    agreeing_agents JSONB DEFAULT '[]'::jsonb,
    domain TEXT,
    debate_id TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_consensus_topic_hash ON consensus(topic_hash);
CREATE INDEX idx_consensus_domain ON consensus(domain);
CREATE INDEX idx_consensus_created ON consensus(created_at DESC);
```

### CritiqueStore Schema

```sql
CREATE TABLE debates (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    final_answer TEXT,
    consensus_reached BOOLEAN DEFAULT FALSE,
    confidence REAL,
    rounds_used INTEGER,
    duration_seconds REAL,
    grounded_verdict JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE critiques (
    id SERIAL PRIMARY KEY,
    debate_id TEXT REFERENCES debates(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    target_agent TEXT,
    issues JSONB DEFAULT '[]'::jsonb,
    suggestions JSONB DEFAULT '[]'::jsonb,
    severity REAL,
    reasoning TEXT,
    led_to_improvement BOOLEAN DEFAULT FALSE,
    expected_usefulness REAL DEFAULT 0.5,
    actual_usefulness REAL,
    prediction_error REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    issue_type TEXT NOT NULL,
    issue_text TEXT NOT NULL,
    suggestion_text TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_severity REAL DEFAULT 0.5,
    surprise_score REAL DEFAULT 0.0,
    base_rate REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE agent_reputation (
    agent_name TEXT PRIMARY KEY,
    proposals_made INTEGER DEFAULT 0,
    proposals_accepted INTEGER DEFAULT 0,
    critiques_given INTEGER DEFAULT 0,
    critiques_valuable INTEGER DEFAULT 0,
    calibration_score REAL DEFAULT 0.5,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Best Practices

### Connection Pooling

PostgreSQL stores use connection pooling via asyncpg:

```python
from aragora.storage.postgres_store import get_postgres_pool

# Get or create the global pool
pool = await get_postgres_pool(
    dsn="postgresql://...",
    min_size=5,   # Minimum connections
    max_size=20,  # Maximum connections
)

# Use with stores
from aragora.memory.postgres_consensus import PostgresConsensusMemory

memory = PostgresConsensusMemory(pool)
await memory.initialize()
```

### Graceful Shutdown

Close the pool on shutdown:

```python
from aragora.storage.postgres_store import close_postgres_pool

# At application shutdown
await close_postgres_pool()
```

### Health Checks

```python
# Check if PostgreSQL is available
async with pool.acquire() as conn:
    await conn.fetchval("SELECT 1")
```

## Troubleshooting

### Connection Errors

```
RuntimeError: PostgreSQL DSN not configured
```

**Solution:** Set `DATABASE_URL` or `ARAGORA_POSTGRES_DSN` environment variable.

### asyncpg Not Installed

```
RuntimeError: PostgreSQL backend requires 'asyncpg' package
```

**Solution:** Install asyncpg: `pip install asyncpg`

### Migration Errors

If migration fails mid-way:
1. Check the error report for specific table failures
2. Fix the issue (e.g., schema mismatch)
3. Re-run with `--skip-existing` to continue from where it stopped

### Performance Issues

1. **Increase pool size** for high-concurrency workloads
2. **Use batch operations** for bulk inserts
3. **Add indexes** for frequent query patterns
4. **Enable connection timeout** to prevent hung connections

## Next Steps

- [x] Migrate EloSystem to PostgreSQL (`PostgresEloDatabase`)
- [x] Migrate ContinuumMemory to PostgreSQL (`PostgresContinuumMemory`)
- [ ] Add PostgreSQL support to remaining stores
- [ ] Implement read replicas for scaling reads
