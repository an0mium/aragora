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

## Unified Schema File

For production deployments, use the consolidated schema file that includes all tables:

```bash
# Initialize database with full schema
psql -U postgres -d aragora -f aragora/db/schema/postgres_schema.sql

# Or use the init script
python scripts/init_postgres_db.py
```

The schema file includes:
- Users and organizations
- Audit and security tables
- Governance and approvals
- Integrations and webhooks
- Marketplace
- Job queue
- Federation
- Gauntlet testing
- Finding workflows
- Notifications
- Sharing
- Decision results

## Docker Deployment

```bash
# Start PostgreSQL with docker-compose
docker compose --profile postgres up -d postgres

# Wait for PostgreSQL to be ready
docker compose --profile postgres exec postgres pg_isready -U aragora

# Initialize schema
docker compose exec aragora python scripts/init_postgres_db.py

# Or start everything together
docker compose --profile postgres up
```

## Next Steps

- [x] Migrate EloSystem to PostgreSQL (`PostgresEloDatabase`)
- [x] Migrate ContinuumMemory to PostgreSQL (`PostgresContinuumMemory`)
- [x] Unified schema file created (`aragora/db/schema/postgres_schema.sql`)
- [x] Database initialization script (`scripts/init_postgres_db.py`)
- [ ] Add PostgreSQL support to remaining stores
- [ ] Implement read replicas for scaling reads

---

## Supabase Deployment (AWS EC2)

This section covers deploying Aragora with Supabase as the PostgreSQL backend on AWS EC2 instances.

### Overview

All PostgreSQL store implementations are complete and ready to use:

| Store | Class | Status |
|-------|-------|--------|
| Workflow Store | `PostgresFindingWorkflowStore` | Ready |
| Job Queue Store | `PostgresJobQueueStore` | Ready |
| Integration Store | `PostgresIntegrationStore` | Ready |
| Checkpoint Store | `PostgresCheckpointStore` | Ready |

### Supabase Setup

1. **Create Supabase Project**
   - Go to https://supabase.com/dashboard
   - Create new project or use existing
   - Note your project credentials

2. **Get Connection String**
   - Go to Settings → Database
   - Copy the connection string (URI format)
   - Use **port 5432** for direct connection (recommended with asyncpg)
   - Use **port 6543** for pooled connection (PgBouncer)

   ```
   postgresql://postgres:[PASSWORD]@db.[PROJECT_ID].supabase.co:5432/postgres
   ```

### AWS EC2 Configuration

#### 1. Store Secrets in AWS Parameter Store

```bash
# Store database URL (SecureString)
aws ssm put-parameter \
    --name "/aragora/prod/DATABASE_URL" \
    --value "postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres" \
    --type SecureString

# Store Supabase service key
aws ssm put-parameter \
    --name "/aragora/prod/SUPABASE_KEY" \
    --value "your-service-role-key" \
    --type SecureString

# Store encryption key
aws ssm put-parameter \
    --name "/aragora/prod/ARAGORA_ENCRYPTION_KEY" \
    --value "$(python -c 'import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())')" \
    --type SecureString
```

#### 2. Configure IAM Role

Add SSM read permissions to your EC2 instance role:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ssm:GetParameter",
                "ssm:GetParameters"
            ],
            "Resource": "arn:aws:ssm:*:*:parameter/aragora/prod/*"
        }
    ]
}
```

#### 3. EC2 User Data Script

```bash
#!/bin/bash
# Fetch secrets from Parameter Store
export DATABASE_URL=$(aws ssm get-parameter \
    --name "/aragora/prod/DATABASE_URL" \
    --with-decryption \
    --query Parameter.Value \
    --output text)

export SUPABASE_KEY=$(aws ssm get-parameter \
    --name "/aragora/prod/SUPABASE_KEY" \
    --with-decryption \
    --query Parameter.Value \
    --output text)

export ARAGORA_ENCRYPTION_KEY=$(aws ssm get-parameter \
    --name "/aragora/prod/ARAGORA_ENCRYPTION_KEY" \
    --with-decryption \
    --query Parameter.Value \
    --output text)

# Set backend mode
export ARAGORA_DB_BACKEND=postgres
export ARAGORA_ENV=production

# Start application
cd /opt/aragora
python -m aragora.server.unified_server --port 8080
```

#### 4. Security Group Rules

Allow outbound connections to Supabase:
- **Port 5432** to Supabase IP ranges (for asyncpg direct connection)
- Consider VPC Peering for private connectivity (Supabase Enterprise)

### Data Migration

If migrating existing SQLite data:

```bash
# Set environment
export DATABASE_URL="postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres"

# Dry run to see what would be migrated
python scripts/migrate_sqlite_to_supabase.py --all --dry-run

# Run migration
python scripts/migrate_sqlite_to_supabase.py --all

# Migrate specific stores
python scripts/migrate_sqlite_to_supabase.py --stores workflow jobs integrations
```

### Verification

#### Check Store Initialization

Application logs should show:
```
INFO     aragora.storage: Using PostgresFindingWorkflowStore
INFO     aragora.storage: Using PostgresJobQueueStore
INFO     aragora.storage: Using PostgresIntegrationStore
INFO     aragora.workflow: Using PostgresCheckpointStore
```

#### Health Endpoint

```bash
curl http://localhost:8080/api/health
# Expected: {"status": "healthy", "database": "connected"}
```

#### Pool Metrics

```bash
curl http://localhost:8080/api/admin/pool-metrics
# Expected: {"pool_size": 20, "free_connections": 18, ...}
```

### Rollback

To revert to SQLite:

```bash
# Set environment variable
export ARAGORA_DB_BACKEND=sqlite

# Restart application
```

SQLite databases remain intact as fallback.

### Production Checklist

- [ ] Supabase project created with connection string
- [ ] Environment variables stored in AWS Parameter Store
- [ ] EC2 IAM role has SSM GetParameter permission
- [ ] Security group allows outbound PostgreSQL (5432)
- [ ] `ARAGORA_DB_BACKEND=postgres` set
- [ ] `ARAGORA_ENV=production` set
- [ ] Application starts and logs show PostgreSQL stores
- [ ] Health endpoint reports database connected
- [ ] Multi-instance test: both EC2s share same data

### Connection Pool Tuning

Supabase has connection limits based on your plan:

| Plan | Direct Connections | Pooled Connections |
|------|-------------------|-------------------|
| Free | 10 | 200 |
| Pro | 60 | 200 |
| Team | 120 | 200 |
| Enterprise | Custom | Custom |

Configure pool size accordingly:

```bash
# For Pro plan with 2 EC2 instances
export ARAGORA_POSTGRES_POOL_MIN=5
export ARAGORA_POSTGRES_POOL_MAX=25  # Leave headroom for other connections
```
