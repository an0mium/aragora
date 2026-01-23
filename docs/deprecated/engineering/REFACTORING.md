# Refactoring Plan: Database Manager Abstraction

> **Deprecated:** Historical refactoring notes. For current architecture, see
> `docs/ARCHITECTURE.md` and `docs/DATABASE.md`.

## Problem Statement

The codebase has 21 duplicate `_init_db()` functions and 39 usages of `get_wal_connection()` spread across modules. This leads to:
- Code duplication
- Inconsistent error handling
- Difficult maintenance
- No centralized connection pooling

## Current State

### Affected Modules (21 stores)
```
aragora/insights/store.py           - InsightStore
aragora/ranking/elo.py              - EloDatabase
aragora/memory/store.py             - CritiqueStore
aragora/memory/streams.py           - MemoryStreamStore
aragora/memory/continuum.py         - ContinuumMemory
aragora/memory/consensus.py         - ConsensusMemory
aragora/learning/meta.py            - MetaLearningStore
aragora/runtime/metadata.py         - MetadataStore
aragora/agents/personas.py          - PersonaManager
aragora/agents/truth_grounding.py   - PositionStore
aragora/agents/laboratory.py        - PersonaLabStore
aragora/agents/calibration.py       - CalibrationTracker
aragora/server/storage.py           - ServerStorage
aragora/debate/traces.py            - DebateTraceStore
aragora/evolution/evolver.py        - EvolutionStore
aragora/persistence/evolution.py    - PersistenceStore
aragora/audience/feedback.py        - FeedbackStore
aragora/tournaments/tournament.py   - TournamentStore
aragora/genesis/breeding.py         - PopulationStore
aragora/genesis/ledger.py           - GenesisLedger
aragora/genesis/genome.py           - GenomeStore
```

### Existing Infrastructure
- `aragora/storage/schema.py` - SchemaManager for migrations
- `aragora/storage/connection.py` - get_wal_connection helper

## Proposed Solution

### 1. Create Base Class

**File:** `aragora/storage/base.py`

```python
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
import sqlite3

from aragora.storage.connection import get_wal_connection, DB_TIMEOUT
from aragora.storage.schema import SchemaManager


class DatabaseStore(ABC):
    """Base class for SQLite-backed stores.

    Provides:
    - Connection management with WAL mode
    - Schema initialization and migrations
    - Common query helpers
    """

    # Subclasses should set these
    SCHEMA_NAME: str = ""
    SCHEMA_VERSION: int = 1

    def __init__(self, db_path: str, timeout: float = DB_TIMEOUT):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._init_db()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection as a context manager."""
        conn = get_wal_connection(str(self.db_path), timeout=self.timeout)
        try:
            yield conn
        finally:
            conn.close()

    @abstractmethod
    def _get_initial_schema(self) -> str:
        """Return the initial schema SQL."""
        pass

    def _init_db(self) -> None:
        """Initialize database schema with migrations."""
        with self.connection() as conn:
            manager = SchemaManager(
                conn, self.SCHEMA_NAME, current_version=self.SCHEMA_VERSION
            )
            manager.ensure_schema(self._get_initial_schema())
            self._register_migrations(manager)
            manager.run_migrations()

    def _register_migrations(self, manager: SchemaManager) -> None:
        """Override to register migrations between versions."""
        pass

    # Query helpers
    def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute SQL without returning results."""
        with self.connection() as conn:
            conn.execute(sql, params)
            conn.commit()

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Fetch single row."""
        with self.connection() as conn:
            return conn.execute(sql, params).fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Fetch all rows."""
        with self.connection() as conn:
            return conn.execute(sql, params).fetchall()
```

### 2. Migration Example

**Before:**
```python
class CritiqueStore:
    def __init__(self, db_path: str = ".nomic/critiques.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = get_wal_connection(self.db_path, timeout=DB_TIMEOUT)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            manager = SchemaManager(conn, "critique_store", current_version=3)
            manager.ensure_schema(INITIAL_SCHEMA)
            # ... migrations ...
```

**After:**
```python
class CritiqueStore(DatabaseStore):
    SCHEMA_NAME = "critique_store"
    SCHEMA_VERSION = 3

    def __init__(self, db_path: str = ".nomic/critiques.db"):
        super().__init__(db_path)

    def _get_initial_schema(self) -> str:
        return INITIAL_SCHEMA

    def _register_migrations(self, manager: SchemaManager) -> None:
        manager.register_migration(1, 2, MIGRATION_1_TO_2)
        manager.register_migration(2, 3, MIGRATION_2_TO_3)
```

### 3. Implementation Phases

**Phase 1: Create Base Class** (Low Risk)
- Add `aragora/storage/base.py` with `DatabaseStore`
- Add tests for base class
- No changes to existing code

**Phase 2: Migrate Low-Risk Modules** (Medium Risk)
- InsightStore, FeedbackStore, MetadataStore
- These are simpler stores with fewer dependencies

**Phase 3: Migrate Core Modules** (Higher Risk)
- CritiqueStore, MemoryStreamStore, EloDatabase
- These are heavily used and need careful testing

**Phase 4: Migrate Complex Modules** (Highest Risk)
- GenomeStore, GenesisLedger, TournamentStore
- These have complex schemas and relationships

### 4. Benefits

1. **Reduced Duplication**: ~500 lines of code eliminated
2. **Consistent Error Handling**: Central connection management
3. **Easier Testing**: Mock the base class
4. **Future Pooling**: Can add connection pooling in one place
5. **Better Observability**: Central logging/metrics

### 5. Risks

1. **Breaking Changes**: Existing stores may have subtle differences
2. **Migration Errors**: Schema migrations need careful testing
3. **Performance**: Base class overhead (minimal)

### 6. Testing Strategy

1. Create comprehensive tests for DatabaseStore base class
2. For each migration:
   - Run existing tests to ensure no regressions
   - Add new tests for base class features
3. Integration tests with actual SQLite files

## Timeline Estimate

- Phase 1: 1-2 hours
- Phase 2: 2-3 hours (3 modules)
- Phase 3: 4-6 hours (3 modules)
- Phase 4: 4-6 hours (3 modules)

**Total: 11-17 hours of focused work**

## Decision

This refactoring is **recommended** but should be done incrementally:
1. Start with Phase 1 (create base class)
2. Migrate one module to validate approach
3. If successful, continue with remaining phases
4. Track progress in todo list
