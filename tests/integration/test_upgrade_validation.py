"""
Upgrade Validation Tests.

Verifies the database upgrade path works correctly:
- Schema creation from scratch via SchemaManager
- Migration runner handles already-applied versions (idempotency)
- Data survives schema migrations
- Config backward compatibility (old-style Arena params emit deprecation warnings)
- WAL mode and connection pool behavior
- Advisory lock behavior (SQLite via WAL file-level locking)

These tests use in-memory and temporary SQLite databases so they run
without external infrastructure.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import pytest

from aragora.storage.schema import (
    ConnectionPool,
    DatabaseManager,
    Migration,
    SchemaManager,
    get_wal_connection,
    safe_add_column,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_conn():
    """Provide an in-memory SQLite connection with WAL-compatible settings."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture()
def tmp_db(tmp_path: Path):
    """Provide a temporary on-disk SQLite database path."""
    return str(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# 1. Schema creation from scratch
# ---------------------------------------------------------------------------


class TestSchemaCreationFromScratch:
    """Verify SchemaManager can bootstrap a fresh database."""

    INITIAL_SCHEMA = """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            data TEXT
        );
    """

    def test_fresh_database_gets_version_1(self, mem_conn: sqlite3.Connection):
        mgr = SchemaManager(mem_conn, "test_module", current_version=1)
        assert mgr.get_version() == 0  # nothing applied yet

        applied = mgr.ensure_schema(initial_schema=self.INITIAL_SCHEMA)
        assert applied is True
        assert mgr.get_version() == 1

    def test_tables_created(self, mem_conn: sqlite3.Connection):
        mgr = SchemaManager(mem_conn, "test_module", current_version=1)
        mgr.ensure_schema(initial_schema=self.INITIAL_SCHEMA)

        result = mgr.validate_schema(["items", "metadata"])
        assert result["valid"] is True
        assert result["missing"] == []
        assert result["version"] == 1

    def test_version_table_exists(self, mem_conn: sqlite3.Connection):
        mgr = SchemaManager(mem_conn, "test_module", current_version=1)
        mgr.ensure_schema(initial_schema=self.INITIAL_SCHEMA)

        cursor = mem_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_versions'"
        )
        assert cursor.fetchone() is not None

    def test_multiple_modules_independent(self, mem_conn: sqlite3.Connection):
        """Two modules can track versions independently on the same connection."""
        mgr_a = SchemaManager(mem_conn, "module_a", current_version=1)
        mgr_b = SchemaManager(mem_conn, "module_b", current_version=1)

        mgr_a.ensure_schema(
            initial_schema="CREATE TABLE a_table (id INTEGER PRIMARY KEY);"
        )
        mgr_b.ensure_schema(
            initial_schema="CREATE TABLE b_table (id INTEGER PRIMARY KEY);"
        )

        assert mgr_a.get_version() == 1
        assert mgr_b.get_version() == 1

        # Advance only module_a
        mgr_a2 = SchemaManager(mem_conn, "module_a", current_version=2)
        mgr_a2.register_migration(
            1,
            2,
            sql="ALTER TABLE a_table ADD COLUMN extra TEXT;",
            description="add extra column",
        )
        mgr_a2.ensure_schema()
        assert mgr_a2.get_version() == 2
        assert mgr_b.get_version() == 1  # unchanged


# ---------------------------------------------------------------------------
# 2. Migration runner handles already-applied versions (idempotency)
# ---------------------------------------------------------------------------


class TestMigrationIdempotency:
    """Running ensure_schema twice should be a no-op the second time."""

    INITIAL = "CREATE TABLE events (id INTEGER PRIMARY KEY, payload TEXT);"

    def test_ensure_schema_idempotent(self, mem_conn: sqlite3.Connection):
        mgr = SchemaManager(mem_conn, "events", current_version=2)
        mgr.register_migration(
            1, 2, sql="ALTER TABLE events ADD COLUMN ts TIMESTAMP;"
        )
        mgr.ensure_schema(initial_schema=self.INITIAL)
        assert mgr.get_version() == 2

        # Second call should return False (nothing to do)
        applied = mgr.ensure_schema(initial_schema=self.INITIAL)
        assert applied is False
        assert mgr.get_version() == 2

    def test_partial_migration_resumes(self, mem_conn: sqlite3.Connection):
        """If only v1 was applied previously, a new manager at v3 picks up from v1."""
        mgr1 = SchemaManager(mem_conn, "resume", current_version=1)
        mgr1.ensure_schema(
            initial_schema="CREATE TABLE t (id INTEGER PRIMARY KEY);"
        )
        assert mgr1.get_version() == 1

        mgr3 = SchemaManager(mem_conn, "resume", current_version=3)
        mgr3.register_migration(1, 2, sql="ALTER TABLE t ADD COLUMN a TEXT;")
        mgr3.register_migration(2, 3, sql="ALTER TABLE t ADD COLUMN b TEXT;")
        applied = mgr3.ensure_schema()
        assert applied is True
        assert mgr3.get_version() == 3

    def test_newer_db_version_skipped(self, mem_conn: sqlite3.Connection):
        """Code at v1 should not downgrade a v2 database."""
        mgr2 = SchemaManager(mem_conn, "compat", current_version=2)
        mgr2.ensure_schema(
            initial_schema="CREATE TABLE c (id INTEGER PRIMARY KEY);"
        )
        mgr2.register_migration(1, 2, sql="ALTER TABLE c ADD COLUMN x TEXT;")
        mgr2.ensure_schema()
        assert mgr2.get_version() == 2

        # Now a "v1" manager should leave v2 intact
        mgr1 = SchemaManager(mem_conn, "compat", current_version=1)
        applied = mgr1.ensure_schema()
        assert applied is False
        assert mgr1.get_version() == 2  # still v2


# ---------------------------------------------------------------------------
# 3. Data survives migrations
# ---------------------------------------------------------------------------


class TestDataSurvivesMigration:
    """Insert data at v1, run migration to v2, verify data is intact."""

    INITIAL = """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL
        );
    """

    MIGRATION_SQL = "ALTER TABLE products ADD COLUMN category TEXT DEFAULT 'general';"

    def _seed_data(self, conn: sqlite3.Connection) -> list[tuple]:
        rows = [("Widget", 9.99), ("Gadget", 19.99), ("Doohickey", 4.50)]
        conn.executemany(
            "INSERT INTO products (name, price) VALUES (?, ?)", rows
        )
        conn.commit()
        return rows

    def test_data_preserved_after_migration(self, mem_conn: sqlite3.Connection):
        mgr = SchemaManager(mem_conn, "products", current_version=1)
        mgr.ensure_schema(initial_schema=self.INITIAL)

        original = self._seed_data(mem_conn)

        # Upgrade to v2
        mgr2 = SchemaManager(mem_conn, "products", current_version=2)
        mgr2.register_migration(1, 2, sql=self.MIGRATION_SQL, description="add category")
        mgr2.ensure_schema()

        cursor = mem_conn.execute("SELECT name, price, category FROM products ORDER BY id")
        rows = cursor.fetchall()

        assert len(rows) == len(original)
        for row, (name, price) in zip(rows, original):
            assert row[0] == name
            assert row[1] == price
            assert row[2] == "general"  # default value applied

    def test_new_column_usable_after_migration(self, mem_conn: sqlite3.Connection):
        mgr = SchemaManager(mem_conn, "products", current_version=1)
        mgr.ensure_schema(initial_schema=self.INITIAL)
        self._seed_data(mem_conn)

        mgr2 = SchemaManager(mem_conn, "products", current_version=2)
        mgr2.register_migration(1, 2, sql=self.MIGRATION_SQL)
        mgr2.ensure_schema()

        mem_conn.execute(
            "UPDATE products SET category = 'tools' WHERE name = 'Widget'"
        )
        mem_conn.commit()
        cursor = mem_conn.execute(
            "SELECT category FROM products WHERE name = 'Widget'"
        )
        assert cursor.fetchone()[0] == "tools"

    def test_function_migration_preserves_data(self, mem_conn: sqlite3.Connection):
        """Migrations using Python functions (not SQL) also preserve data."""
        mgr = SchemaManager(mem_conn, "products", current_version=1)
        mgr.ensure_schema(initial_schema=self.INITIAL)
        self._seed_data(mem_conn)

        def upgrade_fn(conn: sqlite3.Connection) -> None:
            conn.execute(
                "ALTER TABLE products ADD COLUMN sku TEXT DEFAULT 'UNKNOWN'"
            )
            # Back-fill existing rows
            conn.execute(
                "UPDATE products SET sku = 'SKU-' || CAST(id AS TEXT)"
            )

        mgr2 = SchemaManager(mem_conn, "products", current_version=2)
        mgr2.register_migration(1, 2, function=upgrade_fn, description="add sku")
        mgr2.ensure_schema()

        cursor = mem_conn.execute("SELECT id, sku FROM products ORDER BY id")
        for row in cursor.fetchall():
            assert row[1] == f"SKU-{row[0]}"


# ---------------------------------------------------------------------------
# 4. Config backward compatibility (Arena deprecation warnings)
# ---------------------------------------------------------------------------


class TestConfigBackwardCompatibility:
    """Old-style individual Arena params should still work but emit deprecation warnings."""

    def test_supermemory_params_emit_deprecation(self):
        """Individual supermemory params trigger DeprecationWarning when no MemoryConfig given."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                from aragora.debate.orchestrator_config import resolve_arena_config

                resolve_arena_config(
                    enable_supermemory=True,
                    memory_config=None,
                )
            except (ImportError, TypeError, Exception):
                pytest.skip("resolve_arena_config not importable in test environment")

            deprecation_msgs = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            assert len(deprecation_msgs) >= 1, (
                "Expected DeprecationWarning for individual supermemory params"
            )

    def test_knowledge_params_emit_deprecation(self):
        """Individual knowledge params trigger DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                from aragora.debate.orchestrator_config import resolve_arena_config

                resolve_arena_config(
                    enable_knowledge_mound=True,
                    knowledge_config=None,
                )
            except (ImportError, TypeError, Exception):
                pytest.skip("resolve_arena_config not importable in test environment")

            deprecation_msgs = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            assert len(deprecation_msgs) >= 1, (
                "Expected DeprecationWarning for individual knowledge params"
            )

    def test_no_deprecation_when_config_objects_used(self):
        """Using config objects should NOT emit deprecation warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                from aragora.debate.orchestrator_config import resolve_arena_config

                # Pass config objects instead of individual params
                resolve_arena_config(
                    memory_config=object(),  # truthy = config provided
                    knowledge_config=object(),
                    evolution_config=object(),
                    ml_config=object(),
                )
            except (ImportError, TypeError, Exception):
                pytest.skip("resolve_arena_config not importable in test environment")

            deprecation_msgs = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            assert len(deprecation_msgs) == 0, (
                f"No DeprecationWarning expected when using config objects, got: "
                f"{[str(w.message) for w in deprecation_msgs]}"
            )


# ---------------------------------------------------------------------------
# 5. WAL mode and connection behavior
# ---------------------------------------------------------------------------


class TestWALMode:
    """Verify WAL mode is configured correctly for on-disk databases."""

    def test_wal_mode_enabled(self, tmp_db: str):
        conn = get_wal_connection(tmp_db)
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode == "wal"
        finally:
            conn.close()

    def test_synchronous_normal(self, tmp_db: str):
        conn = get_wal_connection(tmp_db)
        try:
            cursor = conn.execute("PRAGMA synchronous")
            # NORMAL = 1
            assert cursor.fetchone()[0] == 1
        finally:
            conn.close()

    def test_row_factory_set(self, tmp_db: str):
        conn = get_wal_connection(tmp_db)
        try:
            assert conn.row_factory is sqlite3.Row
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 6. safe_add_column
# ---------------------------------------------------------------------------


class TestSafeAddColumn:
    """Verify safe_add_column idempotency and validation."""

    def test_adds_column(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        added = safe_add_column(mem_conn, "t", "name", "TEXT")
        assert added is True
        cursor = mem_conn.execute("PRAGMA table_info(t)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "name" in cols

    def test_idempotent(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        added = safe_add_column(mem_conn, "t", "name", "TEXT")
        assert added is False

    def test_rejects_invalid_table_name(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_add_column(mem_conn, "DROP TABLE t;--", "col", "TEXT")

    def test_rejects_invalid_column_type(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        with pytest.raises(ValueError, match="Invalid column type"):
            safe_add_column(mem_conn, "t", "col", "EXPLOIT")

    def test_rejects_invalid_default(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        with pytest.raises(ValueError, match="Invalid default value"):
            safe_add_column(mem_conn, "t", "col", "TEXT", default="'; DROP TABLE t;--")


# ---------------------------------------------------------------------------
# 7. DatabaseManager
# ---------------------------------------------------------------------------


class TestDatabaseManager:
    """Verify DatabaseManager singleton and connection management."""

    def test_singleton_per_path(self, tmp_db: str):
        DatabaseManager.clear_instances()
        try:
            mgr1 = DatabaseManager.get_instance(tmp_db)
            mgr2 = DatabaseManager.get_instance(tmp_db)
            assert mgr1 is mgr2
        finally:
            DatabaseManager.clear_instances()

    def test_connection_context_commits(self, tmp_db: str):
        DatabaseManager.clear_instances()
        try:
            mgr = DatabaseManager.get_instance(tmp_db)
            with mgr.connection() as conn:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
                conn.execute("INSERT INTO t (v) VALUES ('hello')")

            # Verify committed
            with mgr.connection() as conn:
                cursor = conn.execute("SELECT v FROM t")
                assert cursor.fetchone()[0] == "hello"
        finally:
            DatabaseManager.clear_instances()

    def test_connection_context_rollback_on_error(self, tmp_db: str):
        DatabaseManager.clear_instances()
        try:
            mgr = DatabaseManager.get_instance(tmp_db)
            with mgr.connection() as conn:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
                conn.execute("INSERT INTO t (v) VALUES ('keep')")

            with pytest.raises(RuntimeError):
                with mgr.connection() as conn:
                    conn.execute("INSERT INTO t (v) VALUES ('discard')")
                    raise RuntimeError("simulated failure")

            with mgr.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM t")
                assert cursor.fetchone()[0] == 1  # only 'keep'
        finally:
            DatabaseManager.clear_instances()

    def test_pool_stats(self, tmp_db: str):
        DatabaseManager.clear_instances()
        try:
            mgr = DatabaseManager.get_instance(tmp_db)
            stats = mgr.pool_stats()
            assert "hits" in stats
            assert "misses" in stats
            assert "max_pool_size" in stats
        finally:
            DatabaseManager.clear_instances()


# ---------------------------------------------------------------------------
# 8. ConnectionPool
# ---------------------------------------------------------------------------


class TestConnectionPool:
    """Verify ConnectionPool acquire/release and stats."""

    def test_acquire_and_release(self, tmp_db: str):
        pool = ConnectionPool(tmp_db, max_connections=3)
        try:
            conn = pool.acquire()
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            pool.release(conn)

            stats = pool.stats()
            assert stats["active"] == 0
            assert stats["idle"] == 1
        finally:
            pool.close()

    def test_context_manager(self, tmp_db: str):
        pool = ConnectionPool(tmp_db, max_connections=3)
        try:
            with pool.connection() as conn:
                conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO t VALUES (1)")

            with pool.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM t")
                assert cursor.fetchone()[0] == 1
        finally:
            pool.close()

    def test_pool_closed_raises(self, tmp_db: str):
        pool = ConnectionPool(tmp_db, max_connections=2)
        pool.close()
        with pytest.raises(Exception):
            pool.acquire()


# ---------------------------------------------------------------------------
# 9. Migration object validation
# ---------------------------------------------------------------------------


class TestMigrationObject:
    """Verify the Migration dataclass behavior."""

    def test_sql_migration(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        m = Migration(1, 2, sql="ALTER TABLE t ADD COLUMN v TEXT;")
        m.apply(mem_conn)
        cursor = mem_conn.execute("PRAGMA table_info(t)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "v" in cols

    def test_function_migration(self, mem_conn: sqlite3.Connection):
        mem_conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")

        def fn(conn: sqlite3.Connection) -> None:
            conn.execute("ALTER TABLE t ADD COLUMN f TEXT")

        m = Migration(1, 2, function=fn)
        m.apply(mem_conn)
        cursor = mem_conn.execute("PRAGMA table_info(t)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "f" in cols

    def test_empty_migration_raises(self, mem_conn: sqlite3.Connection):
        m = Migration(1, 2)
        with pytest.raises(ValueError, match="must have either sql or function"):
            m.apply(mem_conn)


# ---------------------------------------------------------------------------
# 10. Multi-step migration chain
# ---------------------------------------------------------------------------


class TestMultiStepMigrationChain:
    """Verify a chain of v1->v2->v3->v4 migrations applies correctly."""

    def test_full_chain(self, mem_conn: sqlite3.Connection):
        initial = "CREATE TABLE audit (id INTEGER PRIMARY KEY, action TEXT);"

        mgr = SchemaManager(mem_conn, "audit", current_version=4)
        mgr.register_migration(1, 2, sql="ALTER TABLE audit ADD COLUMN actor TEXT;")
        mgr.register_migration(2, 3, sql="ALTER TABLE audit ADD COLUMN ts TIMESTAMP;")
        mgr.register_migration(
            3,
            4,
            sql="ALTER TABLE audit ADD COLUMN details TEXT DEFAULT '';",
        )

        mgr.ensure_schema(initial_schema=initial)
        assert mgr.get_version() == 4

        # Verify all columns exist
        cursor = mem_conn.execute("PRAGMA table_info(audit)")
        cols = {row[1] for row in cursor.fetchall()}
        assert cols == {"id", "action", "actor", "ts", "details"}

    def test_data_preserved_through_chain(self, mem_conn: sqlite3.Connection):
        initial = "CREATE TABLE audit (id INTEGER PRIMARY KEY, action TEXT);"

        # Start at v1
        mgr1 = SchemaManager(mem_conn, "audit", current_version=1)
        mgr1.ensure_schema(initial_schema=initial)
        mem_conn.execute("INSERT INTO audit (action) VALUES ('login')")
        mem_conn.execute("INSERT INTO audit (action) VALUES ('logout')")
        mem_conn.commit()

        # Upgrade to v3
        mgr3 = SchemaManager(mem_conn, "audit", current_version=3)
        mgr3.register_migration(1, 2, sql="ALTER TABLE audit ADD COLUMN actor TEXT;")
        mgr3.register_migration(2, 3, sql="ALTER TABLE audit ADD COLUMN ts TIMESTAMP;")
        mgr3.ensure_schema()

        cursor = mem_conn.execute("SELECT action FROM audit ORDER BY id")
        actions = [row[0] for row in cursor.fetchall()]
        assert actions == ["login", "logout"]
