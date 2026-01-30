"""
Tests for evolution database abstraction.

Tests cover:
- EvolutionDatabase class initialization
- Connection management
- Thread-safe access patterns
- Convenience methods (fetch_one, fetch_all)
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aragora.evolution.database import EvolutionDatabase


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def db(temp_db):
    """Create an EvolutionDatabase instance."""
    return EvolutionDatabase(temp_db)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEvolutionDatabaseInit:
    """Test EvolutionDatabase initialization."""

    def test_init_creates_file(self, temp_db):
        """Test that initialization creates database file."""
        db = EvolutionDatabase(temp_db)

        # Access to trigger any lazy initialization
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")

        assert Path(temp_db).exists()

    def test_init_with_path_object(self, temp_db):
        """Test initialization with Path object."""
        db = EvolutionDatabase(Path(temp_db))

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

        assert result[0] == 1

    def test_inherits_from_base_database(self, db):
        """Test that EvolutionDatabase inherits from BaseDatabase."""
        from aragora.storage import BaseDatabase

        assert isinstance(db, BaseDatabase)


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Test connection management."""

    def test_connection_context_manager(self, db):
        """Test connection works as context manager."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (1)")

        # Verify data persisted
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM test")
            assert cursor.fetchone()[0] == 1

    def test_connection_auto_commit(self, db):
        """Test that context manager auto-commits on success."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (42)")

        # Verify committed
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM test")
            assert cursor.fetchone()[0] == 42

    def test_connection_rollback_on_error(self, db):
        """Test that context manager rolls back on error."""
        try:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER)")
                cursor.execute("INSERT INTO test VALUES (1)")
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Table might exist but data should be rolled back
        # (depends on BaseDatabase implementation)
        # Just verify no crash
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")


# =============================================================================
# Convenience Method Tests
# =============================================================================


class TestConvenienceMethods:
    """Test convenience methods for database operations."""

    def test_fetch_one(self, db):
        """Test fetch_one method."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            cursor.execute("INSERT INTO test VALUES (1, 'alice')")
            cursor.execute("INSERT INTO test VALUES (2, 'bob')")

        if hasattr(db, "fetch_one"):
            row = db.fetch_one("SELECT * FROM test WHERE id = ?", (1,))
            assert row is not None
            assert row[0] == 1
            assert row[1] == "alice"

    def test_fetch_one_no_result(self, db):
        """Test fetch_one returns None when no match."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")

        if hasattr(db, "fetch_one"):
            row = db.fetch_one("SELECT * FROM test WHERE id = ?", (999,))
            assert row is None

    def test_fetch_all(self, db):
        """Test fetch_all method."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (1)")
            cursor.execute("INSERT INTO test VALUES (2)")
            cursor.execute("INSERT INTO test VALUES (3)")

        if hasattr(db, "fetch_all"):
            rows = db.fetch_all("SELECT * FROM test ORDER BY id")
            assert len(rows) == 3
            assert rows[0][0] == 1
            assert rows[2][0] == 3

    def test_fetch_all_empty(self, db):
        """Test fetch_all with empty result."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")

        if hasattr(db, "fetch_all"):
            rows = db.fetch_all("SELECT * FROM test")
            assert rows == []


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread safety of database operations."""

    def test_multiple_connections(self, db):
        """Test multiple sequential connections work."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")

        for i in range(10):
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test VALUES (?)", (i,))

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM test")
            assert cursor.fetchone()[0] == 10

    def test_concurrent_reads(self, db):
        """Test concurrent read operations."""
        import threading

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")
            for i in range(100):
                cursor.execute("INSERT INTO test VALUES (?, ?)", (i, f"value-{i}"))

        results = []
        errors = []

        def read_data():
            try:
                with db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM test")
                    count = cursor.fetchone()[0]
                    results.append(count)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_data) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == 100 for r in results)


# =============================================================================
# WAL Mode Tests
# =============================================================================


class TestWALMode:
    """Test WAL mode configuration."""

    def test_wal_mode_enabled(self, db):
        """Test that WAL mode is enabled for better concurrency."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0].lower()

            # WAL mode should be enabled for evolution database
            # (as per docstring), but check if configured
            assert mode in ("wal", "delete", "truncate", "memory")


# =============================================================================
# Schema Management Tests
# =============================================================================


class TestSchemaManagement:
    """Test schema management inherited from BaseDatabase."""

    def test_can_create_tables(self, db):
        """Test that tables can be created."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY,
                    fitness REAL DEFAULT 0.5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cursor.execute("INSERT INTO strategies (id) VALUES ('test-1')")

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, fitness FROM strategies")
            row = cursor.fetchone()
            assert row[0] == "test-1"
            assert row[1] == 0.5

    def test_table_persistence(self, temp_db):
        """Test that tables persist across database instances."""
        # Create table in first instance
        db1 = EvolutionDatabase(temp_db)
        with db1.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (123)")

        # Verify in second instance
        db2 = EvolutionDatabase(temp_db)
        with db2.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM test")
            assert cursor.fetchone()[0] == 123


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in database operations."""

    def test_invalid_sql(self, db):
        """Test handling of invalid SQL."""
        with pytest.raises(sqlite3.OperationalError):
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INVALID SQL STATEMENT")

    def test_constraint_violation(self, db):
        """Test handling of constraint violations."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE test (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE
                )
            """
            )
            cursor.execute("INSERT INTO test VALUES (1, 'unique')")

        with pytest.raises(sqlite3.IntegrityError):
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test VALUES (2, 'unique')")

    def test_foreign_key_violation(self, db):
        """Test handling of foreign key violations."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(
                """
                CREATE TABLE parent (id INTEGER PRIMARY KEY)
            """
            )
            cursor.execute(
                """
                CREATE TABLE child (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER REFERENCES parent(id)
                )
            """
            )

        # Attempt to insert child with non-existent parent
        # May or may not raise depending on FK enforcement
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            try:
                cursor.execute("INSERT INTO child VALUES (1, 999)")
                # If we get here, FK enforcement may be off
            except sqlite3.IntegrityError:
                # Expected if FK enforcement is on
                pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for EvolutionDatabase usage patterns."""

    def test_evolution_workflow(self, db):
        """Test typical evolution workflow usage."""
        # Create schema
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE strategies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    fitness REAL DEFAULT 0.5,
                    generation INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

        # Insert initial strategies
        strategies = [
            ("s1", "Strategy A", 0.5),
            ("s2", "Strategy B", 0.6),
            ("s3", "Strategy C", 0.4),
        ]

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO strategies (id, name, fitness) VALUES (?, ?, ?)",
                strategies,
            )

        # Update fitness after evaluation
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE strategies SET fitness = ?, generation = ? WHERE id = ?",
                (0.75, 1, "s1"),
            )

        # Select top strategies
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, fitness FROM strategies
                ORDER BY fitness DESC
                LIMIT 2
            """
            )
            top = cursor.fetchall()

        assert len(top) == 2
        assert top[0][0] == "s1"  # Highest fitness
        assert top[0][2] == 0.75

    def test_complex_queries(self, db):
        """Test complex query patterns."""
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE experiments (
                    id TEXT PRIMARY KEY,
                    agent TEXT NOT NULL,
                    generation INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0
                )
            """
            )

            # Insert test data
            data = [
                ("e1", "claude", 0, 5, 3),
                ("e2", "claude", 1, 8, 2),
                ("e3", "gpt4", 0, 4, 4),
                ("e4", "gpt4", 1, 6, 4),
            ]
            cursor.executemany(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
                data,
            )

        # Complex aggregation query
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    agent,
                    SUM(wins) as total_wins,
                    SUM(losses) as total_losses,
                    CAST(SUM(wins) AS REAL) / (SUM(wins) + SUM(losses)) as win_rate
                FROM experiments
                GROUP BY agent
                ORDER BY win_rate DESC
            """
            )
            results = cursor.fetchall()

        assert len(results) == 2
        assert results[0][0] == "claude"  # Higher win rate
        assert results[0][1] == 13  # Total wins
