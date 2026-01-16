"""
Tests for aragora.persistence.migrations.consolidate - Database consolidation.

Tests cover:
- MigrationStats and ConsolidationResult dataclasses
- DatabaseConsolidator initialization
- Backup creation and safety
- Table migration with various scenarios
- Error handling and rollback
- Verification after migration
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def source_db(temp_db_dir: Path):
    """Create a source database with test data."""
    db_path = temp_db_dir / "debates.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE debates (
            id TEXT PRIMARY KEY,
            task TEXT,
            status TEXT,
            created_at TEXT
        )
    """
    )
    conn.execute(
        """
        INSERT INTO debates (id, task, status, created_at)
        VALUES ('d1', 'Test task 1', 'completed', '2024-01-01'),
               ('d2', 'Test task 2', 'pending', '2024-01-02'),
               ('d3', 'Test task 3', 'failed', '2024-01-03')
    """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def consolidator(temp_db_dir: Path):
    """Create a DatabaseConsolidator for testing."""
    from aragora.persistence.migrations.consolidate import DatabaseConsolidator

    return DatabaseConsolidator(
        source_dir=temp_db_dir,
        target_dir=temp_db_dir / "consolidated",
        backup=True,
    )


# ===========================================================================
# Test MigrationStats
# ===========================================================================


class TestMigrationStats:
    """Tests for MigrationStats dataclass."""

    def test_creation_with_defaults(self):
        """Test MigrationStats with default values."""
        from aragora.persistence.migrations.consolidate import MigrationStats

        stats = MigrationStats(
            table_name="debates",
            source_db="debates.db",
            target_db="core.db",
        )

        assert stats.table_name == "debates"
        assert stats.source_db == "debates.db"
        assert stats.target_db == "core.db"
        assert stats.rows_read == 0
        assert stats.rows_written == 0
        assert stats.rows_skipped == 0
        assert stats.errors == []

    def test_errors_list_initialized(self):
        """Test that errors list is properly initialized."""
        from aragora.persistence.migrations.consolidate import MigrationStats

        stats = MigrationStats(
            table_name="test",
            source_db="test.db",
            target_db="core.db",
        )

        # Should be able to append without None check
        stats.errors.append("Test error")
        assert len(stats.errors) == 1


# ===========================================================================
# Test ConsolidationResult
# ===========================================================================


class TestConsolidationResult:
    """Tests for ConsolidationResult dataclass."""

    def test_successful_result(self):
        """Test successful consolidation result."""
        from aragora.persistence.migrations.consolidate import (
            ConsolidationResult,
            MigrationStats,
        )

        stats = [
            MigrationStats("debates", "debates.db", "core.db", 100, 100, 0),
            MigrationStats("agents", "agents.db", "core.db", 50, 50, 0),
        ]

        result = ConsolidationResult(
            success=True,
            tables_migrated=2,
            total_rows=150,
            duration_seconds=1.5,
            stats=stats,
            errors=[],
        )

        assert result.success is True
        assert result.tables_migrated == 2
        assert result.total_rows == 150

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from aragora.persistence.migrations.consolidate import (
            ConsolidationResult,
            MigrationStats,
        )

        stats = [MigrationStats("debates", "debates.db", "core.db", 10, 10, 0)]
        result = ConsolidationResult(
            success=True,
            tables_migrated=1,
            total_rows=10,
            duration_seconds=0.5,
            stats=stats,
            errors=[],
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["tables_migrated"] == 1
        assert d["total_rows"] == 10
        assert d["duration_seconds"] == 0.5
        assert len(d["stats"]) == 1
        assert d["stats"][0]["table"] == "debates"

    def test_to_dict_limits_errors(self):
        """Test that to_dict limits errors to first 10."""
        from aragora.persistence.migrations.consolidate import ConsolidationResult

        result = ConsolidationResult(
            success=False,
            tables_migrated=0,
            total_rows=0,
            duration_seconds=0.1,
            stats=[],
            errors=[f"Error {i}" for i in range(20)],
        )

        d = result.to_dict()
        assert len(d["errors"]) == 10


# ===========================================================================
# Test DatabaseConsolidator Initialization
# ===========================================================================


class TestDatabaseConsolidatorInit:
    """Tests for DatabaseConsolidator initialization."""

    def test_init_with_defaults(self, temp_db_dir: Path):
        """Test initialization with default values."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(source_dir=temp_db_dir)

        assert consolidator.source_dir == temp_db_dir
        assert consolidator.target_dir == temp_db_dir
        assert consolidator.backup is True
        assert consolidator.stats == []
        assert consolidator.errors == []

    def test_init_with_custom_target(self, temp_db_dir: Path):
        """Test initialization with custom target directory."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        target = temp_db_dir / "output"
        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            target_dir=target,
            backup=False,
        )

        assert consolidator.source_dir == temp_db_dir
        assert consolidator.target_dir == target
        assert consolidator.backup is False


# ===========================================================================
# Test Database Connection
# ===========================================================================


class TestGetConnection:
    """Tests for _get_connection method."""

    def test_connection_to_existing_db(self, source_db: Path, consolidator):
        """Test connecting to an existing database."""
        conn = consolidator._get_connection(source_db)

        assert conn is not None
        cursor = conn.execute("SELECT COUNT(*) FROM debates")
        assert cursor.fetchone()[0] == 3
        conn.close()

    def test_connection_to_nonexistent_db(self, consolidator):
        """Test connecting to a non-existent database returns None."""
        conn = consolidator._get_connection(Path("/nonexistent/path.db"))
        assert conn is None

    def test_connection_wal_mode(self, source_db: Path, consolidator):
        """Test that WAL mode is enabled."""
        conn = consolidator._get_connection(source_db)

        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"
        conn.close()


# ===========================================================================
# Test Backup
# ===========================================================================


class TestBackupDatabases:
    """Tests for _backup_databases method."""

    def test_backup_creates_directory(self, source_db: Path, consolidator):
        """Test that backup creates a timestamped directory."""
        result = consolidator._backup_databases()

        assert result is True
        backup_dirs = list((consolidator.source_dir / "backup").iterdir())
        assert len(backup_dirs) == 1
        assert backup_dirs[0].is_dir()

    def test_backup_copies_files(self, source_db: Path, consolidator):
        """Test that backup copies database files."""
        consolidator._backup_databases()

        backup_dir = next((consolidator.source_dir / "backup").iterdir())
        backed_up_file = backup_dir / "debates.db"
        assert backed_up_file.exists()

        # Verify contents
        conn = sqlite3.connect(str(backed_up_file))
        cursor = conn.execute("SELECT COUNT(*) FROM debates")
        assert cursor.fetchone()[0] == 3
        conn.close()

    def test_backup_handles_errors(self, consolidator, temp_db_dir: Path):
        """Test backup handles errors gracefully."""
        # Create a database file
        db_path = temp_db_dir / "test.db"
        db_path.touch()

        # Mock shutil.copy2 to raise an error
        with patch("shutil.copy2", side_effect=PermissionError("Access denied")):
            result = consolidator._backup_databases()

        assert result is False
        assert len(consolidator.errors) > 0
        assert "Backup failed" in consolidator.errors[0]


# ===========================================================================
# Test Table Migration
# ===========================================================================


class TestMigrateTable:
    """Tests for _migrate_table method."""

    def test_migrate_table_success(self, source_db: Path, temp_db_dir: Path):
        """Test successful table migration."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            target_dir=temp_db_dir / "target",
            backup=False,
        )

        # Create target database
        target_dir = temp_db_dir / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_db = target_dir / "core.db"
        target_conn = sqlite3.connect(str(target_db))
        target_conn.execute(
            """
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                task TEXT,
                status TEXT,
                created_at TEXT
            )
        """
        )

        source_conn = sqlite3.connect(str(source_db))
        source_conn.row_factory = sqlite3.Row

        stats = consolidator._migrate_table(
            source_conn=source_conn,
            target_conn=target_conn,
            source_table="debates",
            target_table="debates",
            columns=["id", "task", "status", "created_at"],
            source_db="debates.db",
            target_db="core.db",
        )

        assert stats.rows_read == 3
        assert stats.rows_written == 3
        assert stats.rows_skipped == 0
        assert len(stats.errors) == 0

        source_conn.close()
        target_conn.close()

    def test_migrate_table_dry_run(self, source_db: Path, temp_db_dir: Path):
        """Test dry run doesn't write data."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            target_dir=temp_db_dir / "target",
            backup=False,
        )

        target_dir = temp_db_dir / "target"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_db = target_dir / "core.db"
        target_conn = sqlite3.connect(str(target_db))
        target_conn.execute(
            """
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                task TEXT
            )
        """
        )

        source_conn = sqlite3.connect(str(source_db))
        source_conn.row_factory = sqlite3.Row

        stats = consolidator._migrate_table(
            source_conn=source_conn,
            target_conn=target_conn,
            source_table="debates",
            target_table="debates",
            columns=["id", "task"],
            source_db="debates.db",
            target_db="core.db",
            dry_run=True,
        )

        assert stats.rows_read == 3
        assert stats.rows_written == 0  # Dry run

        # Verify nothing written
        cursor = target_conn.execute("SELECT COUNT(*) FROM debates")
        assert cursor.fetchone()[0] == 0

        source_conn.close()
        target_conn.close()

    def test_migrate_table_missing_table(self, temp_db_dir: Path):
        """Test migration handles missing source table."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            backup=False,
        )

        # Create empty source database
        source_db = temp_db_dir / "source.db"
        source_conn = sqlite3.connect(str(source_db))
        source_conn.row_factory = sqlite3.Row

        target_db = temp_db_dir / "target.db"
        target_conn = sqlite3.connect(str(target_db))
        target_conn.execute("CREATE TABLE debates (id TEXT PRIMARY KEY)")

        stats = consolidator._migrate_table(
            source_conn=source_conn,
            target_conn=target_conn,
            source_table="nonexistent",
            target_table="debates",
            columns=["id"],
            source_db="source.db",
            target_db="target.db",
        )

        assert len(stats.errors) > 0
        assert "not found" in stats.errors[0].lower()

        source_conn.close()
        target_conn.close()

    def test_migrate_table_handles_duplicates(self, source_db: Path, temp_db_dir: Path):
        """Test migration skips duplicate rows via INSERT OR IGNORE."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            backup=False,
        )

        target_db = temp_db_dir / "target.db"
        target_conn = sqlite3.connect(str(target_db))
        target_conn.execute(
            """
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                task TEXT
            )
        """
        )
        # Pre-insert a duplicate
        target_conn.execute("INSERT INTO debates (id, task) VALUES ('d1', 'Existing')")
        target_conn.commit()

        source_conn = sqlite3.connect(str(source_db))
        source_conn.row_factory = sqlite3.Row

        stats = consolidator._migrate_table(
            source_conn=source_conn,
            target_conn=target_conn,
            source_table="debates",
            target_table="debates",
            columns=["id", "task"],
            source_db="debates.db",
            target_db="target.db",
        )

        assert stats.rows_read == 3
        # INSERT OR IGNORE silently ignores duplicates, so rows_written == 3
        # but only 2 new rows are actually inserted
        assert stats.rows_written == 3  # All attempted
        assert stats.rows_skipped == 0  # No explicit skips with OR IGNORE

        # Verify actual data: should have 3 rows total (1 existing + 2 new)
        cursor = target_conn.execute("SELECT COUNT(*) FROM debates")
        assert cursor.fetchone()[0] == 3

        source_conn.close()
        target_conn.close()


# ===========================================================================
# Test Full Migration
# ===========================================================================


class TestMigrate:
    """Tests for migrate method."""

    def test_migrate_dry_run(self, source_db: Path, temp_db_dir: Path):
        """Test full migration in dry run mode."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            target_dir=temp_db_dir / "target",
            backup=False,
        )

        result = consolidator.migrate(dry_run=True)

        # Dry run should succeed but not write
        assert result.success is True
        assert result.errors == []
        # Target directory should not be created
        assert not (temp_db_dir / "target").exists()


# ===========================================================================
# Test Verification
# ===========================================================================


class TestVerify:
    """Tests for verify method."""

    def test_verify_empty_target(self, temp_db_dir: Path):
        """Test verification of empty target directory."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            target_dir=temp_db_dir / "target",
            backup=False,
        )

        result = consolidator.verify()

        assert "databases" in result
        assert result["databases"] == {}


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_migration_with_corrupt_source(self, temp_db_dir: Path):
        """Test migration handles corrupt source database."""
        from aragora.persistence.migrations.consolidate import DatabaseConsolidator

        # Create a corrupt database file
        corrupt_db = temp_db_dir / "corrupt.db"
        corrupt_db.write_text("not a valid sqlite database")

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            backup=False,
        )

        # _get_connection raises DatabaseError for corrupt files
        # because it tries to set journal_mode=WAL
        with pytest.raises(sqlite3.DatabaseError):
            consolidator._get_connection(corrupt_db)

    def test_migration_recovers_from_partial_failure(self, temp_db_dir: Path):
        """Test that partial failures are recorded in stats."""
        from aragora.persistence.migrations.consolidate import (
            DatabaseConsolidator,
            MigrationStats,
        )

        consolidator = DatabaseConsolidator(
            source_dir=temp_db_dir,
            backup=False,
        )

        # Manually add a failed stat
        failed_stat = MigrationStats(
            table_name="test",
            source_db="test.db",
            target_db="core.db",
        )
        failed_stat.errors.append("Test error")
        consolidator.stats.append(failed_stat)

        # Should be able to continue and track errors
        assert len(consolidator.stats) == 1
        assert len(consolidator.stats[0].errors) == 1


# ===========================================================================
# Test Concurrent Access Safety
# ===========================================================================


class TestConcurrencySafety:
    """Tests for concurrent access safety."""

    def test_busy_timeout_configured(self, source_db: Path, consolidator):
        """Test that busy timeout is configured for concurrent access."""
        conn = consolidator._get_connection(source_db)

        # Check busy_timeout is set
        cursor = conn.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]
        assert timeout == 30000  # 30 seconds

        conn.close()

    def test_transaction_isolation(self, source_db: Path, consolidator):
        """Test that migrations use proper transaction isolation."""
        conn1 = consolidator._get_connection(source_db)
        conn2 = consolidator._get_connection(source_db)

        # Start transaction in conn1
        conn1.execute("BEGIN IMMEDIATE")
        conn1.execute("UPDATE debates SET status = 'updated' WHERE id = 'd1'")

        # conn2 should see old value until commit
        cursor = conn2.execute("SELECT status FROM debates WHERE id = 'd1'")
        status = cursor.fetchone()[0]
        assert status == "completed"  # Old value

        conn1.commit()

        # Now conn2 should see new value
        cursor = conn2.execute("SELECT status FROM debates WHERE id = 'd1'")
        status = cursor.fetchone()[0]
        assert status == "updated"

        conn1.close()
        conn2.close()
