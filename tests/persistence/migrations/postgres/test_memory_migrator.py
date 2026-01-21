"""
Tests for SQLite to PostgreSQL Memory Migrator.

Tests the MemoryMigrator class that migrates ConsensusMemory and CritiqueStore
data from SQLite to PostgreSQL.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


# Check if asyncpg is available
try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class TestMigrationStats:
    """Tests for MigrationStats dataclass."""

    def test_stats_success_when_no_errors(self):
        """MigrationStats.success should be True when no errors."""
        from aragora.persistence.migrations.postgres.memory_migrator import MigrationStats

        stats = MigrationStats(table="test", rows_migrated=100, rows_skipped=5)
        assert stats.success is True

    def test_stats_failure_when_errors(self):
        """MigrationStats.success should be False when errors present."""
        from aragora.persistence.migrations.postgres.memory_migrator import MigrationStats

        stats = MigrationStats(table="test", rows_migrated=50, rows_skipped=10, errors=["Error 1"])
        assert stats.success is False


class TestMigrationReport:
    """Tests for MigrationReport dataclass."""

    def test_report_totals(self):
        """MigrationReport should calculate totals correctly."""
        from aragora.persistence.migrations.postgres.memory_migrator import (
            MigrationReport,
            MigrationStats,
        )

        report = MigrationReport(
            source_path="/test/path",
            target_dsn="postgresql://...",
            started_at=datetime.now(timezone.utc),
            tables=[
                MigrationStats(table="t1", rows_migrated=100, rows_skipped=10),
                MigrationStats(table="t2", rows_migrated=200, rows_skipped=20),
            ],
        )

        assert report.total_migrated == 300
        assert report.total_skipped == 30
        assert report.total_errors == 0
        assert report.success is True

    def test_report_success_false_when_errors(self):
        """MigrationReport.success should be False if any table has errors."""
        from aragora.persistence.migrations.postgres.memory_migrator import (
            MigrationReport,
            MigrationStats,
        )

        report = MigrationReport(
            source_path="/test/path",
            target_dsn="postgresql://...",
            started_at=datetime.now(timezone.utc),
            tables=[
                MigrationStats(table="t1", rows_migrated=100),
                MigrationStats(table="t2", rows_migrated=0, errors=["Failed"]),
            ],
        )

        assert report.success is False
        assert report.total_errors == 1


class TestValueConversion:
    """Tests for value conversion logic."""

    @pytest.fixture
    def migrator(self):
        """Create a migrator instance for testing."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        # Use a dummy path since we won't actually connect
        return MemoryMigrator(sqlite_path="/tmp/test.db", postgres_dsn="postgresql://test")

    def test_convert_boolean_true(self, migrator):
        """Boolean 1 should convert to True."""
        config = {"bool_columns": ["is_active"]}
        result = migrator._convert_value(1, "is_active", config)
        assert result is True

    def test_convert_boolean_false(self, migrator):
        """Boolean 0 should convert to False."""
        config = {"bool_columns": ["is_active"]}
        result = migrator._convert_value(0, "is_active", config)
        assert result is False

    def test_convert_json_string(self, migrator):
        """JSON string should be preserved."""
        config = {"json_columns": ["data"]}
        result = migrator._convert_value('{"key": "value"}', "data", config)
        assert result == '{"key": "value"}'

    def test_convert_json_invalid(self, migrator):
        """Invalid JSON should be returned as-is."""
        config = {"json_columns": ["data"]}
        result = migrator._convert_value("not json", "data", config)
        assert result == "not json"

    def test_convert_timestamp_iso(self, migrator):
        """ISO timestamp should be converted to datetime."""
        config = {"timestamp_columns": ["created_at"]}
        result = migrator._convert_value("2024-01-15T10:30:00", "created_at", config)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_convert_timestamp_with_z(self, migrator):
        """ISO timestamp with Z suffix should be converted."""
        config = {"timestamp_columns": ["created_at"]}
        result = migrator._convert_value("2024-01-15T10:30:00Z", "created_at", config)
        assert isinstance(result, datetime)

    def test_convert_none(self, migrator):
        """None values should remain None."""
        config = {"bool_columns": ["flag"]}
        result = migrator._convert_value(None, "flag", config)
        assert result is None

    def test_convert_regular_value(self, migrator):
        """Regular values should pass through unchanged."""
        config = {}
        result = migrator._convert_value("hello", "name", config)
        assert result == "hello"


class TestTableConfigurations:
    """Tests for table configuration definitions."""

    def test_consensus_tables_defined(self):
        """CONSENSUS_TABLES should define expected tables."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        expected_tables = ["consensus", "dissent", "verified_proofs"]
        for table in expected_tables:
            assert table in MemoryMigrator.CONSENSUS_TABLES

    def test_critique_tables_defined(self):
        """CRITIQUE_TABLES should define expected tables."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        expected_tables = [
            "debates",
            "critiques",
            "patterns",
            "agent_reputation",
            "patterns_archive",
        ]
        for table in expected_tables:
            assert table in MemoryMigrator.CRITIQUE_TABLES

    def test_consensus_columns_match(self):
        """Consensus table configs should have matching sqlite and pg columns."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        for table, config in MemoryMigrator.CONSENSUS_TABLES.items():
            assert config["sqlite_columns"] == config["pg_columns"], f"Column mismatch in {table}"

    def test_critique_columns_match(self):
        """Critique table configs should have matching sqlite and pg columns."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        for table, config in MemoryMigrator.CRITIQUE_TABLES.items():
            assert config["sqlite_columns"] == config["pg_columns"], f"Column mismatch in {table}"


class TestSQLiteOperations:
    """Tests for SQLite operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database with test tables."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                task TEXT,
                consensus_reached INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("""
            INSERT INTO debates (id, task, consensus_reached, created_at)
            VALUES ('d1', 'Test task', 1, '2024-01-15T10:00:00')
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        db_path.unlink(missing_ok=True)

    def test_get_sqlite_conn(self, temp_db):
        """Should successfully connect to SQLite database."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        migrator = MemoryMigrator(sqlite_path=temp_db, postgres_dsn="postgresql://test")
        conn = migrator._get_sqlite_conn()

        try:
            cursor = conn.execute("SELECT id FROM debates")
            row = cursor.fetchone()
            assert row["id"] == "d1"
        finally:
            conn.close()

    def test_table_exists_sqlite_true(self, temp_db):
        """_table_exists_sqlite should return True for existing table."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        migrator = MemoryMigrator(sqlite_path=temp_db, postgres_dsn="postgresql://test")
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row

        try:
            assert migrator._table_exists_sqlite(conn, "debates") is True
        finally:
            conn.close()

    def test_table_exists_sqlite_false(self, temp_db):
        """_table_exists_sqlite should return False for missing table."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        migrator = MemoryMigrator(sqlite_path=temp_db, postgres_dsn="postgresql://test")
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row

        try:
            assert migrator._table_exists_sqlite(conn, "nonexistent") is False
        finally:
            conn.close()

    def test_get_sqlite_columns(self, temp_db):
        """_get_sqlite_columns should return column names."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        migrator = MemoryMigrator(sqlite_path=temp_db, postgres_dsn="postgresql://test")
        conn = sqlite3.connect(str(temp_db))
        conn.row_factory = sqlite3.Row

        try:
            columns = migrator._get_sqlite_columns(conn, "debates")
            assert "id" in columns
            assert "task" in columns
            assert "consensus_reached" in columns
        finally:
            conn.close()


@pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg required")
class TestPostgresOperations:
    """Tests for PostgreSQL operations with mocking."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        return MagicMock()

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.executemany = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)
        return conn

    @pytest.mark.asyncio
    async def test_table_exists_pg_true(self, mock_pool, mock_connection):
        """_table_exists_pg should return True when table exists."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        mock_connection.fetchrow.return_value = [True]
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_connection), __aexit__=AsyncMock()
            )
        )

        migrator = MemoryMigrator(sqlite_path="/tmp/test.db", postgres_dsn="postgresql://test")
        migrator._pool = mock_pool

        # Create proper async context manager
        async def async_ctx():
            return mock_connection

        mock_pool.acquire.return_value.__aenter__ = async_ctx
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await migrator._table_exists_pg(mock_pool, "debates")
        assert result is True


class TestMigratorInit:
    """Tests for MemoryMigrator initialization."""

    def test_init_with_defaults(self):
        """Migrator should initialize with default values."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        migrator = MemoryMigrator(
            sqlite_path="/test/path.db",
            postgres_dsn="postgresql://user:pass@host/db",
        )

        assert migrator.sqlite_path == Path("/test/path.db")
        assert migrator.postgres_dsn == "postgresql://user:pass@host/db"
        assert migrator.batch_size == 1000
        assert migrator.skip_existing is True

    def test_init_with_custom_values(self):
        """Migrator should accept custom values."""
        from aragora.persistence.migrations.postgres.memory_migrator import MemoryMigrator

        migrator = MemoryMigrator(
            sqlite_path="/test/path.db",
            postgres_dsn="postgresql://test",
            batch_size=500,
            skip_existing=False,
        )

        assert migrator.batch_size == 500
        assert migrator.skip_existing is False


class TestReportPrinting:
    """Tests for report printing."""

    def test_print_report_success(self, capsys):
        """print_report should format successful migration."""
        from aragora.persistence.migrations.postgres.memory_migrator import (
            MigrationReport,
            MigrationStats,
            print_report,
        )

        report = MigrationReport(
            source_path="/test/source.db",
            target_dsn="host/db",
            started_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 15, 10, 0, 30, tzinfo=timezone.utc),
            tables=[
                MigrationStats(table="debates", rows_migrated=100),
                MigrationStats(table="patterns", rows_migrated=50, rows_skipped=5),
            ],
        )

        print_report(report)
        captured = capsys.readouterr()

        assert "MEMORY STORE MIGRATION REPORT" in captured.out
        assert "debates" in captured.out
        assert "100 migrated" in captured.out
        assert "SUCCESS" in captured.out

    def test_print_report_with_errors(self, capsys):
        """print_report should show errors."""
        from aragora.persistence.migrations.postgres.memory_migrator import (
            MigrationReport,
            MigrationStats,
            print_report,
        )

        report = MigrationReport(
            source_path="/test/source.db",
            target_dsn="host/db",
            started_at=datetime.now(timezone.utc),
            tables=[
                MigrationStats(table="debates", rows_migrated=0, errors=["Connection failed"]),
            ],
        )
        report.completed_at = datetime.now(timezone.utc)

        print_report(report)
        captured = capsys.readouterr()

        assert "Connection failed" in captured.out
        assert "FAILED" in captured.out
