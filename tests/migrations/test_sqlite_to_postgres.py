"""
Tests for aragora.migrations.sqlite_to_postgres - SQLite-to-PostgreSQL Migration Orchestrator.

Tests cover:
- Type mapping (sqlite_type_to_pg)
- SchemaTranslator DDL generation
- MigrationOrchestrator discovery
- Data coercion
- Report data classes
- Dry-run mode
- End-to-end migration with mock asyncpg (no real PostgreSQL needed)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.migrations.sqlite_to_postgres import (
    DatabaseMigrationResult,
    MigrationOrchestrator,
    MigrationReport,
    SchemaTranslator,
    TableMigrationResult,
    sqlite_type_to_pg,
    SQLITE_TO_PG_TYPES,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with sample SQLite databases."""
    # Core database
    core_db = tmp_path / "core.db"
    conn = sqlite3.connect(str(core_db))
    conn.execute(
        """
        CREATE TABLE debates (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            final_answer TEXT,
            consensus_reached BOOLEAN DEFAULT 0,
            confidence REAL DEFAULT 0.0,
            rounds_used INTEGER DEFAULT 0,
            metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX idx_debates_task ON debates(task)")
    conn.execute(
        "INSERT INTO debates (id, task, final_answer, consensus_reached, confidence, rounds_used) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("d1", "Design a rate limiter", "Token bucket", 1, 0.85, 3),
    )
    conn.execute(
        "INSERT INTO debates (id, task, final_answer, consensus_reached, confidence, rounds_used) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("d2", "Choose a database", "PostgreSQL", 0, 0.60, 5),
    )
    conn.commit()
    conn.close()

    # Analytics database
    analytics_db = tmp_path / "analytics.db"
    conn = sqlite3.connect(str(analytics_db))
    conn.execute(
        """
        CREATE TABLE agent_elo (
            agent_name TEXT PRIMARY KEY,
            rating REAL DEFAULT 1500.0,
            games_played INTEGER DEFAULT 0,
            last_updated TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO agent_elo (agent_name, rating, games_played) VALUES (?, ?, ?)",
        ("claude", 1650.0, 42),
    )
    conn.execute(
        "INSERT INTO agent_elo (agent_name, rating, games_played) VALUES (?, ?, ?)",
        ("gpt4", 1580.0, 38),
    )
    conn.commit()
    conn.close()

    # Empty database (should be handled gracefully)
    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(empty_db))
    conn.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    return tmp_path


@pytest.fixture
def orchestrator(tmp_data_dir: Path) -> MigrationOrchestrator:
    """Create a MigrationOrchestrator for the test data directory."""
    return MigrationOrchestrator(
        data_dir=tmp_data_dir,
        postgres_dsn="postgresql://test:test@localhost/test",
        batch_size=10,
    )


# ===========================================================================
# Test Type Mapping
# ===========================================================================


class TestSqliteTypeToPg:
    """Tests for sqlite_type_to_pg conversion function."""

    def test_standard_types(self):
        """Standard SQLite types map to correct PostgreSQL types."""
        assert sqlite_type_to_pg("TEXT") == "TEXT"
        assert sqlite_type_to_pg("INTEGER") == "BIGINT"
        assert sqlite_type_to_pg("REAL") == "DOUBLE PRECISION"
        assert sqlite_type_to_pg("BLOB") == "BYTEA"
        assert sqlite_type_to_pg("BOOLEAN") == "BOOLEAN"

    def test_case_insensitive(self):
        """Conversion is case-insensitive."""
        assert sqlite_type_to_pg("text") == "TEXT"
        assert sqlite_type_to_pg("integer") == "BIGINT"
        assert sqlite_type_to_pg("Json") == "JSONB"

    def test_parenthesized_types(self):
        """Types with length specifications are handled."""
        assert sqlite_type_to_pg("VARCHAR(255)") == "TEXT"
        assert sqlite_type_to_pg("CHAR(1)") == "TEXT"
        assert sqlite_type_to_pg("NUMERIC(10,2)") == "NUMERIC"

    def test_datetime_types(self):
        """Date/time types map to TIMESTAMPTZ."""
        assert sqlite_type_to_pg("DATETIME") == "TIMESTAMPTZ"
        assert sqlite_type_to_pg("TIMESTAMP") == "TIMESTAMPTZ"

    def test_json_types(self):
        """JSON types map to JSONB."""
        assert sqlite_type_to_pg("JSON") == "JSONB"
        assert sqlite_type_to_pg("JSONB") == "JSONB"

    def test_affinity_rules(self):
        """SQLite type affinity rules work correctly."""
        # INT affinity
        assert sqlite_type_to_pg("BIGINT") == "BIGINT"
        assert sqlite_type_to_pg("MEDIUMINT") == "BIGINT"
        assert sqlite_type_to_pg("INT2") == "BIGINT"

        # TEXT affinity
        assert sqlite_type_to_pg("NVARCHAR(100)") == "TEXT"
        assert sqlite_type_to_pg("CLOB") == "TEXT"

        # REAL affinity
        assert sqlite_type_to_pg("FLOAT") == "DOUBLE PRECISION"
        assert sqlite_type_to_pg("DOUBLE PRECISION") == "DOUBLE PRECISION"

    def test_empty_type(self):
        """Empty type defaults to TEXT."""
        assert sqlite_type_to_pg("") == "TEXT"

    def test_unknown_type(self):
        """Truly unknown types default to TEXT."""
        assert sqlite_type_to_pg("CUSTOM_TYPE") == "TEXT"

    def test_all_map_entries(self):
        """All entries in SQLITE_TO_PG_TYPES are valid and non-empty."""
        for sqlite_type, pg_type in SQLITE_TO_PG_TYPES.items():
            assert pg_type, f"Empty mapping for {sqlite_type}"
            result = sqlite_type_to_pg(sqlite_type)
            assert result == pg_type, f"{sqlite_type} -> {result} != {pg_type}"


# ===========================================================================
# Test SchemaTranslator
# ===========================================================================


class TestSchemaTranslator:
    """Tests for the SchemaTranslator class."""

    def test_translate_simple_table(self, tmp_data_dir: Path):
        """Simple table translates correctly."""
        translator = SchemaTranslator()
        conn = sqlite3.connect(str(tmp_data_dir / "analytics.db"))

        ddl = translator.translate_table(conn, "agent_elo")
        conn.close()

        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert '"agent_elo"' in ddl
        assert '"agent_name" TEXT' in ddl
        assert '"rating" DOUBLE PRECISION' in ddl
        assert '"games_played" BIGINT' in ddl
        assert "PRIMARY KEY" in ddl

    def test_translate_table_with_defaults(self, tmp_data_dir: Path):
        """Table with default values translates correctly."""
        translator = SchemaTranslator()
        conn = sqlite3.connect(str(tmp_data_dir / "core.db"))

        ddl = translator.translate_table(conn, "debates")
        conn.close()

        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert '"debates"' in ddl
        assert '"id" TEXT' in ddl
        assert '"task" TEXT NOT NULL' in ddl
        assert "BOOLEAN" in ddl
        assert "TIMESTAMPTZ" in ddl or "NOW()" in ddl

    def test_translate_table_with_index(self, tmp_data_dir: Path):
        """Indexes are included in the DDL."""
        translator = SchemaTranslator()
        conn = sqlite3.connect(str(tmp_data_dir / "core.db"))

        ddl = translator.translate_table(conn, "debates")
        conn.close()

        assert "CREATE INDEX" in ddl or "CREATE UNIQUE INDEX" in ddl
        assert "idx_debates_task" in ddl

    def test_translate_nonexistent_table(self, tmp_data_dir: Path):
        """Nonexistent table returns empty string."""
        translator = SchemaTranslator()
        conn = sqlite3.connect(str(tmp_data_dir / "core.db"))

        ddl = translator.translate_table(conn, "nonexistent")
        conn.close()

        assert ddl == ""

    def test_custom_schema(self, tmp_data_dir: Path):
        """Custom PostgreSQL schema is used in DDL."""
        translator = SchemaTranslator(pg_schema="aragora")
        conn = sqlite3.connect(str(tmp_data_dir / "core.db"))

        ddl = translator.translate_table(conn, "debates")
        conn.close()

        assert '"aragora"."debates"' in ddl

    def test_translate_empty_table(self, tmp_data_dir: Path):
        """Empty table (columns but no data) translates correctly."""
        translator = SchemaTranslator()
        conn = sqlite3.connect(str(tmp_data_dir / "empty.db"))

        ddl = translator.translate_table(conn, "empty_table")
        conn.close()

        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert '"id"' in ddl


# ===========================================================================
# Test MigrationOrchestrator Discovery
# ===========================================================================


class TestOrchestratorDiscovery:
    """Tests for database and table discovery."""

    def test_discover_databases(self, orchestrator: MigrationOrchestrator, tmp_data_dir: Path):
        """Discovers all .db files in the data directory."""
        db_files = orchestrator.discover_databases()

        names = {f.name for f in db_files}
        assert "core.db" in names
        assert "analytics.db" in names
        assert "empty.db" in names

    def test_discover_databases_with_filter(self, tmp_data_dir: Path):
        """include_databases filters to specific databases."""
        orch = MigrationOrchestrator(
            data_dir=tmp_data_dir,
            postgres_dsn="postgresql://test@localhost/test",
            include_databases={"core"},
        )

        db_files = orch.discover_databases()
        assert len(db_files) == 1
        assert db_files[0].name == "core.db"

    def test_discover_databases_nonexistent_dir(self):
        """Nonexistent directory returns empty list."""
        orch = MigrationOrchestrator(
            data_dir="/nonexistent/path",
            postgres_dsn="postgresql://test@localhost/test",
        )

        db_files = orch.discover_databases()
        assert db_files == []

    def test_discover_tables(self, orchestrator: MigrationOrchestrator, tmp_data_dir: Path):
        """Discovers user tables (excluding sqlite_ internal tables)."""
        tables = orchestrator.discover_tables(tmp_data_dir / "core.db")

        assert "debates" in tables
        assert "sqlite_sequence" not in tables

    def test_discover_tables_empty_db(
        self, orchestrator: MigrationOrchestrator, tmp_data_dir: Path
    ):
        """Empty database returns its table."""
        tables = orchestrator.discover_tables(tmp_data_dir / "empty.db")
        assert "empty_table" in tables


# ===========================================================================
# Test Data Coercion
# ===========================================================================


class TestDataCoercion:
    """Tests for the _coerce_value method."""

    def setup_method(self):
        """Create orchestrator for coercion tests."""
        self.orch = MigrationOrchestrator(
            data_dir=".",
            postgres_dsn="postgresql://test@localhost/test",
        )

    def test_none_passthrough(self):
        """None values pass through unchanged."""
        assert self.orch._coerce_value(None, "TEXT") is None
        assert self.orch._coerce_value(None, "INTEGER") is None
        assert self.orch._coerce_value(None, "BOOLEAN") is None

    def test_boolean_coercion(self):
        """SQLite 0/1 integers coerce to Python bools."""
        assert self.orch._coerce_value(1, "BOOLEAN") is True
        assert self.orch._coerce_value(0, "BOOLEAN") is False

    def test_json_coercion_string(self):
        """Valid JSON strings pass through."""
        value = '{"key": "value"}'
        result = self.orch._coerce_value(value, "JSON")
        assert result == value

    def test_json_coercion_dict(self):
        """Dict values are serialized to JSON."""
        value = {"key": "value"}
        result = self.orch._coerce_value(value, "JSON")
        assert json.loads(result) == value

    def test_json_coercion_invalid_string(self):
        """Invalid JSON strings are wrapped in JSON."""
        value = "not json"
        result = self.orch._coerce_value(value, "JSON")
        assert json.loads(result) == "not json"

    def test_timestamp_coercion_iso(self):
        """ISO format timestamps are parsed."""
        value = "2024-01-15T12:30:00+00:00"
        result = self.orch._coerce_value(value, "DATETIME")
        assert isinstance(result, datetime)

    def test_timestamp_coercion_z_suffix(self):
        """Z-suffix timestamps are parsed."""
        value = "2024-01-15T12:30:00Z"
        result = self.orch._coerce_value(value, "TIMESTAMP")
        assert isinstance(result, datetime)

    def test_text_passthrough(self):
        """Text values pass through unchanged."""
        assert self.orch._coerce_value("hello", "TEXT") == "hello"

    def test_integer_passthrough(self):
        """Integer values pass through unchanged."""
        assert self.orch._coerce_value(42, "INTEGER") == 42

    def test_real_passthrough(self):
        """Float values pass through unchanged."""
        assert self.orch._coerce_value(3.14, "REAL") == 3.14


# ===========================================================================
# Test Report Data Classes
# ===========================================================================


class TestReportDataClasses:
    """Tests for migration report data classes."""

    def test_table_result_success(self):
        """TableMigrationResult.success is True with no errors."""
        result = TableMigrationResult(database="test.db", table="users")
        assert result.success is True

    def test_table_result_failure(self):
        """TableMigrationResult.success is False with errors."""
        result = TableMigrationResult(
            database="test.db",
            table="users",
            errors=["Connection failed"],
        )
        assert result.success is False

    def test_table_result_verified(self):
        """TableMigrationResult.verified checks row counts."""
        result = TableMigrationResult(
            database="test.db",
            table="users",
            rows_total_source=100,
            rows_total_target=100,
        )
        assert result.verified is True

        result.rows_total_target = 50
        assert result.verified is False

    def test_database_result_success(self):
        """DatabaseMigrationResult.success aggregates table results."""
        db_result = DatabaseMigrationResult(
            database="core.db",
            tables=[
                TableMigrationResult(database="core.db", table="t1"),
                TableMigrationResult(database="core.db", table="t2"),
            ],
        )
        assert db_result.success is True

        # Add a failure
        db_result.tables.append(
            TableMigrationResult(database="core.db", table="t3", errors=["fail"])
        )
        assert db_result.success is False

    def test_database_result_total_rows(self):
        """DatabaseMigrationResult.total_rows_migrated sums correctly."""
        db_result = DatabaseMigrationResult(
            database="core.db",
            tables=[
                TableMigrationResult(database="core.db", table="t1", rows_migrated=50),
                TableMigrationResult(database="core.db", table="t2", rows_migrated=30),
            ],
        )
        assert db_result.total_rows_migrated == 80

    def test_migration_report_aggregation(self):
        """MigrationReport aggregates across databases."""
        report = MigrationReport(
            data_dir=".",
            target_dsn_safe="localhost:5432",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            databases=[
                DatabaseMigrationResult(
                    database="core.db",
                    tables=[
                        TableMigrationResult(database="core.db", table="t1", rows_migrated=50),
                    ],
                ),
                DatabaseMigrationResult(
                    database="analytics.db",
                    tables=[
                        TableMigrationResult(
                            database="analytics.db",
                            table="t2",
                            rows_migrated=30,
                        ),
                        TableMigrationResult(
                            database="analytics.db",
                            table="t3",
                            rows_migrated=20,
                            errors=["one error"],
                        ),
                    ],
                ),
            ],
        )
        assert report.total_tables == 3
        assert report.total_rows_migrated == 100
        assert report.total_errors == 1
        assert report.success is False  # Has errors

    def test_report_duration(self):
        """MigrationReport.duration_seconds calculates correctly."""
        start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 12, 1, 30, tzinfo=timezone.utc)

        report = MigrationReport(
            data_dir=".",
            target_dsn_safe="localhost",
            started_at=start,
            completed_at=end,
        )
        assert report.duration_seconds == 90.0

    def test_report_no_completed_at(self):
        """MigrationReport.duration_seconds is 0 when not completed."""
        report = MigrationReport(
            data_dir=".",
            target_dsn_safe="localhost",
            started_at=datetime.now(timezone.utc),
        )
        assert report.duration_seconds == 0.0


# ===========================================================================
# Test Dry-Run Mode
# ===========================================================================


class TestDryRun:
    """Tests for dry-run mode (no PostgreSQL connection needed)."""

    @pytest.mark.asyncio
    async def test_dry_run_generates_ddl(
        self, orchestrator: MigrationOrchestrator, tmp_data_dir: Path
    ):
        """Dry run generates DDL without executing."""
        ddl = await orchestrator.create_schema(
            tmp_data_dir / "core.db",
            "debates",
            dry_run=True,
        )

        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert '"debates"' in ddl
        assert '"id" TEXT' in ddl

    @pytest.mark.asyncio
    async def test_dry_run_full_orchestration(self, orchestrator: MigrationOrchestrator):
        """Full dry run produces a report without touching PostgreSQL."""
        report = await orchestrator.run(dry_run=True)

        assert report.dry_run is True
        assert report.success is True
        assert len(report.databases) > 0

        # All tables should have schema_created=True (DDL generated)
        for db in report.databases:
            for table in db.tables:
                assert table.schema_created is True
                assert table.rows_migrated == 0  # No data migration in dry run


# ===========================================================================
# Test Full Migration with Mocks
# ===========================================================================


class TestMigrationWithMocks:
    """Tests for the full migration using mocked asyncpg."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        mock_conn = AsyncMock()

        # Mock column introspection
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"column_name": "id"},
                {"column_name": "task"},
                {"column_name": "final_answer"},
                {"column_name": "consensus_reached"},
                {"column_name": "confidence"},
                {"column_name": "rounds_used"},
                {"column_name": "metadata"},
                {"column_name": "created_at"},
            ]
        )
        mock_conn.fetchval = AsyncMock(return_value=2)
        mock_conn.execute = AsyncMock()
        mock_conn.executemany = AsyncMock()

        # Make pool.acquire() work as async context manager
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=mock_conn)
        pool.close = AsyncMock()

        return pool

    @pytest.mark.asyncio
    async def test_migrate_table_data(
        self,
        orchestrator: MigrationOrchestrator,
        tmp_data_dir: Path,
        mock_pool: MagicMock,
    ):
        """migrate_table_data copies rows correctly."""
        orchestrator._pool = mock_pool

        result = await orchestrator.migrate_table_data(tmp_data_dir / "core.db", "debates")

        assert result.table == "debates"
        assert result.database == "core.db"
        assert result.rows_migrated == 2
        assert result.success is True
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_verify_table(
        self,
        orchestrator: MigrationOrchestrator,
        tmp_data_dir: Path,
        mock_pool: MagicMock,
    ):
        """verify_table compares row counts."""
        orchestrator._pool = mock_pool

        result = await orchestrator.verify_table(tmp_data_dir / "core.db", "debates")

        assert result.rows_total_source == 2  # From real SQLite
        assert result.rows_total_target == 2  # From mock fetchval
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_rollback_table(
        self,
        orchestrator: MigrationOrchestrator,
        mock_pool: MagicMock,
    ):
        """rollback_table drops the table."""
        orchestrator._pool = mock_pool

        success = await orchestrator.rollback_table("debates")
        assert success is True

        # Verify DROP was called
        conn = mock_pool.acquire.return_value
        conn.execute.assert_called()


# ===========================================================================
# Test print_report (smoke test)
# ===========================================================================


class TestPrintReport:
    """Smoke test for report printing."""

    def test_print_report_success(self, capsys):
        """print_report produces output for successful migration."""
        report = MigrationReport(
            data_dir=".nomic",
            target_dsn_safe="localhost:5432/aragora",
            started_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 15, 12, 1, 30, tzinfo=timezone.utc),
            databases=[
                DatabaseMigrationResult(
                    database="core.db",
                    tables=[
                        TableMigrationResult(
                            database="core.db",
                            table="debates",
                            rows_migrated=1000,
                            rows_total_source=1000,
                            rows_total_target=1000,
                            duration_seconds=5.2,
                        ),
                    ],
                ),
            ],
        )

        MigrationOrchestrator.print_report(report)
        captured = capsys.readouterr()

        assert "MIGRATION REPORT" in captured.out
        assert "debates" in captured.out
        assert "1000" in captured.out
        assert "SUCCESS" in captured.out

    def test_print_report_dry_run(self, capsys):
        """print_report shows DRY RUN mode."""
        report = MigrationReport(
            data_dir=".nomic",
            target_dsn_safe="localhost",
            started_at=datetime.now(timezone.utc),
            dry_run=True,
        )

        MigrationOrchestrator.print_report(report)
        captured = capsys.readouterr()

        assert "DRY RUN" in captured.out

    def test_print_report_with_errors(self, capsys):
        """print_report shows errors."""
        report = MigrationReport(
            data_dir=".nomic",
            target_dsn_safe="localhost",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            databases=[
                DatabaseMigrationResult(
                    database="core.db",
                    tables=[
                        TableMigrationResult(
                            database="core.db",
                            table="debates",
                            errors=["Connection refused"],
                        ),
                    ],
                ),
            ],
        )

        MigrationOrchestrator.print_report(report)
        captured = capsys.readouterr()

        assert "FAILED" in captured.out
        assert "Connection refused" in captured.out
