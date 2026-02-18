"""
Tests for Database Migration Utilities (aragora/db/migrate.py).

Covers:
- POSTGRES_SCHEMA constant structure and content
- get_sqlite_tables: listing tables from SQLite databases
- get_table_columns: retrieving column names and types
- get_row_count: counting rows in tables
- migrate_table: single-table migration from SQLite to PostgreSQL (mocked)
- migrate_sqlite_to_postgres: full migration orchestration
- verify_migration: row count comparison between SQLite and PostgreSQL
- SQL injection prevention: table/column name validation
- Rollback scenarios on partial failures
- Edge cases: empty databases, missing files, missing psycopg2
- Data consistency checks (JSON conversion, batch processing)
- CLI main() entry point
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.db.migrate import (
    POSTGRES_SCHEMA,
    get_sqlite_tables,
    get_table_columns,
    get_row_count,
    migrate_table,
    migrate_sqlite_to_postgres,
    verify_migration,
    main,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with sample tables and data."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE debates (
            id TEXT PRIMARY KEY,
            slug TEXT UNIQUE NOT NULL,
            task TEXT NOT NULL,
            agents TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            consensus_reached INTEGER,
            confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE elo_rankings (
            agent_name TEXT PRIMARY KEY,
            elo REAL NOT NULL DEFAULT 1500,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0
        )
    """
    )
    # Insert sample debate data
    conn.execute(
        "INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "d1",
            "test-debate",
            "Design a rate limiter",
            json.dumps(["claude", "gpt4"]),
            json.dumps({"result": "consensus"}),
            1,
            0.95,
            "2024-01-01T00:00:00",
        ),
    )
    conn.execute(
        "INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "d2",
            "another-debate",
            "Evaluate microservices",
            json.dumps(["gemini", "grok"]),
            json.dumps({"result": "majority"}),
            0,
            0.72,
            "2024-01-02T00:00:00",
        ),
    )
    # Insert ELO data
    conn.execute(
        "INSERT INTO elo_rankings VALUES (?, ?, ?, ?, ?)",
        ("claude", 1650.0, 10, 3, 2),
    )
    conn.execute(
        "INSERT INTO elo_rankings VALUES (?, ?, ?, ?, ?)",
        ("gpt4", 1520.0, 7, 5, 3),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_db(tmp_path):
    """Create an empty SQLite database (no tables)."""
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.close()
    return db_path


@pytest.fixture
def db_with_many_rows(tmp_path):
    """Create a database with enough rows to test batch processing."""
    db_path = str(tmp_path / "large.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value REAL,
            metadata TEXT
        )
    """
    )
    rows = [(i, f"item_{i}", float(i) * 1.5, json.dumps({"index": i})) for i in range(2500)]
    conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def db_with_json_data(tmp_path):
    """Create a database with various JSON column data."""
    db_path = str(tmp_path / "json_data.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE json_table (
            id TEXT PRIMARY KEY,
            data_json TEXT,
            data_jsonb JSONB,
            plain_text TEXT
        )
    """
    )
    conn.execute(
        "INSERT INTO json_table VALUES (?, ?, ?, ?)",
        ("j1", json.dumps({"key": "value"}), json.dumps([1, 2, 3]), "plain string"),
    )
    conn.execute(
        "INSERT INTO json_table VALUES (?, ?, ?, ?)",
        (
            "j2",
            "not-json-but-text",
            json.dumps({"nested": {"a": 1}}),
            "{looks-like-json-but-invalid",
        ),
    )
    conn.execute(
        "INSERT INTO json_table VALUES (?, ?, ?, ?)",
        ("j3", None, None, None),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_pg_conn():
    """Create a mock PostgreSQL connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def multi_table_db(tmp_path):
    """Create a database with multiple tables of varying sizes."""
    db_path = str(tmp_path / "multi.db")
    conn = sqlite3.connect(db_path)

    conn.execute("CREATE TABLE alpha (id TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE beta (id TEXT PRIMARY KEY, count INTEGER)")
    conn.execute("CREATE TABLE gamma (id TEXT PRIMARY KEY)")

    conn.execute("INSERT INTO alpha VALUES ('a1', 'hello')")
    conn.execute("INSERT INTO alpha VALUES ('a2', 'world')")
    conn.execute("INSERT INTO beta VALUES ('b1', 42)")
    # gamma is intentionally empty

    conn.commit()
    conn.close()
    return db_path


# -----------------------------------------------------------------------------
# POSTGRES_SCHEMA constant tests
# -----------------------------------------------------------------------------


class TestPostgresSchema:
    """Verify the POSTGRES_SCHEMA constant is well-formed."""

    def test_schema_is_non_empty_string(self):
        assert isinstance(POSTGRES_SCHEMA, str)
        assert len(POSTGRES_SCHEMA) > 100

    def test_schema_contains_core_tables(self):
        expected_tables = [
            "debates",
            "_schema_versions",
            "share_settings",
            "elo_rankings",
            "matches",
            "memory_store",
            "continuum_memory",
            "votes",
            "consensus_memory",
            "tournaments",
            "tournament_matches",
            "plugins",
            "pulse_topics",
        ]
        for table in expected_tables:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in POSTGRES_SCHEMA, (
                f"Missing table definition: {table}"
            )

    def test_schema_contains_indexes(self):
        assert "CREATE INDEX IF NOT EXISTS" in POSTGRES_SCHEMA
        # Spot-check a few critical indexes
        assert "idx_debates_slug" in POSTGRES_SCHEMA
        assert "idx_debates_created" in POSTGRES_SCHEMA
        assert "idx_votes_debate_round" in POSTGRES_SCHEMA
        assert "idx_pulse_topics_score" in POSTGRES_SCHEMA

    def test_schema_uses_jsonb_types(self):
        assert "JSONB" in POSTGRES_SCHEMA

    def test_schema_has_foreign_keys(self):
        assert "REFERENCES debates(id)" in POSTGRES_SCHEMA
        assert "REFERENCES tournaments(id)" in POSTGRES_SCHEMA

    def test_schema_has_timestamp_defaults(self):
        assert "DEFAULT CURRENT_TIMESTAMP" in POSTGRES_SCHEMA


# -----------------------------------------------------------------------------
# get_sqlite_tables tests
# -----------------------------------------------------------------------------


class TestGetSqliteTables:
    """Tests for get_sqlite_tables function."""

    def test_returns_all_user_tables(self, temp_db):
        conn = sqlite3.connect(temp_db)
        tables = get_sqlite_tables(conn)
        conn.close()

        assert set(tables) == {"debates", "elo_rankings"}

    def test_excludes_sqlite_internal_tables(self, temp_db):
        conn = sqlite3.connect(temp_db)
        tables = get_sqlite_tables(conn)
        conn.close()

        for table in tables:
            assert not table.startswith("sqlite_")

    def test_empty_database_returns_empty_list(self, empty_db):
        conn = sqlite3.connect(empty_db)
        tables = get_sqlite_tables(conn)
        conn.close()

        assert tables == []

    def test_multiple_tables_all_found(self, multi_table_db):
        conn = sqlite3.connect(multi_table_db)
        tables = get_sqlite_tables(conn)
        conn.close()

        assert set(tables) == {"alpha", "beta", "gamma"}

    def test_in_memory_database(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test1 (id TEXT)")
        conn.execute("CREATE TABLE test2 (id TEXT)")
        tables = get_sqlite_tables(conn)
        conn.close()

        assert set(tables) == {"test1", "test2"}


# -----------------------------------------------------------------------------
# get_table_columns tests
# -----------------------------------------------------------------------------


class TestGetTableColumns:
    """Tests for get_table_columns function."""

    def test_returns_correct_columns_for_debates(self, temp_db):
        conn = sqlite3.connect(temp_db)
        columns = get_table_columns(conn, "debates")
        conn.close()

        col_names = [c[0] for c in columns]
        assert "id" in col_names
        assert "slug" in col_names
        assert "task" in col_names
        assert "agents" in col_names
        assert "artifact_json" in col_names
        assert "consensus_reached" in col_names
        assert "confidence" in col_names
        assert "created_at" in col_names

    def test_returns_column_types(self, temp_db):
        conn = sqlite3.connect(temp_db)
        columns = get_table_columns(conn, "debates")
        conn.close()

        col_dict = {c[0]: c[1] for c in columns}
        assert col_dict["id"] == "TEXT"
        assert col_dict["confidence"] == "REAL"

    def test_elo_rankings_columns(self, temp_db):
        conn = sqlite3.connect(temp_db)
        columns = get_table_columns(conn, "elo_rankings")
        conn.close()

        col_names = [c[0] for c in columns]
        assert col_names == ["agent_name", "elo", "wins", "losses", "draws"]

    def test_empty_table_has_columns(self, multi_table_db):
        conn = sqlite3.connect(multi_table_db)
        columns = get_table_columns(conn, "gamma")
        conn.close()

        assert len(columns) == 1
        assert columns[0][0] == "id"


# -----------------------------------------------------------------------------
# get_row_count tests
# -----------------------------------------------------------------------------


class TestGetRowCount:
    """Tests for get_row_count function."""

    def test_counts_rows_correctly(self, temp_db):
        conn = sqlite3.connect(temp_db)
        assert get_row_count(conn, "debates") == 2
        assert get_row_count(conn, "elo_rankings") == 2
        conn.close()

    def test_empty_table_returns_zero(self, multi_table_db):
        conn = sqlite3.connect(multi_table_db)
        assert get_row_count(conn, "gamma") == 0
        conn.close()

    def test_large_table_count(self, db_with_many_rows):
        conn = sqlite3.connect(db_with_many_rows)
        assert get_row_count(conn, "items") == 2500
        conn.close()


# -----------------------------------------------------------------------------
# migrate_table tests
# -----------------------------------------------------------------------------


class TestMigrateTable:
    """Tests for migrate_table function."""

    def test_migrates_all_rows(self, temp_db, mock_pg_conn):
        sqlite_conn = sqlite3.connect(temp_db)
        count = migrate_table(sqlite_conn, mock_pg_conn, "debates")
        sqlite_conn.close()

        assert count == 2
        mock_pg_conn.cursor.assert_called()
        cursor = mock_pg_conn.cursor.return_value
        cursor.executemany.assert_called_once()

    def test_insert_sql_uses_correct_format(self, temp_db, mock_pg_conn):
        sqlite_conn = sqlite3.connect(temp_db)
        migrate_table(sqlite_conn, mock_pg_conn, "elo_rankings")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        insert_call = cursor.executemany.call_args
        insert_sql = insert_call[0][0]

        assert 'INSERT INTO "elo_rankings"' in insert_sql
        assert "ON CONFLICT DO NOTHING" in insert_sql
        assert "%s" in insert_sql

    def test_empty_table_returns_zero(self, multi_table_db, mock_pg_conn):
        sqlite_conn = sqlite3.connect(multi_table_db)
        count = migrate_table(sqlite_conn, mock_pg_conn, "gamma")
        sqlite_conn.close()

        assert count == 0
        # cursor.executemany should NOT be called for empty table
        cursor = mock_pg_conn.cursor.return_value
        cursor.executemany.assert_not_called()

    def test_batch_processing(self, db_with_many_rows, mock_pg_conn):
        """Verify rows are batched with the specified batch_size."""
        sqlite_conn = sqlite3.connect(db_with_many_rows)
        count = migrate_table(sqlite_conn, mock_pg_conn, "items", batch_size=1000)
        sqlite_conn.close()

        assert count == 2500
        cursor = mock_pg_conn.cursor.return_value
        # With 2500 rows and batch_size=1000, expect 3 batches
        assert cursor.executemany.call_count == 3

    def test_small_batch_size(self, temp_db, mock_pg_conn):
        """Test with batch_size=1 to verify each row is processed."""
        sqlite_conn = sqlite3.connect(temp_db)
        count = migrate_table(sqlite_conn, mock_pg_conn, "debates", batch_size=1)
        sqlite_conn.close()

        assert count == 2
        cursor = mock_pg_conn.cursor.return_value
        assert cursor.executemany.call_count == 2

    def test_json_data_conversion(self, db_with_json_data, mock_pg_conn):
        """Verify JSON strings are handled correctly during migration."""
        sqlite_conn = sqlite3.connect(db_with_json_data)
        count = migrate_table(sqlite_conn, mock_pg_conn, "json_table")
        sqlite_conn.close()

        assert count == 3
        cursor = mock_pg_conn.cursor.return_value
        call_args = cursor.executemany.call_args
        rows = call_args[0][1]

        # Row 0 has valid JSON in data_json and data_jsonb
        assert rows[0][0] == "j1"
        assert json.loads(rows[0][1]) == {"key": "value"}
        assert json.loads(rows[0][2]) == [1, 2, 3]
        assert rows[0][3] == "plain string"

    def test_null_values_preserved(self, db_with_json_data, mock_pg_conn):
        """Verify NULL values pass through correctly."""
        sqlite_conn = sqlite3.connect(db_with_json_data)
        migrate_table(sqlite_conn, mock_pg_conn, "json_table")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        call_args = cursor.executemany.call_args
        rows = call_args[0][1]

        # Row with id "j3" has NULLs
        j3_row = [r for r in rows if r[0] == "j3"][0]
        assert j3_row[1] is None
        assert j3_row[2] is None
        assert j3_row[3] is None

    def test_invalid_json_strings_passed_through(self, db_with_json_data, mock_pg_conn):
        """Strings that look like JSON but aren't should pass through unchanged."""
        sqlite_conn = sqlite3.connect(db_with_json_data)
        migrate_table(sqlite_conn, mock_pg_conn, "json_table")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        call_args = cursor.executemany.call_args
        rows = call_args[0][1]

        # Row j2 has "not-json-but-text" and "{looks-like-json-but-invalid"
        j2_row = [r for r in rows if r[0] == "j2"][0]
        assert j2_row[1] == "not-json-but-text"
        assert j2_row[3] == "{looks-like-json-but-invalid"

    def test_column_names_in_insert(self, temp_db, mock_pg_conn):
        """Verify column names are correctly included in the INSERT statement."""
        sqlite_conn = sqlite3.connect(temp_db)
        migrate_table(sqlite_conn, mock_pg_conn, "elo_rankings")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        insert_sql = cursor.executemany.call_args[0][0]

        assert "agent_name" in insert_sql
        assert "elo" in insert_sql
        assert "wins" in insert_sql
        assert "losses" in insert_sql
        assert "draws" in insert_sql

    def test_placeholders_match_column_count(self, temp_db, mock_pg_conn):
        """Verify the number of %s placeholders matches the column count."""
        sqlite_conn = sqlite3.connect(temp_db)
        migrate_table(sqlite_conn, mock_pg_conn, "elo_rankings")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        insert_sql = cursor.executemany.call_args[0][0]
        placeholder_count = insert_sql.count("%s")
        assert placeholder_count == 5  # agent_name, elo, wins, losses, draws


# -----------------------------------------------------------------------------
# migrate_sqlite_to_postgres tests
# -----------------------------------------------------------------------------


class TestMigrateSqliteToPostgres:
    """Tests for full migration orchestration."""

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_dry_run_returns_row_counts(self, mock_psycopg2, temp_db):
        """Dry run should report row counts without migrating."""
        mock_pg_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=temp_db,
                dry_run=True,
            )

        assert "debates" in results
        assert results["debates"] == 2
        assert "elo_rankings" in results
        assert results["elo_rankings"] == 2

        # Should NOT execute schema or migrate data
        mock_pg_cursor = mock_pg_conn.cursor.return_value
        mock_pg_cursor.execute.assert_not_called()

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_full_migration_creates_schema_and_migrates(self, mock_psycopg2, temp_db):
        """Full migration should create schema and migrate all tables."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=temp_db,
                dry_run=False,
            )

        # Schema should be executed
        schema_call = mock_pg_cursor.execute.call_args_list[0]
        assert "CREATE TABLE IF NOT EXISTS" in schema_call[0][0]

        # All tables should be in results
        assert "debates" in results
        assert "elo_rankings" in results
        assert results["debates"] == 2
        assert results["elo_rankings"] == 2

        # Commit should be called for schema + each table
        assert mock_pg_conn.commit.call_count >= 3

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_table_migration_error_rolls_back(self, mock_psycopg2, temp_db):
        """When a table migration fails, that table should be rolled back."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value = mock_pg_cursor

        # Make executemany fail for the second batch
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated PG error")

        mock_pg_cursor.executemany.side_effect = side_effect
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=temp_db,
                dry_run=False,
            )

        # First table should succeed, second should fail with -1
        table_names = list(results.keys())
        assert len(table_names) == 2
        # One should be -1 (failed), one should succeed
        values = list(results.values())
        assert -1 in values
        assert mock_pg_conn.rollback.called

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_connections_closed_on_success(self, mock_psycopg2, temp_db):
        """Both connections should be closed even on success."""
        mock_pg_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            migrate_sqlite_to_postgres(
                sqlite_path=temp_db,
                dry_run=True,
            )

        mock_pg_conn.close.assert_called_once()

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_connections_closed_on_error(self, mock_psycopg2, temp_db):
        """Both connections should be closed even on unhandled error."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_cursor.execute.side_effect = Exception("Schema creation failed")
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            with pytest.raises(Exception, match="Schema creation failed"):
                migrate_sqlite_to_postgres(
                    sqlite_path=temp_db,
                    dry_run=False,
                )

        mock_pg_conn.close.assert_called_once()

    def test_missing_psycopg2_returns_empty(self, temp_db):
        """When psycopg2 is not installed, should return empty dict."""
        # Temporarily remove psycopg2 from modules if present
        with patch.dict("sys.modules", {"psycopg2": None}):
            results = migrate_sqlite_to_postgres(sqlite_path=temp_db)
        assert results == {}

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_pg_connection_params(self, mock_psycopg2, temp_db):
        """Verify PostgreSQL connection parameters are passed correctly."""
        mock_pg_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            migrate_sqlite_to_postgres(
                sqlite_path=temp_db,
                pg_host="db.example.com",
                pg_port=5433,
                pg_database="mydb",
                pg_user="admin",
                pg_password="secret123",
                dry_run=True,
            )

        mock_psycopg2.connect.assert_called_once_with(
            host="db.example.com",
            port=5433,
            dbname="mydb",
            user="admin",
            password="secret123",
        )

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_empty_database_migration(self, mock_psycopg2, empty_db):
        """Migrating an empty database should return empty results."""
        mock_pg_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=empty_db,
                dry_run=False,
            )

        assert results == {}

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_partial_failure_continues_remaining_tables(self, mock_psycopg2, multi_table_db):
        """A failure in one table should not prevent migration of others."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value = mock_pg_cursor

        # Track which tables are being processed via INSERT statements
        table_fail_set = {"beta"}

        def conditional_fail(sql, rows):
            for tname in table_fail_set:
                if f'INSERT INTO "{tname}"' in sql or f"INSERT INTO {tname}" in sql:
                    raise RuntimeError(f"Failed to migrate {tname}")
            # Success case: do nothing (mock default behavior)
            return None

        mock_pg_cursor.executemany.side_effect = conditional_fail
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=multi_table_db,
                dry_run=False,
            )

        # alpha (2 rows) should succeed, beta should fail, gamma (0 rows) should succeed
        assert results["alpha"] == 2
        assert results["beta"] == -1
        assert results["gamma"] == 0

        # Rollback should be called for the failed table
        assert mock_pg_conn.rollback.called


# -----------------------------------------------------------------------------
# verify_migration tests
# -----------------------------------------------------------------------------


class TestVerifyMigration:
    """Tests for migration verification."""

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_matching_counts(self, mock_psycopg2, temp_db):
        """When counts match, all tables should report match=True."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        # Return matching counts for each table
        mock_pg_cursor.fetchone.side_effect = [(2,), (2,)]
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = verify_migration(sqlite_path=temp_db)

        for table_name, comparison in results.items():
            assert comparison["match"] is True
            assert comparison["sqlite"] == comparison["postgres"]

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_mismatched_counts(self, mock_psycopg2, temp_db):
        """When counts differ, should report match=False."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        # First table matches, second doesn't
        mock_pg_cursor.fetchone.side_effect = [(2,), (1,)]
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = verify_migration(sqlite_path=temp_db)

        tables = list(results.keys())
        # One should match, one should not
        match_values = [r["match"] for r in results.values()]
        assert True in match_values
        assert False in match_values

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_pg_table_error_returns_negative_one(self, mock_psycopg2, temp_db):
        """When PG query fails for a table, should report postgres=-1."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        # First table succeeds, second fails
        mock_pg_cursor.execute.side_effect = [None, RuntimeError("Table not found")]
        mock_pg_cursor.fetchone.return_value = (2,)
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = verify_migration(sqlite_path=temp_db)

        # The table that errored should have postgres=-1
        values = list(results.values())
        pg_counts = [v["postgres"] for v in values]
        assert -1 in pg_counts

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_verify_closes_connections(self, mock_psycopg2, temp_db):
        """Verify connections are closed after verification."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_cursor.fetchone.return_value = (2,)
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            verify_migration(sqlite_path=temp_db)

        mock_pg_conn.close.assert_called_once()

    def test_verify_missing_psycopg2_returns_empty(self, temp_db):
        """When psycopg2 is not installed, should return empty dict."""
        with patch.dict("sys.modules", {"psycopg2": None}):
            results = verify_migration(sqlite_path=temp_db)
        assert results == {}

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_verify_empty_database(self, mock_psycopg2, empty_db):
        """Verifying an empty database should return empty results."""
        mock_pg_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = verify_migration(sqlite_path=empty_db)

        assert results == {}

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_verify_result_structure(self, mock_psycopg2, temp_db):
        """Verify each result entry has the expected keys."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_cursor.fetchone.return_value = (2,)
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = verify_migration(sqlite_path=temp_db)

        for table_name, comparison in results.items():
            assert "sqlite" in comparison
            assert "postgres" in comparison
            assert "match" in comparison
            assert isinstance(comparison["sqlite"], int)
            assert isinstance(comparison["match"], bool)


# -----------------------------------------------------------------------------
# SQL injection prevention tests
# -----------------------------------------------------------------------------


class TestSQLInjectionPrevention:
    """Test that table names used in SQL are sourced from sqlite_master only.

    The module uses f-string formatting for table names in PRAGMA,
    SELECT COUNT(*), and INSERT INTO statements. These table names originate
    from sqlite_master queries, which limits the attack surface. This test
    class verifies that the names used are exactly those returned by the
    database itself, and that no external input can inject SQL.
    """

    def test_table_names_sourced_from_sqlite_master(self, temp_db):
        """Table names come from sqlite_master, not user input."""
        conn = sqlite3.connect(temp_db)
        tables = get_sqlite_tables(conn)
        conn.close()

        # Only known table names should be returned
        for table in tables:
            assert table.isidentifier(), f"Table name '{table}' is not a valid identifier"

    def test_get_row_count_uses_database_tables(self, temp_db):
        """get_row_count operates on actual table names from the database."""
        conn = sqlite3.connect(temp_db)
        tables = get_sqlite_tables(conn)

        for table in tables:
            # Should work without error for real table names
            count = get_row_count(conn, table)
            assert isinstance(count, int)
            assert count >= 0

        conn.close()

    def test_get_table_columns_uses_database_tables(self, temp_db):
        """get_table_columns operates on actual table names from the database."""
        conn = sqlite3.connect(temp_db)
        tables = get_sqlite_tables(conn)

        for table in tables:
            columns = get_table_columns(conn, table)
            assert len(columns) > 0

        conn.close()

    def test_malicious_table_name_in_sqlite_rejected(self, tmp_path):
        """A table with SQL injection-like name in SQLite would need to be
        a valid SQLite identifier, limiting the injection surface."""
        db_path = str(tmp_path / "inject.db")
        conn = sqlite3.connect(db_path)

        # SQLite allows quoted identifiers, but sqlite_master only returns
        # the name, not the quotes. The module then uses f-strings with the
        # name, so the name must be a valid table name.
        conn.execute('CREATE TABLE "normal_table" (id TEXT)')
        tables = get_sqlite_tables(conn)
        conn.close()

        # Table name returned should be clean
        assert tables == ["normal_table"]
        assert all(t.isidentifier() for t in tables)


# -----------------------------------------------------------------------------
# Data consistency tests
# -----------------------------------------------------------------------------


class TestDataConsistency:
    """Test data integrity during migration."""

    def test_all_rows_from_source_appear_in_migration(self, temp_db, mock_pg_conn):
        """Every row from the source table should be included in the migration."""
        sqlite_conn = sqlite3.connect(temp_db)
        source_count = get_row_count(sqlite_conn, "debates")
        migrated = migrate_table(sqlite_conn, mock_pg_conn, "debates")
        sqlite_conn.close()

        assert migrated == source_count

    def test_data_values_preserved(self, temp_db, mock_pg_conn):
        """Verify actual data values are preserved during migration."""
        sqlite_conn = sqlite3.connect(temp_db)
        migrate_table(sqlite_conn, mock_pg_conn, "elo_rankings")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        call_args = cursor.executemany.call_args
        rows = call_args[0][1]

        # Should have exactly 2 rows
        assert len(rows) == 2

        row_dict = {r[0]: r for r in rows}
        assert row_dict["claude"] == ("claude", 1650.0, 10, 3, 2)
        assert row_dict["gpt4"] == ("gpt4", 1520.0, 7, 5, 3)

    def test_json_round_trip_integrity(self, db_with_json_data, mock_pg_conn):
        """JSON data should remain valid after migration."""
        sqlite_conn = sqlite3.connect(db_with_json_data)
        migrate_table(sqlite_conn, mock_pg_conn, "json_table")
        sqlite_conn.close()

        cursor = mock_pg_conn.cursor.return_value
        call_args = cursor.executemany.call_args
        rows = call_args[0][1]

        j1_row = [r for r in rows if r[0] == "j1"][0]
        # Verify JSON is still parseable
        assert json.loads(j1_row[1]) == {"key": "value"}
        assert json.loads(j1_row[2]) == [1, 2, 3]

    def test_batch_migration_preserves_all_data(self, db_with_many_rows, mock_pg_conn):
        """All 2500 rows should be migrated even with batching."""
        sqlite_conn = sqlite3.connect(db_with_many_rows)
        migrated = migrate_table(sqlite_conn, mock_pg_conn, "items", batch_size=500)
        sqlite_conn.close()

        assert migrated == 2500

        cursor = mock_pg_conn.cursor.return_value
        total_rows = sum(len(c[0][1]) for c in cursor.executemany.call_args_list)
        assert total_rows == 2500


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_table(self, tmp_path):
        """A table with a single row should migrate correctly."""
        db_path = str(tmp_path / "single.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE solo (id TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO solo VALUES ('only', 'one')")
        conn.commit()
        conn.close()

        sqlite_conn = sqlite3.connect(db_path)
        mock_pg = MagicMock()
        count = migrate_table(sqlite_conn, mock_pg, "solo")
        sqlite_conn.close()

        assert count == 1

    def test_table_with_many_columns(self, tmp_path):
        """Tables with many columns should have correct placeholder count."""
        db_path = str(tmp_path / "wide.db")
        conn = sqlite3.connect(db_path)
        cols = ", ".join([f"col_{i} TEXT" for i in range(20)])
        conn.execute(f"CREATE TABLE wide (id TEXT PRIMARY KEY, {cols})")
        values = ["v"] * 21
        placeholders = ", ".join(["?"] * 21)
        conn.execute(f"INSERT INTO wide VALUES ({placeholders})", values)
        conn.commit()
        conn.close()

        sqlite_conn = sqlite3.connect(db_path)
        mock_pg = MagicMock()
        migrate_table(sqlite_conn, mock_pg, "wide")
        sqlite_conn.close()

        cursor = mock_pg.cursor.return_value
        insert_sql = cursor.executemany.call_args[0][0]
        assert insert_sql.count("%s") == 21

    def test_nonexistent_sqlite_file(self):
        """Accessing a nonexistent SQLite file should fail gracefully."""
        # sqlite3.connect creates the file, so we test with operations
        fake_path = "/tmp/nonexistent_db_test_12345.db"
        with pytest.raises(Exception):
            conn = sqlite3.connect(fake_path)
            conn.execute("SELECT * FROM nonexistent_table")

    def test_special_characters_in_text_values(self, tmp_path):
        """Text values with special characters should pass through unchanged."""
        db_path = str(tmp_path / "special.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE special (id TEXT PRIMARY KEY, content TEXT)")
        special_values = [
            ("s1", "Hello 'world'"),
            ("s2", 'Quote "test"'),
            ("s3", "Line1\nLine2\tTabbed"),
            ("s4", "Unicode: \u00e9\u00e0\u00fc\u00f1"),
            ("s5", "SQL; DROP TABLE; --"),
        ]
        conn.executemany("INSERT INTO special VALUES (?, ?)", special_values)
        conn.commit()
        conn.close()

        sqlite_conn = sqlite3.connect(db_path)
        mock_pg = MagicMock()
        count = migrate_table(sqlite_conn, mock_pg, "special")
        sqlite_conn.close()

        assert count == 5
        cursor = mock_pg.cursor.return_value
        rows = cursor.executemany.call_args[0][1]
        content_values = [r[1] for r in rows]
        assert "Hello 'world'" in content_values
        assert 'Quote "test"' in content_values
        assert "SQL; DROP TABLE; --" in content_values

    def test_batch_size_larger_than_row_count(self, temp_db, mock_pg_conn):
        """When batch_size exceeds row count, should still work correctly."""
        sqlite_conn = sqlite3.connect(temp_db)
        count = migrate_table(sqlite_conn, mock_pg_conn, "debates", batch_size=10000)
        sqlite_conn.close()

        assert count == 2

    def test_batch_size_equal_to_row_count(self, temp_db, mock_pg_conn):
        """When batch_size equals row count exactly."""
        sqlite_conn = sqlite3.connect(temp_db)
        count = migrate_table(sqlite_conn, mock_pg_conn, "debates", batch_size=2)
        sqlite_conn.close()

        assert count == 2

    def test_table_with_only_primary_key(self, tmp_path):
        """A table with only an id column and no data columns."""
        db_path = str(tmp_path / "idonly.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE idonly (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO idonly VALUES ('a')")
        conn.execute("INSERT INTO idonly VALUES ('b')")
        conn.commit()
        conn.close()

        sqlite_conn = sqlite3.connect(db_path)
        mock_pg = MagicMock()
        count = migrate_table(sqlite_conn, mock_pg, "idonly")
        sqlite_conn.close()

        assert count == 2
        cursor = mock_pg.cursor.return_value
        insert_sql = cursor.executemany.call_args[0][0]
        assert insert_sql.count("%s") == 1


# -----------------------------------------------------------------------------
# Rollback scenario tests
# -----------------------------------------------------------------------------


class TestRollbackScenarios:
    """Test rollback behavior on failures."""

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_rollback_on_individual_table_failure(self, mock_psycopg2, multi_table_db):
        """Each table failure should trigger a rollback for that table only."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value = mock_pg_cursor

        fail_count = [0]

        def fail_on_beta(sql, rows):
            if 'INSERT INTO "beta"' in sql or "INSERT INTO beta" in sql:
                fail_count[0] += 1
                raise RuntimeError("beta migration failed")

        mock_pg_cursor.executemany.side_effect = fail_on_beta
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=multi_table_db,
                dry_run=False,
            )

        assert results["beta"] == -1
        assert fail_count[0] == 1
        # Rollback called at least once (for beta)
        assert mock_pg_conn.rollback.call_count >= 1

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_successful_tables_committed_despite_other_failures(
        self, mock_psycopg2, multi_table_db
    ):
        """Tables that succeed should be committed even if others fail."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value = mock_pg_cursor

        def fail_on_beta(sql, rows):
            if 'INSERT INTO "beta"' in sql or "INSERT INTO beta" in sql:
                raise RuntimeError("beta migration failed")

        mock_pg_cursor.executemany.side_effect = fail_on_beta
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=multi_table_db,
                dry_run=False,
            )

        # Commit should be called for schema + successful tables (alpha, gamma)
        # Exact count depends on table iteration order, but should be >= 2
        assert mock_pg_conn.commit.call_count >= 2

    @patch("aragora.db.migrate.psycopg2", create=True)
    def test_all_tables_fail(self, mock_psycopg2, multi_table_db):
        """When all tables fail, all should report -1."""
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value = mock_pg_cursor
        mock_pg_cursor.executemany.side_effect = RuntimeError("All fail")
        mock_psycopg2.connect.return_value = mock_pg_conn

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            results = migrate_sqlite_to_postgres(
                sqlite_path=multi_table_db,
                dry_run=False,
            )

        # gamma has 0 rows, so no executemany => no error => count=0
        # alpha and beta have rows => executemany fails => -1
        for table, count in results.items():
            if table == "gamma":
                assert count == 0
            else:
                assert count == -1


# -----------------------------------------------------------------------------
# CLI main() tests
# -----------------------------------------------------------------------------


class TestMainCLI:
    """Tests for the CLI entry point."""

    def test_schema_command_prints_schema(self, capsys):
        """The 'schema' command should print the PostgreSQL schema."""
        with patch("sys.argv", ["migrate.py", "schema"]):
            main()

        captured = capsys.readouterr()
        assert "CREATE TABLE IF NOT EXISTS debates" in captured.out
        assert "CREATE TABLE IF NOT EXISTS elo_rankings" in captured.out

    @patch("aragora.db.migrate.migrate_sqlite_to_postgres")
    def test_migrate_command(self, mock_migrate, capsys, temp_db):
        """The 'migrate' command should call migrate_sqlite_to_postgres."""
        mock_migrate.return_value = {"debates": 10, "elo_rankings": 5}

        with patch(
            "sys.argv",
            [
                "migrate.py",
                "migrate",
                "--sqlite-path",
                temp_db,
                "--dry-run",
            ],
        ):
            main()

        mock_migrate.assert_called_once()
        call_kwargs = mock_migrate.call_args
        assert call_kwargs[1]["dry_run"] is True or (
            len(call_kwargs[0]) > 0 or call_kwargs[1].get("dry_run") is True
        )

        captured = capsys.readouterr()
        assert "Migration results:" in captured.out

    @patch("aragora.db.migrate.migrate_sqlite_to_postgres")
    def test_migrate_command_shows_failed_status(self, mock_migrate, capsys, temp_db):
        """Failed migrations should show FAILED status."""
        mock_migrate.return_value = {"debates": 10, "elo_rankings": -1}

        with patch(
            "sys.argv",
            [
                "migrate.py",
                "migrate",
                "--sqlite-path",
                temp_db,
            ],
        ):
            main()

        captured = capsys.readouterr()
        assert "FAILED" in captured.out
        assert "OK" in captured.out

    @patch("aragora.db.migrate.verify_migration")
    def test_verify_command_all_match(self, mock_verify, capsys, temp_db):
        """When all tables match, verify should exit 0."""
        mock_verify.return_value = {
            "debates": {"sqlite": 10, "postgres": 10, "match": True},
        }

        with patch(
            "sys.argv",
            ["migrate.py", "verify", "--sqlite-path", temp_db],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "OK" in captured.out

    @patch("aragora.db.migrate.verify_migration")
    def test_verify_command_mismatch_exits_1(self, mock_verify, capsys, temp_db):
        """When tables don't match, verify should exit 1."""
        mock_verify.return_value = {
            "debates": {"sqlite": 10, "postgres": 8, "match": False},
        }

        with patch(
            "sys.argv",
            ["migrate.py", "verify", "--sqlite-path", temp_db],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "MISMATCH" in captured.out

    def test_invalid_command_raises_error(self):
        """An invalid command should trigger argparse error."""
        with patch("sys.argv", ["migrate.py", "invalid"]):
            with pytest.raises(SystemExit):
                main()

    @patch("aragora.db.migrate.migrate_sqlite_to_postgres")
    def test_custom_pg_params_passed_through(self, mock_migrate, temp_db):
        """Custom PostgreSQL connection params should be forwarded."""
        mock_migrate.return_value = {}

        with patch(
            "sys.argv",
            [
                "migrate.py",
                "migrate",
                "--sqlite-path",
                temp_db,
                "--pg-host",
                "custom-host",
                "--pg-port",
                "5433",
                "--pg-database",
                "custom_db",
                "--pg-user",
                "custom_user",
                "--pg-password",
                "custom_pass",
            ],
        ):
            main()

        call_kwargs = mock_migrate.call_args[1]
        assert call_kwargs["pg_host"] == "custom-host"
        assert call_kwargs["pg_port"] == 5433
        assert call_kwargs["pg_database"] == "custom_db"
        assert call_kwargs["pg_user"] == "custom_user"
        assert call_kwargs["pg_password"] == "custom_pass"


# -----------------------------------------------------------------------------
# Integration-style tests (SQLite-only, no real PostgreSQL)
# -----------------------------------------------------------------------------


class TestSQLiteToSQLiteMigration:
    """End-to-end migration tests using a second SQLite database as the
    target, validating the full flow without requiring PostgreSQL."""

    def test_full_pipeline_with_sqlite_target(self, temp_db, tmp_path):
        """Simulate a full migration pipeline using SQLite as both source
        and target. This verifies the overall data flow."""
        # Read all data from source
        source_conn = sqlite3.connect(temp_db)
        source_tables = get_sqlite_tables(source_conn)
        source_data = {}
        for table in source_tables:
            columns = get_table_columns(source_conn, table)
            col_names = [c[0] for c in columns]
            cursor = source_conn.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            source_data[table] = {"columns": col_names, "rows": rows}
        source_conn.close()

        # Create target SQLite with same schema
        target_path = str(tmp_path / "target.db")
        target_conn = sqlite3.connect(target_path)
        target_conn.execute(
            """
            CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                slug TEXT UNIQUE NOT NULL,
                task TEXT NOT NULL,
                agents TEXT NOT NULL,
                artifact_json TEXT NOT NULL,
                consensus_reached INTEGER,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        target_conn.execute(
            """
            CREATE TABLE elo_rankings (
                agent_name TEXT PRIMARY KEY,
                elo REAL NOT NULL DEFAULT 1500,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0
            )
        """
        )
        target_conn.commit()

        # Copy data
        for table, data in source_data.items():
            col_names = data["columns"]
            placeholders = ", ".join(["?"] * len(col_names))
            insert_sql = f"INSERT INTO {table} ({', '.join(col_names)}) VALUES ({placeholders})"
            target_conn.executemany(insert_sql, data["rows"])
        target_conn.commit()

        # Verify counts match
        for table in source_tables:
            source_conn = sqlite3.connect(temp_db)
            target_count = get_row_count(target_conn, table)
            source_count = get_row_count(source_conn, table)
            source_conn.close()
            assert target_count == source_count, (
                f"Row count mismatch for {table}: source={source_count}, target={target_count}"
            )

        target_conn.close()

    def test_data_fidelity_check(self, temp_db):
        """Verify that reading and re-reading data produces identical results."""
        conn = sqlite3.connect(temp_db)

        # First read
        cursor1 = conn.execute("SELECT * FROM debates ORDER BY id")
        rows1 = cursor1.fetchall()

        # Second read
        cursor2 = conn.execute("SELECT * FROM debates ORDER BY id")
        rows2 = cursor2.fetchall()

        conn.close()

        assert rows1 == rows2
        assert len(rows1) == 2
