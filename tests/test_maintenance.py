"""Tests for database maintenance module."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.maintenance.db_maintenance import (
    ALLOWED_CLEANUP_TABLES,
    ALLOWED_TIMESTAMP_COLUMNS,
    DatabaseMaintenance,
    run_startup_maintenance,
    schedule_maintenance,
)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_db(temp_db_dir):
    """Create a sample SQLite database for testing."""
    db_path = temp_db_dir / "test.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO test_table (data) VALUES ('test')")
        conn.commit()
    return db_path


@pytest.fixture
def db_with_timestamps(temp_db_dir):
    """Create a database with timestamped records for cleanup testing."""
    db_path = temp_db_dir / "elo.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE match_history (
                id INTEGER PRIMARY KEY,
                data TEXT,
                created_at TEXT
            )
        """
        )
        # Insert old and new records
        old_date = (datetime.now() - timedelta(days=100)).isoformat()
        new_date = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO match_history (data, created_at) VALUES (?, ?)", ("old", old_date)
        )
        conn.execute(
            "INSERT INTO match_history (data, created_at) VALUES (?, ?)", ("new", new_date)
        )
        conn.commit()
    return db_path


class TestDatabaseMaintenance:
    """Tests for DatabaseMaintenance class."""

    def test_init_with_default_path(self):
        """Test initialization with default path."""
        maintenance = DatabaseMaintenance()
        assert maintenance.db_dir == Path(".nomic")

    def test_init_with_custom_path(self, temp_db_dir):
        """Test initialization with custom path."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        assert maintenance.db_dir == temp_db_dir

    def test_get_databases_empty_dir(self, temp_db_dir):
        """Test get_databases with no databases."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        assert maintenance.get_databases() == []

    def test_get_databases_finds_files(self, temp_db_dir, sample_db):
        """Test get_databases finds database files."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        databases = maintenance.get_databases()
        assert len(databases) == 1
        assert databases[0].name == "test.db"

    def test_get_databases_finds_known_databases(self, temp_db_dir):
        """Test get_databases finds known database names."""
        # Create a known database
        (temp_db_dir / "elo.db").touch()
        maintenance = DatabaseMaintenance(temp_db_dir)
        databases = maintenance.get_databases()
        assert len(databases) == 1
        assert databases[0].name == "elo.db"

    def test_get_databases_finds_nested(self, temp_db_dir):
        """Test get_databases finds databases in subdirectories."""
        traces_dir = temp_db_dir / "traces"
        traces_dir.mkdir()
        (traces_dir / "debate_traces.db").touch()

        maintenance = DatabaseMaintenance(temp_db_dir)
        databases = maintenance.get_databases()
        assert len(databases) == 1

    def test_checkpoint_wal_success(self, temp_db_dir, sample_db):
        """Test successful WAL checkpoint."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.checkpoint_wal(sample_db)
        assert result is True

    def test_checkpoint_wal_missing_file(self, temp_db_dir):
        """Test WAL checkpoint with missing file."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.checkpoint_wal(temp_db_dir / "nonexistent.db")
        assert result is False

    def test_checkpoint_all_wal(self, temp_db_dir, sample_db):
        """Test checkpointing all databases."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        results = maintenance.checkpoint_all_wal()
        assert "test.db" in results
        assert results["test.db"] is True

    def test_vacuum_success(self, temp_db_dir, sample_db):
        """Test successful VACUUM operation."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.vacuum(sample_db)
        assert result is True

    def test_vacuum_missing_file(self, temp_db_dir):
        """Test VACUUM with missing file."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.vacuum(temp_db_dir / "nonexistent.db")
        assert result is False

    def test_vacuum_all(self, temp_db_dir, sample_db):
        """Test vacuuming all databases."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        results = maintenance.vacuum_all()
        assert "test.db" in results
        assert results["test.db"] is True
        assert maintenance._last_vacuum is not None

    def test_analyze_success(self, temp_db_dir, sample_db):
        """Test successful ANALYZE operation."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.analyze(sample_db)
        assert result is True

    def test_analyze_missing_file(self, temp_db_dir):
        """Test ANALYZE with missing file."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.analyze(temp_db_dir / "nonexistent.db")
        assert result is False

    def test_analyze_all(self, temp_db_dir, sample_db):
        """Test analyzing all databases."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        results = maintenance.analyze_all()
        assert "test.db" in results
        assert results["test.db"] is True
        assert maintenance._last_analyze is not None

    def test_cleanup_old_data_default_tables(self, temp_db_dir, db_with_timestamps):
        """Test cleanup with default table configuration."""
        maintenance = DatabaseMaintenance(temp_db_dir)

        # Verify we have 2 records before cleanup
        with sqlite3.connect(str(db_with_timestamps)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM match_history").fetchone()[0]
            assert count == 2

        # Run cleanup
        results = maintenance.cleanup_old_data(days=90)

        # Verify old record was deleted
        with sqlite3.connect(str(db_with_timestamps)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM match_history").fetchone()[0]
            assert count == 1

        assert "elo.db:match_history" in results
        assert results["elo.db:match_history"] == 1

    def test_cleanup_old_data_invalid_table(self, temp_db_dir):
        """Test cleanup rejects invalid table names."""
        maintenance = DatabaseMaintenance(temp_db_dir)

        with pytest.raises(ValueError, match="Invalid table name"):
            maintenance.cleanup_old_data(tables={"test.db": "dangerous_table"})

    def test_cleanup_old_data_allowed_tables(self):
        """Test that all allowed tables are in whitelist."""
        assert "match_history" in ALLOWED_CLEANUP_TABLES
        assert "memories" in ALLOWED_CLEANUP_TABLES
        assert "traces" in ALLOWED_CLEANUP_TABLES

    def test_cleanup_old_data_allowed_columns(self):
        """Test that allowed timestamp columns are defined."""
        assert "created_at" in ALLOWED_TIMESTAMP_COLUMNS
        assert "timestamp" in ALLOWED_TIMESTAMP_COLUMNS

    def test_get_stats(self, temp_db_dir, sample_db):
        """Test getting maintenance statistics."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        stats = maintenance.get_stats()

        assert stats["database_count"] == 1
        assert stats["total_size_mb"] >= 0
        assert stats["last_vacuum"] is None
        assert stats["last_analyze"] is None
        assert "test.db" in stats["databases"]

    def test_get_stats_after_maintenance(self, temp_db_dir, sample_db):
        """Test stats update after maintenance operations."""
        maintenance = DatabaseMaintenance(temp_db_dir)
        maintenance.vacuum_all()
        maintenance.analyze_all()

        stats = maintenance.get_stats()
        assert stats["last_vacuum"] is not None
        assert stats["last_analyze"] is not None


class TestStartupMaintenance:
    """Tests for run_startup_maintenance function."""

    def test_startup_maintenance_empty_dir(self, temp_db_dir):
        """Test startup maintenance with no databases."""
        results = run_startup_maintenance(temp_db_dir)

        assert "wal_checkpoint" in results
        assert "stats" in results
        assert results["stats"]["database_count"] == 0

    def test_startup_maintenance_with_db(self, temp_db_dir, sample_db):
        """Test startup maintenance with databases."""
        results = run_startup_maintenance(temp_db_dir)

        assert "wal_checkpoint" in results
        assert "test.db" in results["wal_checkpoint"]

    def test_startup_maintenance_creates_state_file(self, temp_db_dir, sample_db):
        """Test that startup maintenance creates state file."""
        run_startup_maintenance(temp_db_dir)

        state_file = temp_db_dir / "maintenance_state.json"
        assert state_file.exists()

    def test_startup_maintenance_skips_recent_analyze(self, temp_db_dir, sample_db):
        """Test that ANALYZE is skipped if done recently."""
        state_file = temp_db_dir / "maintenance_state.json"
        with open(state_file, "w") as f:
            json.dump({"last_analyze": datetime.now().isoformat()}, f)

        results = run_startup_maintenance(temp_db_dir)
        assert "analyze" not in results


class TestScheduleMaintenance:
    """Tests for schedule_maintenance function."""

    def test_schedule_maintenance_runs_overdue_tasks(self, temp_db_dir, sample_db):
        """Test that overdue tasks are run."""
        # Create old state file
        state_file = temp_db_dir / "maintenance_state.json"
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        with open(state_file, "w") as f:
            json.dump(
                {
                    "last_vacuum": old_date,
                    "last_analyze": old_date,
                    "last_cleanup": old_date,
                },
                f,
            )

        results = schedule_maintenance(temp_db_dir)

        assert "vacuum" in results["tasks_run"]
        assert "analyze" in results["tasks_run"]
        assert "cleanup" in results["tasks_run"]

    def test_schedule_maintenance_skips_recent_tasks(self, temp_db_dir, sample_db):
        """Test that recent tasks are skipped."""
        state_file = temp_db_dir / "maintenance_state.json"
        now = datetime.now().isoformat()
        with open(state_file, "w") as f:
            json.dump(
                {
                    "last_vacuum": now,
                    "last_analyze": now,
                    "last_cleanup": now,
                },
                f,
            )

        results = schedule_maintenance(temp_db_dir)
        assert results["tasks_run"] == []

    def test_schedule_maintenance_updates_state(self, temp_db_dir, sample_db):
        """Test that state file is updated after tasks run."""
        results = schedule_maintenance(temp_db_dir)

        if results["tasks_run"]:
            state_file = temp_db_dir / "maintenance_state.json"
            assert state_file.exists()
            with open(state_file) as f:
                state = json.load(f)
            assert any(k in state for k in ["last_vacuum", "last_analyze", "last_cleanup"])

    def test_schedule_maintenance_custom_intervals(self, temp_db_dir, sample_db):
        """Test custom maintenance intervals."""
        # Set state 2 days old
        state_file = temp_db_dir / "maintenance_state.json"
        old_date = (datetime.now() - timedelta(days=2)).isoformat()
        with open(state_file, "w") as f:
            json.dump(
                {
                    "last_vacuum": old_date,
                    "last_analyze": old_date,
                },
                f,
            )

        # With 3-day interval, vacuum should not run
        results = schedule_maintenance(
            temp_db_dir,
            vacuum_interval_days=3,
            analyze_interval_hours=1,
        )

        assert "vacuum" not in results["tasks_run"]
        assert "analyze" in results["tasks_run"]


class TestSQLInjectionPrevention:
    """Tests for SQL injection prevention in maintenance module."""

    def test_table_whitelist_prevents_injection(self, temp_db_dir):
        """Test that table whitelist prevents SQL injection."""
        maintenance = DatabaseMaintenance(temp_db_dir)

        # Try various injection attempts
        injection_attempts = [
            "users; DROP TABLE users;--",
            "users UNION SELECT * FROM secrets",
            "'; DROP TABLE users;--",
            "users WHERE 1=1 OR ''='",
        ]

        for injection in injection_attempts:
            with pytest.raises(ValueError, match="Invalid table name"):
                maintenance.cleanup_old_data(tables={"test.db": injection})

    def test_only_whitelisted_tables_allowed(self, temp_db_dir):
        """Test that only whitelisted tables are allowed."""
        maintenance = DatabaseMaintenance(temp_db_dir)

        # These should fail
        with pytest.raises(ValueError):
            maintenance.cleanup_old_data(tables={"test.db": "users"})

        with pytest.raises(ValueError):
            maintenance.cleanup_old_data(tables={"test.db": "passwords"})
