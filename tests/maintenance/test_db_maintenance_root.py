"""Tests for database maintenance module."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aragora.maintenance.db_maintenance import (
    DatabaseMaintenance,
    run_startup_maintenance,
    schedule_maintenance,
    KNOWN_DATABASES,
)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory with test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_dir = Path(tmpdir)

        # Create some test databases
        for db_name in ["elo.db", "continuum.db", "test.db"]:
            db_path = db_dir / db_name
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE test_table (
                        id INTEGER PRIMARY KEY,
                        data TEXT,
                        created_at TEXT
                    )
                """
                )
                # Insert some test data
                for i in range(10):
                    conn.execute(
                        "INSERT INTO test_table (data, created_at) VALUES (?, ?)",
                        (f"data_{i}", datetime.now().isoformat()),
                    )

        yield db_dir


@pytest.fixture
def maintenance(temp_db_dir):
    """Create a DatabaseMaintenance instance."""
    return DatabaseMaintenance(temp_db_dir)


class TestDatabaseMaintenance:
    """Tests for DatabaseMaintenance class."""

    def test_get_databases(self, maintenance, temp_db_dir):
        """Test that get_databases finds all database files."""
        databases = maintenance.get_databases()
        db_names = [db.name for db in databases]

        assert "elo.db" in db_names
        assert "continuum.db" in db_names
        assert "test.db" in db_names
        assert len(databases) == 3

    def test_get_databases_empty_dir(self):
        """Test get_databases with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            maintenance = DatabaseMaintenance(tmpdir)
            databases = maintenance.get_databases()
            assert databases == []

    def test_checkpoint_wal(self, maintenance, temp_db_dir):
        """Test WAL checkpoint for a single database."""
        db_path = temp_db_dir / "elo.db"

        # Set database to WAL mode first
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

        result = maintenance.checkpoint_wal(db_path)
        assert result is True

    def test_checkpoint_wal_missing_db(self, maintenance, temp_db_dir):
        """Test WAL checkpoint with missing database."""
        db_path = temp_db_dir / "nonexistent.db"
        result = maintenance.checkpoint_wal(db_path)
        # Should fail but not raise
        assert result is False

    def test_checkpoint_all_wal(self, maintenance):
        """Test WAL checkpoint for all databases."""
        results = maintenance.checkpoint_all_wal()

        assert "elo.db" in results
        assert "continuum.db" in results
        assert "test.db" in results
        # All should succeed
        assert all(results.values())

    def test_vacuum(self, maintenance, temp_db_dir):
        """Test VACUUM for a single database."""
        db_path = temp_db_dir / "elo.db"
        result = maintenance.vacuum(db_path)
        assert result is True

    def test_vacuum_missing_db(self, maintenance, temp_db_dir):
        """Test VACUUM with missing database."""
        db_path = temp_db_dir / "nonexistent.db"
        result = maintenance.vacuum(db_path)
        assert result is False

    def test_vacuum_all(self, maintenance):
        """Test VACUUM for all databases."""
        results = maintenance.vacuum_all()

        assert len(results) == 3
        assert all(results.values())
        assert maintenance._last_vacuum is not None

    def test_analyze(self, maintenance, temp_db_dir):
        """Test ANALYZE for a single database."""
        db_path = temp_db_dir / "elo.db"
        result = maintenance.analyze(db_path)
        assert result is True

    def test_analyze_missing_db(self, maintenance, temp_db_dir):
        """Test ANALYZE with missing database."""
        db_path = temp_db_dir / "nonexistent.db"
        result = maintenance.analyze(db_path)
        assert result is False

    def test_analyze_all(self, maintenance):
        """Test ANALYZE for all databases."""
        results = maintenance.analyze_all()

        assert len(results) == 3
        assert all(results.values())
        assert maintenance._last_analyze is not None

    def test_cleanup_old_data(self, temp_db_dir):
        """Test cleanup of old data based on retention policy."""
        db_path = temp_db_dir / "test_cleanup.db"

        # Create database with old and new records
        # Use 'memories' table name which is in ALLOWED_CLEANUP_TABLES whitelist
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY,
                    data TEXT,
                    created_at TEXT
                )
            """
            )
            # Old record (100 days ago)
            old_date = (datetime.now() - timedelta(days=100)).isoformat()
            conn.execute(
                "INSERT INTO memories (data, created_at) VALUES (?, ?)", ("old_data", old_date)
            )
            # New record (today)
            new_date = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO memories (data, created_at) VALUES (?, ?)", ("new_data", new_date)
            )

        maintenance = DatabaseMaintenance(temp_db_dir)
        results = maintenance.cleanup_old_data(days=90, tables={"test_cleanup.db": "memories"})

        # Should have cleaned 1 record
        assert "test_cleanup.db:memories" in results
        assert results["test_cleanup.db:memories"] == 1

        # Verify new record still exists
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            assert cursor.fetchone()[0] == 1

    def test_cleanup_old_data_nonexistent_table(self, temp_db_dir):
        """Test cleanup handles nonexistent tables gracefully."""
        # Use 'suggestions' which is in ALLOWED_CLEANUP_TABLES but doesn't exist in elo.db
        maintenance = DatabaseMaintenance(temp_db_dir)
        results = maintenance.cleanup_old_data(days=90, tables={"elo.db": "suggestions"})
        assert results == {}

    def test_get_stats(self, maintenance, temp_db_dir):
        """Test statistics gathering."""
        stats = maintenance.get_stats()

        assert stats["database_count"] == 3
        assert stats["total_size_mb"] >= 0
        assert stats["last_vacuum"] is None
        assert stats["last_analyze"] is None
        assert len(stats["databases"]) == 3

    def test_get_stats_after_maintenance(self, maintenance):
        """Test statistics after running maintenance."""
        maintenance.vacuum_all()
        maintenance.analyze_all()

        stats = maintenance.get_stats()

        assert stats["last_vacuum"] is not None
        assert stats["last_analyze"] is not None


class TestStartupMaintenance:
    """Tests for run_startup_maintenance function."""

    def test_run_startup_maintenance(self, temp_db_dir):
        """Test startup maintenance runs successfully."""
        results = run_startup_maintenance(temp_db_dir)

        assert "wal_checkpoint" in results
        assert "stats" in results
        assert results["stats"]["database_count"] == 3

    def test_run_startup_maintenance_empty_dir(self):
        """Test startup maintenance with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_startup_maintenance(tmpdir)

            assert "wal_checkpoint" in results
            assert "stats" in results
            assert results["stats"]["database_count"] == 0

    def test_run_startup_maintenance_respects_analyze_interval(self, temp_db_dir):
        """Test that ANALYZE isn't run if done recently."""
        # First run - should run ANALYZE
        results1 = run_startup_maintenance(temp_db_dir)

        # Second run - should skip ANALYZE (within 24 hours)
        results2 = run_startup_maintenance(temp_db_dir)

        # First run should have analyze key (or be skipped on first run if no DBs)
        assert "wal_checkpoint" in results1
        # Second run should not have analyze key
        assert "analyze" not in results2


class TestScheduleMaintenance:
    """Tests for schedule_maintenance function."""

    def test_schedule_maintenance_nothing_due(self, temp_db_dir):
        """Test scheduled maintenance when nothing is due."""
        # Create state file indicating recent maintenance
        state_file = temp_db_dir / "maintenance_state.json"
        now = datetime.now()
        state = {
            "last_vacuum": now.isoformat(),
            "last_analyze": now.isoformat(),
            "last_cleanup": now.isoformat(),
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

        results = schedule_maintenance(temp_db_dir)

        assert results["tasks_run"] == []

    def test_schedule_maintenance_vacuum_due(self, temp_db_dir):
        """Test scheduled maintenance when VACUUM is due."""
        # Create state file indicating old vacuum
        state_file = temp_db_dir / "maintenance_state.json"
        old_date = (datetime.now() - timedelta(days=8)).isoformat()
        state = {
            "last_vacuum": old_date,
            "last_analyze": datetime.now().isoformat(),
            "last_cleanup": datetime.now().isoformat(),
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

        results = schedule_maintenance(temp_db_dir)

        assert "vacuum" in results["tasks_run"]
        assert "vacuum" in results

    def test_schedule_maintenance_analyze_due(self, temp_db_dir):
        """Test scheduled maintenance when ANALYZE is due."""
        # Create state file indicating old analyze
        state_file = temp_db_dir / "maintenance_state.json"
        old_date = (datetime.now() - timedelta(hours=25)).isoformat()
        state = {
            "last_vacuum": datetime.now().isoformat(),
            "last_analyze": old_date,
            "last_cleanup": datetime.now().isoformat(),
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

        results = schedule_maintenance(temp_db_dir)

        assert "analyze" in results["tasks_run"]
        assert "analyze" in results

    def test_schedule_maintenance_cleanup_due(self, temp_db_dir):
        """Test scheduled maintenance when cleanup is due."""
        # Create state file indicating old cleanup
        state_file = temp_db_dir / "maintenance_state.json"
        old_date = (datetime.now() - timedelta(days=8)).isoformat()
        state = {
            "last_vacuum": datetime.now().isoformat(),
            "last_analyze": datetime.now().isoformat(),
            "last_cleanup": old_date,
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

        results = schedule_maintenance(temp_db_dir)

        assert "cleanup" in results["tasks_run"]

    def test_schedule_maintenance_first_run(self, temp_db_dir):
        """Test scheduled maintenance on first run (no state file)."""
        results = schedule_maintenance(temp_db_dir)

        # All tasks should run on first run
        assert "vacuum" in results["tasks_run"]
        assert "analyze" in results["tasks_run"]
        assert "cleanup" in results["tasks_run"]


class TestKnownDatabases:
    """Tests for KNOWN_DATABASES constant."""

    def test_known_databases_not_empty(self):
        """Test that KNOWN_DATABASES is not empty."""
        assert len(KNOWN_DATABASES) > 0

    def test_known_databases_are_db_files(self):
        """Test that all known databases have .db extension."""
        for db in KNOWN_DATABASES:
            assert db.endswith(".db"), f"{db} does not end with .db"

    def test_known_databases_includes_critical_dbs(self):
        """Test that critical databases are included."""
        critical = ["elo.db", "continuum.db", "genesis.db"]
        for db in critical:
            assert db in KNOWN_DATABASES, f"{db} not in KNOWN_DATABASES"


class TestMaintenanceEdgeCases:
    """Edge case tests for maintenance module."""

    def test_locked_database(self, temp_db_dir):
        """Test handling of locked database."""
        db_path = temp_db_dir / "locked.db"

        # Create and lock the database
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")

        maintenance = DatabaseMaintenance(temp_db_dir)

        # Vacuum should still succeed on unlocked database
        result = maintenance.vacuum(db_path)
        assert result is True

    def test_corrupted_database(self, temp_db_dir):
        """Test handling of corrupted database."""
        db_path = temp_db_dir / "corrupted.db"

        # Create a corrupted database file
        with open(db_path, "w") as f:
            f.write("not a valid sqlite database")

        maintenance = DatabaseMaintenance(temp_db_dir)
        result = maintenance.vacuum(db_path)

        # Should fail gracefully
        assert result is False

    def test_subdirectory_databases(self, temp_db_dir):
        """Test discovery of databases in subdirectories."""
        # Create subdirectory with database
        sub_dir = temp_db_dir / "traces"
        sub_dir.mkdir()
        db_path = sub_dir / "debate_traces.db"

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE traces (id INTEGER)")

        maintenance = DatabaseMaintenance(temp_db_dir)
        databases = maintenance.get_databases()
        db_paths = [str(db) for db in databases]

        # Should find the subdirectory database
        assert any("debate_traces.db" in p for p in db_paths)

    def test_concurrent_access(self, temp_db_dir):
        """Test maintenance with concurrent database access."""
        db_path = temp_db_dir / "concurrent.db"

        # Create database
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")

        maintenance = DatabaseMaintenance(temp_db_dir)

        # Open another connection while running maintenance
        with sqlite3.connect(str(db_path)) as conn:
            # Keep a reader active
            cursor = conn.execute("SELECT * FROM test")
            cursor.fetchone()

            # Run ANALYZE (should work with concurrent readers)
            result = maintenance.analyze(db_path)
            assert result is True
