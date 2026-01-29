"""Tests for aragora.maintenance.db_maintenance module."""

import sqlite3
from pathlib import Path

import pytest

from aragora.maintenance.db_maintenance import (
    ALLOWED_CLEANUP_TABLES,
    ALLOWED_TIMESTAMP_COLUMNS,
    DEFAULT_DB_DIR,
    KNOWN_DATABASES,
    DatabaseMaintenance,
    run_startup_maintenance,
)


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public symbols are importable."""

    def test_database_maintenance_class(self):
        assert DatabaseMaintenance is not None

    def test_run_startup_maintenance_function(self):
        assert callable(run_startup_maintenance)

    def test_module_constants(self):
        assert isinstance(KNOWN_DATABASES, list)
        assert len(KNOWN_DATABASES) > 0
        assert isinstance(ALLOWED_CLEANUP_TABLES, frozenset)
        assert isinstance(ALLOWED_TIMESTAMP_COLUMNS, frozenset)
        assert isinstance(DEFAULT_DB_DIR, Path)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """DatabaseMaintenance can be created with a tmp_path."""

    def test_init_with_path(self, tmp_path: Path):
        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.db_dir == tmp_path
        assert dm._last_vacuum is None
        assert dm._last_analyze is None

    def test_init_with_string(self, tmp_path: Path):
        dm = DatabaseMaintenance(db_dir=str(tmp_path))
        assert dm.db_dir == tmp_path

    def test_init_default(self):
        dm = DatabaseMaintenance()
        assert dm.db_dir == DEFAULT_DB_DIR


# ---------------------------------------------------------------------------
# get_databases
# ---------------------------------------------------------------------------


class TestGetDatabases:
    """get_databases returns a list of existing .db paths."""

    def test_empty_directory(self, tmp_path: Path):
        dm = DatabaseMaintenance(db_dir=tmp_path)
        result = dm.get_databases()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_discovers_known_databases(self, tmp_path: Path):
        # Create a known database file
        db_file = tmp_path / "elo.db"
        db_file.touch()
        dm = DatabaseMaintenance(db_dir=tmp_path)
        result = dm.get_databases()
        assert db_file in result

    def test_discovers_unknown_databases(self, tmp_path: Path):
        # Create a .db file not in KNOWN_DATABASES
        db_file = tmp_path / "custom.db"
        db_file.touch()
        dm = DatabaseMaintenance(db_dir=tmp_path)
        result = dm.get_databases()
        assert db_file in result

    def test_no_duplicates(self, tmp_path: Path):
        # A known database should appear only once
        db_file = tmp_path / "elo.db"
        db_file.touch()
        dm = DatabaseMaintenance(db_dir=tmp_path)
        result = dm.get_databases()
        assert result.count(db_file) == 1


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """get_stats returns a well-structured dictionary."""

    def test_stats_structure_empty(self, tmp_path: Path):
        dm = DatabaseMaintenance(db_dir=tmp_path)
        stats = dm.get_stats()
        assert stats["database_count"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["last_vacuum"] is None
        assert stats["last_analyze"] is None
        assert stats["databases"] == []

    def test_stats_with_database(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        # Create a real SQLite database so it has non-zero size
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

        dm = DatabaseMaintenance(db_dir=tmp_path)
        stats = dm.get_stats()
        assert stats["database_count"] == 1
        assert stats["total_size_mb"] >= 0
        assert "test.db" in stats["databases"]


# ---------------------------------------------------------------------------
# checkpoint_wal
# ---------------------------------------------------------------------------


class TestCheckpointWal:
    """checkpoint_wal flushes WAL on an empty database."""

    def test_checkpoint_existing_db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.checkpoint_wal(db_path) is True

    def test_checkpoint_missing_db(self, tmp_path: Path):
        db_path = tmp_path / "nonexistent.db"
        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.checkpoint_wal(db_path) is False


# ---------------------------------------------------------------------------
# vacuum
# ---------------------------------------------------------------------------


class TestVacuum:
    """vacuum reclaims space on an empty database."""

    def test_vacuum_existing_db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.vacuum(db_path) is True

    def test_vacuum_missing_db(self, tmp_path: Path):
        db_path = tmp_path / "nonexistent.db"
        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.vacuum(db_path) is False


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    """analyze updates optimizer statistics on an empty database."""

    def test_analyze_existing_db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.analyze(db_path) is True

    def test_analyze_missing_db(self, tmp_path: Path):
        db_path = tmp_path / "nonexistent.db"
        dm = DatabaseMaintenance(db_dir=tmp_path)
        assert dm.analyze(db_path) is False


# ---------------------------------------------------------------------------
# run_startup_maintenance
# ---------------------------------------------------------------------------


class TestRunStartupMaintenance:
    """run_startup_maintenance performs WAL checkpoint and optional ANALYZE."""

    def test_on_empty_directory(self, tmp_path: Path):
        result = run_startup_maintenance(db_dir=tmp_path)
        assert "wal_checkpoint" in result
        assert "stats" in result
        assert isinstance(result["wal_checkpoint"], dict)
        assert isinstance(result["stats"], dict)

    def test_with_database(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

        result = run_startup_maintenance(db_dir=tmp_path)
        assert result["stats"]["database_count"] == 1
        # First run should trigger ANALYZE (no prior state file)
        assert "analyze" in result

    def test_writes_state_file(self, tmp_path: Path):
        run_startup_maintenance(db_dir=tmp_path)
        state_file = tmp_path / "maintenance_state.json"
        assert state_file.exists()
