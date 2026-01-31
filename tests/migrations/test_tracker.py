"""
Tests for aragora.migrations.tracker module.

Covers:
- MigrationTracker initialization and table creation
- Marking migrations as applied/rolled back
- Version tracking (applied, pending)
- Checksum verification
- Rollback SQL storage and retrieval
- Status reporting

Run with:
    python -m pytest tests/migrations/test_tracker.py -v --noconftest --timeout=30
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Helpers: lightweight in-memory SQLite backend that satisfies DatabaseBackend
# ---------------------------------------------------------------------------


class InMemorySQLiteBackend:
    """
    Minimal DatabaseBackend implementation backed by an in-memory SQLite
    database for testing the MigrationTracker.
    """

    backend_type = "sqlite"

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchall()

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    """Provide a fresh in-memory SQLite backend per test."""
    b = InMemorySQLiteBackend()
    yield b
    b.close()


@pytest.fixture()
def tracker(backend):
    """Provide a MigrationTracker wired to the in-memory backend."""
    from aragora.migrations.tracker import MigrationTracker

    return MigrationTracker(backend)


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify the tracker module and its public API can be imported."""

    def test_import_tracker_module(self):
        import aragora.migrations.tracker as mod

        assert hasattr(mod, "MigrationTracker")
        assert hasattr(mod, "AppliedMigration")
        assert hasattr(mod, "compute_migration_checksum")

    def test_import_from_package(self):
        from aragora.migrations import (
            AppliedMigration,
            MigrationTracker,
            compute_migration_checksum,
        )

        assert MigrationTracker is not None
        assert AppliedMigration is not None
        assert callable(compute_migration_checksum)


# ---------------------------------------------------------------------------
# MigrationTracker initialization
# ---------------------------------------------------------------------------


class TestTrackerInit:
    """Tests for tracker initialization and table creation."""

    def test_creates_table_on_first_use(self, backend, tracker):
        # First operation triggers table creation
        tracker.is_applied("v1")

        # Check table exists
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("schema_migrations",),
        )
        assert len(rows) == 1

    def test_table_has_expected_columns(self, backend, tracker):
        tracker.is_applied("v1")

        cursor = backend._conn.execute("PRAGMA table_info(schema_migrations)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "version" in cols
        assert "name" in cols
        assert "applied_at" in cols
        assert "applied_by" in cols
        assert "checksum" in cols
        assert "rollback_sql" in cols

    def test_idempotent_table_creation(self, backend, tracker):
        # Multiple operations should not fail
        tracker.is_applied("v1")
        tracker.is_applied("v2")
        tracker.get_applied_versions()


# ---------------------------------------------------------------------------
# is_applied tests
# ---------------------------------------------------------------------------


class TestIsApplied:
    """Tests for the is_applied method."""

    def test_returns_false_when_not_applied(self, tracker):
        assert tracker.is_applied("v1") is False

    def test_returns_true_after_mark_applied(self, tracker):
        tracker.mark_applied("v1", "First migration")
        assert tracker.is_applied("v1") is True

    def test_returns_false_for_different_version(self, tracker):
        tracker.mark_applied("v1", "First migration")
        assert tracker.is_applied("v2") is False


# ---------------------------------------------------------------------------
# mark_applied tests
# ---------------------------------------------------------------------------


class TestMarkApplied:
    """Tests for the mark_applied method."""

    def test_mark_applied_basic(self, tracker, backend):
        tracker.mark_applied("20240101000000", "Initial schema")

        rows = backend.fetch_all(
            "SELECT version, name FROM schema_migrations WHERE version = ?",
            ("20240101000000",),
        )
        assert len(rows) == 1
        assert rows[0][0] == "20240101000000"
        assert rows[0][1] == "Initial schema"

    def test_mark_applied_with_checksum(self, tracker, backend):
        tracker.mark_applied("v1", "Test", checksum="abc123")

        rows = backend.fetch_all(
            "SELECT checksum FROM schema_migrations WHERE version = ?",
            ("v1",),
        )
        assert rows[0][0] == "abc123"

    def test_mark_applied_with_rollback_sql(self, tracker, backend):
        tracker.mark_applied("v1", "Test", rollback_sql="DROP TABLE test")

        rows = backend.fetch_all(
            "SELECT rollback_sql FROM schema_migrations WHERE version = ?",
            ("v1",),
        )
        assert rows[0][0] == "DROP TABLE test"

    def test_mark_applied_with_all_metadata(self, tracker, backend):
        tracker.mark_applied(
            version="v1",
            name="Full test",
            checksum="sha256hash",
            rollback_sql="DROP TABLE t1; DROP TABLE t2",
        )

        row = backend.fetch_one(
            "SELECT version, name, checksum, rollback_sql, applied_by "
            "FROM schema_migrations WHERE version = ?",
            ("v1",),
        )
        assert row[0] == "v1"
        assert row[1] == "Full test"
        assert row[2] == "sha256hash"
        assert row[3] == "DROP TABLE t1; DROP TABLE t2"
        assert row[4] is not None  # applied_by should be set

    def test_mark_applied_duplicate_raises(self, tracker):
        tracker.mark_applied("v1", "First")

        with pytest.raises(ValueError, match="already applied"):
            tracker.mark_applied("v1", "Duplicate")

    def test_mark_applied_sets_applied_by(self, tracker, backend):
        import os

        tracker.mark_applied("v1", "Test")

        row = backend.fetch_one(
            "SELECT applied_by FROM schema_migrations WHERE version = ?",
            ("v1",),
        )
        applied_by = row[0]
        assert applied_by is not None
        assert ":" in applied_by
        assert str(os.getpid()) in applied_by


# ---------------------------------------------------------------------------
# mark_rolled_back tests
# ---------------------------------------------------------------------------


class TestMarkRolledBack:
    """Tests for the mark_rolled_back method."""

    def test_mark_rolled_back_removes_record(self, tracker):
        tracker.mark_applied("v1", "Test")
        assert tracker.is_applied("v1") is True

        tracker.mark_rolled_back("v1")
        assert tracker.is_applied("v1") is False

    def test_mark_rolled_back_not_applied_raises(self, tracker):
        with pytest.raises(ValueError, match="not applied"):
            tracker.mark_rolled_back("v1")

    def test_mark_rolled_back_idempotent_on_reapply(self, tracker):
        tracker.mark_applied("v1", "Test")
        tracker.mark_rolled_back("v1")

        # Can reapply after rollback
        tracker.mark_applied("v1", "Test reapplied")
        assert tracker.is_applied("v1") is True


# ---------------------------------------------------------------------------
# get_applied_versions tests
# ---------------------------------------------------------------------------


class TestGetAppliedVersions:
    """Tests for the get_applied_versions method."""

    def test_empty_when_none_applied(self, tracker):
        assert tracker.get_applied_versions() == []

    def test_returns_applied_versions(self, tracker):
        tracker.mark_applied("v1", "First")
        tracker.mark_applied("v2", "Second")
        tracker.mark_applied("v3", "Third")

        versions = tracker.get_applied_versions()
        assert len(versions) == 3
        assert "v1" in versions
        assert "v2" in versions
        assert "v3" in versions

    def test_order_by_applied_at(self, tracker):
        tracker.mark_applied("v2", "Second")
        tracker.mark_applied("v1", "First")
        tracker.mark_applied("v3", "Third")

        versions = tracker.get_applied_versions()
        # Should be in order of application, not version number
        assert versions[0] == "v2"
        assert versions[1] == "v1"
        assert versions[2] == "v3"


# ---------------------------------------------------------------------------
# get_applied_migrations tests
# ---------------------------------------------------------------------------


class TestGetAppliedMigrations:
    """Tests for the get_applied_migrations method."""

    def test_returns_applied_migration_objects(self, tracker):
        tracker.mark_applied("v1", "First", checksum="abc", rollback_sql="DROP TABLE t")

        migrations = tracker.get_applied_migrations()
        assert len(migrations) == 1

        m = migrations[0]
        assert m.version == "v1"
        assert m.name == "First"
        assert m.checksum == "abc"
        assert m.rollback_sql == "DROP TABLE t"
        assert m.applied_by is not None
        assert isinstance(m.applied_at, datetime)


# ---------------------------------------------------------------------------
# get_pending_migrations tests
# ---------------------------------------------------------------------------


class TestGetPendingMigrations:
    """Tests for the get_pending_migrations method."""

    def test_all_pending_when_none_applied(self, tracker):
        available = ["v1", "v2", "v3"]
        pending = tracker.get_pending_migrations(available)
        assert pending == ["v1", "v2", "v3"]

    def test_excludes_applied(self, tracker):
        tracker.mark_applied("v1", "First")
        tracker.mark_applied("v3", "Third")

        available = ["v1", "v2", "v3", "v4"]
        pending = tracker.get_pending_migrations(available)
        assert pending == ["v2", "v4"]

    def test_preserves_order(self, tracker):
        available = ["v3", "v1", "v2"]
        pending = tracker.get_pending_migrations(available)
        assert pending == ["v3", "v1", "v2"]

    def test_empty_when_all_applied(self, tracker):
        tracker.mark_applied("v1", "First")
        tracker.mark_applied("v2", "Second")

        available = ["v1", "v2"]
        pending = tracker.get_pending_migrations(available)
        assert pending == []


# ---------------------------------------------------------------------------
# get_migration tests
# ---------------------------------------------------------------------------


class TestGetMigration:
    """Tests for the get_migration method."""

    def test_returns_none_when_not_found(self, tracker):
        assert tracker.get_migration("v999") is None

    def test_returns_migration_details(self, tracker):
        tracker.mark_applied("v1", "Test", checksum="hash123", rollback_sql="DROP t")

        m = tracker.get_migration("v1")
        assert m is not None
        assert m.version == "v1"
        assert m.name == "Test"
        assert m.checksum == "hash123"
        assert m.rollback_sql == "DROP t"


# ---------------------------------------------------------------------------
# get_rollback_sql tests
# ---------------------------------------------------------------------------


class TestGetRollbackSql:
    """Tests for the get_rollback_sql method."""

    def test_returns_none_when_not_found(self, tracker):
        assert tracker.get_rollback_sql("v1") is None

    def test_returns_none_when_no_rollback_stored(self, tracker):
        tracker.mark_applied("v1", "Test")
        assert tracker.get_rollback_sql("v1") is None

    def test_returns_stored_rollback_sql(self, tracker):
        tracker.mark_applied("v1", "Test", rollback_sql="DROP TABLE users")
        assert tracker.get_rollback_sql("v1") == "DROP TABLE users"

    def test_returns_multi_statement_rollback(self, tracker):
        sql = "DROP TABLE a; DROP TABLE b; DROP INDEX idx"
        tracker.mark_applied("v1", "Test", rollback_sql=sql)
        assert tracker.get_rollback_sql("v1") == sql


# ---------------------------------------------------------------------------
# verify_checksum tests
# ---------------------------------------------------------------------------


class TestVerifyChecksum:
    """Tests for the verify_checksum method."""

    def test_returns_true_when_not_applied(self, tracker):
        # Migration not applied, nothing to verify
        assert tracker.verify_checksum("v1", "any_checksum") is True

    def test_returns_true_when_no_stored_checksum(self, tracker):
        tracker.mark_applied("v1", "Test")  # No checksum
        assert tracker.verify_checksum("v1", "any_checksum") is True

    def test_returns_true_when_checksums_match(self, tracker):
        tracker.mark_applied("v1", "Test", checksum="abc123")
        assert tracker.verify_checksum("v1", "abc123") is True

    def test_returns_false_when_checksums_mismatch(self, tracker):
        tracker.mark_applied("v1", "Test", checksum="abc123")
        assert tracker.verify_checksum("v1", "different_hash") is False


# ---------------------------------------------------------------------------
# get_checksum_mismatches tests
# ---------------------------------------------------------------------------


class TestGetChecksumMismatches:
    """Tests for the get_checksum_mismatches method."""

    def test_returns_empty_when_no_mismatches(self, tracker):
        tracker.mark_applied("v1", "Test", checksum="hash1")
        tracker.mark_applied("v2", "Test2", checksum="hash2")

        mismatches = tracker.get_checksum_mismatches({"v1": "hash1", "v2": "hash2"})
        assert mismatches == []

    def test_returns_mismatches(self, tracker):
        tracker.mark_applied("v1", "Test", checksum="stored_hash")

        mismatches = tracker.get_checksum_mismatches({"v1": "current_hash"})
        assert len(mismatches) == 1
        assert mismatches[0] == ("v1", "stored_hash", "current_hash")

    def test_ignores_null_checksums(self, tracker):
        tracker.mark_applied("v1", "Test")  # No checksum stored

        mismatches = tracker.get_checksum_mismatches({"v1": "any_hash"})
        assert mismatches == []

    def test_ignores_unapplied_migrations(self, tracker):
        tracker.mark_applied("v1", "Test", checksum="hash1")

        # v2 not applied, should be ignored
        mismatches = tracker.get_checksum_mismatches({"v1": "hash1", "v2": "hash2"})
        assert mismatches == []


# ---------------------------------------------------------------------------
# status tests
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for the status method."""

    def test_status_empty(self, tracker):
        status = tracker.status()
        assert status["table_exists"] is True
        assert status["applied_count"] == 0
        assert status["applied_versions"] == []
        assert status["latest_version"] is None

    def test_status_with_applied(self, tracker):
        tracker.mark_applied("v1", "First")
        tracker.mark_applied("v2", "Second")

        status = tracker.status()
        assert status["applied_count"] == 2
        assert "v1" in status["applied_versions"]
        assert "v2" in status["applied_versions"]
        assert status["latest_version"] == "v2"


# ---------------------------------------------------------------------------
# compute_migration_checksum tests
# ---------------------------------------------------------------------------


class TestComputeMigrationChecksum:
    """Tests for the compute_migration_checksum helper function."""

    def test_computes_sha256(self):
        from aragora.migrations.tracker import compute_migration_checksum

        result = compute_migration_checksum("CREATE TABLE users (id INT)")
        assert len(result) == 64  # SHA-256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        from aragora.migrations.tracker import compute_migration_checksum

        content = "SELECT * FROM users"
        result1 = compute_migration_checksum(content)
        result2 = compute_migration_checksum(content)
        assert result1 == result2

    def test_different_content_different_hash(self):
        from aragora.migrations.tracker import compute_migration_checksum

        hash1 = compute_migration_checksum("CREATE TABLE a (id INT)")
        hash2 = compute_migration_checksum("CREATE TABLE b (id INT)")
        assert hash1 != hash2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end tests for the tracker."""

    def test_full_lifecycle(self, tracker):
        # Initially nothing applied
        assert tracker.get_applied_versions() == []
        assert tracker.get_pending_migrations(["v1", "v2", "v3"]) == ["v1", "v2", "v3"]

        # Apply first migration
        tracker.mark_applied(
            "v1",
            "Create users table",
            checksum="hash1",
            rollback_sql="DROP TABLE users",
        )
        assert tracker.is_applied("v1")
        assert tracker.get_pending_migrations(["v1", "v2", "v3"]) == ["v2", "v3"]

        # Apply second migration
        tracker.mark_applied(
            "v2",
            "Add posts table",
            checksum="hash2",
            rollback_sql="DROP TABLE posts",
        )
        assert tracker.get_applied_versions() == ["v1", "v2"]

        # Verify checksums
        assert tracker.verify_checksum("v1", "hash1")
        assert not tracker.verify_checksum("v1", "wrong_hash")

        # Get rollback SQL
        assert tracker.get_rollback_sql("v1") == "DROP TABLE users"

        # Rollback
        tracker.mark_rolled_back("v2")
        assert not tracker.is_applied("v2")
        assert tracker.get_applied_versions() == ["v1"]

        # Reapply
        tracker.mark_applied("v2", "Add posts table (reapplied)", checksum="hash2_v2")
        assert tracker.is_applied("v2")

    def test_with_migration_runner(self, backend):
        """Test that tracker table doesn't conflict with runner table."""
        from aragora.migrations.runner import MigrationRunner
        from aragora.migrations.tracker import MigrationTracker

        # Both can coexist
        runner = MigrationRunner(backend=backend)
        tracker = MigrationTracker(backend)

        # Each has its own table
        runner_tables = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("_aragora_migrations",),
        )
        assert len(runner_tables) == 1

        tracker.is_applied("v1")
        tracker_tables = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("schema_migrations",),
        )
        assert len(tracker_tables) == 1
