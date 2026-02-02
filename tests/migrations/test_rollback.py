"""
Comprehensive tests for migration rollback functionality.

Covers:
- Single migration rollback
- Multi-step rollback (rollback_steps)
- Rollback to specific version (downgrade with target_version)
- Rollback ordering (reverse of apply order)
- Partial rollback (when some migrations lack down_fn)
- Rollback validation (pre-flight checks)
- Rollback history tracking
- Dry-run rollback
- Error handling during rollback
- Stored rollback SQL usage

Run with:
    python -m pytest tests/migrations/test_rollback.py -v --timeout=30
"""

from __future__ import annotations

import sqlite3
from typing import Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers: lightweight in-memory SQLite backend
# ---------------------------------------------------------------------------


class InMemorySQLiteBackend:
    """
    Minimal DatabaseBackend implementation backed by an in-memory SQLite
    database for testing.
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

    def connection(self):
        return self._conn


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
def runner(backend):
    """Provide a MigrationRunner wired to the in-memory backend."""
    from aragora.migrations.runner import MigrationRunner

    return MigrationRunner(backend=backend)


def make_migration(version, name, up_sql="SELECT 1", down_sql=None, down_fn=None):
    """Helper to create Migration instances."""
    from aragora.migrations.runner import Migration

    return Migration(
        version=version,
        name=name,
        up_sql=up_sql,
        down_sql=down_sql,
        down_fn=down_fn,
    )


# ---------------------------------------------------------------------------
# Test: Single Migration Rollback
# ---------------------------------------------------------------------------


class TestSingleMigrationRollback:
    """Tests for rolling back a single migration."""

    def test_rollback_single_migration_with_down_sql(self, runner, backend):
        """Rolling back a single migration should execute down_sql."""
        runner.register(
            make_migration(
                1,
                "create_users",
                up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY)",
                down_sql="DROP TABLE users",
            )
        )
        runner.upgrade()

        # Verify table exists
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert len(rows) == 1

        # Rollback
        rolled = runner.downgrade()
        assert len(rolled) == 1
        assert rolled[0].version == 1

        # Table should be gone
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert len(rows) == 0

    def test_rollback_single_migration_with_down_fn(self, runner, backend):
        """Rolling back should execute down_fn when provided."""
        from aragora.migrations.runner import Migration

        call_log = []

        def up(be):
            be.execute_write("CREATE TABLE fn_test (id INTEGER PRIMARY KEY)")

        def down(be):
            call_log.append("down_called")
            be.execute_write("DROP TABLE fn_test")

        m = Migration(version=1, name="fn-migration", up_fn=up, down_fn=down)
        runner.register(m)
        runner.upgrade()
        runner.downgrade()

        assert "down_called" in call_log
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fn_test'"
        )
        assert len(rows) == 0

    def test_rollback_removes_tracking_record(self, runner, backend):
        """Rollback should remove the migration from the tracking table."""
        runner.register(make_migration(1, "tracked", down_sql="SELECT 1"))
        runner.upgrade()
        assert 1 in runner.get_applied_versions()

        runner.downgrade()
        assert 1 not in runner.get_applied_versions()

    def test_rollback_without_down_stops(self, runner, backend):
        """Migrations without down_sql/down_fn cannot be rolled back."""
        runner.register(make_migration(1, "no-rollback"))
        runner.upgrade()

        rolled = runner.downgrade()
        assert len(rolled) == 0
        assert 1 in runner.get_applied_versions()


# ---------------------------------------------------------------------------
# Test: Multiple Migration Rollback
# ---------------------------------------------------------------------------


class TestMultipleMigrationRollback:
    """Tests for rolling back multiple migrations."""

    def test_rollback_multiple_with_target_version(self, runner, backend):
        """Rollback to a target version should rollback all migrations above it."""
        runner.register(make_migration(1, "first", down_sql="SELECT 1"))
        runner.register(make_migration(2, "second", down_sql="SELECT 1"))
        runner.register(make_migration(3, "third", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.downgrade(target_version=1)
        assert len(rolled) == 2
        versions = {m.version for m in rolled}
        assert versions == {2, 3}
        assert runner.get_applied_versions() == {1}

    def test_rollback_steps_one(self, runner, backend):
        """rollback_steps(1) should rollback the most recent migration."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(make_migration(2, "b", down_sql="SELECT 1"))
        runner.register(make_migration(3, "c", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.rollback_steps(steps=1)
        assert len(rolled) == 1
        assert rolled[0].version == 3
        assert runner.get_applied_versions() == {1, 2}

    def test_rollback_steps_multiple(self, runner, backend):
        """rollback_steps(N) should rollback N migrations in reverse order."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(make_migration(2, "b", down_sql="SELECT 1"))
        runner.register(make_migration(3, "c", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.rollback_steps(steps=2)
        assert len(rolled) == 2
        versions = [m.version for m in rolled]
        # Should be in reverse order (3, 2)
        assert versions == [3, 2]
        assert runner.get_applied_versions() == {1}

    def test_rollback_all_migrations(self, runner, backend):
        """Rolling back all migrations should leave no applied migrations."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(make_migration(2, "b", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.downgrade(target_version=0)
        assert len(rolled) == 2
        assert runner.get_applied_versions() == set()


# ---------------------------------------------------------------------------
# Test: Rollback Ordering
# ---------------------------------------------------------------------------


class TestRollbackOrdering:
    """Tests to verify rollback happens in reverse order of application."""

    def test_rollback_executes_in_reverse_order(self, runner, backend):
        """Rollback should execute migrations in reverse version order."""
        execution_order = []

        def make_tracking_migration(version):
            from aragora.migrations.runner import Migration

            def down(be):
                execution_order.append(version)

            return Migration(
                version=version,
                name=f"m{version}",
                up_sql="SELECT 1",
                down_fn=down,
            )

        runner.register(make_tracking_migration(1))
        runner.register(make_tracking_migration(2))
        runner.register(make_tracking_migration(3))
        runner.upgrade()

        runner.downgrade(target_version=0)

        # Should execute in reverse order: 3, 2, 1
        assert execution_order == [3, 2, 1]

    def test_rollback_with_non_sequential_versions(self, runner, backend):
        """Rollback works correctly with non-sequential version numbers."""
        runner.register(make_migration(100, "first", down_sql="SELECT 1"))
        runner.register(make_migration(250, "second", down_sql="SELECT 1"))
        runner.register(make_migration(500, "third", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.downgrade(target_version=100)
        versions = [m.version for m in rolled]
        assert versions == [500, 250]


# ---------------------------------------------------------------------------
# Test: Partial Rollback (some migrations lack down_fn)
# ---------------------------------------------------------------------------


class TestPartialRollback:
    """Tests for partial rollback when some migrations lack rollback support."""

    def test_rollback_stops_at_migration_without_down(self, runner, backend):
        """Rollback should stop when it encounters a migration without down."""
        runner.register(make_migration(1, "no-down"))  # No down_sql or down_fn
        runner.register(make_migration(2, "has-down", down_sql="SELECT 1"))
        runner.upgrade()

        # Start at version 2, try to rollback to 0
        rolled = runner.downgrade(target_version=0)

        # Should only rollback version 2, stop at version 1
        assert len(rolled) == 1
        assert rolled[0].version == 2
        assert runner.get_applied_versions() == {1}

    def test_validate_rollback_reports_missing_down(self, runner, backend):
        """Validation should report migrations without rollback support."""
        runner.register(make_migration(1, "no-down"))
        runner.upgrade()

        validation = runner.validate_rollback()
        assert validation.safe is False
        assert any("no rollback defined" in e for e in validation.errors)


# ---------------------------------------------------------------------------
# Test: Rollback Validation (pre-flight checks)
# ---------------------------------------------------------------------------


class TestRollbackValidation:
    """Tests for pre-rollback validation."""

    def test_validate_with_no_applied_migrations(self, runner):
        """Validation should fail when no migrations are applied."""
        validation = runner.validate_rollback()
        assert validation.safe is False
        assert "No migrations have been applied" in validation.errors[0]

    def test_validate_with_rollback_support(self, runner, backend):
        """Validation should pass when all migrations have rollback support."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.upgrade()

        validation = runner.validate_rollback()
        assert validation.safe is True
        assert len(validation.errors) == 0
        assert 1 in validation.migrations_to_rollback

    def test_validate_target_version_too_high(self, runner, backend):
        """Validation should fail when target >= current version."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.upgrade()

        validation = runner.validate_rollback(target_version=2)
        assert validation.safe is False
        assert any("must be less than" in e for e in validation.errors)

    def test_validate_negative_target_version(self, runner, backend):
        """Validation should fail for negative target version."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.upgrade()

        validation = runner.validate_rollback(target_version=-1)
        assert validation.safe is False
        assert any("must be >= 0" in e for e in validation.errors)

    def test_validate_steps_zero(self, runner, backend):
        """Validation should fail when steps=0."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.upgrade()

        validation = runner.validate_rollback(steps=0)
        assert validation.safe is False
        assert any("must be >= 1" in e for e in validation.errors)

    def test_validate_large_rollback_warning(self, runner, backend):
        """Validation should warn for large rollback operations."""
        for i in range(1, 8):
            runner.register(make_migration(i, f"m{i}", down_sql="SELECT 1"))
        runner.upgrade()

        validation = runner.validate_rollback(target_version=0)
        assert validation.safe is True
        assert any("large operation" in w for w in validation.warnings)


# ---------------------------------------------------------------------------
# Test: Rollback History Tracking
# ---------------------------------------------------------------------------


class TestRollbackHistory:
    """Tests for rollback history tracking."""

    def test_empty_history(self, runner):
        """Empty rollback history should return empty list."""
        history = runner.get_rollback_history()
        assert history == []

    def test_downgrade_records_history(self, runner, backend):
        """Downgrade should record entry in rollback history."""
        runner.register(make_migration(1, "test_migration", down_sql="SELECT 1"))
        runner.upgrade()
        runner.downgrade()

        history = runner.get_rollback_history()
        assert len(history) == 1
        assert history[0].version == 1
        assert history[0].name == "test_migration"
        assert ":" in history[0].rolled_back_by  # hostname:pid format

    def test_rollback_records_reason(self, runner, backend):
        """Reason should be stored in rollback history."""
        runner.register(make_migration(1, "test", down_sql="SELECT 1"))
        runner.upgrade()
        runner.downgrade(reason="Bug discovered in migration")

        history = runner.get_rollback_history()
        assert len(history) == 1
        assert history[0].reason == "Bug discovered in migration"

    def test_multi_step_rollback_records_all(self, runner, backend):
        """Multi-step rollback should record each migration individually."""
        runner.register(make_migration(1, "first", down_sql="SELECT 1"))
        runner.register(make_migration(2, "second", down_sql="SELECT 1"))
        runner.register(make_migration(3, "third", down_sql="SELECT 1"))
        runner.upgrade()

        runner.downgrade(target_version=1, reason="bulk rollback")

        history = runner.get_rollback_history()
        assert len(history) == 2
        versions = {h.version for h in history}
        assert versions == {2, 3}
        for h in history:
            assert h.reason == "bulk rollback"

    def test_history_ordered_most_recent_first(self, runner, backend):
        """History should be ordered most recent first."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(make_migration(2, "b", down_sql="SELECT 1"))
        runner.upgrade()

        runner.downgrade()  # Rollback v2
        runner.upgrade()  # Re-apply v2
        runner.downgrade()  # Rollback v2 again

        history = runner.get_rollback_history()
        assert len(history) == 2
        assert history[0].id > history[1].id  # Most recent first


# ---------------------------------------------------------------------------
# Test: Dry-Run Rollback
# ---------------------------------------------------------------------------


class TestDryRunRollback:
    """Tests for dry-run rollback mode."""

    def test_dry_run_returns_candidates(self, runner, backend):
        """Dry-run should return what would be rolled back."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(make_migration(2, "b", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.downgrade(dry_run=True)
        assert len(rolled) == 1
        assert rolled[0].version == 2

        # Nothing actually rolled back
        assert runner.get_applied_versions() == {1, 2}

    def test_dry_run_with_target_version(self, runner, backend):
        """Dry-run with target_version should preview multi-step rollback."""
        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(make_migration(2, "b", down_sql="SELECT 1"))
        runner.register(make_migration(3, "c", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.downgrade(target_version=1, dry_run=True)
        assert len(rolled) == 2
        versions = {m.version for m in rolled}
        assert versions == {2, 3}

        # Nothing actually rolled back
        assert runner.get_applied_versions() == {1, 2, 3}

    def test_dry_run_does_not_record_history(self, runner, backend):
        """Dry-run should not create rollback history entries."""
        runner.register(make_migration(1, "test", down_sql="SELECT 1"))
        runner.upgrade()
        runner.downgrade(dry_run=True)

        history = runner.get_rollback_history()
        assert len(history) == 0

    def test_dry_run_stops_at_no_rollback(self, runner, backend):
        """Dry-run should indicate when migration has no rollback."""
        runner.register(make_migration(1, "no-down"))
        runner.upgrade()

        rolled = runner.downgrade(dry_run=True)
        assert len(rolled) == 0

    def test_dry_run_does_not_acquire_lock(self, backend):
        """Dry-run should not acquire advisory lock."""
        from aragora.migrations.runner import Migration, MigrationRunner

        runner = MigrationRunner(backend=backend)
        lock_acquired = []

        original_acquire = runner._acquire_migration_lock

        def mock_acquire(timeout_seconds=30.0):
            lock_acquired.append(True)
            return original_acquire(timeout_seconds)

        runner._acquire_migration_lock = mock_acquire

        m = Migration(version=1, name="test", up_sql="SELECT 1", down_sql="SELECT 1")
        runner.register(m)
        runner.upgrade()

        lock_acquired.clear()
        runner.downgrade(dry_run=True)
        assert len(lock_acquired) == 0


# ---------------------------------------------------------------------------
# Test: Error Handling During Rollback
# ---------------------------------------------------------------------------


class TestRollbackErrorHandling:
    """Tests for error handling during rollback."""

    def test_rollback_raises_on_error(self, runner, backend):
        """Rollback should raise when down_fn fails."""
        from aragora.migrations.runner import Migration

        def bad_down(be):
            raise RuntimeError("rollback failed")

        m = Migration(version=1, name="fail", up_sql="SELECT 1", down_fn=bad_down)
        runner.register(m)
        runner.upgrade()

        with pytest.raises(RuntimeError, match="rollback failed"):
            runner.downgrade()

    def test_multi_step_rollback_stops_on_error(self, runner, backend):
        """Multi-step rollback should stop on first error."""
        from aragora.migrations.runner import Migration

        call_count = [0]

        def failing_down(be):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second rollback failed")

        runner.register(make_migration(1, "a", down_sql="SELECT 1"))
        runner.register(Migration(version=2, name="b", up_sql="SELECT 1", down_fn=failing_down))
        runner.register(Migration(version=3, name="c", up_sql="SELECT 1", down_fn=failing_down))
        runner.upgrade()

        with pytest.raises(RuntimeError, match="second rollback failed"):
            runner.downgrade(target_version=0)

        # Version 3 should be rolled back, but 2 and 1 remain
        # Note: due to error, v2 is still applied
        applied = runner.get_applied_versions()
        assert 1 in applied
        assert 2 in applied
        assert 3 not in applied

    def test_invalid_steps_raises_value_error(self, runner):
        """rollback_steps with invalid steps should raise ValueError."""
        with pytest.raises(ValueError, match="steps must be >= 1"):
            runner.rollback_steps(steps=0)

        with pytest.raises(ValueError, match="steps must be >= 1"):
            runner.rollback_steps(steps=-1)


# ---------------------------------------------------------------------------
# Test: Stored Rollback SQL Usage
# ---------------------------------------------------------------------------


class TestStoredRollbackSQL:
    """Tests for using stored rollback SQL from database."""

    def test_stored_rollback_sql_used(self, runner, backend):
        """use_stored_rollback should use SQL from database."""
        # Apply migration with down_sql
        runner.register(
            make_migration(
                1,
                "with_stored_sql",
                up_sql="CREATE TABLE stored_test (id INTEGER)",
                down_sql="DROP TABLE stored_test",
            )
        )
        runner.upgrade()

        # Verify stored SQL
        stored = runner.get_stored_rollback_sql(1)
        assert stored == "DROP TABLE stored_test"

        # Rollback using stored SQL
        rolled = runner.downgrade(use_stored_rollback=True)
        assert len(rolled) == 1

        # Table should be dropped
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stored_test'"
        )
        assert len(rows) == 0

    def test_stored_rollback_sql_fallback(self, runner, backend):
        """Should fallback to migration's down_sql if no stored SQL."""
        from aragora.migrations.runner import Migration

        m = Migration(
            version=1,
            name="no_stored",
            up_sql="SELECT 1",
            down_sql="SELECT 2",  # This will be used
        )
        runner.register(m)
        runner.upgrade()

        # Remove stored SQL to test fallback
        backend.execute_write(
            f"UPDATE {runner.MIGRATIONS_TABLE} SET rollback_sql = NULL WHERE version = 1"
        )

        # Should fallback to migration's down_sql
        rolled = runner.downgrade(use_stored_rollback=True)
        assert len(rolled) == 1


# ---------------------------------------------------------------------------
# Test: End-to-End Rollback Lifecycle
# ---------------------------------------------------------------------------


class TestRollbackLifecycle:
    """End-to-end tests for rollback operations."""

    def test_upgrade_rollback_reapply_cycle(self, backend):
        """Test full upgrade -> rollback -> re-upgrade cycle."""
        from aragora.migrations.runner import Migration, MigrationRunner

        runner = MigrationRunner(backend=backend)

        m1 = Migration(
            version=1,
            name="create users",
            up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY)",
            down_sql="DROP TABLE users",
        )
        m2 = Migration(
            version=2,
            name="create posts",
            up_sql="CREATE TABLE posts (id INTEGER PRIMARY KEY)",
            down_sql="DROP TABLE posts",
        )
        runner.register(m1)
        runner.register(m2)

        # Upgrade
        applied = runner.upgrade()
        assert len(applied) == 2

        # Rollback one
        rolled = runner.downgrade(reason="posts not needed")
        assert len(rolled) == 1
        assert rolled[0].version == 2

        # Verify posts gone
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
        )
        assert len(rows) == 0

        # Users still exists
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert len(rows) == 1

        # Re-apply
        applied = runner.upgrade()
        assert len(applied) == 1
        assert applied[0].version == 2

        # Both tables exist
        for table in ["users", "posts"]:
            rows = backend.fetch_all(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            assert len(rows) == 1

        # Check history
        history = runner.get_rollback_history()
        assert len(history) == 1
        assert history[0].version == 2
        assert history[0].reason == "posts not needed"

    def test_validate_then_execute_rollback(self, backend):
        """Test validation followed by actual rollback."""
        from aragora.migrations.runner import Migration, MigrationRunner

        runner = MigrationRunner(backend=backend)

        for i in range(1, 4):
            runner.register(
                Migration(
                    version=i,
                    name=f"m{i}",
                    up_sql=f"CREATE TABLE t{i} (id INTEGER)",
                    down_sql=f"DROP TABLE t{i}",
                )
            )
        runner.upgrade()

        # Validate
        validation = runner.validate_rollback(steps=2)
        assert validation.safe is True
        assert sorted(validation.migrations_to_rollback) == [2, 3]

        # Execute
        rolled = runner.rollback_steps(steps=2)
        assert len(rolled) == 2
        assert runner.get_applied_versions() == {1}

        # Verify tables
        rows = backend.fetch_all("SELECT name FROM sqlite_master WHERE type='table' AND name='t1'")
        assert len(rows) == 1

        for i in [2, 3]:
            rows = backend.fetch_all(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='t{i}'"
            )
            assert len(rows) == 0


# ---------------------------------------------------------------------------
# Test: Rollback with Timestamp-Based Versions
# ---------------------------------------------------------------------------


class TestTimestampVersionRollback:
    """Tests for rollback with timestamp-based version numbers."""

    def test_rollback_timestamp_versions(self, runner, backend):
        """Rollback should work correctly with timestamp-based versions."""
        runner.register(make_migration(20260101000000, "initial", down_sql="SELECT 1"))
        runner.register(make_migration(20260115120000, "second", down_sql="SELECT 1"))
        runner.register(make_migration(20260201093000, "third", down_sql="SELECT 1"))
        runner.upgrade()

        rolled = runner.downgrade(target_version=20260101000000)
        assert len(rolled) == 2
        versions = {m.version for m in rolled}
        assert versions == {20260115120000, 20260201093000}

        assert runner.get_applied_versions() == {20260101000000}
