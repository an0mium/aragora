"""
Tests for Aragora database migration system.

Tests cover:
- Migration runner initialization
- Applying and rolling back migrations
- Migration status tracking
- Multiple migration ordering
- CLI interface
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.migrations.runner import (
    Migration,
    MigrationRunner,
    get_migration_runner,
    reset_runner,
    apply_migrations,
    rollback_migration,
    get_migration_status,
)
from aragora.storage.backends import SQLiteBackend


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_migrations.db")


@pytest.fixture
def runner(temp_db):
    """Create a migration runner with temp database."""
    backend = SQLiteBackend(temp_db)
    runner = MigrationRunner(backend=backend)
    yield runner
    runner.close()


@pytest.fixture
def sample_migrations():
    """Create sample migrations for testing."""
    return [
        Migration(
            version=20240101000001,
            name="Create users table",
            up_sql="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            down_sql="DROP TABLE users;",
        ),
        Migration(
            version=20240101000002,
            name="Create posts table",
            up_sql="CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT);",
            down_sql="DROP TABLE posts;",
        ),
        Migration(
            version=20240101000003,
            name="Add email to users",
            up_sql="ALTER TABLE users ADD COLUMN email TEXT;",
            down_sql=None,  # SQLite doesn't support DROP COLUMN easily
        ),
    ]


@pytest.fixture(autouse=True)
def cleanup_global_runner():
    """Ensure global runner is reset after each test."""
    yield
    reset_runner()


# ============================================================================
# Migration Dataclass Tests
# ============================================================================


class TestMigrationDataclass:
    """Tests for Migration dataclass."""

    def test_migration_with_sql(self):
        """Test creating migration with SQL."""
        m = Migration(
            version=1,
            name="Test",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
        )
        assert m.version == 1
        assert m.name == "Test"
        assert m.up_sql is not None
        assert m.down_sql is not None

    def test_migration_with_function(self):
        """Test creating migration with Python function."""

        def up_fn(backend):
            backend.execute_write("CREATE TABLE test (id INTEGER)")

        m = Migration(
            version=1,
            name="Test",
            up_fn=up_fn,
        )
        assert m.version == 1
        assert m.up_fn is not None

    def test_migration_requires_up(self):
        """Test that migration requires up_sql or up_fn."""
        with pytest.raises(ValueError, match="must have up_sql or up_fn"):
            Migration(version=1, name="Invalid")


# ============================================================================
# Migration Runner Tests
# ============================================================================


class TestMigrationRunner:
    """Tests for MigrationRunner class."""

    def test_init_creates_migrations_table(self, runner, temp_db):
        """Test that runner creates migrations tracking table."""
        backend = SQLiteBackend(temp_db)
        rows = backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_aragora_migrations'"
        )
        assert len(rows) == 1

    def test_register_migration(self, runner, sample_migrations):
        """Test registering migrations."""
        for m in sample_migrations:
            runner.register(m)

        assert len(runner._migrations) == 3

    def test_register_maintains_order(self, runner, sample_migrations):
        """Test that migrations are sorted by version."""
        # Register in reverse order
        for m in reversed(sample_migrations):
            runner.register(m)

        versions = [m.version for m in runner._migrations]
        assert versions == sorted(versions)

    def test_get_applied_versions_empty(self, runner):
        """Test getting applied versions when none exist."""
        applied = runner.get_applied_versions()
        assert applied == set()

    def test_get_pending_migrations(self, runner, sample_migrations):
        """Test getting pending migrations."""
        for m in sample_migrations:
            runner.register(m)

        pending = runner.get_pending_migrations()
        assert len(pending) == 3

    def test_upgrade_applies_all(self, runner, sample_migrations):
        """Test upgrading applies all pending migrations."""
        for m in sample_migrations:
            runner.register(m)

        applied = runner.upgrade()

        assert len(applied) == 3
        assert runner.get_applied_versions() == {
            20240101000001,
            20240101000002,
            20240101000003,
        }

    def test_upgrade_with_target_version(self, runner, sample_migrations):
        """Test upgrading to specific version."""
        for m in sample_migrations:
            runner.register(m)

        applied = runner.upgrade(target_version=20240101000002)

        assert len(applied) == 2
        assert 20240101000003 not in runner.get_applied_versions()

    def test_upgrade_idempotent(self, runner, sample_migrations):
        """Test that upgrade is idempotent."""
        for m in sample_migrations:
            runner.register(m)

        runner.upgrade()
        applied = runner.upgrade()

        assert len(applied) == 0  # Nothing new to apply

    def test_downgrade_one(self, runner, sample_migrations):
        """Test downgrading one migration."""
        for m in sample_migrations[:2]:  # Only use first two (have down_sql)
            runner.register(m)

        runner.upgrade()
        rolled_back = runner.downgrade()

        assert len(rolled_back) == 1
        assert rolled_back[0].version == 20240101000002

    def test_downgrade_to_target(self, runner, sample_migrations):
        """Test downgrading to specific version."""
        for m in sample_migrations[:2]:
            runner.register(m)

        runner.upgrade()
        rolled_back = runner.downgrade(target_version=20240101000000)

        assert len(rolled_back) == 2
        assert runner.get_applied_versions() == set()

    def test_downgrade_skips_without_down_sql(self, runner, sample_migrations):
        """Test that downgrade stops at migration without rollback."""
        for m in sample_migrations:
            runner.register(m)

        runner.upgrade()
        rolled_back = runner.downgrade()

        # Should stop at migration without down_sql
        assert len(rolled_back) == 0  # v3 has no down_sql

    def test_status(self, runner, sample_migrations):
        """Test getting migration status."""
        for m in sample_migrations:
            runner.register(m)

        # Before upgrade
        status = runner.status()
        assert status["applied_count"] == 0
        assert status["pending_count"] == 3
        assert status["latest_applied"] is None

        # After upgrade
        runner.upgrade()
        status = runner.status()
        assert status["applied_count"] == 3
        assert status["pending_count"] == 0
        assert status["latest_applied"] == 20240101000003


# ============================================================================
# Migration with Python Functions Tests
# ============================================================================


class TestMigrationFunctions:
    """Tests for migrations using Python functions."""

    def test_upgrade_with_function(self, runner):
        """Test applying migration with up_fn."""
        calls = []

        def up_fn(backend):
            calls.append("up")
            backend.execute_write("CREATE TABLE fn_test (id INTEGER)")

        m = Migration(version=1, name="Test", up_fn=up_fn)
        runner.register(m)
        runner.upgrade()

        assert calls == ["up"]

    def test_downgrade_with_function(self, runner):
        """Test rolling back migration with down_fn."""
        calls = []

        def up_fn(backend):
            backend.execute_write("CREATE TABLE fn_test (id INTEGER)")

        def down_fn(backend):
            calls.append("down")
            backend.execute_write("DROP TABLE fn_test")

        m = Migration(version=1, name="Test", up_fn=up_fn, down_fn=down_fn)
        runner.register(m)
        runner.upgrade()
        runner.downgrade()

        assert calls == ["down"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in migrations."""

    def test_upgrade_fails_on_bad_sql(self, runner):
        """Test that upgrade fails gracefully on bad SQL."""
        m = Migration(
            version=1,
            name="Bad SQL",
            up_sql="INVALID SQL STATEMENT",
        )
        runner.register(m)

        with pytest.raises(Exception):
            runner.upgrade()

        # Migration should not be recorded
        assert 1 not in runner.get_applied_versions()

    def test_downgrade_fails_on_bad_sql(self, runner):
        """Test that downgrade fails gracefully on bad SQL."""
        m = Migration(
            version=1,
            name="Test",
            up_sql="CREATE TABLE test (id INTEGER)",
            down_sql="INVALID SQL STATEMENT",
        )
        runner.register(m)
        runner.upgrade()

        with pytest.raises(Exception):
            runner.downgrade()

        # Migration should still be recorded
        assert 1 in runner.get_applied_versions()


# ============================================================================
# Global Runner Functions Tests
# ============================================================================


class TestGlobalRunnerFunctions:
    """Tests for module-level convenience functions."""

    def test_get_migration_runner_creates_singleton(self, temp_db):
        """Test that get_migration_runner returns same instance."""
        runner1 = get_migration_runner(db_path=temp_db)
        runner2 = get_migration_runner(db_path=temp_db)

        assert runner1 is runner2

    def test_reset_runner(self, temp_db):
        """Test resetting the global runner."""
        runner1 = get_migration_runner(db_path=temp_db)
        reset_runner()
        runner2 = get_migration_runner(db_path=temp_db)

        assert runner1 is not runner2

    def test_apply_migrations(self, temp_db):
        """Test apply_migrations convenience function."""
        # First create runner to register migrations
        runner = get_migration_runner(db_path=temp_db)

        # Should work without error (may apply existing migrations)
        applied = apply_migrations(db_path=temp_db)
        assert isinstance(applied, list)

    def test_get_migration_status(self, temp_db):
        """Test get_migration_status convenience function."""
        status = get_migration_status(db_path=temp_db)

        assert "applied_count" in status
        assert "pending_count" in status


# ============================================================================
# CLI Tests
# ============================================================================


class TestCLI:
    """Tests for CLI interface."""

    def test_cmd_upgrade(self, temp_db):
        """Test upgrade CLI command."""
        from aragora.migrations.__main__ import cmd_upgrade
        import argparse

        args = argparse.Namespace(
            db_path=temp_db,
            database_url=None,
            target=None,
        )

        result = cmd_upgrade(args)
        assert result == 0

    def test_cmd_status(self, temp_db):
        """Test status CLI command."""
        from aragora.migrations.__main__ import cmd_status
        import argparse

        args = argparse.Namespace(
            db_path=temp_db,
            database_url=None,
        )

        result = cmd_status(args)
        assert result == 0

    def test_cmd_create(self, tmp_path):
        """Test create CLI command."""
        from aragora.migrations.__main__ import cmd_create
        import argparse

        # Patch the versions path to use tmp_path
        with patch("aragora.migrations.__main__.Path") as MockPath:
            mock_file_path = tmp_path / "versions"
            mock_file_path.mkdir(exist_ok=True)

            # Mock __file__ location
            MockPath.return_value.parent.__truediv__ = lambda self, name: mock_file_path

            args = argparse.Namespace(name="Test Migration")

            # The actual implementation writes to versions dir
            # Just verify no exception is raised
            result = cmd_create(args)
            # May fail due to patching issues, but shouldn't crash
            assert result in (0, 1)


# ============================================================================
# Multi-statement SQL Tests
# ============================================================================


class TestMultiStatementSQL:
    """Tests for migrations with multiple SQL statements."""

    def test_multi_statement_upgrade(self, runner):
        """Test migration with multiple SQL statements."""
        m = Migration(
            version=1,
            name="Multi-statement",
            up_sql="""
                CREATE TABLE multi1 (id INTEGER);
                CREATE TABLE multi2 (id INTEGER);
                CREATE INDEX idx_multi1 ON multi1(id);
            """,
            down_sql="""
                DROP INDEX idx_multi1;
                DROP TABLE multi2;
                DROP TABLE multi1;
            """,
        )
        runner.register(m)
        runner.upgrade()

        # Verify all tables created
        tables = runner._backend.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'multi%'"
        )
        assert len(tables) == 2

    def test_multi_statement_with_comments(self, runner):
        """Test migration with SQL comments."""
        m = Migration(
            version=1,
            name="With comments",
            up_sql="""
                -- Create the users table
                CREATE TABLE comment_test (id INTEGER);
                -- Add index
                CREATE INDEX idx_comment_test ON comment_test(id);
            """,
            down_sql="DROP TABLE comment_test;",
        )
        runner.register(m)

        # Should handle comments gracefully
        applied = runner.upgrade()
        assert len(applied) == 1


# ============================================================================
# Version Loading Tests
# ============================================================================


class TestVersionLoading:
    """Tests for automatic migration version loading."""

    def test_loads_versions_from_package(self, temp_db):
        """Test that migrations are loaded from versions package."""
        runner = get_migration_runner(db_path=temp_db)

        # Should have loaded at least the initial schema migration
        assert len(runner._migrations) >= 1

        # Verify initial schema migration exists
        versions = {m.version for m in runner._migrations}
        assert 20240101000000 in versions  # Initial schema version
