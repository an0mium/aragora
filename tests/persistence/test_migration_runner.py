"""
Tests for aragora.persistence.migrations.runner - Migration runner.

Tests cover:
- Migration discovery
- Migration execution
- Rollback functionality
- Dry-run mode
- Error handling
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.persistence.migrations.runner import (
    MigrationFile,
    MigrationRunner,
    MigrationStatus,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def runner(temp_db_dir: Path):
    """Create a MigrationRunner for testing."""
    return MigrationRunner(
        nomic_dir=temp_db_dir,
        db_paths={"test": "test.db"},
    )


@pytest.fixture
def test_db(temp_db_dir: Path):
    """Create a test database with schema version tracking."""
    db_path = temp_db_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _schema_versions (
            module TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO _schema_versions (module, version, updated_at)
        VALUES ('test', 1, CURRENT_TIMESTAMP)
    """
    )
    conn.commit()
    conn.close()
    return db_path


# ===========================================================================
# Test MigrationFile
# ===========================================================================


class TestMigrationFile:
    """Tests for MigrationFile dataclass."""

    def test_creation(self):
        """Test MigrationFile creation."""
        path = Path("/tmp/migrations/001_initial.py")
        mf = MigrationFile(
            version=1,
            name="initial",
            path=path,
            description="Initial migration",
        )

        assert mf.version == 1
        assert mf.name == "initial"
        assert mf.path == path
        assert mf.description == "Initial migration"

    def test_module_name(self):
        """Test module_name property."""
        path = Path("/tmp/migrations/001_initial.py")
        mf = MigrationFile(version=1, name="initial", path=path)

        assert mf.module_name == "001_initial"


# ===========================================================================
# Test MigrationRunner
# ===========================================================================


class TestMigrationRunnerInit:
    """Tests for MigrationRunner initialization."""

    def test_default_init(self, temp_db_dir: Path):
        """Test MigrationRunner with defaults."""
        runner = MigrationRunner(nomic_dir=temp_db_dir)

        assert runner.nomic_dir == temp_db_dir
        assert "elo" in runner.db_paths
        assert "memory" in runner.db_paths

    def test_custom_db_paths(self, temp_db_dir: Path):
        """Test MigrationRunner with custom db paths."""
        custom_paths = {"custom": "custom.db"}
        runner = MigrationRunner(nomic_dir=temp_db_dir, db_paths=custom_paths)

        assert runner.db_paths == custom_paths


class TestMigrationRunnerGetDbPath:
    """Tests for get_db_path method."""

    def test_get_db_path(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test getting database path."""
        path = runner.get_db_path("test")

        assert path == temp_db_dir / "test.db"

    def test_get_db_path_unknown(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test getting path for unknown database uses default naming."""
        path = runner.get_db_path("unknown")

        # Should return default path based on db_name
        assert path == temp_db_dir / "aragora_unknown.db"


# ===========================================================================
# Test Rollback
# ===========================================================================


class TestRollback:
    """Tests for rollback functionality."""

    def test_rollback_no_migrations(self, runner: MigrationRunner):
        """Test rollback when no migrations exist."""
        with patch.object(runner, "discover_migrations", return_value=[]):
            result = runner.rollback("test")

        assert result["status"] == "no_migrations"
        assert "No migrations found" in result["message"]

    def test_rollback_no_database(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test rollback when database doesn't exist."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )
        with patch.object(runner, "discover_migrations", return_value=[migration]):
            result = runner.rollback("test")

        assert result["status"] == "no_database"
        assert "does not exist" in result["message"]

    def test_rollback_nothing_to_rollback(
        self, runner: MigrationRunner, temp_db_dir: Path
    ):
        """Test rollback when version is 0."""
        # Create database with version 0
        db_path = temp_db_dir / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO _schema_versions (module, version, updated_at)
            VALUES ('test', 0, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )
        with patch.object(runner, "discover_migrations", return_value=[migration]):
            result = runner.rollback("test")

        assert result["status"] == "nothing_to_rollback"
        assert result["current_version"] == 0

    def test_rollback_dry_run(self, runner: MigrationRunner, test_db: Path):
        """Test rollback in dry-run mode."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
            description="Initial migration",
        )
        with patch.object(runner, "discover_migrations", return_value=[migration]):
            result = runner.rollback("test", dry_run=True)

        assert result["status"] == "dry_run"
        assert result["current_version"] == 1
        assert result["would_rollback"]["version"] == 1
        assert result["would_rollback"]["name"] == "initial"

    def test_rollback_migration_not_found(
        self, runner: MigrationRunner, test_db: Path
    ):
        """Test rollback when migration file is missing."""
        # Database is at version 1, but we only have version 2 migration
        migration = MigrationFile(
            version=2,
            name="second",
            path=Path("/fake/path/002_second.py"),
        )
        with patch.object(runner, "discover_migrations", return_value=[migration]):
            result = runner.rollback("test")

        assert result["status"] == "migration_not_found"
        assert "not found" in result["message"]

    def test_rollback_no_downgrade_function(
        self, runner: MigrationRunner, test_db: Path
    ):
        """Test rollback when migration has no downgrade function."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )

        # Mock module without downgrade
        mock_module = MagicMock(spec=[])  # No downgrade attribute

        with patch.object(runner, "discover_migrations", return_value=[migration]):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                result = runner.rollback("test")

        assert result["status"] == "no_downgrade"
        assert "downgrade()" in result["message"]

    def test_rollback_empty_downgrade(self, runner: MigrationRunner, test_db: Path):
        """Test rollback when downgrade function is empty."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )

        mock_module = MagicMock()
        mock_module.downgrade = lambda conn: None  # Empty function

        with patch.object(runner, "discover_migrations", return_value=[migration]):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "_is_empty_migration", return_value=True):
                    result = runner.rollback("test")

        assert result["status"] == "empty_downgrade"
        assert "implement it first" in result["message"]

    def test_rollback_success(self, runner: MigrationRunner, test_db: Path):
        """Test successful rollback."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
            description="Initial migration",
        )

        mock_module = MagicMock()
        mock_module.downgrade = MagicMock()

        with patch.object(runner, "discover_migrations", return_value=[migration]):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "_is_empty_migration", return_value=False):
                    result = runner.rollback("test")

        assert result["status"] == "completed"
        assert result["previous_version"] == 1
        assert result["current_version"] == 0
        assert result["rolled_back"] == 1

        # Verify downgrade was called
        mock_module.downgrade.assert_called_once()

    def test_rollback_to_previous_version(
        self, runner: MigrationRunner, temp_db_dir: Path
    ):
        """Test rollback sets version to previous migration."""
        # Create database at version 2
        db_path = temp_db_dir / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO _schema_versions (module, version, updated_at)
            VALUES ('test', 2, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migrations = [
            MigrationFile(version=1, name="first", path=Path("/fake/001.py")),
            MigrationFile(version=2, name="second", path=Path("/fake/002.py")),
        ]

        mock_module = MagicMock()
        mock_module.downgrade = MagicMock()

        with patch.object(runner, "discover_migrations", return_value=migrations):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "_is_empty_migration", return_value=False):
                    result = runner.rollback("test")

        assert result["status"] == "completed"
        assert result["previous_version"] == 2
        assert result["current_version"] == 1
        assert result["rolled_back"] == 2

    def test_rollback_failure(self, runner: MigrationRunner, test_db: Path):
        """Test rollback when downgrade raises exception."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )

        mock_module = MagicMock()
        mock_module.downgrade = MagicMock(side_effect=Exception("Downgrade failed"))

        with patch.object(runner, "discover_migrations", return_value=[migration]):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "_is_empty_migration", return_value=False):
                    result = runner.rollback("test")

        assert result["status"] == "failed"
        assert "Downgrade failed" in result["error"]


# ===========================================================================
# Test CLI Argument Parsing
# ===========================================================================


class TestCLI:
    """Tests for CLI argument handling."""

    def test_rollback_requires_db(self):
        """Test that rollback requires --db argument."""
        from aragora.persistence.migrations.runner import main

        with patch("sys.argv", ["runner.py", "--rollback"]):
            with patch("builtins.print") as mock_print:
                result = main()

        assert result == 1
        # Check that error message was printed
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("--db is required" in str(call) for call in calls)

    def test_rollback_with_dry_run(self, temp_db_dir: Path):
        """Test rollback with dry-run flag."""
        from aragora.persistence.migrations.runner import main

        # Create test database
        db_path = temp_db_dir / "elo" / "aragora_elo.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO _schema_versions (module, version, updated_at)
            VALUES ('elo', 1, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/001_initial.py"),
            description="Initial",
        )

        with patch("sys.argv", ["runner.py", "--rollback", "--dry-run", "--db", "elo", "--nomic-dir", str(temp_db_dir)]):
            with patch.object(MigrationRunner, "discover_migrations", return_value=[migration]):
                with patch("builtins.print") as mock_print:
                    result = main()

        assert result == 0
