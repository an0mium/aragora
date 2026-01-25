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

    def test_rollback_nothing_to_rollback(self, runner: MigrationRunner, temp_db_dir: Path):
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

    def test_rollback_migration_not_found(self, runner: MigrationRunner, test_db: Path):
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

    def test_rollback_no_downgrade_function(self, runner: MigrationRunner, test_db: Path):
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

    def test_rollback_to_previous_version(self, runner: MigrationRunner, temp_db_dir: Path):
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

        with patch(
            "sys.argv",
            [
                "runner.py",
                "--rollback",
                "--dry-run",
                "--db",
                "elo",
                "--nomic-dir",
                str(temp_db_dir),
            ],
        ):
            with patch.object(MigrationRunner, "discover_migrations", return_value=[migration]):
                with patch("builtins.print") as mock_print:
                    result = main()

        assert result == 0


# ===========================================================================
# Test SQLite Table Recreation Downgrade Pattern
# ===========================================================================


class TestSQLiteTableRecreationDowngrade:
    """Tests for SQLite table recreation downgrade pattern (used when DROP COLUMN not supported)."""

    def test_table_recreation_preserves_data(self, temp_db_dir: Path):
        """Test that table recreation pattern preserves existing data."""
        db_path = temp_db_dir / "test_recreation.db"
        conn = sqlite3.connect(str(db_path))

        # Step 1: Create original schema (like 001_initial)
        conn.executescript(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                name TEXT DEFAULT '',
                org_id TEXT,
                role TEXT DEFAULT 'member',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT,
                is_active INTEGER DEFAULT 1,
                email_verified INTEGER DEFAULT 0,
                avatar_url TEXT,
                preferences TEXT DEFAULT '{}'
            );
            CREATE INDEX idx_users_email ON users(email);
            CREATE INDEX idx_users_org ON users(org_id);
        """
        )

        # Insert test data
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, password_salt, name, role)
            VALUES ('user1', 'test@example.com', 'hash1', 'salt1', 'Test User', 'admin')
        """
        )
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, password_salt, name)
            VALUES ('user2', 'user2@example.com', 'hash2', 'salt2', 'Second User')
        """
        )
        conn.commit()

        # Step 2: Apply upgrade (like 002_add_lockout)
        conn.execute("ALTER TABLE users ADD COLUMN locked_until TEXT")
        conn.execute("ALTER TABLE users ADD COLUMN failed_login_count INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE users ADD COLUMN lockout_reason TEXT")
        conn.execute("ALTER TABLE users ADD COLUMN last_activity_at TEXT")
        conn.execute("ALTER TABLE users ADD COLUMN last_debate_at TEXT")

        # Update some lockout data
        conn.execute(
            "UPDATE users SET failed_login_count = 3, lockout_reason = 'test' WHERE id = 'user1'"
        )
        conn.commit()

        # Verify new columns exist
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "locked_until" in columns
        assert "failed_login_count" in columns

        # Step 3: Apply downgrade (table recreation pattern from 002_add_lockout)
        conn.executescript(
            """
            -- Backup current data
            CREATE TABLE users_backup AS SELECT * FROM users;
            -- Drop original table
            DROP TABLE users;
            -- Recreate table WITHOUT the new columns
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                name TEXT DEFAULT '',
                org_id TEXT,
                role TEXT DEFAULT 'member',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT,
                is_active INTEGER DEFAULT 1,
                email_verified INTEGER DEFAULT 0,
                avatar_url TEXT,
                preferences TEXT DEFAULT '{}'
            );
            -- Copy data back (original columns only)
            INSERT INTO users (
                id, email, password_hash, password_salt, name, org_id, role,
                created_at, updated_at, last_login_at, is_active, email_verified,
                avatar_url, preferences
            )
            SELECT
                id, email, password_hash, password_salt, name, org_id, role,
                created_at, updated_at, last_login_at, is_active, email_verified,
                avatar_url, preferences
            FROM users_backup;
            -- Drop backup
            DROP TABLE users_backup;
            -- Recreate indexes
            CREATE INDEX idx_users_email ON users(email);
            CREATE INDEX idx_users_org ON users(org_id);
        """
        )
        conn.commit()

        # Verify columns are removed
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "locked_until" not in columns
        assert "failed_login_count" not in columns
        assert "lockout_reason" not in columns
        assert "last_activity_at" not in columns
        assert "last_debate_at" not in columns

        # Verify original columns still exist
        assert "id" in columns
        assert "email" in columns
        assert "password_hash" in columns
        assert "name" in columns
        assert "role" in columns

        # Verify data is preserved
        cursor = conn.execute("SELECT id, email, name, role FROM users ORDER BY id")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0] == ("user1", "test@example.com", "Test User", "admin")
        assert rows[1] == ("user2", "user2@example.com", "Second User", "member")

        # Verify indexes are recreated
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='users'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_users_email" in indexes
        assert "idx_users_org" in indexes

        conn.close()

    def test_downgrade_with_actual_migration(self, temp_db_dir: Path):
        """Test the actual 002_add_lockout downgrade function."""
        db_path = temp_db_dir / "test_migration.db"
        conn = sqlite3.connect(str(db_path))

        # Create base schema from 001
        conn.executescript(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                name TEXT DEFAULT '',
                org_id TEXT,
                role TEXT DEFAULT 'member',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT,
                is_active INTEGER DEFAULT 1,
                email_verified INTEGER DEFAULT 0,
                avatar_url TEXT,
                preferences TEXT DEFAULT '{}'
            );
            CREATE INDEX idx_users_email ON users(email);
            CREATE INDEX idx_users_org ON users(org_id);
        """
        )

        # Insert test user
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, password_salt)
            VALUES ('u1', 'a@b.com', 'h', 's')
        """
        )
        conn.commit()

        # Import and apply the actual upgrade (module name starts with number, use importlib)
        import importlib

        m002 = importlib.import_module("aragora.persistence.migrations.users.002_add_lockout")
        m002.upgrade(conn)
        conn.commit()

        # Verify upgrade worked
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "locked_until" in columns
        assert "failed_login_count" in columns

        # Apply the actual downgrade
        m002.downgrade(conn)
        conn.commit()

        # Verify downgrade worked
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "locked_until" not in columns
        assert "failed_login_count" not in columns

        # Verify data preserved
        cursor = conn.execute("SELECT id, email FROM users")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0] == ("u1", "a@b.com")

        conn.close()
