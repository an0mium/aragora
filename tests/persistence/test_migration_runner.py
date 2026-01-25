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


# ===========================================================================
# Test Rollback to Version
# ===========================================================================


class TestRollbackToVersion:
    """Tests for rollback_to_version functionality (multi-step rollback)."""

    def test_rollback_to_version_no_migrations(self, runner: MigrationRunner):
        """Test rollback_to_version when no migrations exist."""
        with patch.object(runner, "discover_migrations", return_value=[]):
            result = runner.rollback_to_version("test", target_version=0)

        assert result["status"] == "no_migrations"
        assert "No migrations found" in result["message"]

    def test_rollback_to_version_no_database(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test rollback_to_version when database doesn't exist."""
        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )
        with patch.object(runner, "discover_migrations", return_value=[migration]):
            result = runner.rollback_to_version("test", target_version=0)

        assert result["status"] == "no_database"
        assert "does not exist" in result["message"]

    def test_rollback_to_version_nothing_to_rollback(
        self, runner: MigrationRunner, temp_db_dir: Path
    ):
        """Test rollback_to_version when already at target version."""
        # Create database with version 1
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

        migration = MigrationFile(
            version=1,
            name="initial",
            path=Path("/fake/path/001_initial.py"),
        )
        with patch.object(runner, "discover_migrations", return_value=[migration]):
            result = runner.rollback_to_version("test", target_version=1)

        assert result["status"] == "nothing_to_rollback"
        assert result["current_version"] == 1
        assert result["target_version"] == 1

    def test_rollback_to_version_dry_run(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test rollback_to_version in dry-run mode."""
        # Create database at version 3
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
            VALUES ('test', 3, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migrations = [
            MigrationFile(
                version=1, name="first", path=Path("/fake/001.py"), description="First migration"
            ),
            MigrationFile(
                version=2, name="second", path=Path("/fake/002.py"), description="Second migration"
            ),
            MigrationFile(
                version=3, name="third", path=Path("/fake/003.py"), description="Third migration"
            ),
        ]
        with patch.object(runner, "discover_migrations", return_value=migrations):
            result = runner.rollback_to_version("test", target_version=1, dry_run=True)

        assert result["status"] == "dry_run"
        assert result["current_version"] == 3
        assert result["target_version"] == 1
        assert len(result["would_rollback"]) == 2
        # Should be in reverse order (3, 2)
        assert result["would_rollback"][0]["version"] == 3
        assert result["would_rollback"][1]["version"] == 2

    def test_rollback_to_version_success(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test successful multi-step rollback."""
        # Create database at version 3
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
            VALUES ('test', 3, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migrations = [
            MigrationFile(version=1, name="first", path=Path("/fake/001.py")),
            MigrationFile(version=2, name="second", path=Path("/fake/002.py")),
            MigrationFile(version=3, name="third", path=Path("/fake/003.py")),
        ]

        mock_module = MagicMock()
        mock_module.downgrade = MagicMock()

        with patch.object(runner, "discover_migrations", return_value=migrations):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "_is_empty_migration", return_value=False):
                    with patch.object(
                        runner, "create_pre_migration_backup", return_value="backup-123"
                    ):
                        result = runner.rollback_to_version("test", target_version=1)

        assert result["status"] == "completed"
        assert result["initial_version"] == 3
        assert result["final_version"] == 1
        assert result["rolled_back"] == [3, 2]
        assert result["backup_id"] == "backup-123"

        # Verify downgrade was called twice (for version 3 and 2)
        assert mock_module.downgrade.call_count == 2

    def test_rollback_to_version_partial_failure(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test rollback that fails mid-way."""
        # Create database at version 3
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
            VALUES ('test', 3, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migrations = [
            MigrationFile(version=1, name="first", path=Path("/fake/001.py")),
            MigrationFile(version=2, name="second", path=Path("/fake/002.py")),
            MigrationFile(version=3, name="third", path=Path("/fake/003.py")),
        ]

        call_count = [0]

        def failing_downgrade(conn):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call (version 2)
                raise Exception("Rollback failed at version 2")

        mock_module = MagicMock()
        mock_module.downgrade = failing_downgrade

        with patch.object(runner, "discover_migrations", return_value=migrations):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "_is_empty_migration", return_value=False):
                    with patch.object(runner, "create_pre_migration_backup", return_value=None):
                        result = runner.rollback_to_version("test", target_version=0)

        assert result["status"] == "partial"
        assert result["initial_version"] == 3
        assert result["rolled_back"] == [3]  # Only version 3 was rolled back
        assert len(result["errors"]) == 1
        assert result["errors"][0]["version"] == 2

    def test_rollback_to_version_no_downgrade_function(
        self, runner: MigrationRunner, temp_db_dir: Path
    ):
        """Test rollback_to_version when a migration has no downgrade function."""
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

        mock_module = MagicMock(spec=[])  # No downgrade attribute

        with patch.object(runner, "discover_migrations", return_value=migrations):
            with patch.object(runner, "_load_migration_module", return_value=mock_module):
                with patch.object(runner, "create_pre_migration_backup", return_value=None):
                    result = runner.rollback_to_version("test", target_version=0)

        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        assert "No downgrade" in result["errors"][0]["error"]


# ===========================================================================
# Test Backup Support
# ===========================================================================


class TestBackupSupport:
    """Tests for backup support in migration runner."""

    def test_init_with_backup_disabled(self, temp_db_dir: Path):
        """Test initialization with backup disabled."""
        runner = MigrationRunner(
            nomic_dir=temp_db_dir,
            backup_before_migrate=False,
        )
        assert runner.backup_before_migrate is False

    def test_init_with_custom_backup_dir(self, temp_db_dir: Path):
        """Test initialization with custom backup directory."""
        custom_backup_dir = temp_db_dir / "custom_backups"
        runner = MigrationRunner(
            nomic_dir=temp_db_dir,
            backup_dir=custom_backup_dir,
        )
        assert runner.backup_dir == custom_backup_dir

    def test_default_backup_dir(self, temp_db_dir: Path):
        """Test default backup directory is set correctly."""
        runner = MigrationRunner(nomic_dir=temp_db_dir)
        assert runner.backup_dir == temp_db_dir / "migration_backups"

    def test_create_pre_migration_backup_disabled(self, temp_db_dir: Path):
        """Test create_pre_migration_backup returns None when disabled."""
        runner = MigrationRunner(
            nomic_dir=temp_db_dir,
            backup_before_migrate=False,
        )
        result = runner.create_pre_migration_backup(temp_db_dir / "test.db", "test")
        assert result is None

    def test_create_pre_migration_backup_no_db(self, temp_db_dir: Path):
        """Test create_pre_migration_backup returns None when db doesn't exist."""
        runner = MigrationRunner(nomic_dir=temp_db_dir)
        result = runner.create_pre_migration_backup(temp_db_dir / "nonexistent.db", "test")
        assert result is None


# ===========================================================================
# Test Checksum Verification
# ===========================================================================


class TestChecksumVerification:
    """Tests for checksum verification in migration runner."""

    def test_migration_file_compute_checksum(self, temp_db_dir: Path):
        """Test MigrationFile checksum computation."""
        # Create a migration file
        migration_path = temp_db_dir / "001_test.py"
        migration_path.write_text("def upgrade(conn): pass")

        mf = MigrationFile(version=1, name="test", path=migration_path)
        checksum = mf.compute_checksum()

        assert checksum is not None
        assert len(checksum) == 16  # SHA-256 truncated to 16 chars

    def test_migration_file_checksum_nonexistent(self):
        """Test MigrationFile checksum for non-existent file."""
        mf = MigrationFile(
            version=1,
            name="test",
            path=Path("/nonexistent/001_test.py"),
        )
        checksum = mf.compute_checksum()
        assert checksum == ""

    def test_migration_file_checksum_consistency(self, temp_db_dir: Path):
        """Test that checksum is consistent for same content."""
        migration_path = temp_db_dir / "001_test.py"
        migration_path.write_text("def upgrade(conn): pass")

        mf1 = MigrationFile(version=1, name="test", path=migration_path)
        mf2 = MigrationFile(version=1, name="test", path=migration_path)

        assert mf1.compute_checksum() == mf2.compute_checksum()

    def test_migration_file_checksum_changes_with_content(self, temp_db_dir: Path):
        """Test that checksum changes when content changes."""
        migration_path = temp_db_dir / "001_test.py"

        migration_path.write_text("def upgrade(conn): pass")
        mf = MigrationFile(version=1, name="test", path=migration_path)
        checksum1 = mf.compute_checksum()

        migration_path.write_text("def upgrade(conn): conn.execute('SELECT 1')")
        checksum2 = mf.compute_checksum()

        assert checksum1 != checksum2

    def test_ensure_migration_tracking_creates_table(
        self, runner: MigrationRunner, temp_db_dir: Path
    ):
        """Test that _ensure_migration_tracking creates the tracking table."""
        db_path = temp_db_dir / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner._ensure_migration_tracking(conn, "test")

        # Check table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_migration_checksums'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_record_migration_stores_checksum(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test that _record_migration stores the checksum."""
        db_path = temp_db_dir / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner._ensure_migration_tracking(conn, "test")

        migration = MigrationFile(
            version=1,
            name="test",
            path=Path("/fake/001_test.py"),
            checksum="abc123",
        )
        runner._record_migration(conn, migration)
        conn.commit()

        cursor = conn.execute(
            "SELECT version, name, checksum FROM _migration_checksums WHERE version = 1"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == "test"
        assert row[2] == "abc123"

        conn.close()

    def test_get_applied_checksums(self, runner: MigrationRunner, temp_db_dir: Path):
        """Test _get_applied_checksums retrieves stored checksums."""
        db_path = temp_db_dir / "test.db"
        conn = sqlite3.connect(str(db_path))

        runner._ensure_migration_tracking(conn, "test")

        # Insert some checksums
        conn.execute(
            "INSERT INTO _migration_checksums (version, name, checksum, applied_at) VALUES (1, 'first', 'aaa', '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO _migration_checksums (version, name, checksum, applied_at) VALUES (2, 'second', 'bbb', '2024-01-02')"
        )
        conn.commit()

        checksums = runner._get_applied_checksums(conn)

        assert checksums == {1: "aaa", 2: "bbb"}

        conn.close()


# ===========================================================================
# Test CLI --rollback-to
# ===========================================================================


class TestCLIRollbackTo:
    """Tests for --rollback-to CLI argument."""

    def test_rollback_to_requires_db(self):
        """Test that --rollback-to requires --db argument."""
        from aragora.persistence.migrations.runner import main

        with patch("sys.argv", ["runner.py", "--rollback-to", "5"]):
            with patch("builtins.print") as mock_print:
                result = main()

        assert result == 1
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("--db is required" in str(call) for call in calls)

    def test_rollback_to_with_dry_run(self, temp_db_dir: Path):
        """Test --rollback-to with --dry-run flag."""
        from aragora.persistence.migrations.runner import main

        # Create test database at version 3
        db_path = temp_db_dir / "aragora_test.db"
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
            VALUES ('test', 3, CURRENT_TIMESTAMP)
        """
        )
        conn.commit()
        conn.close()

        migrations = [
            MigrationFile(version=1, name="first", path=Path("/fake/001.py"), description="First"),
            MigrationFile(
                version=2, name="second", path=Path("/fake/002.py"), description="Second"
            ),
            MigrationFile(version=3, name="third", path=Path("/fake/003.py"), description="Third"),
        ]

        with patch(
            "sys.argv",
            [
                "runner.py",
                "--rollback-to",
                "1",
                "--dry-run",
                "--db",
                "test",
                "--nomic-dir",
                str(temp_db_dir),
            ],
        ):
            with patch.object(MigrationRunner, "discover_migrations", return_value=migrations):
                with patch("builtins.print") as mock_print:
                    result = main()

        assert result == 0
        # Verify dry-run output
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "dry run" in printed.lower()

    def test_rollback_to_parses_version(self):
        """Test that --rollback-to correctly parses the version number."""
        from aragora.persistence.migrations.runner import main
        import argparse

        with patch("sys.argv", ["runner.py", "--rollback-to", "5", "--db", "test"]):
            with patch.object(MigrationRunner, "rollback_to_version") as mock_rollback:
                mock_rollback.return_value = {
                    "status": "completed",
                    "initial_version": 10,
                    "final_version": 5,
                    "target_version": 5,
                    "rolled_back": [10, 9, 8, 7, 6],
                    "errors": [],
                }
                with patch("builtins.print"):
                    result = main()

        # Verify rollback_to_version was called with correct version
        mock_rollback.assert_called_once()
        call_args = mock_rollback.call_args
        assert call_args[0][0] == "test"  # db_name
        assert call_args[0][1] == 5  # target_version
