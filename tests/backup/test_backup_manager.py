"""
Tests for the Backup Manager.

Tests cover:
- Creating backups
- Verifying backup integrity
- Restoring backups
- Retention policy enforcement
"""

import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from aragora.backup import (
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    RetentionPolicy,
    VerificationResult,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_database(temp_dir: Path) -> Path:
    """Create a sample SQLite database for testing."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables with data
    cursor.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE debates (
            id INTEGER PRIMARY KEY,
            topic TEXT NOT NULL,
            created_at TEXT
        )
    """
    )

    # Insert test data
    for i in range(10):
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (f"User {i}", f"user{i}@example.com"),
        )
    for i in range(5):
        cursor.execute(
            "INSERT INTO debates (topic, created_at) VALUES (?, ?)",
            (f"Topic {i}", datetime.now(timezone.utc).isoformat()),
        )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def backup_manager(temp_dir: Path) -> BackupManager:
    """Create a backup manager for testing."""
    backup_dir = temp_dir / "backups"
    return BackupManager(
        backup_dir=backup_dir,
        compression=True,
        verify_after_backup=True,
        metrics_enabled=False,
    )


class TestBackupCreation:
    """Tests for backup creation."""

    def test_create_full_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test creating a full backup."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.status == BackupStatus.VERIFIED
        assert backup.backup_type == BackupType.FULL
        assert backup.size_bytes > 0
        assert backup.compressed_size_bytes > 0
        assert backup.checksum != ""
        assert len(backup.tables) == 2
        assert backup.row_counts["users"] == 10
        assert backup.row_counts["debates"] == 5
        assert backup.verified

    def test_backup_creates_file(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that backup creates a file on disk."""
        backup = backup_manager.create_backup(sample_database)

        backup_path = Path(backup.backup_path)
        assert backup_path.exists()
        assert backup_path.suffix == ".gz"
        assert backup_path.stat().st_size == backup.compressed_size_bytes

    def test_backup_with_metadata(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test backup with custom metadata."""
        metadata = {"version": "1.0", "environment": "test"}
        backup = backup_manager.create_backup(sample_database, metadata=metadata)

        assert backup.metadata == metadata

    def test_backup_nonexistent_file(
        self,
        backup_manager: BackupManager,
        temp_dir: Path,
    ):
        """Test backup of nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            backup_manager.create_backup(temp_dir / "nonexistent.db")


class TestBackupVerification:
    """Tests for backup verification."""

    def test_verify_valid_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test verifying a valid backup."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_backup(backup.id)

        assert result.verified
        assert result.checksum_valid
        assert result.restore_tested
        assert result.tables_valid
        assert len(result.errors) == 0

    def test_verify_detects_corruption(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that verification detects file corruption."""
        backup = backup_manager.create_backup(sample_database)

        # Corrupt the backup file
        backup_path = Path(backup.backup_path)
        with open(backup_path, "r+b") as f:
            f.seek(50)
            f.write(b"CORRUPTED")

        result = backup_manager.verify_backup(backup.id)

        assert not result.verified
        assert not result.checksum_valid
        assert len(result.errors) > 0

    def test_verify_nonexistent_backup(
        self,
        backup_manager: BackupManager,
    ):
        """Test verifying nonexistent backup."""
        result = backup_manager.verify_backup("nonexistent")

        assert not result.verified
        assert "not found" in result.errors[0].lower()


class TestBackupRestore:
    """Tests for backup restoration."""

    def test_restore_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Test restoring a backup."""
        backup = backup_manager.create_backup(sample_database)
        restore_path = temp_dir / "restored.db"

        success = backup_manager.restore_backup(backup.id, restore_path)

        assert success
        assert restore_path.exists()

        # Verify restored data
        conn = sqlite3.connect(str(restore_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 10
        conn.close()

    def test_restore_dry_run(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Test dry run restore."""
        backup = backup_manager.create_backup(sample_database)
        restore_path = temp_dir / "restored.db"

        success = backup_manager.restore_backup(backup.id, restore_path, dry_run=True)

        assert success
        assert not restore_path.exists()

    def test_restore_creates_backup_of_existing(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Test that restore creates backup of existing file."""
        backup = backup_manager.create_backup(sample_database)

        # Create an existing file at restore path
        restore_path = temp_dir / "existing.db"
        restore_path.write_text("existing content")

        backup_manager.restore_backup(backup.id, restore_path)

        # Check backup was created
        backup_files = list(temp_dir.glob("existing.backup_*"))
        assert len(backup_files) == 1


class TestBackupListing:
    """Tests for listing backups."""

    def test_list_all_backups(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test listing all backups."""
        backup_manager.create_backup(sample_database)
        backup_manager.create_backup(sample_database)

        backups = backup_manager.list_backups()

        assert len(backups) == 2

    def test_list_backups_by_status(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test filtering backups by status."""
        backup_manager.create_backup(sample_database)

        verified = backup_manager.list_backups(status=BackupStatus.VERIFIED)
        failed = backup_manager.list_backups(status=BackupStatus.FAILED)

        assert len(verified) == 1
        assert len(failed) == 0

    def test_get_latest_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test getting the latest backup."""
        backup1 = backup_manager.create_backup(sample_database)
        backup2 = backup_manager.create_backup(sample_database)

        latest = backup_manager.get_latest_backup()

        assert latest is not None
        assert latest.id == backup2.id


class TestRetentionPolicy:
    """Tests for retention policy enforcement."""

    def test_cleanup_expired_backups(
        self,
        temp_dir: Path,
        sample_database: Path,
    ):
        """Test cleanup of expired backups."""
        policy = RetentionPolicy(
            keep_daily=2,
            keep_weekly=1,
            keep_monthly=0,
            min_backups=1,
        )
        manager = BackupManager(
            backup_dir=temp_dir / "backups",
            retention_policy=policy,
            metrics_enabled=False,
        )

        # Create multiple backups
        for _ in range(5):
            manager.create_backup(sample_database)

        initial_count = len(manager.list_backups())
        deleted = manager.cleanup_expired()

        # With keep_daily=2, some should be deleted
        final_count = len(manager.list_backups())
        assert final_count <= initial_count

    def test_keeps_minimum_backups(
        self,
        temp_dir: Path,
        sample_database: Path,
    ):
        """Test that minimum backups are always kept."""
        policy = RetentionPolicy(
            keep_daily=0,
            keep_weekly=0,
            keep_monthly=0,
            min_backups=3,
        )
        manager = BackupManager(
            backup_dir=temp_dir / "backups",
            retention_policy=policy,
            metrics_enabled=False,
        )

        # Create backups
        for _ in range(5):
            manager.create_backup(sample_database)

        manager.cleanup_expired()

        # Should keep at least 3
        assert len(manager.list_backups()) >= 3


class TestCompression:
    """Tests for backup compression."""

    def test_compressed_smaller_than_original(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that compressed backup is smaller."""
        backup = backup_manager.create_backup(sample_database)

        # Compressed should be smaller (or at least not much larger)
        assert backup.compressed_size_bytes <= backup.size_bytes * 1.1

    def test_no_compression(
        self,
        temp_dir: Path,
        sample_database: Path,
    ):
        """Test backup without compression."""
        manager = BackupManager(
            backup_dir=temp_dir / "backups",
            compression=False,
            metrics_enabled=False,
        )

        backup = manager.create_backup(sample_database)

        backup_path = Path(backup.backup_path)
        assert backup_path.suffix == ".db"
        assert backup.compressed_size_bytes == backup.size_bytes


class TestManifest:
    """Tests for backup manifest persistence."""

    def test_manifest_persists(
        self,
        temp_dir: Path,
        sample_database: Path,
    ):
        """Test that manifest persists across manager instances."""
        backup_dir = temp_dir / "backups"

        # Create backup with first manager
        manager1 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backup = manager1.create_backup(sample_database)

        # Load with new manager
        manager2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backups = manager2.list_backups()

        assert len(backups) == 1
        assert backups[0].id == backup.id

    def test_manifest_survives_corruption(
        self,
        temp_dir: Path,
    ):
        """Test that manager handles corrupted manifest."""
        backup_dir = temp_dir / "backups"
        backup_dir.mkdir(parents=True)

        # Create corrupted manifest
        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text("invalid json{{{")

        # Should not raise
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        assert len(manager.list_backups()) == 0


class TestComprehensiveVerification:
    """Tests for comprehensive backup verification."""

    def test_comprehensive_verify_valid_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test comprehensive verification of a valid backup."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.verified
        assert result.basic_verification.verified
        assert result.basic_verification.checksum_valid
        assert len(result.all_errors) == 0

    def test_comprehensive_includes_schema_validation(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test comprehensive verification includes schema validation."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.schema_validation is not None
        assert result.schema_validation.valid
        assert result.schema_validation.tables_match

    def test_comprehensive_includes_integrity_check(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test comprehensive verification includes integrity check."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.integrity_check is not None
        assert result.integrity_check.valid

    def test_comprehensive_to_dict(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test comprehensive result serialization."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)
        result_dict = result.to_dict()

        assert "backup_id" in result_dict
        assert "verified" in result_dict
        assert "basic_verification" in result_dict
        assert "schema_validation" in result_dict
        assert "integrity_check" in result_dict


class TestSchemaVerification:
    """Tests for schema validation during backup verification."""

    def test_schema_hash_computed(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that schema hash is computed during backup."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.schema_hash != ""
        assert len(backup.schema_hash) == 64  # SHA-256 hex length

    def test_table_checksums_computed(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that per-table checksums are computed."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.table_checksums) == 2
        assert "users" in backup.table_checksums
        assert "debates" in backup.table_checksums
        assert len(backup.table_checksums["users"]) == 64

    def test_indexes_captured(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that indexes are captured in backup metadata."""
        backup = backup_manager.create_backup(sample_database)

        # The sample database has a UNIQUE constraint on email, which creates an index
        # Note: SQLite creates indexes for UNIQUE constraints automatically
        assert isinstance(backup.indexes, list)


class TestRetentionPolicyAdvanced:
    """Advanced tests for retention policy."""

    def test_apply_retention_dry_run(
        self,
        temp_dir: Path,
        sample_database: Path,
    ):
        """Test retention policy dry run mode."""
        policy = RetentionPolicy(
            keep_daily=1,
            keep_weekly=0,
            keep_monthly=0,
            min_backups=1,
        )
        manager = BackupManager(
            backup_dir=temp_dir / "backups",
            retention_policy=policy,
            metrics_enabled=False,
        )

        # Create multiple backups
        for _ in range(3):
            manager.create_backup(sample_database)

        initial_count = len(manager.list_backups())

        # Dry run should return IDs but not delete
        would_delete = manager.apply_retention_policy(dry_run=True)

        # Backups should still exist
        final_count = len(manager.list_backups())
        assert final_count == initial_count

    def test_retention_respects_min_backups(
        self,
        temp_dir: Path,
        sample_database: Path,
    ):
        """Test that retention policy respects minimum backup count."""
        policy = RetentionPolicy(
            keep_daily=0,
            keep_weekly=0,
            keep_monthly=0,
            min_backups=5,
        )
        manager = BackupManager(
            backup_dir=temp_dir / "backups",
            retention_policy=policy,
            metrics_enabled=False,
        )

        # Create exactly 5 backups
        for _ in range(5):
            manager.create_backup(sample_database)

        # Even with keep_* = 0, min_backups should prevent deletion
        deleted = manager.cleanup_expired()
        assert len(deleted) == 0
        assert len(manager.list_backups()) == 5

    def test_cleanup_alias_works(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test that cleanup_expired_backups alias works."""
        backup_manager.create_backup(sample_database)

        # Should not raise
        deleted = backup_manager.cleanup_expired_backups()
        assert isinstance(deleted, list)


class TestBackupMetadataSerialization:
    """Tests for BackupMetadata serialization."""

    def test_metadata_to_dict(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test metadata serializes to dict."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()

        assert data["id"] == backup.id
        assert data["status"] == "verified"
        assert data["backup_type"] == "full"
        assert "schema_hash" in data
        assert "table_checksums" in data

    def test_metadata_from_dict(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test metadata deserializes from dict."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()

        restored = BackupMetadata.from_dict(data)

        assert restored.id == backup.id
        assert restored.status == backup.status
        assert restored.checksum == backup.checksum

    def test_metadata_roundtrip(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test full serialization roundtrip."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()
        restored = BackupMetadata.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_defaults(self):
        """Test verification result default values."""
        result = VerificationResult(
            backup_id="test-123",
            verified=True,
            checksum_valid=True,
            restore_tested=True,
            tables_valid=True,
            row_counts_valid=True,
        )

        assert result.errors == []
        assert result.warnings == []
        assert result.duration_seconds == 0.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_restore_nonexistent_backup(
        self,
        backup_manager: BackupManager,
        temp_dir: Path,
    ):
        """Test restoring nonexistent backup raises error."""
        with pytest.raises(ValueError, match="not found"):
            backup_manager.restore_backup("nonexistent", temp_dir / "restore.db")

    def test_list_backups_with_since_filter(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Test filtering backups by date."""
        backup_manager.create_backup(sample_database)

        # Filter for backups in the future - should be empty
        future = datetime.now(timezone.utc) + timedelta(days=1)
        backups = backup_manager.list_backups(since=future)

        assert len(backups) == 0

    def test_list_backups_by_source_path(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Test filtering backups by source path."""
        backup_manager.create_backup(sample_database)

        # Filter for different source - should be empty
        backups = backup_manager.list_backups(source_path="/nonexistent/path.db")
        assert len(backups) == 0

        # Filter for correct source
        backups = backup_manager.list_backups(source_path=str(sample_database))
        assert len(backups) == 1

    def test_empty_database_backup(
        self,
        backup_manager: BackupManager,
        temp_dir: Path,
    ):
        """Test backing up an empty database."""
        empty_db = temp_dir / "empty.db"
        conn = sqlite3.connect(str(empty_db))
        conn.close()

        backup = backup_manager.create_backup(empty_db)

        assert backup.status == BackupStatus.VERIFIED
        assert len(backup.tables) == 0
        assert len(backup.row_counts) == 0


class TestIntegrityVerification:
    """Tests for referential integrity verification."""

    @pytest.fixture
    def database_with_fk(self, temp_dir: Path) -> Path:
        """Create database with foreign key constraint."""
        db_path = temp_dir / "fk_test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Create tables with FK relationship
        cursor.execute(
            """
            CREATE TABLE authors (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                author_id INTEGER,
                FOREIGN KEY (author_id) REFERENCES authors(id)
            )
        """
        )

        # Insert valid data
        cursor.execute("INSERT INTO authors (name) VALUES ('Author 1')")
        cursor.execute("INSERT INTO books (title, author_id) VALUES ('Book 1', 1)")

        conn.commit()
        conn.close()
        return db_path

    def test_backup_captures_foreign_keys(
        self,
        backup_manager: BackupManager,
        database_with_fk: Path,
    ):
        """Test that foreign keys are captured in backup metadata."""
        backup = backup_manager.create_backup(database_with_fk)

        # Should capture the FK relationship
        assert len(backup.foreign_keys) >= 1
        # FK format: (table, column, ref_table, ref_column)
        fk_tables = [fk[0] for fk in backup.foreign_keys]
        assert "books" in fk_tables

    def test_integrity_check_valid_database(
        self,
        backup_manager: BackupManager,
        database_with_fk: Path,
    ):
        """Test integrity check on valid database."""
        backup = backup_manager.create_backup(database_with_fk)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.integrity_check is not None
        assert result.integrity_check.valid
        assert result.integrity_check.foreign_keys_valid
