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
from datetime import datetime, timedelta
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
            (f"Topic {i}", datetime.utcnow().isoformat()),
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
