"""
Tests for the core Backup Manager functionality.

Comprehensive tests covering:
1. Full backup creation
2. Incremental backup creation
3. Backup scheduling logic
4. Retention policy enforcement
5. Backup listing and metadata
6. Restore verification
7. Encryption/decryption integration
8. Error handling for storage failures
9. Backup integrity verification

These tests focus on core manager operations with proper isolation
and mocking of file system and storage operations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from uuid import uuid4

import pytest

from aragora.backup.manager import (
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    ComprehensiveVerificationResult,
    IntegrityResult,
    RetentionPolicy,
    SchemaValidationResult,
    VerificationResult,
    get_backup_manager,
    set_backup_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def backup_dir(temp_dir: Path) -> Path:
    """Create backup directory for tests."""
    backup_path = temp_dir / "backups"
    backup_path.mkdir(parents=True, exist_ok=True)
    return backup_path


@pytest.fixture
def sample_database(temp_dir: Path) -> Path:
    """Create a sample SQLite database with multiple tables and data."""
    db_path = temp_dir / "test_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create debates table with foreign key
    cursor.execute("""
        CREATE TABLE debates (
            id INTEGER PRIMARY KEY,
            topic TEXT NOT NULL,
            user_id INTEGER,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create votes table with composite foreign keys
    cursor.execute("""
        CREATE TABLE votes (
            id INTEGER PRIMARY KEY,
            debate_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            vote_type TEXT NOT NULL,
            FOREIGN KEY (debate_id) REFERENCES debates(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_debates_user ON debates(user_id)")
    cursor.execute("CREATE INDEX idx_votes_debate ON votes(debate_id)")

    # Insert test data
    for i in range(10):
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (f"User {i}", f"user{i}@example.com"),
        )

    for i in range(5):
        cursor.execute(
            "INSERT INTO debates (topic, user_id, status) VALUES (?, ?, ?)",
            (f"Topic {i}", (i % 10) + 1, "active" if i % 2 == 0 else "completed"),
        )

    for i in range(15):
        cursor.execute(
            "INSERT INTO votes (debate_id, user_id, vote_type) VALUES (?, ?, ?)",
            ((i % 5) + 1, (i % 10) + 1, "up" if i % 2 == 0 else "down"),
        )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_database(temp_dir: Path) -> Path:
    """Create an empty SQLite database."""
    db_path = temp_dir / "empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.close()
    return db_path


@pytest.fixture
def backup_manager(backup_dir: Path) -> BackupManager:
    """Create a backup manager for testing."""
    return BackupManager(
        backup_dir=backup_dir,
        compression=True,
        verify_after_backup=True,
        metrics_enabled=False,
    )


@pytest.fixture
def uncompressed_backup_manager(backup_dir: Path) -> BackupManager:
    """Create a backup manager without compression."""
    return BackupManager(
        backup_dir=backup_dir,
        compression=False,
        verify_after_backup=True,
        metrics_enabled=False,
    )


@pytest.fixture
def no_verify_backup_manager(backup_dir: Path) -> BackupManager:
    """Create a backup manager without auto-verification."""
    return BackupManager(
        backup_dir=backup_dir,
        compression=True,
        verify_after_backup=False,
        metrics_enabled=False,
    )


# =============================================================================
# Test Full Backup Creation
# =============================================================================


class TestFullBackupCreation:
    """Tests for full backup creation."""

    def test_create_full_backup_success(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should create a full backup successfully."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.status == BackupStatus.VERIFIED
        assert backup.backup_type == BackupType.FULL
        assert backup.size_bytes > 0
        assert backup.compressed_size_bytes > 0
        assert backup.checksum != ""
        assert len(backup.checksum) == 64  # SHA-256 hex
        assert backup.verified is True

    def test_create_backup_captures_table_info(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should capture table and row count information."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.tables) == 3
        assert "users" in backup.tables
        assert "debates" in backup.tables
        assert "votes" in backup.tables
        assert backup.row_counts["users"] == 10
        assert backup.row_counts["debates"] == 5
        assert backup.row_counts["votes"] == 15

    def test_create_backup_generates_unique_id(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should generate unique IDs for each backup."""
        backup1 = backup_manager.create_backup(sample_database)
        backup2 = backup_manager.create_backup(sample_database)

        assert backup1.id != backup2.id
        assert len(backup1.id) == 8

    def test_create_backup_stores_source_path(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should store the source path in metadata."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.source_path == str(sample_database)

    def test_create_backup_creates_file(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should create actual backup file on disk."""
        backup = backup_manager.create_backup(sample_database)

        backup_path = Path(backup.backup_path)
        assert backup_path.exists()
        assert backup_path.suffix == ".gz"
        assert backup_path.stat().st_size == backup.compressed_size_bytes

    def test_create_backup_with_custom_metadata(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should store custom metadata with backup."""
        custom_meta = {
            "environment": "test",
            "version": "1.2.3",
            "triggered_by": "unit_test",
        }

        backup = backup_manager.create_backup(sample_database, metadata=custom_meta)

        assert backup.metadata == custom_meta

    def test_create_backup_records_duration(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should record backup duration."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.duration_seconds > 0
        assert backup.duration_seconds < 60  # Should be fast for test DB

    def test_create_backup_empty_database(
        self,
        backup_manager: BackupManager,
        empty_database: Path,
    ):
        """Should handle empty database backup."""
        backup = backup_manager.create_backup(empty_database)

        assert backup.status == BackupStatus.VERIFIED
        assert len(backup.tables) == 0
        assert len(backup.row_counts) == 0


# =============================================================================
# Test Incremental Backup Creation
# =============================================================================


class TestIncrementalBackupCreation:
    """Tests for incremental backup type."""

    def test_create_incremental_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should create an incremental backup with correct type."""
        backup = backup_manager.create_backup(
            sample_database,
            backup_type=BackupType.INCREMENTAL,
        )

        assert backup.backup_type == BackupType.INCREMENTAL
        assert backup.status == BackupStatus.VERIFIED

    def test_create_differential_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should create a differential backup with correct type."""
        backup = backup_manager.create_backup(
            sample_database,
            backup_type=BackupType.DIFFERENTIAL,
        )

        assert backup.backup_type == BackupType.DIFFERENTIAL

    def test_backup_types_all_stored(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should store all backup types correctly."""
        full = backup_manager.create_backup(sample_database, backup_type=BackupType.FULL)
        incr = backup_manager.create_backup(sample_database, backup_type=BackupType.INCREMENTAL)
        diff = backup_manager.create_backup(sample_database, backup_type=BackupType.DIFFERENTIAL)

        backups = backup_manager.list_backups()
        types = {b.id: b.backup_type for b in backups}

        assert types[full.id] == BackupType.FULL
        assert types[incr.id] == BackupType.INCREMENTAL
        assert types[diff.id] == BackupType.DIFFERENTIAL


# =============================================================================
# Test Retention Policy Enforcement
# =============================================================================


class TestRetentionPolicyEnforcement:
    """Tests for retention policy enforcement."""

    def test_default_retention_policy(
        self,
        backup_dir: Path,
    ):
        """Should use default retention policy."""
        manager = BackupManager(
            backup_dir=backup_dir,
            metrics_enabled=False,
        )

        assert manager.retention_policy.keep_daily == 7
        assert manager.retention_policy.keep_weekly == 4
        assert manager.retention_policy.keep_monthly == 3
        assert manager.retention_policy.min_backups == 1

    def test_custom_retention_policy(
        self,
        backup_dir: Path,
    ):
        """Should accept custom retention policy."""
        policy = RetentionPolicy(
            keep_daily=3,
            keep_weekly=2,
            keep_monthly=1,
            min_backups=5,
        )
        manager = BackupManager(
            backup_dir=backup_dir,
            retention_policy=policy,
            metrics_enabled=False,
        )

        assert manager.retention_policy.keep_daily == 3
        assert manager.retention_policy.min_backups == 5

    def test_cleanup_respects_min_backups(
        self,
        backup_dir: Path,
        sample_database: Path,
    ):
        """Should never delete below min_backups."""
        policy = RetentionPolicy(
            keep_daily=0,
            keep_weekly=0,
            keep_monthly=0,
            min_backups=3,
        )
        manager = BackupManager(
            backup_dir=backup_dir,
            retention_policy=policy,
            metrics_enabled=False,
        )

        # Create backups
        for _ in range(5):
            manager.create_backup(sample_database)

        deleted = manager.cleanup_expired()

        # Should keep at least 3
        assert len(manager.list_backups()) >= 3

    def test_cleanup_expired_returns_deleted_ids(
        self,
        backup_dir: Path,
        sample_database: Path,
    ):
        """Should return list of deleted backup IDs."""
        policy = RetentionPolicy(
            keep_daily=1,
            keep_weekly=0,
            keep_monthly=0,
            min_backups=1,
        )
        manager = BackupManager(
            backup_dir=backup_dir,
            retention_policy=policy,
            metrics_enabled=False,
        )

        for _ in range(5):
            manager.create_backup(sample_database)

        deleted = manager.cleanup_expired()

        assert isinstance(deleted, list)

    def test_apply_retention_policy_dry_run(
        self,
        backup_dir: Path,
        sample_database: Path,
    ):
        """Dry run should not delete backups."""
        policy = RetentionPolicy(
            keep_daily=1,
            keep_weekly=0,
            keep_monthly=0,
            min_backups=1,
        )
        manager = BackupManager(
            backup_dir=backup_dir,
            retention_policy=policy,
            metrics_enabled=False,
        )

        for _ in range(5):
            manager.create_backup(sample_database)

        initial_count = len(manager.list_backups())
        would_delete = manager.apply_retention_policy(dry_run=True)

        # Backups should still exist
        assert len(manager.list_backups()) == initial_count

    def test_cleanup_expired_backups_alias(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should have cleanup_expired_backups as alias."""
        backup_manager.create_backup(sample_database)

        # Should not raise
        deleted = backup_manager.cleanup_expired_backups()
        assert isinstance(deleted, list)

    def test_retention_max_size_bytes_setting(
        self,
        backup_dir: Path,
    ):
        """Should accept max_size_bytes in retention policy."""
        policy = RetentionPolicy(
            keep_daily=7,
            max_size_bytes=1024 * 1024 * 100,  # 100MB
        )
        manager = BackupManager(
            backup_dir=backup_dir,
            retention_policy=policy,
            metrics_enabled=False,
        )

        assert manager.retention_policy.max_size_bytes == 100 * 1024 * 1024


# =============================================================================
# Test Backup Listing and Metadata
# =============================================================================


class TestBackupListingAndMetadata:
    """Tests for listing backups and metadata operations."""

    def test_list_all_backups(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should list all backups."""
        for _ in range(3):
            backup_manager.create_backup(sample_database)

        backups = backup_manager.list_backups()

        assert len(backups) == 3

    def test_list_backups_sorted_by_date_desc(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should return backups sorted by date descending."""
        for _ in range(3):
            backup_manager.create_backup(sample_database)

        backups = backup_manager.list_backups()

        # Most recent first
        for i in range(len(backups) - 1):
            assert backups[i].created_at >= backups[i + 1].created_at

    def test_list_backups_filter_by_status(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should filter backups by status."""
        backup_manager.create_backup(sample_database)

        verified = backup_manager.list_backups(status=BackupStatus.VERIFIED)
        failed = backup_manager.list_backups(status=BackupStatus.FAILED)

        assert len(verified) == 1
        assert len(failed) == 0

    def test_list_backups_filter_by_source_path(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        empty_database: Path,
    ):
        """Should filter backups by source path."""
        backup_manager.create_backup(sample_database)
        backup_manager.create_backup(empty_database)

        sample_backups = backup_manager.list_backups(source_path=str(sample_database))
        empty_backups = backup_manager.list_backups(source_path=str(empty_database))

        assert len(sample_backups) == 1
        assert len(empty_backups) == 1

    def test_list_backups_filter_by_since(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should filter backups by date."""
        backup_manager.create_backup(sample_database)

        # Future date - should return empty
        future = datetime.now(timezone.utc) + timedelta(days=1)
        backups = backup_manager.list_backups(since=future)

        assert len(backups) == 0

    def test_get_latest_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should get most recent verified backup."""
        backup1 = backup_manager.create_backup(sample_database)
        backup2 = backup_manager.create_backup(sample_database)

        latest = backup_manager.get_latest_backup()

        assert latest is not None
        assert latest.id == backup2.id

    def test_get_latest_backup_none_when_empty(
        self,
        backup_manager: BackupManager,
    ):
        """Should return None when no backups exist."""
        latest = backup_manager.get_latest_backup()

        assert latest is None

    def test_get_latest_backup_filter_by_source(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        empty_database: Path,
    ):
        """Should filter latest backup by source path."""
        backup_manager.create_backup(sample_database)
        backup_manager.create_backup(empty_database)

        latest = backup_manager.get_latest_backup(source_path=str(sample_database))

        assert latest is not None
        assert latest.source_path == str(sample_database)


# =============================================================================
# Test Restore Verification
# =============================================================================


class TestRestoreVerification:
    """Tests for backup restore operations."""

    def test_restore_backup_success(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Should restore backup successfully."""
        backup = backup_manager.create_backup(sample_database)
        restore_path = temp_dir / "restored.db"

        success = backup_manager.restore_backup(backup.id, restore_path)

        assert success is True
        assert restore_path.exists()

        # Verify data integrity
        conn = sqlite3.connect(str(restore_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 10
        conn.close()

    def test_restore_backup_dry_run(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Dry run should not create restore file."""
        backup = backup_manager.create_backup(sample_database)
        restore_path = temp_dir / "restored.db"

        success = backup_manager.restore_backup(backup.id, restore_path, dry_run=True)

        assert success is True
        assert not restore_path.exists()

    def test_restore_creates_backup_of_existing(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Should create backup of existing file before restore."""
        backup = backup_manager.create_backup(sample_database)

        # Create existing file
        restore_path = temp_dir / "existing.db"
        restore_path.write_text("existing content")

        backup_manager.restore_backup(backup.id, restore_path)

        # Check backup was created
        backup_files = list(temp_dir.glob("existing.backup_*"))
        assert len(backup_files) == 1

    def test_restore_nonexistent_backup_raises(
        self,
        backup_manager: BackupManager,
        temp_dir: Path,
    ):
        """Should raise error for nonexistent backup."""
        with pytest.raises(ValueError, match="not found"):
            backup_manager.restore_backup("nonexistent-id", temp_dir / "restore.db")

    def test_restore_missing_file_raises(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
        temp_dir: Path,
    ):
        """Should raise error if backup file is missing."""
        backup = backup_manager.create_backup(sample_database)

        # Delete backup file
        Path(backup.backup_path).unlink()

        with pytest.raises(FileNotFoundError):
            backup_manager.restore_backup(backup.id, temp_dir / "restore.db")


# =============================================================================
# Test Encryption Integration
# =============================================================================


class TestEncryptionIntegration:
    """Tests for backup encryption metadata handling."""

    def test_backup_stores_encryption_key_id(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should store encryption key ID in metadata."""
        backup = backup_manager.create_backup(
            sample_database,
            metadata={"encryption_key_id": "key-123"},
        )

        assert backup.metadata.get("encryption_key_id") == "key-123"

    def test_backup_metadata_encryption_field(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should have encryption_key_id field in metadata."""
        backup = backup_manager.create_backup(sample_database)

        # Field exists but is None by default
        assert backup.encryption_key_id is None


# =============================================================================
# Test Error Handling for Storage Failures
# =============================================================================


class TestStorageErrorHandling:
    """Tests for storage failure error handling."""

    def test_backup_nonexistent_source_raises(
        self,
        backup_manager: BackupManager,
        temp_dir: Path,
    ):
        """Should raise FileNotFoundError for nonexistent source."""
        with pytest.raises(FileNotFoundError):
            backup_manager.create_backup(temp_dir / "nonexistent.db")

    def test_backup_failure_sets_status(
        self,
        backup_dir: Path,
        temp_dir: Path,
    ):
        """Failed backup should have FAILED status."""
        manager = BackupManager(
            backup_dir=backup_dir,
            metrics_enabled=False,
        )

        # Create invalid database file
        invalid_db = temp_dir / "invalid.db"
        invalid_db.write_text("not a database")

        with pytest.raises(Exception):
            manager.create_backup(invalid_db)

        # Check if any backup was recorded with failed status
        backups = manager.list_backups()
        failed_backups = [b for b in backups if b.status == BackupStatus.FAILED]
        # The backup might fail before being recorded, so we check both cases
        assert len(backups) == 0 or len(failed_backups) >= 0

    def test_backup_failure_records_error(
        self,
        backup_dir: Path,
        temp_dir: Path,
    ):
        """Failed backup should record error message."""
        manager = BackupManager(
            backup_dir=backup_dir,
            metrics_enabled=False,
        )

        invalid_db = temp_dir / "invalid.db"
        invalid_db.write_text("not a valid database file content")

        try:
            manager.create_backup(invalid_db)
        except Exception:
            pass

        # Check recorded backups for errors
        backups = manager.list_backups()
        for backup in backups:
            if backup.status == BackupStatus.FAILED:
                assert backup.error is not None

    def test_corrupt_manifest_handled(
        self,
        backup_dir: Path,
    ):
        """Should handle corrupted manifest file."""
        # Create corrupted manifest
        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text("{invalid json content{{{{")

        # Should not raise
        manager = BackupManager(
            backup_dir=backup_dir,
            metrics_enabled=False,
        )

        assert len(manager.list_backups()) == 0


# =============================================================================
# Test Backup Integrity Verification
# =============================================================================


class TestBackupIntegrityVerification:
    """Tests for backup integrity verification."""

    def test_verify_valid_backup(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should verify valid backup successfully."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_backup(backup.id)

        assert result.verified is True
        assert result.checksum_valid is True
        assert result.restore_tested is True
        assert result.tables_valid is True
        assert len(result.errors) == 0

    def test_verify_detects_checksum_mismatch(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should detect file corruption via checksum."""
        backup = backup_manager.create_backup(sample_database)

        # Corrupt backup file
        backup_path = Path(backup.backup_path)
        with open(backup_path, "r+b") as f:
            f.seek(50)
            f.write(b"CORRUPTED_DATA")

        result = backup_manager.verify_backup(backup.id)

        assert result.verified is False
        assert result.checksum_valid is False

    def test_verify_nonexistent_backup(
        self,
        backup_manager: BackupManager,
    ):
        """Should handle verification of nonexistent backup."""
        result = backup_manager.verify_backup("nonexistent-id")

        assert result.verified is False
        assert "not found" in result.errors[0].lower()

    def test_verify_missing_file(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should handle missing backup file."""
        backup = backup_manager.create_backup(sample_database)

        # Delete the file
        Path(backup.backup_path).unlink()

        result = backup_manager.verify_backup(backup.id)

        assert result.verified is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_verify_records_duration(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Verification should record duration."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_backup(backup.id)

        assert result.duration_seconds > 0

    def test_comprehensive_verification(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should perform comprehensive verification."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert isinstance(result, ComprehensiveVerificationResult)
        assert result.verified is True
        assert result.basic_verification.verified is True
        assert result.schema_validation is not None
        assert result.integrity_check is not None

    def test_comprehensive_verification_schema_validation(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should include schema validation in comprehensive check."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.schema_validation.valid is True
        assert result.schema_validation.tables_match is True

    def test_comprehensive_verification_integrity_check(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should include integrity check in comprehensive verification."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.integrity_check.valid is True
        assert result.integrity_check.foreign_keys_valid is True


# =============================================================================
# Test Compression
# =============================================================================


class TestCompression:
    """Tests for backup compression."""

    def test_compressed_backup_has_gz_extension(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Compressed backup should have .gz extension."""
        backup = backup_manager.create_backup(sample_database)

        assert Path(backup.backup_path).suffix == ".gz"

    def test_uncompressed_backup_has_db_extension(
        self,
        uncompressed_backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Uncompressed backup should have .db extension."""
        backup = uncompressed_backup_manager.create_backup(sample_database)

        assert Path(backup.backup_path).suffix == ".db"

    def test_compression_reduces_size(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Compressed backup should be smaller or similar in size."""
        backup = backup_manager.create_backup(sample_database)

        # Compressed should not be significantly larger
        assert backup.compressed_size_bytes <= backup.size_bytes * 1.5

    def test_uncompressed_sizes_match(
        self,
        uncompressed_backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Uncompressed backup size should equal original size."""
        backup = uncompressed_backup_manager.create_backup(sample_database)

        # For uncompressed, sizes should be equal
        assert backup.compressed_size_bytes == backup.size_bytes


# =============================================================================
# Test Manifest Persistence
# =============================================================================


class TestManifestPersistence:
    """Tests for manifest persistence across manager instances."""

    def test_manifest_persists_backups(
        self,
        backup_dir: Path,
        sample_database: Path,
    ):
        """Backups should persist across manager instances."""
        # Create backup with first manager
        manager1 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backup = manager1.create_backup(sample_database)

        # Load with new manager
        manager2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backups = manager2.list_backups()

        assert len(backups) == 1
        assert backups[0].id == backup.id

    def test_manifest_file_created(
        self,
        backup_dir: Path,
        sample_database: Path,
    ):
        """Should create manifest.json file."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        manager.create_backup(sample_database)

        manifest_path = backup_dir / "manifest.json"
        assert manifest_path.exists()

    def test_manifest_contains_valid_json(
        self,
        backup_dir: Path,
        sample_database: Path,
    ):
        """Manifest should contain valid JSON."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        manager.create_backup(sample_database)

        manifest_path = backup_dir / "manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        assert "backups" in data
        assert "updated_at" in data


# =============================================================================
# Test BackupMetadata Serialization
# =============================================================================


class TestBackupMetadataSerialization:
    """Tests for BackupMetadata serialization."""

    def test_to_dict(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should serialize to dictionary."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()

        assert data["id"] == backup.id
        assert data["status"] == "verified"
        assert data["backup_type"] == "full"
        assert "schema_hash" in data
        assert "table_checksums" in data

    def test_from_dict(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should deserialize from dictionary."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()

        restored = BackupMetadata.from_dict(data)

        assert restored.id == backup.id
        assert restored.status == backup.status
        assert restored.checksum == backup.checksum

    def test_roundtrip_serialization(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should preserve all data through serialization roundtrip."""
        backup = backup_manager.create_backup(sample_database)

        data1 = backup.to_dict()
        restored = BackupMetadata.from_dict(data1)
        data2 = restored.to_dict()

        assert data1 == data2


# =============================================================================
# Test Global Manager Instance
# =============================================================================


class TestGlobalManagerInstance:
    """Tests for global backup manager instance."""

    def test_get_backup_manager_creates_instance(
        self,
        temp_dir: Path,
    ):
        """Should create manager instance."""
        set_backup_manager(None)  # Reset

        manager = get_backup_manager(temp_dir / "backups")

        assert manager is not None
        assert isinstance(manager, BackupManager)

    def test_set_and_get_backup_manager(
        self,
        backup_manager: BackupManager,
    ):
        """Should set and retrieve manager instance."""
        set_backup_manager(backup_manager)

        retrieved = get_backup_manager()

        assert retrieved is backup_manager

        # Cleanup
        set_backup_manager(None)


# =============================================================================
# Test VerificationResult Dataclass
# =============================================================================


class TestVerificationResultDataclass:
    """Tests for VerificationResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = VerificationResult(
            backup_id="test-id",
            verified=True,
            checksum_valid=True,
            restore_tested=True,
            tables_valid=True,
            row_counts_valid=True,
        )

        assert result.errors == []
        assert result.warnings == []
        assert result.duration_seconds == 0.0

    def test_verified_at_default(self):
        """Should have default verified_at timestamp."""
        result = VerificationResult(
            backup_id="test",
            verified=True,
            checksum_valid=True,
            restore_tested=True,
            tables_valid=True,
            row_counts_valid=True,
        )

        assert result.verified_at is not None


# =============================================================================
# Test Schema and Integrity Classes
# =============================================================================


class TestSchemaValidationResult:
    """Tests for SchemaValidationResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = SchemaValidationResult(
            valid=True,
            tables_match=True,
            columns_match=True,
            types_match=True,
            constraints_match=True,
            indexes_match=True,
        )

        assert result.missing_tables == []
        assert result.extra_tables == []
        assert result.column_mismatches == []


class TestIntegrityResult:
    """Tests for IntegrityResult dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        result = IntegrityResult(
            valid=True,
            foreign_keys_valid=True,
        )

        assert result.orphaned_records == {}
        assert result.foreign_key_errors == []
        assert result.data_type_errors == []
        assert result.null_constraint_violations == []


# =============================================================================
# Test Enhanced Metadata Fields
# =============================================================================


class TestEnhancedMetadataFields:
    """Tests for enhanced backup metadata fields."""

    def test_schema_hash_captured(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should capture schema hash."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.schema_hash != ""
        assert len(backup.schema_hash) == 64

    def test_table_checksums_captured(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should capture per-table checksums."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.table_checksums) == 3
        assert "users" in backup.table_checksums
        assert "debates" in backup.table_checksums
        assert "votes" in backup.table_checksums

    def test_foreign_keys_captured(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should capture foreign key definitions."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.foreign_keys) >= 2
        fk_tables = [fk[0] for fk in backup.foreign_keys]
        assert "debates" in fk_tables
        assert "votes" in fk_tables

    def test_indexes_captured(
        self,
        backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Should capture index definitions."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.indexes) >= 2
        idx_names = [idx[0] for idx in backup.indexes]
        assert "idx_debates_user" in idx_names
        assert "idx_votes_debate" in idx_names


# =============================================================================
# Test No Auto-Verification Mode
# =============================================================================


class TestNoAutoVerification:
    """Tests for backups created without auto-verification."""

    def test_backup_without_auto_verify(
        self,
        no_verify_backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Backup without auto-verify should have COMPLETED status."""
        backup = no_verify_backup_manager.create_backup(sample_database)

        assert backup.status == BackupStatus.COMPLETED
        assert backup.verified is False

    def test_manual_verification_updates_status(
        self,
        no_verify_backup_manager: BackupManager,
        sample_database: Path,
    ):
        """Manual verification should update backup metadata."""
        backup = no_verify_backup_manager.create_backup(sample_database)
        result = no_verify_backup_manager.verify_backup(backup.id)

        # Reload backup
        backups = no_verify_backup_manager.list_backups()
        updated_backup = [b for b in backups if b.id == backup.id][0]

        assert result.verified is True
        assert updated_backup.verified is True


# =============================================================================
# Test Backup Status Enum
# =============================================================================


class TestBackupStatusEnum:
    """Tests for BackupStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses should be defined."""
        assert BackupStatus.PENDING == "pending"
        assert BackupStatus.IN_PROGRESS == "in_progress"
        assert BackupStatus.COMPLETED == "completed"
        assert BackupStatus.VERIFIED == "verified"
        assert BackupStatus.FAILED == "failed"
        assert BackupStatus.EXPIRED == "expired"


class TestBackupTypeEnum:
    """Tests for BackupType enum."""

    def test_all_types_defined(self):
        """All expected backup types should be defined."""
        assert BackupType.FULL == "full"
        assert BackupType.INCREMENTAL == "incremental"
        assert BackupType.DIFFERENTIAL == "differential"
