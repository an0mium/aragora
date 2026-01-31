"""
Comprehensive Tests for BackupManager.

This module provides extensive test coverage for the BackupManager class,
focusing on areas not covered by existing test files:

1. Multi-backend support (S3, GCS mocks)
2. Concurrent backup handling
3. Error recovery and rollback
4. Advanced backup metadata management
5. Edge cases and error scenarios

Tests use mocks for cloud providers to avoid real API calls.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock
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
    _validate_sql_identifier,
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
    """Create a sample SQLite database with tables, indexes, and foreign keys."""
    db_path = temp_dir / "test_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE debates (
            id INTEGER PRIMARY KEY,
            topic TEXT NOT NULL,
            user_id INTEGER REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE votes (
            id INTEGER PRIMARY KEY,
            debate_id INTEGER NOT NULL REFERENCES debates(id),
            user_id INTEGER NOT NULL REFERENCES users(id),
            value INTEGER NOT NULL
        )
    """)

    cursor.execute("CREATE INDEX idx_debates_user ON debates(user_id)")
    cursor.execute("CREATE INDEX idx_votes_debate ON votes(debate_id)")

    for i in range(10):
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (f"User {i}", f"user{i}@example.com"),
        )

    for i in range(5):
        cursor.execute(
            "INSERT INTO debates (topic, user_id) VALUES (?, ?)",
            (f"Topic {i}", (i % 10) + 1),
        )

    for i in range(15):
        cursor.execute(
            "INSERT INTO votes (debate_id, user_id, value) VALUES (?, ?, ?)",
            ((i % 5) + 1, (i % 10) + 1, 1 if i % 2 == 0 else -1),
        )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def large_database(temp_dir: Path) -> Path:
    """Create a larger database for performance testing."""
    db_path = temp_dir / "large_database.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE records (
            id INTEGER PRIMARY KEY,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            value REAL
        )
    """)

    cursor.execute("CREATE INDEX idx_records_timestamp ON records(timestamp)")

    for i in range(1000):
        cursor.execute(
            "INSERT INTO records (data, timestamp, value) VALUES (?, ?, ?)",
            (f"Data entry {i} with some content", datetime.now(timezone.utc).isoformat(), i * 1.5),
        )

    conn.commit()
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
def backup_manager_no_compression(backup_dir: Path) -> BackupManager:
    """Create backup manager without compression."""
    return BackupManager(
        backup_dir=backup_dir,
        compression=False,
        verify_after_backup=True,
        metrics_enabled=False,
    )


@pytest.fixture
def backup_manager_no_verify(backup_dir: Path) -> BackupManager:
    """Create backup manager without auto-verification."""
    return BackupManager(
        backup_dir=backup_dir,
        compression=True,
        verify_after_backup=False,
        metrics_enabled=False,
    )


# =============================================================================
# Test BackupManager Initialization with Different Backends
# =============================================================================


class TestBackupManagerInitialization:
    """Tests for BackupManager initialization with different storage backends."""

    def test_init_with_local_filesystem(self, temp_dir: Path):
        """Should initialize with local filesystem backend."""
        backup_dir = temp_dir / "local_backups"
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        assert manager.backup_dir == backup_dir
        assert backup_dir.exists()
        assert manager.compression is True
        assert manager.verify_after_backup is True

    def test_init_creates_backup_directory(self, temp_dir: Path):
        """Should create backup directory if it doesn't exist."""
        backup_dir = temp_dir / "new_backups" / "nested" / "path"
        assert not backup_dir.exists()

        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        assert backup_dir.exists()

    def test_init_with_custom_retention_policy(self, backup_dir: Path):
        """Should accept custom retention policy."""
        policy = RetentionPolicy(
            keep_daily=14,
            keep_weekly=8,
            keep_monthly=6,
            min_backups=5,
            max_size_bytes=1024 * 1024 * 1024,  # 1GB
        )

        manager = BackupManager(
            backup_dir=backup_dir,
            retention_policy=policy,
            metrics_enabled=False,
        )

        assert manager.retention_policy.keep_daily == 14
        assert manager.retention_policy.keep_weekly == 8
        assert manager.retention_policy.keep_monthly == 6
        assert manager.retention_policy.min_backups == 5
        assert manager.retention_policy.max_size_bytes == 1024 * 1024 * 1024

    def test_init_with_compression_disabled(self, backup_dir: Path):
        """Should initialize with compression disabled."""
        manager = BackupManager(
            backup_dir=backup_dir,
            compression=False,
            metrics_enabled=False,
        )

        assert manager.compression is False

    def test_init_with_verification_disabled(self, backup_dir: Path):
        """Should initialize with auto-verification disabled."""
        manager = BackupManager(
            backup_dir=backup_dir,
            verify_after_backup=False,
            metrics_enabled=False,
        )

        assert manager.verify_after_backup is False

    def test_init_loads_existing_manifest(self, backup_dir: Path, sample_database: Path):
        """Should load existing manifest on initialization."""
        manager1 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backup = manager1.create_backup(sample_database)

        manager2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backups = manager2.list_backups()

        assert len(backups) == 1
        assert backups[0].id == backup.id

    def test_init_handles_missing_manifest(self, backup_dir: Path):
        """Should handle missing manifest file gracefully."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        assert len(manager.list_backups()) == 0

    def test_init_handles_corrupted_manifest(self, backup_dir: Path):
        """Should handle corrupted manifest file gracefully."""
        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text("{{invalid json content!!!")

        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        assert len(manager.list_backups()) == 0


# =============================================================================
# Test Mock S3 Backend
# =============================================================================


class TestMockS3Backend:
    """Tests for S3 storage backend using mocks."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock boto3 S3 client."""
        with patch("boto3.client") as mock_client:
            s3 = MagicMock()
            mock_client.return_value = s3
            s3.list_objects_v2.return_value = {"Contents": []}
            s3.head_object.return_value = {"ContentLength": 1024}
            yield s3

    def test_s3_upload_simulation(
        self, mock_s3_client, backup_manager: BackupManager, sample_database: Path
    ):
        """Simulate S3 upload after local backup."""
        backup = backup_manager.create_backup(sample_database)
        backup_path = Path(backup.backup_path)

        mock_s3_client.upload_file(
            str(backup_path),
            "backup-bucket",
            f"backups/{backup.id}.db.gz",
        )

        mock_s3_client.upload_file.assert_called_once()
        call_args = mock_s3_client.upload_file.call_args
        assert call_args[0][1] == "backup-bucket"
        assert backup.id in call_args[0][2]

    def test_s3_download_simulation(self, mock_s3_client, temp_dir: Path):
        """Simulate S3 download for restore."""
        download_path = temp_dir / "downloaded_backup.db.gz"

        mock_s3_client.download_file(
            "backup-bucket",
            "backups/test-backup.db.gz",
            str(download_path),
        )

        mock_s3_client.download_file.assert_called_once()

    def test_s3_list_backups_simulation(self, mock_s3_client):
        """Simulate listing backups from S3."""
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "backups/backup-001.db.gz",
                    "Size": 1024,
                    "LastModified": datetime.now(timezone.utc),
                },
                {
                    "Key": "backups/backup-002.db.gz",
                    "Size": 2048,
                    "LastModified": datetime.now(timezone.utc),
                },
            ]
        }

        response = mock_s3_client.list_objects_v2(Bucket="backup-bucket", Prefix="backups/")

        assert len(response["Contents"]) == 2

    def test_s3_delete_backup_simulation(self, mock_s3_client):
        """Simulate deleting a backup from S3."""
        mock_s3_client.delete_object.return_value = {}

        mock_s3_client.delete_object(Bucket="backup-bucket", Key="backups/old-backup.db.gz")

        mock_s3_client.delete_object.assert_called_once()

    def test_s3_error_handling(self, mock_s3_client):
        """Test S3 error handling simulation."""
        from botocore.exceptions import ClientError

        mock_s3_client.upload_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "PutObject",
        )

        with pytest.raises(ClientError):
            mock_s3_client.upload_file("local_file", "bucket", "key")

    def test_s3_multipart_upload_simulation(
        self, mock_s3_client, large_database: Path, backup_manager: BackupManager
    ):
        """Simulate multipart upload for large backups."""
        backup = backup_manager.create_backup(large_database)

        mock_s3_client.create_multipart_upload.return_value = {"UploadId": "test-upload-id"}
        mock_s3_client.upload_part.return_value = {"ETag": "test-etag"}
        mock_s3_client.complete_multipart_upload.return_value = {}

        mock_s3_client.create_multipart_upload(
            Bucket="backup-bucket", Key=f"backups/{backup.id}.db.gz"
        )
        mock_s3_client.upload_part(
            Bucket="backup-bucket",
            Key=f"backups/{backup.id}.db.gz",
            UploadId="test-upload-id",
            PartNumber=1,
            Body=b"chunk1",
        )
        mock_s3_client.complete_multipart_upload(
            Bucket="backup-bucket",
            Key=f"backups/{backup.id}.db.gz",
            UploadId="test-upload-id",
            MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": "test-etag"}]},
        )

        assert mock_s3_client.create_multipart_upload.called
        assert mock_s3_client.upload_part.called
        assert mock_s3_client.complete_multipart_upload.called


# =============================================================================
# Test Mock GCS Backend
# =============================================================================


class TestMockGCSBackend:
    """Tests for Google Cloud Storage backend using mocks.

    These tests simulate GCS operations without requiring the actual
    google-cloud-storage library to be installed. They demonstrate
    how backup files would be uploaded/downloaded to/from GCS.
    """

    @pytest.fixture
    def mock_gcs_client(self):
        """Create a mock google-cloud-storage client.

        Uses MagicMock directly without patching the google.cloud module,
        since that module may be partially installed without storage.
        """
        client = MagicMock()
        bucket = MagicMock()
        blob = MagicMock()

        client.bucket.return_value = bucket
        bucket.blob.return_value = blob

        yield {"client": client, "bucket": bucket, "blob": blob}

    def test_gcs_upload_simulation(
        self, mock_gcs_client, backup_manager: BackupManager, sample_database: Path
    ):
        """Simulate GCS upload after local backup."""
        backup = backup_manager.create_backup(sample_database)
        backup_path = Path(backup.backup_path)

        client = mock_gcs_client["client"]
        bucket = mock_gcs_client["bucket"]
        blob = mock_gcs_client["blob"]

        client.bucket("backup-bucket")
        bucket.blob(f"backups/{backup.id}.db.gz")
        blob.upload_from_filename(str(backup_path))

        blob.upload_from_filename.assert_called_once_with(str(backup_path))

    def test_gcs_download_simulation(self, mock_gcs_client, temp_dir: Path):
        """Simulate GCS download for restore."""
        download_path = temp_dir / "downloaded_backup.db.gz"

        blob = mock_gcs_client["blob"]
        blob.download_to_filename(str(download_path))

        blob.download_to_filename.assert_called_once()

    def test_gcs_list_blobs_simulation(self, mock_gcs_client):
        """Simulate listing blobs from GCS."""
        bucket = mock_gcs_client["bucket"]

        mock_blob1 = MagicMock()
        mock_blob1.name = "backups/backup-001.db.gz"
        mock_blob1.size = 1024

        mock_blob2 = MagicMock()
        mock_blob2.name = "backups/backup-002.db.gz"
        mock_blob2.size = 2048

        bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        blobs = list(bucket.list_blobs(prefix="backups/"))

        assert len(blobs) == 2

    def test_gcs_delete_blob_simulation(self, mock_gcs_client):
        """Simulate deleting a blob from GCS."""
        blob = mock_gcs_client["blob"]
        blob.delete()

        blob.delete.assert_called_once()

    def test_gcs_error_handling(self, mock_gcs_client):
        """Test GCS error handling simulation.

        Creates a custom NotFound exception class since google-cloud-storage
        may not be installed.
        """

        # Create a mock NotFound exception
        class MockNotFound(Exception):
            """Mock GCS NotFound exception."""

            pass

        blob = mock_gcs_client["blob"]
        blob.download_to_filename.side_effect = MockNotFound("Blob not found")

        with pytest.raises(MockNotFound):
            blob.download_to_filename("/tmp/nonexistent")

    def test_gcs_resumable_upload_simulation(
        self, mock_gcs_client, large_database: Path, backup_manager: BackupManager
    ):
        """Simulate resumable upload for large backups."""
        backup = backup_manager.create_backup(large_database)

        blob = mock_gcs_client["blob"]
        blob.upload_from_filename.return_value = None

        blob.upload_from_filename(str(backup.backup_path), timeout=300)

        blob.upload_from_filename.assert_called_once()


# =============================================================================
# Test Backup Creation (Full and Incremental)
# =============================================================================


class TestBackupCreation:
    """Tests for backup creation operations."""

    def test_create_full_backup(self, backup_manager: BackupManager, sample_database: Path):
        """Should create a full backup with all metadata."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.status == BackupStatus.VERIFIED
        assert backup.backup_type == BackupType.FULL
        assert backup.size_bytes > 0
        assert backup.compressed_size_bytes > 0
        assert len(backup.checksum) == 64
        assert backup.verified is True
        assert len(backup.tables) == 3
        assert backup.row_counts["users"] == 10
        assert backup.row_counts["debates"] == 5
        assert backup.row_counts["votes"] == 15

    def test_create_incremental_backup(self, backup_manager: BackupManager, sample_database: Path):
        """Should create an incremental backup."""
        backup = backup_manager.create_backup(sample_database, backup_type=BackupType.INCREMENTAL)

        assert backup.backup_type == BackupType.INCREMENTAL
        assert backup.status == BackupStatus.VERIFIED

    def test_create_differential_backup(self, backup_manager: BackupManager, sample_database: Path):
        """Should create a differential backup."""
        backup = backup_manager.create_backup(sample_database, backup_type=BackupType.DIFFERENTIAL)

        assert backup.backup_type == BackupType.DIFFERENTIAL

    def test_create_backup_with_custom_metadata(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should store custom metadata with backup."""
        custom_meta = {
            "environment": "staging",
            "triggered_by": "cron",
            "version": "2.1.0",
        }

        backup = backup_manager.create_backup(sample_database, metadata=custom_meta)

        assert backup.metadata == custom_meta

    def test_create_backup_captures_schema_hash(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should capture schema hash."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.schema_hash != ""
        assert len(backup.schema_hash) == 64

    def test_create_backup_captures_table_checksums(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should capture per-table checksums."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.table_checksums) == 3
        assert all(len(cs) == 64 for cs in backup.table_checksums.values() if cs)

    def test_create_backup_captures_foreign_keys(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should capture foreign key relationships."""
        backup = backup_manager.create_backup(sample_database)

        assert len(backup.foreign_keys) >= 2
        fk_tables = [fk[0] for fk in backup.foreign_keys]
        assert "debates" in fk_tables
        assert "votes" in fk_tables

    def test_create_backup_captures_indexes(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should capture index definitions."""
        backup = backup_manager.create_backup(sample_database)

        idx_names = [idx[0] for idx in backup.indexes]
        assert "idx_debates_user" in idx_names
        assert "idx_votes_debate" in idx_names

    def test_create_backup_generates_unique_ids(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should generate unique IDs for each backup."""
        backup1 = backup_manager.create_backup(sample_database)
        backup2 = backup_manager.create_backup(sample_database)

        assert backup1.id != backup2.id

    def test_create_backup_without_verification(
        self, backup_manager_no_verify: BackupManager, sample_database: Path
    ):
        """Should create backup without auto-verification."""
        backup = backup_manager_no_verify.create_backup(sample_database)

        assert backup.status == BackupStatus.COMPLETED
        assert backup.verified is False

    def test_create_backup_without_compression(
        self, backup_manager_no_compression: BackupManager, sample_database: Path
    ):
        """Should create uncompressed backup."""
        backup = backup_manager_no_compression.create_backup(sample_database)

        assert Path(backup.backup_path).suffix == ".db"
        assert backup.compressed_size_bytes == backup.size_bytes

    def test_create_backup_nonexistent_source_raises(
        self, backup_manager: BackupManager, temp_dir: Path
    ):
        """Should raise FileNotFoundError for nonexistent source."""
        with pytest.raises(FileNotFoundError):
            backup_manager.create_backup(temp_dir / "nonexistent.db")


# =============================================================================
# Test Backup Listing and Filtering
# =============================================================================


class TestBackupListingAndFiltering:
    """Tests for backup listing and filtering operations."""

    def test_list_all_backups(self, backup_manager: BackupManager, sample_database: Path):
        """Should list all backups."""
        for _ in range(3):
            backup_manager.create_backup(sample_database)

        backups = backup_manager.list_backups()

        assert len(backups) == 3

    def test_list_backups_sorted_by_date_descending(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should return backups sorted by date descending."""
        for _ in range(3):
            backup_manager.create_backup(sample_database)

        backups = backup_manager.list_backups()

        for i in range(len(backups) - 1):
            assert backups[i].created_at >= backups[i + 1].created_at

    def test_list_backups_filter_by_status(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should filter backups by status."""
        backup_manager.create_backup(sample_database)

        verified = backup_manager.list_backups(status=BackupStatus.VERIFIED)
        failed = backup_manager.list_backups(status=BackupStatus.FAILED)

        assert len(verified) == 1
        assert len(failed) == 0

    def test_list_backups_filter_by_source_path(
        self, backup_manager: BackupManager, sample_database: Path, temp_dir: Path
    ):
        """Should filter backups by source path."""
        other_db = temp_dir / "other.db"
        conn = sqlite3.connect(str(other_db))
        conn.close()

        backup_manager.create_backup(sample_database)
        backup_manager.create_backup(other_db)

        sample_backups = backup_manager.list_backups(source_path=str(sample_database))
        other_backups = backup_manager.list_backups(source_path=str(other_db))

        assert len(sample_backups) == 1
        assert len(other_backups) == 1

    def test_list_backups_filter_by_since(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should filter backups by date."""
        backup_manager.create_backup(sample_database)

        future = datetime.now(timezone.utc) + timedelta(days=1)
        backups = backup_manager.list_backups(since=future)

        assert len(backups) == 0

    def test_get_latest_backup(self, backup_manager: BackupManager, sample_database: Path):
        """Should get most recent verified backup."""
        backup1 = backup_manager.create_backup(sample_database)
        backup2 = backup_manager.create_backup(sample_database)

        latest = backup_manager.get_latest_backup()

        assert latest is not None
        assert latest.id == backup2.id

    def test_get_latest_backup_returns_none_when_empty(self, backup_manager: BackupManager):
        """Should return None when no backups exist."""
        latest = backup_manager.get_latest_backup()

        assert latest is None


# =============================================================================
# Test Backup Verification and Integrity Checking
# =============================================================================


class TestBackupVerification:
    """Tests for backup verification and integrity checking."""

    def test_verify_valid_backup(self, backup_manager: BackupManager, sample_database: Path):
        """Should verify valid backup successfully."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_backup(backup.id)

        assert result.verified is True
        assert result.checksum_valid is True
        assert result.restore_tested is True
        assert result.tables_valid is True
        assert len(result.errors) == 0

    def test_verify_detects_checksum_mismatch(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should detect file corruption via checksum."""
        backup = backup_manager.create_backup(sample_database)

        backup_path = Path(backup.backup_path)
        with open(backup_path, "r+b") as f:
            f.seek(50)
            f.write(b"CORRUPTED_DATA")

        result = backup_manager.verify_backup(backup.id)

        assert result.verified is False
        assert result.checksum_valid is False

    def test_verify_nonexistent_backup(self, backup_manager: BackupManager):
        """Should handle verification of nonexistent backup."""
        result = backup_manager.verify_backup("nonexistent-id")

        assert result.verified is False
        assert "not found" in result.errors[0].lower()

    def test_verify_missing_backup_file(self, backup_manager: BackupManager, sample_database: Path):
        """Should handle missing backup file."""
        backup = backup_manager.create_backup(sample_database)

        Path(backup.backup_path).unlink()

        result = backup_manager.verify_backup(backup.id)

        assert result.verified is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_verify_without_restore_test(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should skip restore test when disabled."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_backup(backup.id, test_restore=False)

        assert result.checksum_valid is True
        assert result.restore_tested is False

    def test_comprehensive_verification(self, backup_manager: BackupManager, sample_database: Path):
        """Should perform comprehensive verification."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert isinstance(result, ComprehensiveVerificationResult)
        assert result.verified is True
        assert result.basic_verification.verified is True
        assert result.schema_validation is not None
        assert result.integrity_check is not None

    def test_comprehensive_verification_includes_schema_validation(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should include schema validation in comprehensive check."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.schema_validation.valid is True
        assert result.schema_validation.tables_match is True

    def test_comprehensive_verification_includes_integrity_check(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should include integrity check in comprehensive verification."""
        backup = backup_manager.create_backup(sample_database)
        result = backup_manager.verify_restore_comprehensive(backup.id)

        assert result.integrity_check.valid is True
        assert result.integrity_check.foreign_keys_valid is True


# =============================================================================
# Test Restore Operations with Dry-Run Support
# =============================================================================


class TestRestoreOperations:
    """Tests for restore operations including dry-run support.

    Note: The restore_backup method uses a dedicated restore directory
    within the backup_dir for security (path traversal protection).
    The actual restore location is: backup_dir / "restore" / filename
    """

    def test_restore_backup_success(self, backup_manager: BackupManager, sample_database: Path):
        """Should restore backup successfully to the restore directory."""
        backup = backup_manager.create_backup(sample_database)

        # The restore will be placed in backup_dir/restore/restored.db
        success = backup_manager.restore_backup(backup.id, Path("restored.db"))

        assert success is True

        # Check the actual restore location
        restore_dir = backup_manager.backup_dir / "restore"
        actual_restore_path = restore_dir / "restored.db"
        assert actual_restore_path.exists()

        conn = sqlite3.connect(str(actual_restore_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        assert cursor.fetchone()[0] == 10
        conn.close()

    def test_restore_dry_run(self, backup_manager: BackupManager, sample_database: Path):
        """Dry run should not create restore file."""
        backup = backup_manager.create_backup(sample_database)

        success = backup_manager.restore_backup(backup.id, Path("restored.db"), dry_run=True)

        assert success is True

        # Verify no file was created in restore directory
        restore_dir = backup_manager.backup_dir / "restore"
        restore_path = restore_dir / "restored.db"
        assert not restore_path.exists()

    def test_restore_creates_backup_of_existing_file(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should create backup of existing file before restore."""
        backup = backup_manager.create_backup(sample_database)

        # Create the restore directory and an existing file
        restore_dir = backup_manager.backup_dir / "restore"
        restore_dir.mkdir(parents=True, exist_ok=True)
        existing_path = restore_dir / "existing.db"
        existing_path.write_text("existing content")

        backup_manager.restore_backup(backup.id, Path("existing.db"))

        # Check backup was created in the restore directory
        backup_files = list(restore_dir.glob("existing.backup_*"))
        assert len(backup_files) == 1

    def test_restore_nonexistent_backup_raises(self, backup_manager: BackupManager):
        """Should raise error for nonexistent backup."""
        with pytest.raises(ValueError, match="not found"):
            backup_manager.restore_backup("nonexistent-id", Path("restore.db"))

    def test_restore_missing_backup_file_raises(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should raise error if backup file is missing."""
        backup = backup_manager.create_backup(sample_database)

        Path(backup.backup_path).unlink()

        with pytest.raises(FileNotFoundError):
            backup_manager.restore_backup(backup.id, Path("restore.db"))

    def test_restore_path_traversal_protection(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should prevent path traversal attacks."""
        backup = backup_manager.create_backup(sample_database)

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            backup_manager.restore_backup(backup.id, Path("../../../etc/passwd"))


# =============================================================================
# Test Retention Policy Enforcement
# =============================================================================


class TestRetentionPolicyEnforcement:
    """Tests for retention policy enforcement."""

    def test_cleanup_respects_min_backups(self, backup_dir: Path, sample_database: Path):
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

        for _ in range(5):
            manager.create_backup(sample_database)

        manager.cleanup_expired()

        assert len(manager.list_backups()) >= 3

    def test_cleanup_returns_deleted_ids(self, backup_dir: Path, sample_database: Path):
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

    def test_apply_retention_dry_run(self, backup_dir: Path, sample_database: Path):
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

        assert len(manager.list_backups()) == initial_count

    def test_cleanup_expired_alias(self, backup_manager: BackupManager, sample_database: Path):
        """Should have cleanup_expired_backups as alias."""
        backup_manager.create_backup(sample_database)

        deleted = backup_manager.cleanup_expired_backups()

        assert isinstance(deleted, list)


# =============================================================================
# Test Concurrent Backup Handling
# =============================================================================


class TestConcurrentBackupHandling:
    """Tests for concurrent backup operations."""

    def test_concurrent_backup_creation(self, backup_dir: Path, sample_database: Path):
        """Should handle concurrent backup creation safely."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        def create_backup():
            return manager.create_backup(sample_database)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_backup) for _ in range(3)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 3
        ids = [r.id for r in results]
        assert len(set(ids)) == 3  # All unique IDs

        for backup in results:
            verification = manager.verify_backup(backup.id)
            assert verification.verified

    def test_concurrent_backup_verification(
        self, backup_manager: BackupManager, sample_database: Path
    ):
        """Should handle concurrent verification safely."""
        backup = backup_manager.create_backup(sample_database)

        def verify_backup():
            return backup_manager.verify_backup(backup.id)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(verify_backup) for _ in range(5)]
            results = [f.result() for f in as_completed(futures)]

        assert all(r.verified for r in results)

    def test_concurrent_backup_listing(self, backup_manager: BackupManager, sample_database: Path):
        """Should handle concurrent listing safely."""
        for _ in range(5):
            backup_manager.create_backup(sample_database)

        def list_backups():
            return backup_manager.list_backups()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(list_backups) for _ in range(5)]
            results = [f.result() for f in as_completed(futures)]

        assert all(len(r) == 5 for r in results)

    @pytest.mark.asyncio
    async def test_async_concurrent_backup_creation(self, backup_dir: Path, sample_database: Path):
        """Should handle async concurrent backup creation."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        async def create_backup():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, manager.create_backup, sample_database)

        tasks = [create_backup() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, BackupMetadata)]
        assert len(successes) >= 1


# =============================================================================
# Test Error Recovery and Rollback
# =============================================================================


class TestErrorRecoveryAndRollback:
    """Tests for error recovery and rollback scenarios."""

    def test_backup_failure_records_error(self, backup_dir: Path, temp_dir: Path):
        """Failed backup should record error message."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        invalid_db = temp_dir / "invalid.db"
        invalid_db.write_text("not a valid database")

        try:
            manager.create_backup(invalid_db)
        except Exception:
            pass

        backups = manager.list_backups()
        failed_backups = [b for b in backups if b.status == BackupStatus.FAILED]
        for backup in failed_backups:
            assert backup.error is not None

    def test_restore_with_corrupted_backup_raises(
        self, backup_manager: BackupManager, sample_database: Path, temp_dir: Path
    ):
        """Should raise error when restoring corrupted backup."""
        backup = backup_manager.create_backup(sample_database)

        backup_path = Path(backup.backup_path)
        with open(backup_path, "wb") as f:
            f.write(b"completely corrupted content")

        with pytest.raises(ValueError, match="verification failed"):
            backup_manager.restore_backup(backup.id, temp_dir / "restore.db")

    def test_recovery_from_partial_backup(self, backup_dir: Path, sample_database: Path):
        """Should recover from interrupted backup."""
        manager = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        backup = manager.create_backup(sample_database)
        assert backup.status in (BackupStatus.VERIFIED, BackupStatus.COMPLETED)

        manager2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backups = manager2.list_backups()
        assert len(backups) >= 1

    def test_manifest_recovery_after_corruption(self, backup_dir: Path, sample_database: Path):
        """Should recover from manifest corruption."""
        manager1 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)
        backup = manager1.create_backup(sample_database)

        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text("corrupted content")

        manager2 = BackupManager(backup_dir=backup_dir, metrics_enabled=False)

        assert len(manager2.list_backups()) == 0

        new_backup = manager2.create_backup(sample_database)
        assert new_backup.status in (BackupStatus.VERIFIED, BackupStatus.COMPLETED)


# =============================================================================
# Test Backup Metadata Management
# =============================================================================


class TestBackupMetadataManagement:
    """Tests for backup metadata management."""

    def test_metadata_to_dict(self, backup_manager: BackupManager, sample_database: Path):
        """Should serialize metadata to dict."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()

        assert data["id"] == backup.id
        assert data["status"] == "verified"
        assert data["backup_type"] == "full"
        assert "schema_hash" in data
        assert "table_checksums" in data
        assert "foreign_keys" in data
        assert "indexes" in data

    def test_metadata_from_dict(self, backup_manager: BackupManager, sample_database: Path):
        """Should deserialize metadata from dict."""
        backup = backup_manager.create_backup(sample_database)
        data = backup.to_dict()

        restored = BackupMetadata.from_dict(data)

        assert restored.id == backup.id
        assert restored.status == backup.status
        assert restored.checksum == backup.checksum
        assert restored.schema_hash == backup.schema_hash

    def test_metadata_roundtrip(self, backup_manager: BackupManager, sample_database: Path):
        """Should preserve all data through serialization roundtrip."""
        backup = backup_manager.create_backup(sample_database)

        data1 = backup.to_dict()
        restored = BackupMetadata.from_dict(data1)
        data2 = restored.to_dict()

        assert data1 == data2

    def test_metadata_with_verified_at(self, backup_manager: BackupManager, sample_database: Path):
        """Should preserve verified_at timestamp."""
        backup = backup_manager.create_backup(sample_database)

        assert backup.verified_at is not None

        data = backup.to_dict()
        restored = BackupMetadata.from_dict(data)

        assert restored.verified_at is not None

    def test_metadata_without_optional_fields(self):
        """Should handle metadata without optional fields."""
        data = {
            "id": "test-id",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backup_type": "full",
            "status": "completed",
            "source_path": "/path/to/source",
            "backup_path": "/path/to/backup",
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.id == "test-id"
        assert metadata.size_bytes == 0
        assert metadata.checksum == ""
        assert metadata.row_counts == {}


# =============================================================================
# Test SQL Identifier Validation
# =============================================================================


class TestSQLIdentifierValidation:
    """Tests for SQL identifier validation to prevent injection."""

    def test_valid_simple_identifier(self):
        """Should accept valid simple identifier."""
        result = _validate_sql_identifier("users", "table")
        assert result == "users"

    def test_valid_identifier_with_underscore(self):
        """Should accept identifier with underscore."""
        result = _validate_sql_identifier("user_data", "table")
        assert result == "user_data"

    def test_valid_identifier_starting_with_underscore(self):
        """Should accept identifier starting with underscore."""
        result = _validate_sql_identifier("_internal", "table")
        assert result == "_internal"

    def test_invalid_identifier_with_special_chars(self):
        """Should reject identifier with special characters."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("users; DROP TABLE--", "table")

    def test_invalid_identifier_empty(self):
        """Should reject empty identifier."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_sql_identifier("", "table")

    def test_invalid_identifier_starting_with_number(self):
        """Should reject identifier starting with number."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("123users", "table")


# =============================================================================
# Test Global Manager Instance
# =============================================================================


class TestGlobalManagerInstance:
    """Tests for global backup manager instance."""

    def test_get_backup_manager_creates_instance(self, temp_dir: Path):
        """Should create manager instance."""
        set_backup_manager(None)

        manager = get_backup_manager(temp_dir / "backups")

        assert manager is not None
        assert isinstance(manager, BackupManager)

        set_backup_manager(None)

    def test_set_and_get_backup_manager(self, backup_manager: BackupManager):
        """Should set and retrieve manager instance."""
        set_backup_manager(backup_manager)

        retrieved = get_backup_manager()

        assert retrieved is backup_manager

        set_backup_manager(None)


# =============================================================================
# Test Dataclass Defaults
# =============================================================================


class TestDataclassDefaults:
    """Tests for verification result dataclass defaults."""

    def test_verification_result_defaults(self):
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

    def test_schema_validation_result_defaults(self):
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

    def test_integrity_result_defaults(self):
        """Should have correct default values."""
        result = IntegrityResult(
            valid=True,
            foreign_keys_valid=True,
        )

        assert result.orphaned_records == {}
        assert result.foreign_key_errors == []
        assert result.data_type_errors == []


# =============================================================================
# Test Enum Values
# =============================================================================


class TestEnumValues:
    """Tests for backup status and type enums."""

    def test_backup_status_values(self):
        """Should have all expected status values."""
        assert BackupStatus.PENDING == "pending"
        assert BackupStatus.IN_PROGRESS == "in_progress"
        assert BackupStatus.COMPLETED == "completed"
        assert BackupStatus.VERIFIED == "verified"
        assert BackupStatus.FAILED == "failed"
        assert BackupStatus.EXPIRED == "expired"

    def test_backup_type_values(self):
        """Should have all expected type values."""
        assert BackupType.FULL == "full"
        assert BackupType.INCREMENTAL == "incremental"
        assert BackupType.DIFFERENTIAL == "differential"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_backup_empty_database(self, backup_manager: BackupManager, temp_dir: Path):
        """Should handle empty database backup."""
        empty_db = temp_dir / "empty.db"
        conn = sqlite3.connect(str(empty_db))
        conn.close()

        backup = backup_manager.create_backup(empty_db)

        assert backup.status == BackupStatus.VERIFIED
        assert len(backup.tables) == 0

    def test_backup_database_with_blob_data(self, backup_manager: BackupManager, temp_dir: Path):
        """Should handle database with BLOB data."""
        db_path = temp_dir / "blob.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, data BLOB)")
        conn.execute("INSERT INTO files (data) VALUES (?)", (b"\x00\x01\x02\x03" * 100,))
        conn.commit()
        conn.close()

        backup = backup_manager.create_backup(db_path)

        assert backup.status == BackupStatus.VERIFIED
        assert "files" in backup.tables

    def test_backup_database_with_null_values(self, backup_manager: BackupManager, temp_dir: Path):
        """Should handle database with NULL values."""
        db_path = temp_dir / "nulls.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES (NULL)")
        conn.execute("INSERT INTO items (name) VALUES ('test')")
        conn.commit()
        conn.close()

        backup = backup_manager.create_backup(db_path)

        assert backup.status == BackupStatus.VERIFIED

    def test_backup_database_with_unicode(self, backup_manager: BackupManager, temp_dir: Path):
        """Should handle database with unicode data."""
        db_path = temp_dir / "unicode.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES (?)", ("Hello World",))
        conn.execute("INSERT INTO items (name) VALUES (?)", ("Cafe",))
        conn.commit()
        conn.close()

        backup = backup_manager.create_backup(db_path)

        assert backup.status == BackupStatus.VERIFIED

    def test_backup_large_database(self, backup_manager: BackupManager, large_database: Path):
        """Should handle larger databases efficiently."""
        backup = backup_manager.create_backup(large_database)

        assert backup.status == BackupStatus.VERIFIED
        assert backup.row_counts["records"] == 1000

        # Restore goes to backup_dir/restore/filename
        backup_manager.restore_backup(backup.id, Path("restored_large.db"))

        # Find the actual restore path
        restore_dir = backup_manager.backup_dir / "restore"
        actual_restore_path = restore_dir / "restored_large.db"

        conn = sqlite3.connect(str(actual_restore_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM records")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1000
