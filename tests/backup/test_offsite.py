"""
Tests for Offsite Backup Automation and Restore Drill Evidence.

Tests cover:
- OffsiteBackupConfig construction
- Upload, list, download operations (mocked cloud storage)
- Restore drill execution and evidence collection
- Backup integrity verification
- Drill history persistence and retrieval
- Error handling for missing backups and failed operations
- State persistence (records and drill history)
"""

import gzip
import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.backup.offsite import (
    IntegrityResult,
    OffsiteBackupConfig,
    OffsiteBackupManager,
    OffsiteBackupRecord,
    RestoreDrillResult,
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
def config():
    """Create a default S3 offsite backup config."""
    return OffsiteBackupConfig(
        provider="s3",
        bucket="test-backups",
        prefix="aragora/test",
        encryption_key_id="key-001",
        retention_days=30,
        schedule_cron="0 2 * * *",
    )


@pytest.fixture
def state_dir(temp_dir):
    """Create a state directory for the manager."""
    state = temp_dir / "state"
    state.mkdir()
    return state


@pytest.fixture
def sample_db(temp_dir) -> Path:
    """Create a sample SQLite database for testing."""
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    cursor.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    for i in range(5):
        cursor.execute("INSERT INTO users (name) VALUES (?)", (f"User {i}",))
    cursor.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("version", "1.0"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_backup(temp_dir, sample_db) -> Path:
    """Create a gzip-compressed backup of the sample database."""
    backup_path = temp_dir / "backup.db.gz"
    with open(sample_db, "rb") as f_in:
        with gzip.open(backup_path, "wb") as f_out:
            f_out.write(f_in.read())
    return backup_path


@pytest.fixture
def mock_cloud():
    """Mock all cloud storage operations to use local file copies."""
    storage: dict[str, Path] = {}

    def fake_upload(local_path: Path, remote_key: str) -> None:
        tmp = Path(tempfile.mktemp(suffix="_cloud"))
        shutil.copy(local_path, tmp)
        storage[remote_key] = tmp

    def fake_download(remote_key: str, local_path: Path) -> None:
        src = storage.get(remote_key)
        if src is None or not src.exists():
            raise RuntimeError(f"Object not found: {remote_key}")
        shutil.copy(src, local_path)

    return storage, fake_upload, fake_download


@pytest.fixture
def manager(config, state_dir, mock_cloud):
    """Create an OffsiteBackupManager with mocked cloud operations."""
    _storage, fake_upload, fake_download = mock_cloud
    mgr = OffsiteBackupManager(config, state_dir=state_dir)
    mgr._cloud_upload = fake_upload
    mgr._cloud_download = fake_download
    return mgr


# =============================================================================
# Config Tests
# =============================================================================


class TestOffsiteBackupConfig:
    """Tests for OffsiteBackupConfig dataclass."""

    def test_default_values(self):
        cfg = OffsiteBackupConfig(provider="s3", bucket="my-bucket")
        assert cfg.provider == "s3"
        assert cfg.bucket == "my-bucket"
        assert cfg.prefix == "aragora/backups"
        assert cfg.encryption_key_id is None
        assert cfg.retention_days == 90
        assert cfg.schedule_cron == "0 3 * * *"

    def test_custom_values(self):
        cfg = OffsiteBackupConfig(
            provider="gcs",
            bucket="gcs-bucket",
            prefix="custom/prefix",
            encryption_key_id="kms-key-123",
            retention_days=365,
            schedule_cron="0 0 * * 0",
            region="us-central1",
        )
        assert cfg.provider == "gcs"
        assert cfg.region == "us-central1"
        assert cfg.retention_days == 365

    def test_azure_provider(self):
        cfg = OffsiteBackupConfig(provider="azure", bucket="container-name")
        assert cfg.provider == "azure"


# =============================================================================
# Record / Result Dataclass Tests
# =============================================================================


class TestOffsiteBackupRecord:
    """Tests for OffsiteBackupRecord serialization."""

    def test_to_dict_and_from_dict(self):
        now = datetime.now(timezone.utc)
        record = OffsiteBackupRecord(
            id="abc123",
            timestamp=now,
            size_bytes=1024,
            checksum="sha256abc",
            provider="s3",
            path="prefix/2026/02/23/file.db.gz",
            metadata={"source": "test"},
        )
        d = record.to_dict()
        assert d["id"] == "abc123"
        assert d["size_bytes"] == 1024
        assert d["metadata"] == {"source": "test"}

        restored = OffsiteBackupRecord.from_dict(d)
        assert restored.id == record.id
        assert restored.checksum == record.checksum
        assert restored.metadata == record.metadata


class TestRestoreDrillResult:
    """Tests for RestoreDrillResult serialization."""

    def test_to_dict_and_from_dict(self):
        now = datetime.now(timezone.utc)
        result = RestoreDrillResult(
            drill_id="drill-001",
            backup_id="bk-001",
            started_at=now,
            completed_at=now,
            success=True,
            duration_seconds=2.5,
            tables_verified=3,
            rows_verified=100,
            errors=[],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["tables_verified"] == 3

        restored = RestoreDrillResult.from_dict(d)
        assert restored.drill_id == "drill-001"
        assert restored.rows_verified == 100


class TestIntegrityResult:
    """Tests for IntegrityResult."""

    def test_valid_result(self):
        result = IntegrityResult(valid=True, checksum_match=True, size_match=True)
        d = result.to_dict()
        assert d["valid"] is True
        assert d["errors"] == []

    def test_invalid_result(self):
        result = IntegrityResult(
            valid=False,
            checksum_match=False,
            size_match=True,
            errors=["checksum mismatch"],
        )
        assert not result.valid
        assert len(result.errors) == 1


# =============================================================================
# Upload Tests
# =============================================================================


class TestUploadBackup:
    """Tests for uploading backups to offsite storage."""

    def test_upload_success(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup, {"env": "test"})

        assert record.id is not None
        assert record.size_bytes > 0
        assert record.checksum != ""
        assert record.provider == "s3"
        assert "aragora/test" in record.path
        assert record.metadata == {"env": "test"}

    def test_upload_file_not_found(self, manager):
        with pytest.raises(FileNotFoundError, match="not found"):
            manager.upload_backup("/nonexistent/path.db.gz")

    def test_upload_cloud_failure(self, manager, sample_backup):
        manager._cloud_upload = MagicMock(side_effect=RuntimeError("Network error"))
        with pytest.raises(RuntimeError, match="Offsite upload failed"):
            manager.upload_backup(sample_backup)

    def test_upload_persists_record(self, manager, sample_backup, state_dir):
        record = manager.upload_backup(sample_backup)

        # Verify record is persisted
        records_file = state_dir / "offsite_records.json"
        assert records_file.exists()
        data = json.loads(records_file.read_text())
        assert record.id in data


# =============================================================================
# List Tests
# =============================================================================


class TestListOffsiteBackups:
    """Tests for listing offsite backups."""

    def test_list_empty(self, manager):
        assert manager.list_offsite_backups() == []

    def test_list_after_upload(self, manager, sample_backup):
        manager.upload_backup(sample_backup)
        manager.upload_backup(sample_backup)

        backups = manager.list_offsite_backups()
        assert len(backups) == 2
        # Most recent first
        assert backups[0].timestamp >= backups[1].timestamp

    def test_list_with_limit(self, manager, sample_backup):
        for _ in range(5):
            manager.upload_backup(sample_backup)

        assert len(manager.list_offsite_backups(limit=3)) == 3
        assert len(manager.list_offsite_backups(limit=10)) == 5


# =============================================================================
# Download Tests
# =============================================================================


class TestDownloadBackup:
    """Tests for downloading offsite backups."""

    def test_download_success(self, manager, sample_backup, temp_dir):
        record = manager.upload_backup(sample_backup)

        target = temp_dir / "downloaded.db.gz"
        result_path = manager.download_backup(record.id, target)

        assert result_path == target
        assert target.exists()
        assert target.stat().st_size == record.size_bytes

    def test_download_not_found(self, manager, temp_dir):
        with pytest.raises(ValueError, match="not found"):
            manager.download_backup("nonexistent", temp_dir / "x.db")

    def test_download_cloud_failure(self, manager, sample_backup, temp_dir):
        record = manager.upload_backup(sample_backup)
        manager._cloud_download = MagicMock(side_effect=RuntimeError("Download failed"))
        with pytest.raises(RuntimeError, match="Offsite download failed"):
            manager.download_backup(record.id, temp_dir / "fail.db")


# =============================================================================
# Restore Drill Tests
# =============================================================================


class TestRunRestoreDrill:
    """Tests for restore drill execution and evidence collection."""

    def test_drill_success_with_compressed_backup(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        result = manager.run_restore_drill(record.id)

        assert result.success is True
        assert result.backup_id == record.id
        assert result.drill_id is not None
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_seconds > 0
        assert result.tables_verified == 2  # users + settings
        assert result.rows_verified == 6  # 5 users + 1 setting
        assert result.errors == []

    def test_drill_success_with_plain_db(self, manager, sample_db):
        """Test drill with uncompressed SQLite backup."""
        record = manager.upload_backup(sample_db)
        result = manager.run_restore_drill(record.id)

        assert result.success is True
        assert result.tables_verified == 2
        assert result.rows_verified == 6

    def test_drill_backup_not_found(self, manager):
        result = manager.run_restore_drill("missing-id")
        assert result.success is False
        assert "not found" in result.errors[0]

    def test_drill_checksum_mismatch(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        # Tamper with the stored checksum
        manager._records[record.id].checksum = "tampered_checksum"

        result = manager.run_restore_drill(record.id)
        assert result.success is False
        assert any("Checksum mismatch" in e for e in result.errors)

    def test_drill_download_failure(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        manager._cloud_download = MagicMock(side_effect=RuntimeError("Simulated failure"))
        result = manager.run_restore_drill(record.id)
        assert result.success is False
        assert len(result.errors) > 0

    def test_drill_recorded_in_history(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        manager.run_restore_drill(record.id)
        manager.run_restore_drill(record.id)

        history = manager.get_drill_history()
        assert len(history) == 2
        # Most recent first
        assert history[0].started_at >= history[1].started_at


# =============================================================================
# Integrity Verification Tests
# =============================================================================


class TestVerifyBackupIntegrity:
    """Tests for offsite backup integrity verification."""

    def test_integrity_valid(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        result = manager.verify_backup_integrity(record.id)

        assert result.valid is True
        assert result.checksum_match is True
        assert result.size_match is True
        assert result.errors == []

    def test_integrity_checksum_mismatch(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        manager._records[record.id].checksum = "wrong_checksum"

        result = manager.verify_backup_integrity(record.id)
        assert result.valid is False
        assert result.checksum_match is False
        assert any("Checksum" in e for e in result.errors)

    def test_integrity_size_mismatch(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        manager._records[record.id].size_bytes = 999999

        result = manager.verify_backup_integrity(record.id)
        assert result.valid is False
        assert result.size_match is False

    def test_integrity_not_found(self, manager):
        result = manager.verify_backup_integrity("missing")
        assert result.valid is False
        assert "not found" in result.errors[0]

    def test_integrity_download_failure(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        manager._cloud_download = MagicMock(side_effect=RuntimeError("Cloud down"))
        result = manager.verify_backup_integrity(record.id)
        assert result.valid is False
        assert len(result.errors) > 0


# =============================================================================
# Drill History Tests
# =============================================================================


class TestDrillHistory:
    """Tests for drill history retrieval and persistence."""

    def test_empty_history(self, manager):
        assert manager.get_drill_history() == []

    def test_history_ordered_by_time(self, manager, sample_backup):
        record = manager.upload_backup(sample_backup)
        manager.run_restore_drill(record.id)
        manager.run_restore_drill(record.id)
        manager.run_restore_drill(record.id)

        history = manager.get_drill_history()
        assert len(history) == 3
        for i in range(len(history) - 1):
            assert history[i].started_at >= history[i + 1].started_at

    def test_history_persisted_to_disk(self, manager, sample_backup, state_dir, config, mock_cloud):
        record = manager.upload_backup(sample_backup)
        manager.run_restore_drill(record.id)

        # Create a new manager instance to test persistence
        _storage, fake_upload, fake_download = mock_cloud
        mgr2 = OffsiteBackupManager(config, state_dir=state_dir)
        mgr2._cloud_upload = fake_upload
        mgr2._cloud_download = fake_download

        history = mgr2.get_drill_history()
        assert len(history) == 1
        assert history[0].backup_id == record.id


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for record and drill history persistence across restarts."""

    def test_records_survive_restart(self, config, state_dir, sample_backup, mock_cloud):
        _storage, fake_upload, fake_download = mock_cloud

        mgr1 = OffsiteBackupManager(config, state_dir=state_dir)
        mgr1._cloud_upload = fake_upload
        mgr1._cloud_download = fake_download
        record = mgr1.upload_backup(sample_backup, {"test": True})

        # Simulate restart
        mgr2 = OffsiteBackupManager(config, state_dir=state_dir)
        mgr2._cloud_upload = fake_upload
        mgr2._cloud_download = fake_download

        backups = mgr2.list_offsite_backups()
        assert len(backups) == 1
        assert backups[0].id == record.id
        assert backups[0].metadata == {"test": True}

    def test_corrupted_state_handled_gracefully(self, config, state_dir):
        (state_dir / "offsite_records.json").write_text("NOT VALID JSON")
        (state_dir / "drill_history.json").write_text("NOT VALID JSON")

        # Should not raise
        mgr = OffsiteBackupManager(config, state_dir=state_dir)
        assert mgr.list_offsite_backups() == []
        assert mgr.get_drill_history() == []


# =============================================================================
# Provider-Specific Client Tests
# =============================================================================


class TestCloudClients:
    """Test lazy cloud client instantiation."""

    def test_s3_client_lazy_import(self):
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            from aragora.backup.offsite import _get_s3_client

            client = _get_s3_client(region="us-east-1")
            assert client is not None

    def test_gcs_client_lazy_import(self):
        mock_storage = MagicMock()
        mock_google = MagicMock()
        mock_google.cloud.storage = mock_storage
        with patch.dict(
            "sys.modules",
            {
                "google": mock_google,
                "google.cloud": mock_google.cloud,
                "google.cloud.storage": mock_storage,
            },
        ):
            from aragora.backup.offsite import _get_gcs_client

            client = _get_gcs_client()
            assert client is not None

    def test_unsupported_provider_raises(self, state_dir):
        cfg = OffsiteBackupConfig(
            provider="invalid",  # type: ignore[arg-type]
            bucket="b",
        )
        mgr = OffsiteBackupManager(cfg, state_dir=state_dir)
        with pytest.raises(ValueError, match="Unsupported provider"):
            mgr._cloud_upload(Path("/tmp/test"), "key")

    def test_unsupported_provider_download_raises(self, state_dir):
        cfg = OffsiteBackupConfig(
            provider="invalid",  # type: ignore[arg-type]
            bucket="b",
        )
        mgr = OffsiteBackupManager(cfg, state_dir=state_dir)
        with pytest.raises(ValueError, match="Unsupported provider"):
            mgr._cloud_download("key", Path("/tmp/test"))
