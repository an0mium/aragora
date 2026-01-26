"""
Tests for BackupHandler - Backup HTTP endpoints.

Tests cover:
- List backups with filtering and pagination
- Create new backup
- Get single backup by ID
- Verify backup integrity
- Comprehensive verification
- Restore test (dry-run)
- Delete backup
- Cleanup expired backups
- Statistics endpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.backup_handler import (
    BackupHandler,
    create_backup_handler,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockBackupMetadata:
    """Mock backup metadata for testing."""

    id: str = "backup-001"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    backup_type: Any = "full"
    status: Any = "verified"
    source_path: str = "/data/aragora.db"
    backup_path: str = "/backups/aragora_20231215_120000_abc123.db.gz"
    size_bytes: int = 1048576
    compressed_size_bytes: int = 524288
    checksum: str = "sha256:abc123"
    row_counts: Dict[str, int] = field(default_factory=dict)
    tables: List[str] = field(default_factory=list)
    duration_seconds: float = 5.0
    verified: bool = True
    verified_at: Optional[datetime] = None
    restore_tested: bool = True
    error: Optional[str] = None
    storage_backend: str = "local"
    encryption_key_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_hash: str = ""
    table_checksums: Dict[str, str] = field(default_factory=dict)
    foreign_keys: List[tuple] = field(default_factory=list)
    indexes: List[tuple] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "backup_type": self.backup_type,
            "status": self.status,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "size_bytes": self.size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "checksum": self.checksum,
            "row_counts": self.row_counts,
            "tables": self.tables,
            "duration_seconds": self.duration_seconds,
            "verified": self.verified,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "restore_tested": self.restore_tested,
            "error": self.error,
            "storage_backend": self.storage_backend,
            "encryption_key_id": self.encryption_key_id,
            "metadata": self.metadata,
            "schema_hash": self.schema_hash,
            "table_checksums": self.table_checksums,
            "foreign_keys": [list(fk) for fk in self.foreign_keys],
            "indexes": [list(idx) for idx in self.indexes],
        }


@dataclass
class MockVerificationResult:
    """Mock verification result."""

    backup_id: str
    verified: bool
    checksum_valid: bool
    restore_tested: bool
    tables_valid: bool
    row_counts_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 1.0


@dataclass
class MockComprehensiveResult:
    """Mock comprehensive verification result."""

    backup_id: str
    verified: bool
    basic_verification: MockVerificationResult = field(default_factory=MockVerificationResult)
    schema_validation: Optional[Any] = None
    integrity_check: Optional[Any] = None
    table_checksums_valid: bool = True
    table_checksum_errors: List[str] = field(default_factory=list)
    all_errors: List[str] = field(default_factory=list)
    all_warnings: List[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "verified": self.verified,
            "basic_verification": {
                "checksum_valid": self.basic_verification.checksum_valid,
                "restore_tested": self.basic_verification.restore_tested,
                "tables_valid": self.basic_verification.tables_valid,
                "row_counts_valid": self.basic_verification.row_counts_valid,
            },
            "schema_validation": None,
            "integrity_check": None,
            "table_checksums_valid": self.table_checksums_valid,
            "all_errors": self.all_errors,
            "all_warnings": self.all_warnings,
            "verified_at": self.verified_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class MockRetentionPolicy:
    """Mock retention policy."""

    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 3
    max_size_bytes: Optional[int] = None
    min_backups: int = 1


class MockBackupManager:
    """Mock backup manager for testing."""

    def __init__(self):
        self._backups: Dict[str, MockBackupMetadata] = {}
        self.retention_policy = MockRetentionPolicy()
        self.compression = True
        self.verify_after_backup = True
        self.backup_dir = MagicMock()
        self.backup_dir.exists.return_value = True
        self.backup_dir.is_dir.return_value = True

    def list_backups(
        self,
        source_path: Optional[str] = None,
        status: Optional[Any] = None,
        since: Optional[datetime] = None,
    ) -> List[MockBackupMetadata]:
        return list(self._backups.values())

    def get_latest_backup(self, source_path: Optional[str] = None) -> Optional[MockBackupMetadata]:
        backups = self.list_backups()
        return backups[0] if backups else None

    def create_backup(
        self,
        source_path: str,
        backup_type: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MockBackupMetadata:
        backup = MockBackupMetadata(
            id="new-backup-001",
            source_path=source_path,
        )
        self._backups[backup.id] = backup
        return backup

    def verify_backup(
        self,
        backup_id: str,
        backup_meta: Optional[Any] = None,
        test_restore: bool = True,
    ) -> MockVerificationResult:
        return MockVerificationResult(
            backup_id=backup_id,
            verified=True,
            checksum_valid=True,
            restore_tested=test_restore,
            tables_valid=True,
            row_counts_valid=True,
        )

    def verify_restore_comprehensive(
        self,
        backup_id: str,
        backup_meta: Optional[Any] = None,
    ) -> MockComprehensiveResult:
        return MockComprehensiveResult(
            backup_id=backup_id,
            verified=True,
            basic_verification=MockVerificationResult(
                backup_id=backup_id,
                verified=True,
                checksum_valid=True,
                restore_tested=True,
                tables_valid=True,
                row_counts_valid=True,
            ),
        )

    def restore_backup(
        self,
        backup_id: str,
        target_path: str,
        dry_run: bool = False,
    ) -> bool:
        return True

    def apply_retention_policy(self, dry_run: bool = False) -> List[str]:
        return []

    def _save_manifest(self) -> None:
        pass


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def mock_backup_manager():
    """Create mock backup manager with sample data."""
    manager = MockBackupManager()
    manager._backups["backup-001"] = MockBackupMetadata()
    manager._backups["backup-002"] = MockBackupMetadata(
        id="backup-002",
        status="completed",
        verified=False,
    )
    return manager


@pytest.fixture
def handler(mock_server_context, mock_backup_manager):
    """Create handler with mocked dependencies."""
    with patch(
        "aragora.backup.manager.get_backup_manager",
        return_value=mock_backup_manager,
    ):
        h = BackupHandler(mock_server_context)
        h._manager = mock_backup_manager
        return h


# ===========================================================================
# Handler Tests
# ===========================================================================


class TestBackupHandlerRouting:
    """Test request routing."""

    def test_can_handle_backups_path(self, handler):
        """Test that handler recognizes backup paths."""
        assert handler.can_handle("/api/v2/backups", "GET")
        assert handler.can_handle("/api/v2/backups", "POST")
        assert handler.can_handle("/api/v2/backups/backup-001", "GET")
        assert handler.can_handle("/api/v2/backups/backup-001/verify", "POST")
        assert handler.can_handle("/api/v2/backups/backup-001", "DELETE")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-backup paths."""
        assert not handler.can_handle("/api/v2/receipts", "GET")
        assert not handler.can_handle("/api/v1/backups", "GET")


class TestListBackups:
    """Test list backups endpoint."""

    @pytest.mark.asyncio
    async def test_list_backups_success(self, handler):
        """Test listing backups returns correct format."""
        with patch.object(handler, "_list_backups") as mock_list:
            mock_list.return_value = MagicMock(
                status_code=200,
                body=b'{"backups": [], "pagination": {}}',
            )
            result = await handler.handle("GET", "/api/v2/backups")
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_backups_with_filters(self, handler):
        """Test listing backups with query filters."""
        result = await handler.handle(
            "GET",
            "/api/v2/backups",
            query_params={"status": "verified", "limit": "10"},
        )
        assert result.status_code == 200


class TestGetBackup:
    """Test get single backup endpoint."""

    @pytest.mark.asyncio
    async def test_get_backup_success(self, handler, mock_backup_manager):
        """Test getting a specific backup."""
        result = await handler.handle("GET", "/api/v2/backups/backup-001")
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_backup_not_found(self, handler):
        """Test getting non-existent backup returns 404."""
        result = await handler.handle("GET", "/api/v2/backups/nonexistent")
        assert result.status_code == 404


class TestCreateBackup:
    """Test create backup endpoint."""

    @pytest.mark.asyncio
    async def test_create_backup_success(self, handler):
        """Test creating a new backup."""
        result = await handler.handle(
            "POST",
            "/api/v2/backups",
            body={"source_path": "/data/test.db"},
        )
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_backup_missing_source(self, handler):
        """Test creating backup without source_path fails."""
        result = await handler.handle("POST", "/api/v2/backups", body={})
        assert result.status_code == 400


class TestVerifyBackup:
    """Test backup verification endpoints."""

    @pytest.mark.asyncio
    async def test_verify_backup_success(self, handler):
        """Test verifying a backup."""
        result = await handler.handle(
            "POST",
            "/api/v2/backups/backup-001/verify",
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_comprehensive_verify_success(self, handler):
        """Test comprehensive backup verification."""
        result = await handler.handle(
            "POST",
            "/api/v2/backups/backup-001/verify-comprehensive",
        )
        assert result.status_code == 200


class TestRestoreTest:
    """Test restore test endpoint."""

    @pytest.mark.asyncio
    async def test_restore_test_success(self, handler):
        """Test dry-run restore."""
        result = await handler.handle(
            "POST",
            "/api/v2/backups/backup-001/restore-test",
            body={"target_path": "/tmp/test.db"},
        )
        assert result.status_code == 200


class TestDeleteBackup:
    """Test delete backup endpoint."""

    @pytest.mark.asyncio
    async def test_delete_backup_not_found(self, handler):
        """Test deleting non-existent backup."""
        result = await handler.handle("DELETE", "/api/v2/backups/nonexistent")
        assert result.status_code == 404


class TestCleanupExpired:
    """Test cleanup expired endpoint."""

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, handler):
        """Test cleanup with dry run."""
        result = await handler.handle(
            "POST",
            "/api/v2/backups/cleanup",
            body={"dry_run": True},
        )
        assert result.status_code == 200


class TestBackupStats:
    """Test backup statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats(self, handler):
        """Test getting backup statistics."""
        result = await handler.handle("GET", "/api/v2/backups/stats")
        assert result.status_code == 200


class TestFactoryFunction:
    """Test handler factory function."""

    def test_create_backup_handler(self, mock_server_context):
        """Test factory function creates handler."""
        handler = create_backup_handler(mock_server_context)
        assert isinstance(handler, BackupHandler)
