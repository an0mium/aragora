"""
Tests for DRHandler - Disaster Recovery HTTP endpoints.

Tests cover:
- DR status endpoint
- DR drill execution
- RPO/RTO objectives
- Configuration validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.dr_handler import (
    DRHandler,
    create_dr_handler,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockBackupMetadata:
    """Mock backup metadata for DR testing."""

    id: str = "backup-001"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "verified"
    verified: bool = True
    compressed_size_bytes: int = 524288

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "verified": self.verified,
            "compressed_size_bytes": self.compressed_size_bytes,
        }


@dataclass
class MockVerificationResult:
    """Mock verification result for DR testing."""

    backup_id: str
    verified: bool = True
    checksum_valid: bool = True
    restore_tested: bool = True
    tables_valid: bool = True
    row_counts_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 1.0


@dataclass
class MockComprehensiveResult:
    """Mock comprehensive verification result."""

    backup_id: str
    verified: bool = True
    schema_validation: Optional[Any] = None
    integrity_check: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "verified": self.verified,
            "basic_verification": {"checksum_valid": True},
            "schema_validation": None,
            "integrity_check": None,
            "table_checksums_valid": True,
            "all_errors": [],
            "all_warnings": [],
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 1.0,
        }


@dataclass
class MockRetentionPolicy:
    """Mock retention policy."""

    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 3
    min_backups: int = 1


class MockBackupManager:
    """Mock backup manager for DR testing."""

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

    def verify_backup(
        self,
        backup_id: str,
        backup_meta: Optional[Any] = None,
        test_restore: bool = True,
    ) -> MockVerificationResult:
        return MockVerificationResult(backup_id=backup_id)

    def verify_restore_comprehensive(
        self,
        backup_id: str,
        backup_meta: Optional[Any] = None,
    ) -> MockComprehensiveResult:
        return MockComprehensiveResult(backup_id=backup_id)

    def restore_backup(
        self,
        backup_id: str,
        target_path: str,
        dry_run: bool = False,
    ) -> bool:
        return True


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def mock_backup_manager():
    """Create mock backup manager with sample data."""
    manager = MockBackupManager()
    # Add a recent verified backup
    manager._backups["backup-001"] = MockBackupMetadata(
        created_at=datetime.now(timezone.utc) - timedelta(hours=6)
    )
    return manager


@pytest.fixture
def handler(mock_server_context, mock_backup_manager):
    """Create handler with mocked dependencies."""
    with patch(
        "aragora.backup.manager.get_backup_manager",
        return_value=mock_backup_manager,
    ):
        h = DRHandler(mock_server_context)
        h._manager = mock_backup_manager
        return h


# ===========================================================================
# Handler Tests
# ===========================================================================


class TestDRHandlerRouting:
    """Test request routing."""

    def test_can_handle_dr_paths(self, handler):
        """Test that handler recognizes DR paths."""
        assert handler.can_handle("/api/v2/dr/status", "GET")
        assert handler.can_handle("/api/v2/dr/drill", "POST")
        assert handler.can_handle("/api/v2/dr/objectives", "GET")
        assert handler.can_handle("/api/v2/dr/validate", "POST")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-DR paths."""
        assert not handler.can_handle("/api/v2/backups", "GET")
        assert not handler.can_handle("/api/v1/dr/status", "GET")


class TestDRStatus:
    """Test DR status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_healthy(self, handler):
        """Test DR status returns healthy with recent backup."""
        result = await handler.handle("GET", "/api/v2/dr/status")
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_status_no_backups(self, handler, mock_backup_manager):
        """Test DR status handles no backups."""
        mock_backup_manager._backups.clear()
        result = await handler.handle("GET", "/api/v2/dr/status")
        assert result.status_code == 200


class TestDRDrill:
    """Test DR drill endpoint."""

    @pytest.mark.asyncio
    async def test_run_restore_test_drill(self, handler):
        """Test restore_test drill type."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "restore_test"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_run_full_recovery_drill(self, handler):
        """Test full_recovery_sim drill type."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "full_recovery_sim"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_run_failover_drill(self, handler):
        """Test failover_test drill type."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "failover_test"},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_run_drill_invalid_type(self, handler):
        """Test invalid drill type returns error."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "invalid_type"},
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_run_drill_no_backup(self, handler, mock_backup_manager):
        """Test drill fails gracefully without backups."""
        mock_backup_manager._backups.clear()
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "restore_test"},
        )
        assert result.status_code == 400


class TestDRObjectives:
    """Test DR objectives endpoint."""

    @pytest.mark.asyncio
    async def test_get_objectives(self, handler):
        """Test getting RPO/RTO objectives."""
        result = await handler.handle("GET", "/api/v2/dr/objectives")
        assert result.status_code == 200


class TestDRValidate:
    """Test DR validation endpoint."""

    @pytest.mark.asyncio
    async def test_validate_configuration(self, handler):
        """Test configuration validation."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/validate",
            body={"check_storage": True},
        )
        assert result.status_code == 200


class TestFactoryFunction:
    """Test handler factory function."""

    def test_create_dr_handler(self, mock_server_context):
        """Test factory function creates handler."""
        handler = create_dr_handler(mock_server_context)
        assert isinstance(handler, DRHandler)
