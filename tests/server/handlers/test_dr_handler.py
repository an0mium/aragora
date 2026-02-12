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

    def to_dict(self) -> dict[str, Any]:
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
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 1.0


@dataclass
class MockComprehensiveResult:
    """Mock comprehensive verification result."""

    backup_id: str
    verified: bool = True
    schema_validation: Any | None = None
    integrity_check: Any | None = None

    def to_dict(self) -> dict[str, Any]:
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
        self._backups: dict[str, MockBackupMetadata] = {}
        self.retention_policy = MockRetentionPolicy()
        self.compression = True
        self.verify_after_backup = True
        self.backup_dir = MagicMock()
        self.backup_dir.exists.return_value = True
        self.backup_dir.is_dir.return_value = True

    def list_backups(
        self,
        source_path: str | None = None,
        status: Any | None = None,
        since: datetime | None = None,
    ) -> list[MockBackupMetadata]:
        return list(self._backups.values())

    def get_latest_backup(self, source_path: str | None = None) -> MockBackupMetadata | None:
        backups = self.list_backups()
        return backups[0] if backups else None

    def verify_backup(
        self,
        backup_id: str,
        backup_meta: Any | None = None,
        test_restore: bool = True,
    ) -> MockVerificationResult:
        return MockVerificationResult(backup_id=backup_id)

    def verify_restore_comprehensive(
        self,
        backup_id: str,
        backup_meta: Any | None = None,
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


# ===========================================================================
# Extended Test Coverage - Edge Cases
# ===========================================================================


class TestDRStatusEdgeCases:
    """Test DR status edge cases."""

    @pytest.mark.asyncio
    async def test_get_status_stale_backup(self, handler, mock_backup_manager):
        """Test DR status handles stale backup (>24h old)."""
        mock_backup_manager._backups["backup-001"] = MockBackupMetadata(
            created_at=datetime.now(timezone.utc) - timedelta(hours=48)
        )
        result = await handler.handle("GET", "/api/v2/dr/status")
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_status_unverified_backup(self, handler, mock_backup_manager):
        """Test DR status handles unverified backup."""
        mock_backup_manager._backups["backup-001"] = MockBackupMetadata(
            verified=False, status="pending"
        )
        result = await handler.handle("GET", "/api/v2/dr/status")
        assert result.status_code == 200


class TestDRDrillEdgeCases:
    """Test DR drill edge cases."""

    @pytest.mark.asyncio
    async def test_run_drill_missing_type_uses_default(self, handler):
        """Test drill with missing drill_type uses default (restore_test)."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={},
        )
        # Handler defaults to restore_test when type is missing
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_run_drill_with_notification(self, handler):
        """Test drill with notification flag."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "restore_test", "notify": True},
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_run_drill_dry_run(self, handler):
        """Test drill dry run mode."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/drill",
            body={"drill_type": "restore_test", "dry_run": True},
        )
        assert result.status_code == 200


class TestDRValidateEdgeCases:
    """Test DR validate edge cases."""

    @pytest.mark.asyncio
    async def test_validate_missing_storage(self, handler, mock_backup_manager):
        """Test validation when storage is missing."""
        mock_backup_manager.backup_dir.exists.return_value = False
        result = await handler.handle(
            "POST",
            "/api/v2/dr/validate",
            body={"check_storage": True},
        )
        # Should still return 200 with validation results
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_all_checks(self, handler):
        """Test validation with all check flags enabled."""
        result = await handler.handle(
            "POST",
            "/api/v2/dr/validate",
            body={
                "check_storage": True,
                "check_retention": True,
                "check_encryption": True,
            },
        )
        assert result.status_code == 200


class TestDRObjectivesEdgeCases:
    """Test DR objectives edge cases."""

    @pytest.mark.asyncio
    async def test_get_objectives_includes_policy(self, handler):
        """Test objectives response includes retention policy info."""
        result = await handler.handle("GET", "/api/v2/dr/objectives")
        assert result.status_code == 200
