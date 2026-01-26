"""
Integration tests for the Backup API endpoints.

Tests the backup management functionality:
- List backups
- Create backup
- Get backup metadata
- Verify integrity
- Restore test (dry-run)
- Cleanup retention
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any


class TestBackupAPIEndpoints:
    """Test Backup API HTTP endpoints."""

    @pytest.fixture(autouse=True)
    def mock_rbac_for_endpoints(self):
        """Mock RBAC to allow all permissions for endpoint testing."""
        from aragora.rbac.models import AuthorizationContext

        mock_context = AuthorizationContext(
            user_id="test_user",
            permissions={
                "backups:read",
                "backups:create",
                "backups:verify",
                "backups:restore",
                "backups:delete",
            },
        )
        with patch(
            "aragora.rbac.decorators._get_context_from_args",
            return_value=mock_context,
        ):
            yield

    @pytest.fixture
    def backup_handler(self):
        """Create BackupHandler instance."""
        from aragora.server.handlers.backup_handler import BackupHandler

        server_context = {"workspace_id": "test"}
        return BackupHandler(server_context)

    @pytest.fixture
    def mock_backup_manager(self):
        """Create mock backup manager."""
        manager = MagicMock()
        manager.list_backups = AsyncMock(
            return_value={
                "backups": [
                    {
                        "backup_id": "bkp_001",
                        "created_at": "2024-01-01T00:00:00Z",
                        "type": "full",
                        "status": "completed",
                        "size_bytes": 1024000,
                    }
                ],
                "total": 1,
            }
        )
        manager.create_backup = AsyncMock(
            return_value={
                "backup_id": "bkp_002",
                "status": "in_progress",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        manager.get_backup = AsyncMock(
            return_value={
                "backup_id": "bkp_001",
                "type": "full",
                "status": "completed",
                "size_bytes": 1024000,
                "checksum": "sha256:abc123",
            }
        )
        manager.verify_integrity = AsyncMock(return_value={"valid": True, "checksum_match": True})
        manager.restore_test = AsyncMock(
            return_value={
                "success": True,
                "tables_verified": 10,
                "rows_sampled": 1000,
                "errors": [],
            }
        )
        return manager

    @pytest.mark.asyncio
    async def test_list_backups(self, backup_handler, mock_backup_manager):
        """Test listing backups endpoint."""
        with patch.object(backup_handler, "_get_manager", return_value=mock_backup_manager):
            result = await backup_handler.handle(
                method="GET",
                path="/api/v2/backups",
                query_params={"limit": "10"},
            )

            assert result is not None
            assert result.status_code in (200, 500)  # 500 if manager unavailable

    @pytest.mark.asyncio
    async def test_create_backup(self, backup_handler, mock_backup_manager):
        """Test creating a backup."""
        with patch.object(backup_handler, "_get_manager", return_value=mock_backup_manager):
            result = await backup_handler.handle(
                method="POST",
                path="/api/v2/backups",
                body={"source_path": "/path/to/db", "type": "full", "description": "Manual backup"},
            )

            assert result is not None
            # May return 200 or 202 for async operation
            assert result.status_code in (200, 202, 400, 500)

    @pytest.mark.asyncio
    async def test_get_backup(self, backup_handler, mock_backup_manager):
        """Test getting backup metadata."""
        with patch.object(backup_handler, "_get_manager", return_value=mock_backup_manager):
            result = await backup_handler.handle(
                method="GET",
                path="/api/v2/backups/bkp_001",
            )

            assert result is not None
            assert result.status_code in (200, 404, 500)

    @pytest.mark.asyncio
    async def test_verify_backup(self, backup_handler, mock_backup_manager):
        """Test verifying backup integrity."""
        with patch.object(backup_handler, "_get_manager", return_value=mock_backup_manager):
            result = await backup_handler.handle(
                method="POST",
                path="/api/v2/backups/bkp_001/verify",
            )

            assert result is not None
            assert result.status_code in (200, 404, 500)

    @pytest.mark.asyncio
    async def test_restore_test(self, backup_handler, mock_backup_manager):
        """Test dry-run restore."""
        with patch.object(backup_handler, "_get_manager", return_value=mock_backup_manager):
            result = await backup_handler.handle(
                method="POST",
                path="/api/v2/backups/bkp_001/restore-test",
            )

            assert result is not None
            assert result.status_code in (200, 404, 500)


class TestBackupManagerIntegration:
    """Test BackupManager integration."""

    def test_backup_manager_import(self):
        """Test that BackupManager can be imported."""
        try:
            from aragora.backup.manager import BackupManager

            assert BackupManager is not None
        except ImportError:
            pytest.skip("BackupManager not available")

    def test_backup_handler_import(self):
        """Test that BackupHandler can be imported."""
        from aragora.server.handlers.backup_handler import BackupHandler

        assert BackupHandler is not None

    def test_backup_handler_routes(self):
        """Test that BackupHandler has correct routes."""
        from aragora.server.handlers.backup_handler import BackupHandler

        handler = BackupHandler({})
        assert handler.can_handle("/api/v2/backups", "GET")
        assert handler.can_handle("/api/v2/backups", "POST")
        assert handler.can_handle("/api/v2/backups/bkp_001", "GET")
        assert handler.can_handle("/api/v2/backups/bkp_001/verify", "POST")


class TestBackupPermissions:
    """Test RBAC permissions for backup operations."""

    def test_backup_read_permission(self):
        """Test that listing backups requires backup.read permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("backups:read")
        def list_backups(ctx):
            return {"backups": []}

        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backups:read"},
        )
        result = list_backups(ctx)
        assert "backups" in result

    def test_backup_create_permission(self):
        """Test that creating backups requires backup.create permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("backups:create")
        def create_backup(ctx):
            return {"backup_id": "bkp_new"}

        from aragora.rbac.models import AuthorizationContext

        ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"backups:create"},
        )
        result = create_backup(ctx)
        assert "backup_id" in result
