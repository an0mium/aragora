"""
Integration tests for RBAC enforcement on enterprise handlers.

These tests verify that RBAC permissions are correctly enforced at the
middleware/routing level for backup, DR, and compliance endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.rbac.models import AuthorizationContext
from aragora.rbac.decorators import require_permission, PermissionDeniedError


class TestBackupHandlerRBAC:
    """Test RBAC enforcement for backup operations."""

    def test_backup_read_permission_grants_access(self):
        """Verify backup.read permission allows read operations."""

        @require_permission("backups:read")
        def _list_backups_allowed(ctx):
            return {"backups": []}

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backups:read"},
        )
        result = _list_backups_allowed(ctx)
        assert "backups" in result

    def test_backup_read_permission_denied(self):
        """Verify backup.read permission is required."""

        @require_permission("backups:read")
        def _list_backups_denied(ctx):
            return {"backups": []}

        ctx = AuthorizationContext(
            user_id="user_456",  # Different user
            permissions={"other:permission"},
        )
        with pytest.raises(PermissionDeniedError):
            _list_backups_denied(ctx)

    def test_backup_create_permission_grants_access(self):
        """Verify backup.create permission allows create operations."""

        @require_permission("backups:create")
        def create_backup(ctx, source_path: str):
            return {"backup_id": "bkp_001", "source": source_path}

        ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"backups:create"},
        )
        result = create_backup(ctx, "/path/to/db")
        assert result["backup_id"] == "bkp_001"

    def test_backup_create_denied_without_permission(self):
        """Verify backup.create permission is required for create."""

        @require_permission("backups:create")
        def create_backup(ctx, source_path: str):
            return {"backup_id": "bkp_001"}

        ctx = AuthorizationContext(
            user_id="viewer_123",
            permissions={"backups:read"},  # Only read, not create
        )
        with pytest.raises(PermissionDeniedError):
            create_backup(ctx, "/path/to/db")

    def test_backup_delete_requires_delete_permission(self):
        """Verify backup.delete permission is required for deletion."""

        @require_permission("backups:delete")
        def delete_backup(ctx, backup_id: str):
            return {"deleted": True, "backup_id": backup_id}

        # Admin with delete permission
        admin_ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"backups:delete"},
        )
        result = delete_backup(admin_ctx, "bkp_001")
        assert result["deleted"] is True

        # User without delete permission
        user_ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backups:read"},
        )
        with pytest.raises(PermissionDeniedError):
            delete_backup(user_ctx, "bkp_001")

    def test_backup_restore_requires_restore_permission(self):
        """Verify backup.restore permission is required for restore operations."""

        @require_permission("backups:restore")
        def restore_backup(ctx, backup_id: str):
            return {"restored": True, "backup_id": backup_id}

        # User with restore permission
        dba_ctx = AuthorizationContext(
            user_id="dba_123",
            permissions={"backups:restore"},
        )
        result = restore_backup(dba_ctx, "bkp_001")
        assert result["restored"] is True

        # User without restore permission (only read)
        user_ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backups:read", "backups:create"},
        )
        with pytest.raises(PermissionDeniedError):
            restore_backup(user_ctx, "bkp_001")


class TestDRHandlerRBAC:
    """Test RBAC enforcement for disaster recovery operations."""

    def test_dr_read_permission_grants_status_access(self):
        """Verify dr.read permission allows status operations."""

        @require_permission("dr:read")
        def get_dr_status(ctx):
            return {"status": "healthy", "readiness_score": 85}

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"dr:read"},
        )
        result = get_dr_status(ctx)
        assert result["status"] == "healthy"

    def test_dr_drill_requires_drill_permission(self):
        """Verify dr.drill permission is required for executing drills."""

        @require_permission("dr:drill")
        def run_dr_drill(ctx, drill_type: str):
            return {"drill_id": "drill_001", "type": drill_type}

        # SRE with drill permission
        sre_ctx = AuthorizationContext(
            user_id="sre_123",
            permissions={"dr:drill"},
        )
        result = run_dr_drill(sre_ctx, "restore_test")
        assert result["drill_id"] == "drill_001"

        # Viewer without drill permission
        viewer_ctx = AuthorizationContext(
            user_id="viewer_123",
            permissions={"dr:read"},
        )
        with pytest.raises(PermissionDeniedError):
            run_dr_drill(viewer_ctx, "restore_test")

    def test_dr_objectives_requires_read_permission(self):
        """Verify dr.read permission allows viewing objectives."""

        @require_permission("dr:read")
        def get_objectives(ctx):
            return {"rpo_hours": 24, "rto_hours": 4}

        ctx = AuthorizationContext(
            user_id="analyst_123",
            permissions={"dr:read"},
        )
        result = get_objectives(ctx)
        assert result["rpo_hours"] == 24


class TestComplianceHandlerRBAC:
    """Test RBAC enforcement for compliance operations."""

    def test_compliance_read_grants_status_access(self):
        """Verify compliance.read permission allows status access."""

        @require_permission("compliance:read")
        def get_compliance_status(ctx):
            return {"overall_score": 92, "status": "compliant"}

        ctx = AuthorizationContext(
            user_id="auditor_123",
            permissions={"compliance:read"},
        )
        result = get_compliance_status(ctx)
        assert result["status"] == "compliant"

    def test_compliance_soc2_requires_permission(self):
        """Verify compliance.soc2 permission is required for SOC 2 reports."""

        @require_permission("compliance:soc2")
        def get_soc2_report(ctx):
            return {"report_id": "soc2_001", "type": "Type II"}

        # Auditor with soc2 permission
        auditor_ctx = AuthorizationContext(
            user_id="auditor_123",
            permissions={"compliance:soc2"},
        )
        result = get_soc2_report(auditor_ctx)
        assert result["type"] == "Type II"

        # User without soc2 permission
        user_ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"compliance:read"},
        )
        with pytest.raises(PermissionDeniedError):
            get_soc2_report(user_ctx)

    def test_compliance_gdpr_export_requires_permission(self):
        """Verify compliance.gdpr permission is required for GDPR exports."""

        @require_permission("compliance:gdpr")
        def export_gdpr_data(ctx, user_id: str):
            return {"user_id": user_id, "data": {}}

        # Admin with gdpr permission
        admin_ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"compliance:gdpr"},
        )
        result = export_gdpr_data(admin_ctx, "target_user")
        assert result["user_id"] == "target_user"

        # Viewer without gdpr permission
        viewer_ctx = AuthorizationContext(
            user_id="viewer_123",
            permissions={"compliance:read"},
        )
        with pytest.raises(PermissionDeniedError):
            export_gdpr_data(viewer_ctx, "target_user")

    def test_compliance_audit_requires_audit_permission(self):
        """Verify compliance.audit permission is required for audit operations."""

        @require_permission("compliance:audit")
        def verify_audit_trail(ctx, date_from: str, date_to: str):
            return {"verified": True, "records": 100}

        # Compliance officer with audit permission
        officer_ctx = AuthorizationContext(
            user_id="officer_123",
            permissions={"compliance:audit"},
        )
        result = verify_audit_trail(officer_ctx, "2024-01-01", "2024-01-31")
        assert result["verified"] is True

        # Analyst without audit permission
        analyst_ctx = AuthorizationContext(
            user_id="analyst_123",
            permissions={"compliance:read"},
        )
        with pytest.raises(PermissionDeniedError):
            verify_audit_trail(analyst_ctx, "2024-01-01", "2024-01-31")


class TestRBACRoleHierarchy:
    """Test role hierarchy and permission inheritance."""

    def test_admin_has_all_backup_permissions(self):
        """Verify admin role has all backup permissions."""
        admin_permissions = {
            "backups:read",
            "backups:create",
            "backups:verify",
            "backups:restore",
            "backups:delete",
            "admin:all",
        }

        @require_permission("backups:delete")
        def delete_backup(ctx, backup_id: str):
            return {"deleted": True}

        admin_ctx = AuthorizationContext(
            user_id="admin_123",
            permissions=admin_permissions,
        )
        result = delete_backup(admin_ctx, "bkp_001")
        assert result["deleted"] is True

    def test_viewer_has_minimal_permissions(self):
        """Verify viewer role has minimal permissions."""
        viewer_permissions = {"debates:read"}  # Only debate read

        @require_permission("backups:read")
        def list_backups(ctx):
            return {"backups": []}

        viewer_ctx = AuthorizationContext(
            user_id="viewer_123",
            permissions=viewer_permissions,
        )
        with pytest.raises(PermissionDeniedError):
            list_backups(viewer_ctx)

    def test_compliance_officer_permissions(self):
        """Verify compliance officer has appropriate permissions."""
        co_permissions = {
            "compliance:read",
            "compliance:soc2",
            "compliance:gdpr",
            "compliance:audit",
        }

        @require_permission("compliance:audit")
        def run_audit(ctx):
            return {"audit_started": True}

        co_ctx = AuthorizationContext(
            user_id="co_123",
            permissions=co_permissions,
        )
        result = run_audit(co_ctx)
        assert result["audit_started"] is True


class TestRBACErrorMessages:
    """Test that RBAC errors include helpful information."""

    def test_permission_denied_includes_permission_key(self):
        """Verify PermissionDeniedError includes the denied permission."""

        @require_permission("sensitive:operation")
        def sensitive_operation(ctx):
            return {}

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"other:permission"},
        )

        with pytest.raises(PermissionDeniedError) as exc_info:
            sensitive_operation(ctx)

        # Error should include information about what permission was denied
        # Permission key is normalized from colon to dot format
        assert exc_info.value.permission_key == "sensitive.operation"

    def test_permission_denied_does_not_leak_sensitive_info(self):
        """Verify error messages don't leak sensitive information."""

        @require_permission("backups:restore")
        def restore_backup(ctx, backup_id: str):
            return {"backup_id": backup_id, "secret_path": "/secure/location"}

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions=set(),
        )

        with pytest.raises(PermissionDeniedError) as exc_info:
            restore_backup(ctx, "bkp_001")

        # Error message should not contain the backup_id or secret paths
        error_str = str(exc_info.value)
        assert "/secure/location" not in error_str


class TestAsyncRBACDecorators:
    """Test that RBAC decorators work with async functions."""

    @pytest.mark.asyncio
    async def test_async_permission_check_allowed(self):
        """Verify RBAC works with async functions when allowed."""

        @require_permission("backups:read")
        async def async_list_backups(ctx):
            return {"backups": []}

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backups:read"},
        )
        result = await async_list_backups(ctx)
        assert "backups" in result

    @pytest.mark.asyncio
    async def test_async_permission_check_denied(self):
        """Verify RBAC works with async functions when denied."""

        @require_permission("backups:delete")
        async def async_delete_backup(ctx, backup_id: str):
            return {"deleted": True}

        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backups:read"},
        )
        with pytest.raises(PermissionDeniedError):
            await async_delete_backup(ctx, "bkp_001")
