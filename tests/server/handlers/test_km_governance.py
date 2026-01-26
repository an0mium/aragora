"""
Tests for KnowledgeMound Governance Handler operations.

Tests RBAC and audit trail functionality:
- Role creation and management
- Role assignment/revocation
- Permission checking
- Audit trail querying
- User activity tracking
- Governance statistics
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_mound():
    """Create mock KnowledgeMound instance."""
    mound = MagicMock()

    # Mock role creation
    mock_role = MagicMock()
    mock_role.to_dict.return_value = {
        "id": "role_123",
        "name": "Custom Editor",
        "permissions": ["read", "create", "update"],
        "description": "Can read and edit items",
        "workspace_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    mound.create_role = AsyncMock(return_value=mock_role)

    # Mock role assignment
    mock_assignment = MagicMock()
    mock_assignment.to_dict.return_value = {
        "id": "assign_456",
        "user_id": "user_123",
        "role_id": "role_123",
        "workspace_id": None,
        "assigned_by": "admin_001",
        "assigned_at": datetime.now(timezone.utc).isoformat(),
    }
    mound.assign_role = AsyncMock(return_value=mock_assignment)

    # Mock role revocation
    mound.revoke_role = AsyncMock(return_value=True)

    # Mock permissions retrieval
    from aragora.knowledge.mound.ops.governance import Permission

    mound.get_user_permissions = AsyncMock(return_value={Permission.READ, Permission.CREATE})

    # Mock permission check
    mound.check_permission = AsyncMock(return_value=True)

    # Mock audit trail
    mock_audit_entry = MagicMock()
    mock_audit_entry.to_dict.return_value = {
        "id": "audit_001",
        "actor_id": "user_123",
        "action": "create",
        "resource_id": "node_456",
        "resource_type": "node",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": {},
    }
    mound.query_audit = AsyncMock(return_value=[mock_audit_entry])

    # Mock user activity
    mound.get_user_activity = AsyncMock(
        return_value={
            "user_id": "user_123",
            "total_actions": 42,
            "actions_by_type": {"create": 20, "update": 15, "read": 7},
            "most_active_day": "2024-01-15",
        }
    )

    # Mock governance stats
    mound.get_governance_stats = MagicMock(
        return_value={
            "total_roles": 5,
            "total_assignments": 25,
            "total_audit_entries": 1000,
            "active_users": 15,
        }
    )

    return mound


@pytest.fixture
def governance_mixin(mock_mound):
    """Create handler instance with governance mixin."""
    from aragora.server.handlers.knowledge_base.mound.governance import (
        GovernanceOperationsMixin,
    )

    class MockHandler(GovernanceOperationsMixin):
        def __init__(self, mound):
            self.ctx = {}
            self._mound = mound

        def _get_mound(self):
            return self._mound

    return MockHandler(mock_mound)


# ===========================================================================
# Test Role Creation
# ===========================================================================


class TestCreateRole:
    """Tests for create_role endpoint."""

    async def test_create_role_success(self, governance_mixin, mock_mound):
        """Test successful role creation."""
        result = await governance_mixin.create_role(
            name="Custom Editor",
            permissions=["read", "create", "update"],
            description="Can read and edit items",
            created_by="admin_001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert "role" in body
        assert body["role"]["name"] == "Custom Editor"
        mock_mound.create_role.assert_called_once()

    async def test_create_role_missing_name(self, governance_mixin):
        """Test role creation fails without name."""
        result = await governance_mixin.create_role(
            name="",
            permissions=["read"],
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert "name" in body.get("error", "").lower()

    async def test_create_role_missing_permissions(self, governance_mixin):
        """Test role creation fails without permissions."""
        result = await governance_mixin.create_role(
            name="Empty Role",
            permissions=[],
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert "permissions" in body.get("error", "").lower()

    async def test_create_role_invalid_permission(self, governance_mixin):
        """Test role creation fails with invalid permission."""
        result = await governance_mixin.create_role(
            name="Bad Role",
            permissions=["invalid_permission"],
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert "invalid permission" in body.get("error", "").lower()

    async def test_create_role_mound_unavailable(self, governance_mixin):
        """Test role creation when mound unavailable."""
        governance_mixin._mound = None

        result = await governance_mixin.create_role(
            name="Test Role",
            permissions=["read"],
        )

        assert result.status_code == 503

    async def test_create_role_with_workspace(self, governance_mixin, mock_mound):
        """Test role creation scoped to workspace."""
        result = await governance_mixin.create_role(
            name="Workspace Role",
            permissions=["read", "create"],
            workspace_id="ws_123",
        )

        assert result.status_code == 200
        mock_mound.create_role.assert_called_once()
        call_kwargs = mock_mound.create_role.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws_123"


# ===========================================================================
# Test Role Assignment
# ===========================================================================


class TestAssignRole:
    """Tests for assign_role endpoint."""

    async def test_assign_role_success(self, governance_mixin, mock_mound):
        """Test successful role assignment."""
        result = await governance_mixin.assign_role(
            user_id="user_123",
            role_id="role_456",
            assigned_by="admin_001",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert "assignment" in body
        mock_mound.assign_role.assert_called_once()

    async def test_assign_role_missing_user_id(self, governance_mixin):
        """Test assignment fails without user_id."""
        result = await governance_mixin.assign_role(
            user_id="",
            role_id="role_456",
        )

        assert result.status_code == 400

    async def test_assign_role_missing_role_id(self, governance_mixin):
        """Test assignment fails without role_id."""
        result = await governance_mixin.assign_role(
            user_id="user_123",
            role_id="",
        )

        assert result.status_code == 400

    async def test_assign_role_not_found(self, governance_mixin, mock_mound):
        """Test assignment fails when role not found."""
        mock_mound.assign_role = AsyncMock(side_effect=ValueError("Role not found: role_999"))

        result = await governance_mixin.assign_role(
            user_id="user_123",
            role_id="role_999",
        )

        assert result.status_code == 404

    async def test_assign_role_with_workspace(self, governance_mixin, mock_mound):
        """Test role assignment scoped to workspace."""
        result = await governance_mixin.assign_role(
            user_id="user_123",
            role_id="role_456",
            workspace_id="ws_789",
        )

        assert result.status_code == 200
        call_kwargs = mock_mound.assign_role.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws_789"


# ===========================================================================
# Test Role Revocation
# ===========================================================================


class TestRevokeRole:
    """Tests for revoke_role endpoint."""

    async def test_revoke_role_success(self, governance_mixin, mock_mound):
        """Test successful role revocation."""
        result = await governance_mixin.revoke_role(
            user_id="user_123",
            role_id="role_456",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        mock_mound.revoke_role.assert_called_once()

    async def test_revoke_role_missing_user_id(self, governance_mixin):
        """Test revocation fails without user_id."""
        result = await governance_mixin.revoke_role(
            user_id="",
            role_id="role_456",
        )

        assert result.status_code == 400

    async def test_revoke_role_not_found(self, governance_mixin, mock_mound):
        """Test revocation fails when assignment not found."""
        mock_mound.revoke_role = AsyncMock(return_value=False)

        result = await governance_mixin.revoke_role(
            user_id="user_123",
            role_id="role_999",
        )

        assert result.status_code == 404


# ===========================================================================
# Test Permission Checking
# ===========================================================================


class TestGetUserPermissions:
    """Tests for get_user_permissions endpoint."""

    async def test_get_permissions_success(self, governance_mixin, mock_mound):
        """Test getting user permissions."""
        result = await governance_mixin.get_user_permissions(
            user_id="user_123",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["user_id"] == "user_123"
        assert "permissions" in body
        assert len(body["permissions"]) == 2
        mock_mound.get_user_permissions.assert_called_once()

    async def test_get_permissions_missing_user_id(self, governance_mixin):
        """Test fails without user_id."""
        result = await governance_mixin.get_user_permissions(
            user_id="",
        )

        assert result.status_code == 400

    async def test_get_permissions_with_workspace(self, governance_mixin, mock_mound):
        """Test getting permissions for specific workspace."""
        result = await governance_mixin.get_user_permissions(
            user_id="user_123",
            workspace_id="ws_456",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["workspace_id"] == "ws_456"


class TestCheckPermission:
    """Tests for check_permission endpoint."""

    async def test_check_permission_allowed(self, governance_mixin, mock_mound):
        """Test permission check when allowed."""
        result = await governance_mixin.check_permission(
            user_id="user_123",
            permission="read",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["has_permission"] is True
        mock_mound.check_permission.assert_called_once()

    async def test_check_permission_denied(self, governance_mixin, mock_mound):
        """Test permission check when denied."""
        mock_mound.check_permission = AsyncMock(return_value=False)

        result = await governance_mixin.check_permission(
            user_id="user_123",
            permission="admin",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["has_permission"] is False

    async def test_check_permission_invalid(self, governance_mixin):
        """Test check fails with invalid permission."""
        result = await governance_mixin.check_permission(
            user_id="user_123",
            permission="invalid_permission",
        )

        assert result.status_code == 400

    async def test_check_permission_missing_params(self, governance_mixin):
        """Test check fails without required params."""
        result = await governance_mixin.check_permission(
            user_id="",
            permission="read",
        )
        assert result.status_code == 400

        result = await governance_mixin.check_permission(
            user_id="user_123",
            permission="",
        )
        assert result.status_code == 400


# ===========================================================================
# Test Audit Trail
# ===========================================================================


class TestQueryAuditTrail:
    """Tests for query_audit_trail endpoint."""

    async def test_query_audit_success(self, governance_mixin, mock_mound):
        """Test querying audit trail."""
        result = await governance_mixin.query_audit_trail()

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert "entries" in body
        assert body["count"] == 1
        mock_mound.query_audit.assert_called_once()

    async def test_query_audit_with_filters(self, governance_mixin, mock_mound):
        """Test querying audit with filters."""
        result = await governance_mixin.query_audit_trail(
            actor_id="user_123",
            action="item.create",  # Use valid AuditAction enum value
            workspace_id="ws_456",
            limit=50,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["filters"]["actor_id"] == "user_123"
        assert body["filters"]["action"] == "item.create"

    async def test_query_audit_invalid_action(self, governance_mixin):
        """Test audit query with invalid action."""
        result = await governance_mixin.query_audit_trail(
            action="invalid_action",
        )

        assert result.status_code == 400


class TestGetUserActivity:
    """Tests for get_user_activity endpoint."""

    async def test_get_activity_success(self, governance_mixin, mock_mound):
        """Test getting user activity."""
        result = await governance_mixin.get_user_activity(
            user_id="user_123",
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["user_id"] == "user_123"
        assert body["total_actions"] == 42
        mock_mound.get_user_activity.assert_called_once()

    async def test_get_activity_missing_user_id(self, governance_mixin):
        """Test activity fails without user_id."""
        result = await governance_mixin.get_user_activity(
            user_id="",
        )

        assert result.status_code == 400

    async def test_get_activity_with_days(self, governance_mixin, mock_mound):
        """Test activity with custom days parameter."""
        result = await governance_mixin.get_user_activity(
            user_id="user_123",
            days=7,
        )

        assert result.status_code == 200
        call_kwargs = mock_mound.get_user_activity.call_args.kwargs
        assert call_kwargs["days"] == 7


# ===========================================================================
# Test Governance Stats
# ===========================================================================


class TestGetGovernanceStats:
    """Tests for get_governance_stats endpoint."""

    async def test_get_stats_success(self, governance_mixin, mock_mound):
        """Test getting governance statistics."""
        result = await governance_mixin.get_governance_stats()

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["total_roles"] == 5
        assert body["total_assignments"] == 25
        assert body["active_users"] == 15
        mock_mound.get_governance_stats.assert_called_once()

    async def test_get_stats_mound_unavailable(self, governance_mixin):
        """Test stats when mound unavailable."""
        governance_mixin._mound = None

        result = await governance_mixin.get_governance_stats()

        assert result.status_code == 503


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in governance operations."""

    async def test_create_role_exception(self, governance_mixin, mock_mound):
        """Test role creation handles exceptions."""
        mock_mound.create_role = AsyncMock(side_effect=Exception("Database error"))

        result = await governance_mixin.create_role(
            name="Test Role",
            permissions=["read"],
        )

        assert result.status_code == 500

    async def test_audit_query_exception(self, governance_mixin, mock_mound):
        """Test audit query handles exceptions."""
        mock_mound.query_audit = AsyncMock(side_effect=Exception("Query failed"))

        result = await governance_mixin.query_audit_trail()

        assert result.status_code == 500

    async def test_permission_check_exception(self, governance_mixin, mock_mound):
        """Test permission check handles exceptions."""
        mock_mound.check_permission = AsyncMock(side_effect=Exception("Permission check failed"))

        result = await governance_mixin.check_permission(
            user_id="user_123",
            permission="read",
        )

        assert result.status_code == 500
