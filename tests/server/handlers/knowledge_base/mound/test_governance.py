"""
Tests for GovernanceOperationsMixin.

Tests RBAC and audit trail API endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.governance import (
    GovernanceOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


class Permission(str, Enum):
    """Mock permission enum."""

    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ADMIN = "admin"


class AuditAction(str, Enum):
    """Mock audit action enum."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACCESSED = "accessed"
    SHARED = "shared"


@dataclass
class MockRole:
    """Mock role object."""

    id: str
    name: str
    permissions: Set[Permission]
    description: str = ""
    workspace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "description": self.description,
            "workspace_id": self.workspace_id,
        }


@dataclass
class MockRoleAssignment:
    """Mock role assignment."""

    user_id: str
    role_id: str
    workspace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role_id": self.role_id,
            "workspace_id": self.workspace_id,
        }


@dataclass
class MockAuditEntry:
    """Mock audit entry."""

    id: str
    actor_id: str
    action: str
    resource_id: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp,
        }


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    create_role: AsyncMock = field(default_factory=AsyncMock)
    assign_role: AsyncMock = field(default_factory=AsyncMock)
    revoke_role: AsyncMock = field(default_factory=AsyncMock)
    get_user_permissions: AsyncMock = field(default_factory=AsyncMock)
    check_permission: AsyncMock = field(default_factory=AsyncMock)
    query_audit: AsyncMock = field(default_factory=AsyncMock)
    get_user_activity: AsyncMock = field(default_factory=AsyncMock)
    get_governance_stats: MagicMock = field(default_factory=MagicMock)


class GovernanceHandler(GovernanceOperationsMixin):
    """Handler implementation for testing GovernanceOperationsMixin."""

    def __init__(self, mound: Optional[MockKnowledgeMound] = None):
        self._mound = mound
        self.ctx = {}

    def _get_mound(self):
        return self._mound


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    return MockKnowledgeMound()


@pytest.fixture
def handler(mock_mound):
    """Create a test handler with mock mound."""
    return GovernanceHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a test handler without mound."""
    return GovernanceHandler(mound=None)


# =============================================================================
# Test create_role
# =============================================================================


class TestCreateRole:
    """Tests for create_role endpoint."""

    @pytest.mark.asyncio
    async def test_create_role_success(self, handler, mock_mound):
        """Test successful role creation."""
        mock_role = MockRole(
            id="role-123",
            name="Editor",
            permissions={Permission.READ, Permission.CREATE, Permission.UPDATE},
            description="Can edit items",
        )
        mock_mound.create_role.return_value = mock_role

        result = await handler.create_role(
            name="Editor",
            permissions=["read", "create", "update"],
            description="Can edit items",
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["role"]["name"] == "Editor"

    @pytest.mark.asyncio
    async def test_create_role_missing_name(self, handler):
        """Test role creation with missing name."""
        result = await handler.create_role(
            name="",
            permissions=["read"],
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_create_role_missing_permissions(self, handler):
        """Test role creation with missing permissions."""
        result = await handler.create_role(
            name="TestRole",
            permissions=[],
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_create_role_invalid_permission(self, handler):
        """Test role creation with invalid permission."""
        result = await handler.create_role(
            name="TestRole",
            permissions=["invalid_permission"],
        )

        assert result.status_code == 400
        assert "Invalid permission" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_create_role_no_mound(self, handler_no_mound):
        """Test role creation when mound not available."""
        result = await handler_no_mound.create_role(
            name="TestRole",
            permissions=["read"],
        )

        assert result.status_code == 503


# =============================================================================
# Test assign_role
# =============================================================================


class TestAssignRole:
    """Tests for assign_role endpoint."""

    @pytest.mark.asyncio
    async def test_assign_role_success(self, handler, mock_mound):
        """Test successful role assignment."""
        assignment = MockRoleAssignment(
            user_id="user-123",
            role_id="role-456",
            workspace_id="ws-789",
        )
        mock_mound.assign_role.return_value = assignment

        result = await handler.assign_role(
            user_id="user-123",
            role_id="role-456",
            workspace_id="ws-789",
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["assignment"]["user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_assign_role_missing_user_id(self, handler):
        """Test assignment with missing user_id."""
        result = await handler.assign_role(
            user_id="",
            role_id="role-456",
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_assign_role_missing_role_id(self, handler):
        """Test assignment with missing role_id."""
        result = await handler.assign_role(
            user_id="user-123",
            role_id="",
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_assign_role_not_found(self, handler, mock_mound):
        """Test assignment when role not found."""
        mock_mound.assign_role.side_effect = ValueError("Role not found")

        result = await handler.assign_role(
            user_id="user-123",
            role_id="nonexistent",
        )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_assign_role_no_mound(self, handler_no_mound):
        """Test assignment when mound not available."""
        result = await handler_no_mound.assign_role(
            user_id="user-123",
            role_id="role-456",
        )

        assert result.status_code == 503


# =============================================================================
# Test revoke_role
# =============================================================================


class TestRevokeRole:
    """Tests for revoke_role endpoint."""

    @pytest.mark.asyncio
    async def test_revoke_role_success(self, handler, mock_mound):
        """Test successful role revocation."""
        mock_mound.revoke_role.return_value = True

        result = await handler.revoke_role(
            user_id="user-123",
            role_id="role-456",
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert "revoked" in parse_response(result)["message"]

    @pytest.mark.asyncio
    async def test_revoke_role_not_found(self, handler, mock_mound):
        """Test revocation when assignment not found."""
        mock_mound.revoke_role.return_value = False

        result = await handler.revoke_role(
            user_id="user-123",
            role_id="role-456",
        )

        assert result.status_code == 404
        assert "not found" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_revoke_role_missing_params(self, handler):
        """Test revocation with missing parameters."""
        result = await handler.revoke_role(
            user_id="",
            role_id="role-456",
        )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_revoke_role_no_mound(self, handler_no_mound):
        """Test revocation when mound not available."""
        result = await handler_no_mound.revoke_role(
            user_id="user-123",
            role_id="role-456",
        )

        assert result.status_code == 503


# =============================================================================
# Test get_user_permissions
# =============================================================================


class TestGetUserPermissions:
    """Tests for get_user_permissions endpoint."""

    @pytest.mark.asyncio
    async def test_get_permissions_success(self, handler, mock_mound):
        """Test successful permissions retrieval."""
        mock_mound.get_user_permissions.return_value = {
            Permission.READ,
            Permission.CREATE,
        }

        result = await handler.get_user_permissions(
            user_id="user-123",
            workspace_id="ws-456",
        )

        assert result.status_code == 200
        assert parse_response(result)["user_id"] == "user-123"
        assert "read" in parse_response(result)["permissions"]
        assert "create" in parse_response(result)["permissions"]

    @pytest.mark.asyncio
    async def test_get_permissions_missing_user_id(self, handler):
        """Test permissions with missing user_id."""
        result = await handler.get_user_permissions(user_id="")

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_get_permissions_no_mound(self, handler_no_mound):
        """Test permissions when mound not available."""
        result = await handler_no_mound.get_user_permissions(user_id="user-123")

        assert result.status_code == 503


# =============================================================================
# Test check_permission
# =============================================================================


class TestCheckPermission:
    """Tests for check_permission endpoint."""

    @pytest.mark.asyncio
    async def test_check_permission_has_access(self, handler, mock_mound):
        """Test permission check when user has access."""
        mock_mound.check_permission.return_value = True

        result = await handler.check_permission(
            user_id="user-123",
            permission="read",
        )

        assert result.status_code == 200
        assert parse_response(result)["has_permission"] is True

    @pytest.mark.asyncio
    async def test_check_permission_no_access(self, handler, mock_mound):
        """Test permission check when user lacks access."""
        mock_mound.check_permission.return_value = False

        result = await handler.check_permission(
            user_id="user-123",
            permission="delete",
        )

        assert result.status_code == 200
        assert parse_response(result)["has_permission"] is False

    @pytest.mark.asyncio
    async def test_check_permission_invalid(self, handler):
        """Test check with invalid permission."""
        result = await handler.check_permission(
            user_id="user-123",
            permission="invalid",
        )

        assert result.status_code == 400
        assert "Invalid permission" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_check_permission_missing_params(self, handler):
        """Test check with missing parameters."""
        result = await handler.check_permission(
            user_id="",
            permission="read",
        )

        assert result.status_code == 400


# =============================================================================
# Test query_audit_trail
# =============================================================================


class TestQueryAuditTrail:
    """Tests for query_audit_trail endpoint."""

    @pytest.mark.asyncio
    async def test_query_audit_success(self, handler, mock_mound):
        """Test successful audit trail query."""
        entries = [
            MockAuditEntry(
                id="audit-1",
                actor_id="user-123",
                action="created",
                resource_id="item-456",
                timestamp="2026-01-27T12:00:00Z",
            ),
        ]
        mock_mound.query_audit.return_value = entries

        result = await handler.query_audit_trail(
            actor_id="user-123",
            limit=50,
        )

        assert result.status_code == 200
        assert parse_response(result)["count"] == 1
        assert len(parse_response(result)["entries"]) == 1

    @pytest.mark.asyncio
    async def test_query_audit_with_action_filter(self, handler, mock_mound):
        """Test audit query with action filter."""
        mock_mound.query_audit.return_value = []

        result = await handler.query_audit_trail(action="item.create")

        assert result.status_code == 200
        assert parse_response(result)["filters"]["action"] == "item.create"

    @pytest.mark.asyncio
    async def test_query_audit_invalid_action(self, handler):
        """Test audit query with invalid action."""
        result = await handler.query_audit_trail(action="invalid_action")

        assert result.status_code == 400
        assert "Invalid action" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_query_audit_no_mound(self, handler_no_mound):
        """Test audit query when mound not available."""
        result = await handler_no_mound.query_audit_trail()

        assert result.status_code == 503


# =============================================================================
# Test get_user_activity
# =============================================================================


class TestGetUserActivity:
    """Tests for get_user_activity endpoint."""

    @pytest.mark.asyncio
    async def test_get_activity_success(self, handler, mock_mound):
        """Test successful activity retrieval."""
        mock_activity = {
            "user_id": "user-123",
            "total_actions": 150,
            "by_action": {"created": 50, "updated": 80, "deleted": 20},
        }
        mock_mound.get_user_activity.return_value = mock_activity

        result = await handler.get_user_activity(
            user_id="user-123",
            days=30,
        )

        assert result.status_code == 200
        assert parse_response(result)["total_actions"] == 150

    @pytest.mark.asyncio
    async def test_get_activity_missing_user_id(self, handler):
        """Test activity with missing user_id."""
        result = await handler.get_user_activity(user_id="")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_get_activity_no_mound(self, handler_no_mound):
        """Test activity when mound not available."""
        result = await handler_no_mound.get_user_activity(user_id="user-123")

        assert result.status_code == 503


# =============================================================================
# Test get_governance_stats
# =============================================================================


class TestGetGovernanceStats:
    """Tests for get_governance_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_mound):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_roles": 10,
            "total_assignments": 50,
            "audit_entries_last_30d": 1000,
        }
        mock_mound.get_governance_stats.return_value = mock_stats

        result = await handler.get_governance_stats()

        assert result.status_code == 200
        assert parse_response(result)["total_roles"] == 10

    @pytest.mark.asyncio
    async def test_get_stats_no_mound(self, handler_no_mound):
        """Test stats when mound not available."""
        result = await handler_no_mound.get_governance_stats()

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_stats_error(self, handler, mock_mound):
        """Test stats error handling."""
        mock_mound.get_governance_stats.side_effect = Exception("Error")

        result = await handler.get_governance_stats()

        assert result.status_code == 500
