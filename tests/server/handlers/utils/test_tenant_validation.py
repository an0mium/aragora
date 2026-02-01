"""
Tests for tenant access validation utilities.

Covers:
- Valid tenant access (user is member)
- Invalid tenant access (user is not member)
- Admin override (system admins bypass restrictions)
- Audit log generation for cross-tenant access attempts
- Workspace validation (alias for tenant validation)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.utils.tenant_validation import (
    TenantAccessDeniedError,
    _get_user_tenant_memberships,
    _is_admin_user,
    audit_cross_tenant_attempt,
    get_validated_tenant_id,
    validate_tenant_access,
    validate_tenant_access_sync,
    validate_workspace_access,
    validate_workspace_access_sync,
)


def _get_response_status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _get_response_body(result) -> dict:
    """Extract and parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# ===========================================================================
# Test fixtures and helpers
# ===========================================================================


@dataclass
class MockUser:
    """Mock user context for testing."""

    user_id: str = "user-123"
    tenant_id: Optional[str] = None
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None
    roles: Optional[set[str]] = None
    role: Optional[str] = None
    permissions: Optional[set[str]] = None
    is_admin: bool = False
    is_superadmin: bool = False
    workspace_memberships: Optional[list[dict]] = None
    tenant_memberships: Optional[list[dict]] = None


# ===========================================================================
# Test _get_user_tenant_memberships
# ===========================================================================


class TestGetUserTenantMemberships:
    """Tests for extracting tenant memberships from user context."""

    def test_extracts_tenant_id(self):
        user = MockUser(tenant_id="tenant-1")
        memberships = _get_user_tenant_memberships(user)
        assert "tenant-1" in memberships

    def test_extracts_org_id(self):
        user = MockUser(org_id="org-1")
        memberships = _get_user_tenant_memberships(user)
        assert "org-1" in memberships

    def test_extracts_workspace_id(self):
        user = MockUser(workspace_id="ws-1")
        memberships = _get_user_tenant_memberships(user)
        assert "ws-1" in memberships

    def test_extracts_workspace_memberships_dict(self):
        user = MockUser(
            workspace_memberships=[
                {"workspace_id": "ws-1"},
                {"workspace_id": "ws-2"},
            ]
        )
        memberships = _get_user_tenant_memberships(user)
        assert "ws-1" in memberships
        assert "ws-2" in memberships

    def test_extracts_workspace_memberships_id_fallback(self):
        user = MockUser(
            workspace_memberships=[
                {"id": "ws-3"},
            ]
        )
        memberships = _get_user_tenant_memberships(user)
        assert "ws-3" in memberships

    def test_extracts_tenant_memberships_dict(self):
        user = MockUser(
            tenant_memberships=[
                {"tenant_id": "t-1"},
                {"tenant_id": "t-2"},
            ]
        )
        memberships = _get_user_tenant_memberships(user)
        assert "t-1" in memberships
        assert "t-2" in memberships

    def test_combines_all_sources(self):
        user = MockUser(
            tenant_id="tenant-1",
            org_id="org-1",
            workspace_id="ws-1",
            workspace_memberships=[{"workspace_id": "ws-2"}],
            tenant_memberships=[{"tenant_id": "t-2"}],
        )
        memberships = _get_user_tenant_memberships(user)
        assert memberships == {"tenant-1", "org-1", "ws-1", "ws-2", "t-2"}

    def test_handles_empty_lists(self):
        user = MockUser(
            workspace_memberships=[],
            tenant_memberships=[],
        )
        memberships = _get_user_tenant_memberships(user)
        assert memberships == set()

    def test_handles_none_values(self):
        user = MockUser()
        memberships = _get_user_tenant_memberships(user)
        assert memberships == set()


# ===========================================================================
# Test _is_admin_user
# ===========================================================================


class TestIsAdminUser:
    """Tests for admin user detection."""

    def test_is_admin_flag(self):
        user = MockUser(is_admin=True)
        assert _is_admin_user(user) is True

    def test_is_superadmin_flag(self):
        user = MockUser(is_superadmin=True)
        assert _is_admin_user(user) is True

    def test_admin_role_in_roles_set(self):
        user = MockUser(roles={"admin", "member"})
        assert _is_admin_user(user) is True

    def test_super_admin_role_in_roles_set(self):
        user = MockUser(roles={"super_admin"})
        assert _is_admin_user(user) is True

    def test_admin_role_attribute(self):
        user = MockUser(role="admin")
        assert _is_admin_user(user) is True

    def test_admin_permission(self):
        user = MockUser(permissions={"admin", "read"})
        assert _is_admin_user(user) is True

    def test_wildcard_permission(self):
        user = MockUser(permissions={"*"})
        assert _is_admin_user(user) is True

    def test_not_admin_regular_user(self):
        user = MockUser(roles={"member"}, role="member", permissions={"read"})
        assert _is_admin_user(user) is False

    def test_not_admin_empty_user(self):
        user = MockUser()
        assert _is_admin_user(user) is False

    def test_roles_as_list(self):
        user = MockUser()
        user.roles = ["admin", "member"]  # List instead of set
        assert _is_admin_user(user) is True


# ===========================================================================
# Test validate_tenant_access (async)
# ===========================================================================


class TestValidateTenantAccess:
    """Tests for tenant access validation (async version)."""

    @pytest.mark.asyncio
    async def test_allows_none_tenant_when_allow_none_true(self):
        user = MockUser(tenant_id="tenant-1")
        result = await validate_tenant_access(user, None, allow_none=True)
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_rejects_none_tenant_when_allow_none_false(self):
        user = MockUser(tenant_id="tenant-1")
        result = await validate_tenant_access(user, None, allow_none=False)
        assert result is not None
        assert _get_response_status(result) == 400

    @pytest.mark.asyncio
    async def test_allows_user_tenant(self):
        user = MockUser(tenant_id="tenant-1")
        result = await validate_tenant_access(user, "tenant-1")
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_allows_user_org_id(self):
        user = MockUser(org_id="org-1")
        result = await validate_tenant_access(user, "org-1")
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_allows_user_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = await validate_tenant_access(user, "ws-1")
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_allows_workspace_membership(self):
        user = MockUser(
            workspace_memberships=[
                {"workspace_id": "ws-1"},
                {"workspace_id": "ws-2"},
            ]
        )
        result = await validate_tenant_access(user, "ws-2")
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_rejects_default_without_membership(self):
        # validate_tenant_access does NOT automatically allow "default"
        # Use validate_workspace_access for that behavior
        user = MockUser(tenant_id="tenant-1")
        result = await validate_tenant_access(user, "default")
        # "default" is not in memberships, so should be rejected
        assert result is not None
        assert _get_response_status(result) == 403

    @pytest.mark.asyncio
    async def test_rejects_unauthorized_tenant(self):
        user = MockUser(tenant_id="tenant-1")
        result = await validate_tenant_access(user, "tenant-2")
        assert result is not None
        assert _get_response_status(result) == 403
        body = _get_response_body(result)
        assert body["error"]["code"] == "TENANT_ACCESS_DENIED"

    @pytest.mark.asyncio
    async def test_admin_bypasses_restrictions(self):
        user = MockUser(tenant_id="tenant-1", is_admin=True)
        result = await validate_tenant_access(user, "tenant-2")
        assert result is None  # Admin can access any tenant

    @pytest.mark.asyncio
    async def test_superadmin_bypasses_restrictions(self):
        user = MockUser(tenant_id="tenant-1", is_superadmin=True)
        result = await validate_tenant_access(user, "tenant-other")
        assert result is None  # Superadmin can access any tenant


# ===========================================================================
# Test validate_tenant_access_sync
# ===========================================================================


class TestValidateTenantAccessSync:
    """Tests for tenant access validation (sync version)."""

    def test_allows_user_tenant(self):
        user = MockUser(tenant_id="tenant-1")
        result = validate_tenant_access_sync(user, "tenant-1")
        assert result is None  # No error

    def test_rejects_unauthorized_tenant(self):
        user = MockUser(tenant_id="tenant-1")
        result = validate_tenant_access_sync(user, "tenant-2")
        assert result is not None
        assert _get_response_status(result) == 403

    def test_admin_bypasses_restrictions(self):
        user = MockUser(tenant_id="tenant-1", is_admin=True)
        result = validate_tenant_access_sync(user, "tenant-2")
        assert result is None

    def test_rejects_default_without_membership(self):
        # validate_tenant_access_sync does NOT automatically allow "default"
        # Use validate_workspace_access_sync for that behavior
        user = MockUser(tenant_id="tenant-1")
        result = validate_tenant_access_sync(user, "default")
        # "default" is not in memberships, so should be rejected
        assert result is not None
        assert _get_response_status(result) == 403


# ===========================================================================
# Test validate_workspace_access
# ===========================================================================


class TestValidateWorkspaceAccess:
    """Tests for workspace access validation (async version)."""

    @pytest.mark.asyncio
    async def test_allows_default_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = await validate_workspace_access(user, "default")
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_allows_user_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = await validate_workspace_access(user, "ws-1")
        assert result is None  # No error

    @pytest.mark.asyncio
    async def test_rejects_unauthorized_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = await validate_workspace_access(user, "ws-other")
        assert result is not None
        assert _get_response_status(result) == 403


# ===========================================================================
# Test validate_workspace_access_sync
# ===========================================================================


class TestValidateWorkspaceAccessSync:
    """Tests for workspace access validation (sync version)."""

    def test_allows_default_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = validate_workspace_access_sync(user, "default")
        assert result is None  # No error

    def test_allows_user_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = validate_workspace_access_sync(user, "ws-1")
        assert result is None  # No error

    def test_rejects_unauthorized_workspace(self):
        user = MockUser(workspace_id="ws-1")
        result = validate_workspace_access_sync(user, "ws-other")
        assert result is not None
        assert _get_response_status(result) == 403

    def test_disallow_default_option(self):
        user = MockUser(workspace_id="ws-1")
        result = validate_workspace_access_sync(user, "default", allow_default=False)
        assert result is not None
        assert _get_response_status(result) == 403


# ===========================================================================
# Test get_validated_tenant_id
# ===========================================================================


class TestGetValidatedTenantId:
    """Tests for validated tenant ID retrieval."""

    def test_returns_requested_tenant_if_provided(self):
        user = MockUser(tenant_id="tenant-1")
        result = get_validated_tenant_id(user, "tenant-2")
        assert result == "tenant-2"

    def test_falls_back_to_user_tenant_id(self):
        user = MockUser(tenant_id="tenant-1")
        result = get_validated_tenant_id(user, None)
        assert result == "tenant-1"

    def test_falls_back_to_user_org_id(self):
        user = MockUser(org_id="org-1")
        result = get_validated_tenant_id(user, None)
        assert result == "org-1"

    def test_falls_back_to_user_workspace_id(self):
        user = MockUser(workspace_id="ws-1")
        result = get_validated_tenant_id(user, None)
        assert result == "ws-1"

    def test_returns_none_if_no_fallback(self):
        user = MockUser()
        result = get_validated_tenant_id(user, None)
        assert result is None


# ===========================================================================
# Test audit_cross_tenant_attempt
# ===========================================================================


class TestAuditCrossTenantAttempt:
    """Tests for cross-tenant access audit logging."""

    def test_logs_security_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            audit_cross_tenant_attempt(
                user_id="user-123",
                user_tenant_id="tenant-1",
                requested_tenant_id="tenant-2",
                endpoint="/api/v2/integrations",
                ip_address="192.168.1.1",
            )

        # Check the log contains key information
        assert "user-123" in caplog.text
        assert "tenant-1" in caplog.text
        assert "tenant-2" in caplog.text
        assert "Cross-tenant access attempt" in caplog.text

    def test_logs_with_additional_context(self, caplog):
        with caplog.at_level(logging.WARNING):
            audit_cross_tenant_attempt(
                user_id="user-456",
                user_tenant_id="tenant-a",
                requested_tenant_id="tenant-b",
                endpoint="/api/test",
                additional_context={"action": "delete"},
            )

        assert "user-456" in caplog.text


# ===========================================================================
# Test TenantAccessDeniedError
# ===========================================================================


class TestTenantAccessDeniedError:
    """Tests for TenantAccessDeniedError exception."""

    def test_exception_attributes(self):
        error = TenantAccessDeniedError(
            user_id="user-1",
            requested_tenant_id="tenant-2",
            message="Custom message",
        )
        assert error.user_id == "user-1"
        assert error.requested_tenant_id == "tenant-2"
        assert str(error) == "Custom message"

    def test_default_message(self):
        error = TenantAccessDeniedError(
            user_id="user-1",
            requested_tenant_id="tenant-2",
        )
        assert "Access denied" in str(error)
