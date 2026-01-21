"""
Tests for FindingWorkflowHandler RBAC enforcement.

Verifies that all finding workflow endpoints enforce proper RBAC permissions.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockRequest:
    """Mock HTTP request for testing."""

    def __init__(
        self,
        headers: Dict[str, str] | None = None,
        body: bytes = b"{}",
    ):
        self.headers = headers or {}
        self._body = body

    async def read(self) -> bytes:
        return self._body


class MockServerContext:
    """Mock server context for handler initialization."""

    def __init__(self):
        self.config = {}
        self.state = {}


class TestFindingWorkflowRBAC:
    """Test RBAC enforcement on finding workflow endpoints."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        return FindingWorkflowHandler(server_context=MockServerContext())

    @pytest.fixture
    def admin_request(self):
        """Request with admin role."""
        return MockRequest(
            headers={
                "X-User-ID": "admin-123",
                "X-User-Roles": "admin",
                "X-Org-ID": "org-1",
            }
        )

    @pytest.fixture
    def member_request(self):
        """Request with member role."""
        return MockRequest(
            headers={
                "X-User-ID": "user-456",
                "X-User-Roles": "member",
                "X-Org-ID": "org-1",
            }
        )

    @pytest.fixture
    def viewer_request(self):
        """Request with viewer role (read-only)."""
        return MockRequest(
            headers={
                "X-User-ID": "viewer-789",
                "X-User-Roles": "viewer",
                "X-Org-ID": "org-1",
            }
        )

    @pytest.fixture
    def anonymous_request(self):
        """Request with no authentication."""
        return MockRequest(headers={})

    def test_check_permission_returns_none_for_admin(self, handler, admin_request):
        """Admin should have all findings permissions."""
        # Admin has all permissions
        result = handler._check_permission(admin_request, "findings.read")
        assert result is None  # None means allowed

        result = handler._check_permission(admin_request, "findings.update")
        assert result is None

        result = handler._check_permission(admin_request, "findings.assign")
        assert result is None

        result = handler._check_permission(admin_request, "findings.bulk")
        assert result is None

    def test_check_permission_returns_none_for_member_read(
        self, handler, member_request
    ):
        """Member should have read and update permissions."""
        result = handler._check_permission(member_request, "findings.read")
        assert result is None

        result = handler._check_permission(member_request, "findings.update")
        assert result is None

    def test_check_permission_denies_bulk_for_member(self, handler, member_request):
        """Member should not have bulk permission."""
        result = handler._check_permission(member_request, "findings.bulk")
        # Should return error response dict
        assert result is not None
        assert result.get("status") == 403 or "Permission denied" in str(
            result.get("error", "")
        )

    def test_check_permission_allows_read_for_viewer(self, handler, viewer_request):
        """Viewer should have read permission."""
        result = handler._check_permission(viewer_request, "findings.read")
        assert result is None

    def test_check_permission_denies_update_for_viewer(self, handler, viewer_request):
        """Viewer should not have update permission."""
        result = handler._check_permission(viewer_request, "findings.update")
        assert result is not None
        assert result.get("status") == 403 or "Permission denied" in str(
            result.get("error", "")
        )

    def test_get_auth_context_extracts_user_info(self, handler, admin_request):
        """Auth context should extract user info from headers."""
        ctx = handler._get_auth_context(admin_request)

        assert ctx.user_id == "admin-123"
        assert ctx.org_id == "org-1"
        assert "admin" in ctx.roles

    def test_get_auth_context_defaults_for_anonymous(self, handler, anonymous_request):
        """Anonymous requests should get anonymous user_id."""
        ctx = handler._get_auth_context(anonymous_request)

        assert ctx.user_id == "anonymous"
        assert "member" in ctx.roles  # Default role


class TestFindingWorkflowEndpointPermissions:
    """Test that each endpoint checks the correct permission."""

    @pytest.fixture
    def handler(self):
        """Create handler instance with mocked store."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        h = FindingWorkflowHandler(server_context=MockServerContext())
        return h

    @pytest.fixture
    def mock_store(self):
        """Mock finding workflow store."""
        store = MagicMock()
        store.get_workflow.return_value = {
            "finding_id": "finding-1",
            "status": "open",
            "priority": 3,
            "comments": [],
            "history": [],
        }
        store.get_by_assignee.return_value = []
        store.get_overdue.return_value = []
        return store

    @pytest.mark.asyncio
    async def test_update_status_requires_update_permission(self, handler):
        """_update_status should check findings.update permission."""
        request = MockRequest(
            headers={"X-User-ID": "viewer-1", "X-User-Roles": "viewer"},
            body=json.dumps({"status": "triaging"}).encode(),
        )

        result = await handler._update_status(request, "finding-1")

        # Viewer doesn't have update permission
        assert result.get("status") == 403 or "Permission denied" in str(result)

    @pytest.mark.asyncio
    async def test_assign_requires_assign_permission(self, handler):
        """_assign should check findings.assign permission."""
        request = MockRequest(
            headers={"X-User-ID": "viewer-1", "X-User-Roles": "viewer"},
            body=json.dumps({"user_id": "user-2"}).encode(),
        )

        result = await handler._assign(request, "finding-1")

        # Viewer doesn't have assign permission
        assert result.get("status") == 403 or "Permission denied" in str(result)

    @pytest.mark.asyncio
    async def test_bulk_action_requires_bulk_permission(self, handler):
        """_bulk_action should check findings.bulk permission."""
        request = MockRequest(
            headers={"X-User-ID": "member-1", "X-User-Roles": "member"},
            body=json.dumps({
                "finding_ids": ["f1", "f2"],
                "action": "update_status",
                "params": {"status": "resolved"},
            }).encode(),
        )

        result = await handler._bulk_action(request)

        # Member doesn't have bulk permission
        assert result.get("status") == 403 or "Permission denied" in str(result)

    @pytest.mark.asyncio
    async def test_get_comments_requires_read_permission(self, handler):
        """_get_comments should check findings.read permission."""
        # Create request with no roles (minimal permissions)
        request = MockRequest(
            headers={"X-User-ID": "anon", "X-User-Roles": ""},
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.get_finding_workflow_store"
        ) as mock_get_store:
            mock_store = MagicMock()
            mock_store.get_workflow.return_value = None
            mock_get_store.return_value = mock_store

            result = await handler._get_comments(request, "finding-1")

        # Empty roles should deny access
        assert result.get("status") == 403 or "Permission denied" in str(result)


class TestFindingsPermissionsInDefaults:
    """Test that findings permissions are properly registered."""

    def test_findings_permissions_exist(self):
        """All findings permissions should be registered."""
        from aragora.rbac.defaults import SYSTEM_PERMISSIONS

        assert "findings.read" in SYSTEM_PERMISSIONS
        assert "findings.update" in SYSTEM_PERMISSIONS
        assert "findings.assign" in SYSTEM_PERMISSIONS
        assert "findings.bulk" in SYSTEM_PERMISSIONS

    def test_admin_has_all_findings_permissions(self):
        """Admin role should have all findings permissions."""
        from aragora.rbac import get_role_permissions

        admin_perms = get_role_permissions("admin", include_inherited=True)

        assert "findings.read" in admin_perms
        assert "findings.update" in admin_perms
        assert "findings.assign" in admin_perms
        assert "findings.bulk" in admin_perms

    def test_debate_creator_has_findings_permissions(self):
        """Debate creator should have read/update/assign but not bulk."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("debate_creator", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" in perms
        assert "findings.assign" in perms
        # Debate creator should NOT have bulk
        assert "findings.bulk" not in perms

    def test_member_has_limited_findings_permissions(self):
        """Member should have read and update but not assign or bulk."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("member", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" in perms
        # Member should NOT have assign or bulk
        assert "findings.assign" not in perms
        assert "findings.bulk" not in perms

    def test_analyst_has_read_only_findings(self):
        """Analyst should only have read permission."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("analyst", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" not in perms
        assert "findings.assign" not in perms
        assert "findings.bulk" not in perms

    def test_viewer_has_read_only_findings(self):
        """Viewer should only have read permission."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("viewer", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" not in perms


class TestRoutePermissionsForFindings:
    """Test that route permissions are properly configured."""

    def _get_pattern_str(self, rp) -> str:
        """Get pattern as string, handling both str and compiled regex."""
        pattern = rp.pattern
        if hasattr(pattern, "pattern"):
            return pattern.pattern
        return str(pattern)

    def test_findings_routes_in_middleware(self):
        """All findings routes should have permissions configured."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        # Find all findings routes
        findings_routes = [
            rp for rp in DEFAULT_ROUTE_PERMISSIONS if "findings" in self._get_pattern_str(rp)
        ]

        # Should have multiple findings routes
        assert len(findings_routes) >= 10

        # Check specific routes exist
        route_patterns = [self._get_pattern_str(rp) for rp in findings_routes]

        assert any("bulk-action" in p for p in route_patterns)
        assert any("my-assignments" in p for p in route_patterns)
        assert any("status" in p for p in route_patterns)
        assert any("assign" in p for p in route_patterns)
        assert any("comments" in p for p in route_patterns)
        assert any("history" in p for p in route_patterns)
        assert any("priority" in p for p in route_patterns)
        assert any("due-date" in p for p in route_patterns)

    def test_bulk_action_requires_bulk_permission(self):
        """Bulk action route should require findings.bulk permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        bulk_routes = [
            rp for rp in DEFAULT_ROUTE_PERMISSIONS if "bulk-action" in self._get_pattern_str(rp)
        ]

        assert len(bulk_routes) == 1
        assert bulk_routes[0].permission_key == "findings.bulk"

    def test_status_update_requires_update_permission(self):
        """Status update route should require findings.update permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        status_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "/status" in self._get_pattern_str(rp) and "findings" in self._get_pattern_str(rp)
        ]

        assert len(status_routes) >= 1
        assert all(rp.permission_key == "findings.update" for rp in status_routes)

    def test_assign_routes_require_assign_permission(self):
        """Assign/unassign routes should require findings.assign permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        # Look for routes that end with /assign or /unassign (not my-assignments)
        assign_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if (self._get_pattern_str(rp).endswith("/assign$") or
                self._get_pattern_str(rp).endswith("/unassign$"))
            and "findings" in self._get_pattern_str(rp)
        ]

        assert len(assign_routes) >= 2
        assert all(rp.permission_key == "findings.assign" for rp in assign_routes)
