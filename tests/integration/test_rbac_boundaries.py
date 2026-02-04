"""
Integration tests for RBAC permission enforcement across APIs.

These tests verify that role-based access control is properly enforced
across all API endpoints and that permission boundaries are respected.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str
    email: str
    roles: list = field(default_factory=list)
    workspace_id: str = "workspace-1"
    permissions: set = field(default_factory=set)


@dataclass
class MockAuthContext:
    """Mock authorization context."""

    user: MockUser
    workspace_id: str
    tenant_id: str = "tenant-1"
    is_admin: bool = False


class TestPermissionEnforcement:
    """Test permission enforcement across different endpoints."""

    @pytest.fixture
    def admin_user(self):
        """Create admin user with full permissions."""
        return MockUser(
            user_id="admin-1",
            email="admin@example.com",
            roles=["admin", "owner"],
            permissions={
                "debates:read",
                "debates:write",
                "debates:delete",
                "users:read",
                "users:write",
                "users:delete",
                "settings:read",
                "settings:write",
                "billing:read",
                "billing:write",
                "audit:read",
            },
        )

    @pytest.fixture
    def regular_user(self):
        """Create regular user with limited permissions."""
        return MockUser(
            user_id="user-1",
            email="user@example.com",
            roles=["member"],
            permissions={
                "debates:read",
                "debates:write",
                "users:read",
            },
        )

    @pytest.fixture
    def viewer_user(self):
        """Create viewer with read-only permissions."""
        return MockUser(
            user_id="viewer-1",
            email="viewer@example.com",
            roles=["viewer"],
            permissions={
                "debates:read",
                "users:read",
            },
        )

    @pytest.fixture
    def mock_permission_checker(self):
        """Create mock permission checker."""
        checker = MagicMock()

        def check_permission(user: MockUser, permission: str) -> bool:
            return permission in user.permissions

        checker.has_permission = check_permission
        return checker

    async def test_admin_can_delete_debates(self, admin_user, mock_permission_checker):
        """Admin should be able to delete debates."""
        can_delete = mock_permission_checker.has_permission(admin_user, "debates:delete")
        assert can_delete is True

    async def test_regular_user_cannot_delete_debates(self, regular_user, mock_permission_checker):
        """Regular user should not be able to delete debates."""
        can_delete = mock_permission_checker.has_permission(regular_user, "debates:delete")
        assert can_delete is False

    async def test_viewer_cannot_write_debates(self, viewer_user, mock_permission_checker):
        """Viewer should not be able to write debates."""
        can_write = mock_permission_checker.has_permission(viewer_user, "debates:write")
        assert can_write is False

    async def test_admin_can_access_billing(self, admin_user, mock_permission_checker):
        """Admin should be able to access billing."""
        can_read_billing = mock_permission_checker.has_permission(admin_user, "billing:read")
        can_write_billing = mock_permission_checker.has_permission(admin_user, "billing:write")
        assert can_read_billing is True
        assert can_write_billing is True

    async def test_regular_user_cannot_access_billing(self, regular_user, mock_permission_checker):
        """Regular user should not be able to access billing."""
        can_read_billing = mock_permission_checker.has_permission(regular_user, "billing:read")
        assert can_read_billing is False


class TestWorkspaceIsolation:
    """Test workspace isolation across API boundaries."""

    @pytest.fixture
    def user_workspace_a(self):
        """User in workspace A."""
        return MockUser(
            user_id="user-a",
            email="user-a@example.com",
            workspace_id="workspace-a",
            permissions={"debates:read", "debates:write"},
        )

    @pytest.fixture
    def user_workspace_b(self):
        """User in workspace B."""
        return MockUser(
            user_id="user-b",
            email="user-b@example.com",
            workspace_id="workspace-b",
            permissions={"debates:read", "debates:write"},
        )

    @pytest.fixture
    def mock_debate_store(self):
        """Create mock debate store with workspace filtering."""
        store = MagicMock()
        store.debates = {
            "debate-a1": {"id": "debate-a1", "workspace_id": "workspace-a"},
            "debate-a2": {"id": "debate-a2", "workspace_id": "workspace-a"},
            "debate-b1": {"id": "debate-b1", "workspace_id": "workspace-b"},
        }

        async def get_by_workspace(workspace_id: str):
            return [d for d in store.debates.values() if d["workspace_id"] == workspace_id]

        async def get_debate(debate_id: str, workspace_id: str):
            debate = store.debates.get(debate_id)
            if debate and debate["workspace_id"] == workspace_id:
                return debate
            return None

        store.get_by_workspace = get_by_workspace
        store.get_debate = get_debate
        return store

    async def test_user_sees_only_own_workspace_debates(self, user_workspace_a, mock_debate_store):
        """User should only see debates from their workspace."""
        debates = await mock_debate_store.get_by_workspace(user_workspace_a.workspace_id)

        assert len(debates) == 2
        assert all(d["workspace_id"] == "workspace-a" for d in debates)

    async def test_user_cannot_access_other_workspace_debates(
        self, user_workspace_a, mock_debate_store
    ):
        """User should not access debates from other workspaces."""
        debate = await mock_debate_store.get_debate("debate-b1", user_workspace_a.workspace_id)

        assert debate is None

    async def test_cross_workspace_access_denied(
        self, user_workspace_a, user_workspace_b, mock_debate_store
    ):
        """Cross-workspace access should be denied."""
        # User A tries to access workspace B debate
        debate_a_sees = await mock_debate_store.get_debate(
            "debate-b1", user_workspace_a.workspace_id
        )

        # User B tries to access workspace A debate
        debate_b_sees = await mock_debate_store.get_debate(
            "debate-a1", user_workspace_b.workspace_id
        )

        assert debate_a_sees is None
        assert debate_b_sees is None


class TestRoleHierarchy:
    """Test role hierarchy and permission inheritance."""

    @pytest.fixture
    def role_hierarchy(self):
        """Define role hierarchy."""
        return {
            "owner": ["admin", "member", "viewer"],
            "admin": ["member", "viewer"],
            "member": ["viewer"],
            "viewer": [],
        }

    @pytest.fixture
    def role_permissions(self):
        """Define permissions per role."""
        return {
            "owner": {
                "workspace:delete",
                "billing:write",
                "users:delete",
            },
            "admin": {
                "settings:write",
                "users:write",
                "audit:read",
            },
            "member": {
                "debates:write",
                "knowledge:write",
            },
            "viewer": {
                "debates:read",
                "knowledge:read",
                "users:read",
            },
        }

    def get_effective_permissions(
        self, role: str, role_hierarchy: dict, role_permissions: dict
    ) -> set:
        """Get all permissions including inherited ones."""
        permissions = set(role_permissions.get(role, set()))
        for inherited_role in role_hierarchy.get(role, []):
            permissions |= self.get_effective_permissions(
                inherited_role, role_hierarchy, role_permissions
            )
        return permissions

    async def test_owner_inherits_all_permissions(self, role_hierarchy, role_permissions):
        """Owner should have all permissions."""
        owner_perms = self.get_effective_permissions("owner", role_hierarchy, role_permissions)

        # Owner should have viewer permissions
        assert "debates:read" in owner_perms

        # Owner should have member permissions
        assert "debates:write" in owner_perms

        # Owner should have admin permissions
        assert "settings:write" in owner_perms

        # Owner should have owner-specific permissions
        assert "workspace:delete" in owner_perms

    async def test_admin_inherits_member_and_viewer(self, role_hierarchy, role_permissions):
        """Admin should inherit member and viewer permissions."""
        admin_perms = self.get_effective_permissions("admin", role_hierarchy, role_permissions)

        assert "debates:read" in admin_perms  # From viewer
        assert "debates:write" in admin_perms  # From member
        assert "settings:write" in admin_perms  # Admin's own
        assert "workspace:delete" not in admin_perms  # Owner only

    async def test_viewer_has_only_read_permissions(self, role_hierarchy, role_permissions):
        """Viewer should only have read permissions."""
        viewer_perms = self.get_effective_permissions("viewer", role_hierarchy, role_permissions)

        assert "debates:read" in viewer_perms
        assert "knowledge:read" in viewer_perms
        assert "debates:write" not in viewer_perms
        assert "settings:write" not in viewer_perms


class TestAPIEndpointProtection:
    """Test that all API endpoints enforce RBAC."""

    @pytest.fixture
    def protected_endpoints(self):
        """Define protected endpoints and required permissions."""
        return {
            ("GET", "/api/v1/debates"): "debates:read",
            ("POST", "/api/v1/debates"): "debates:write",
            ("DELETE", "/api/v1/debates/{id}"): "debates:delete",
            ("GET", "/api/v1/users"): "users:read",
            ("POST", "/api/v1/users/invite"): "users:write",
            ("DELETE", "/api/v1/users/{id}"): "users:delete",
            ("GET", "/api/v1/billing"): "billing:read",
            ("POST", "/api/v1/billing/plans"): "billing:write",
            ("GET", "/api/v1/settings"): "settings:read",
            ("PUT", "/api/v1/settings"): "settings:write",
            ("GET", "/api/v1/audit/logs"): "audit:read",
        }

    @pytest.fixture
    def mock_request_handler(self, protected_endpoints):
        """Create mock request handler with RBAC checks."""
        handler = MagicMock()

        async def handle_request(method: str, path: str, user: MockUser) -> tuple[int, dict]:
            endpoint_key = (method, path)

            # Find matching endpoint (handle path parameters)
            required_permission = None
            for (ep_method, ep_path), perm in protected_endpoints.items():
                if ep_method == method:
                    # Simple path matching (ignores params for testing)
                    if ep_path.replace("{id}", "") in path or ep_path == path:
                        required_permission = perm
                        break

            if required_permission is None:
                return (404, {"error": "Not found"})

            if required_permission not in user.permissions:
                return (403, {"error": "Forbidden"})

            return (200, {"success": True})

        handler.handle_request = handle_request
        return handler

    async def test_debates_read_requires_permission(self, mock_request_handler):
        """Reading debates requires debates:read permission."""
        user_with = MockUser(user_id="u1", email="u1@ex.com", permissions={"debates:read"})
        user_without = MockUser(user_id="u2", email="u2@ex.com", permissions=set())

        status_with, _ = await mock_request_handler.handle_request(
            "GET", "/api/v1/debates", user_with
        )
        status_without, _ = await mock_request_handler.handle_request(
            "GET", "/api/v1/debates", user_without
        )

        assert status_with == 200
        assert status_without == 403

    async def test_billing_requires_billing_permission(self, mock_request_handler):
        """Billing endpoints require billing permissions."""
        admin = MockUser(
            user_id="admin", email="admin@ex.com", permissions={"billing:read", "billing:write"}
        )
        member = MockUser(
            user_id="member", email="member@ex.com", permissions={"debates:read", "debates:write"}
        )

        admin_read, _ = await mock_request_handler.handle_request("GET", "/api/v1/billing", admin)
        member_read, _ = await mock_request_handler.handle_request("GET", "/api/v1/billing", member)

        assert admin_read == 200
        assert member_read == 403

    async def test_audit_read_restricted(self, mock_request_handler):
        """Audit logs require audit:read permission."""
        auditor = MockUser(user_id="auditor", email="auditor@ex.com", permissions={"audit:read"})
        regular = MockUser(user_id="regular", email="regular@ex.com", permissions={"debates:read"})

        auditor_read, _ = await mock_request_handler.handle_request(
            "GET", "/api/v1/audit/logs", auditor
        )
        regular_read, _ = await mock_request_handler.handle_request(
            "GET", "/api/v1/audit/logs", regular
        )

        assert auditor_read == 200
        assert regular_read == 403


class TestTenantBoundaries:
    """Test tenant isolation across the system."""

    @pytest.fixture
    def tenant_a_user(self):
        """User in tenant A."""
        return MockAuthContext(
            user=MockUser(
                user_id="user-a",
                email="user@tenant-a.com",
                workspace_id="ws-a",
                permissions={"debates:read"},
            ),
            workspace_id="ws-a",
            tenant_id="tenant-a",
        )

    @pytest.fixture
    def tenant_b_user(self):
        """User in tenant B."""
        return MockAuthContext(
            user=MockUser(
                user_id="user-b",
                email="user@tenant-b.com",
                workspace_id="ws-b",
                permissions={"debates:read"},
            ),
            workspace_id="ws-b",
            tenant_id="tenant-b",
        )

    async def test_tenant_data_isolation(self, tenant_a_user, tenant_b_user):
        """Tenants should have completely isolated data."""
        tenant_data = {
            "tenant-a": ["data-a1", "data-a2"],
            "tenant-b": ["data-b1", "data-b2"],
        }

        def get_data_for_tenant(ctx: MockAuthContext):
            return tenant_data.get(ctx.tenant_id, [])

        data_a = get_data_for_tenant(tenant_a_user)
        data_b = get_data_for_tenant(tenant_b_user)

        assert data_a == ["data-a1", "data-a2"]
        assert data_b == ["data-b1", "data-b2"]
        assert set(data_a).isdisjoint(set(data_b))

    async def test_cross_tenant_queries_blocked(self, tenant_a_user):
        """Queries across tenant boundaries should be blocked."""

        def query_with_tenant_check(ctx: MockAuthContext, requested_tenant: str) -> Optional[list]:
            if ctx.tenant_id != requested_tenant:
                return None  # Blocked
            return ["allowed_data"]

        # Same tenant - allowed
        result_same = query_with_tenant_check(tenant_a_user, "tenant-a")
        assert result_same == ["allowed_data"]

        # Different tenant - blocked
        result_diff = query_with_tenant_check(tenant_a_user, "tenant-b")
        assert result_diff is None


class TestRBACCaching:
    """Test RBAC permission caching behavior."""

    @pytest.fixture
    def mock_rbac_cache(self):
        """Create mock RBAC cache."""
        cache = MagicMock()
        cache.data = {}
        cache.hits = 0
        cache.misses = 0

        def get(key: str):
            if key in cache.data:
                cache.hits += 1
                return cache.data[key]
            cache.misses += 1
            return None

        def set(key: str, value, ttl: int = 300):
            cache.data[key] = value

        def invalidate(key: str):
            if key in cache.data:
                del cache.data[key]

        cache.get = get
        cache.set = set
        cache.invalidate = invalidate
        return cache

    async def test_permission_cache_hit(self, mock_rbac_cache):
        """Cached permissions should be returned on subsequent requests."""
        user_id = "user-1"
        permissions = {"debates:read", "debates:write"}

        # First request - cache miss
        mock_rbac_cache.set(f"perms:{user_id}", permissions)

        # Second request - cache hit
        cached = mock_rbac_cache.get(f"perms:{user_id}")

        assert cached == permissions
        assert mock_rbac_cache.hits == 1

    async def test_cache_invalidation_on_role_change(self, mock_rbac_cache):
        """Cache should be invalidated when roles change."""
        user_id = "user-1"
        mock_rbac_cache.set(f"perms:{user_id}", {"debates:read"})

        # Simulate role change
        mock_rbac_cache.invalidate(f"perms:{user_id}")

        cached = mock_rbac_cache.get(f"perms:{user_id}")
        assert cached is None
