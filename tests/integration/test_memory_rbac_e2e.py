"""
End-to-end integration tests for Memory RBAC and tenant isolation.

Validates that:
- RBAC permissions for memory management exist and are correctly assigned to roles
- ContinuumMemory tenant_id isolation works at the data layer
- PermissionChecker correctly grants/denies memory:manage based on role
- Full pipeline: memory entries + RBAC checks + tenant filtering
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aragora.memory.continuum.core import ContinuumMemory
from aragora.memory.tier_manager import MemoryTier
from aragora.rbac.checker import PermissionChecker
from aragora.rbac.defaults.permissions.knowledge import PERM_MEMORY_MANAGE
from aragora.rbac.defaults.roles import (
    ROLE_ADMIN,
    ROLE_ANALYST,
    ROLE_MEMBER,
    ROLE_OWNER,
    ROLE_VIEWER,
)
from aragora.rbac.models import AuthorizationContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return a temporary SQLite database path for ContinuumMemory."""
    return tmp_path / "test_continuum.db"


@pytest.fixture()
def memory(tmp_db: Path) -> ContinuumMemory:
    """Create a fresh ContinuumMemory backed by a temporary database."""
    return ContinuumMemory(db_path=tmp_db)


@pytest.fixture()
def checker() -> PermissionChecker:
    """Create a fresh PermissionChecker with caching disabled to avoid stale results."""
    return PermissionChecker(enable_cache=False)


def _make_context(user_id: str, roles: set[str], org_id: str = "org-1") -> AuthorizationContext:
    """Helper to build an AuthorizationContext for testing."""
    return AuthorizationContext(user_id=user_id, org_id=org_id, roles=roles)


# ---------------------------------------------------------------------------
# 1. PERM_MEMORY_MANAGE exists with correct key
# ---------------------------------------------------------------------------


class TestPermMemoryManageDefinition:
    """Verify the PERM_MEMORY_MANAGE permission object is correctly defined."""

    def test_perm_memory_manage_key(self) -> None:
        """PERM_MEMORY_MANAGE.key must equal 'memory.manage'."""
        assert PERM_MEMORY_MANAGE.key == "memory.manage"

    def test_perm_memory_manage_resource_and_action(self) -> None:
        """PERM_MEMORY_MANAGE must target the MEMORY resource with MANAGE action."""
        from aragora.rbac.models import Action, ResourceType

        assert PERM_MEMORY_MANAGE.resource == ResourceType.MEMORY
        assert PERM_MEMORY_MANAGE.action == Action.MANAGE


# ---------------------------------------------------------------------------
# 2. Role-level permission assignments
# ---------------------------------------------------------------------------


class TestRoleMemoryPermissions:
    """Verify which roles include memory.manage and which do not."""

    def test_admin_has_memory_manage(self) -> None:
        assert PERM_MEMORY_MANAGE.key in ROLE_ADMIN.permissions

    def test_owner_has_memory_manage(self) -> None:
        assert PERM_MEMORY_MANAGE.key in ROLE_OWNER.permissions

    def test_viewer_lacks_memory_manage(self) -> None:
        assert PERM_MEMORY_MANAGE.key not in ROLE_VIEWER.permissions

    def test_member_lacks_memory_manage(self) -> None:
        assert PERM_MEMORY_MANAGE.key not in ROLE_MEMBER.permissions

    def test_analyst_lacks_memory_manage(self) -> None:
        assert PERM_MEMORY_MANAGE.key not in ROLE_ANALYST.permissions


# ---------------------------------------------------------------------------
# 3. ContinuumMemory add with tenant_id stores it in metadata
# ---------------------------------------------------------------------------


class TestMemoryAddTenantId:
    def test_add_with_tenant_id_stores_in_metadata(self, memory: ContinuumMemory) -> None:
        entry = memory.add(
            id="mem-1",
            content="test content",
            tier=MemoryTier.FAST,
            importance=0.5,
            tenant_id="tenant-alpha",
        )
        assert entry.metadata.get("tenant_id") == "tenant-alpha"

    def test_add_with_tenant_id_persists_to_db(self, memory: ContinuumMemory) -> None:
        memory.add(
            id="mem-2",
            content="persisted content",
            tier=MemoryTier.MEDIUM,
            importance=0.6,
            tenant_id="tenant-beta",
        )
        fetched = memory.get("mem-2")
        assert fetched is not None
        assert fetched.metadata.get("tenant_id") == "tenant-beta"


# ---------------------------------------------------------------------------
# 4. ContinuumMemory get filters by tenant_id
# ---------------------------------------------------------------------------


class TestMemoryGetTenantFilter:
    def test_get_returns_entry_for_matching_tenant(self, memory: ContinuumMemory) -> None:
        memory.add(id="mem-t1", content="alpha data", tenant_id="tenant-alpha")
        result = memory.get("mem-t1", tenant_id="tenant-alpha")
        assert result is not None
        assert result.id == "mem-t1"

    def test_get_returns_none_for_wrong_tenant(self, memory: ContinuumMemory) -> None:
        memory.add(id="mem-t2", content="alpha data", tenant_id="tenant-alpha")
        result = memory.get("mem-t2", tenant_id="tenant-beta")
        assert result is None


# ---------------------------------------------------------------------------
# 5. ContinuumMemory retrieve returns only matching tenant entries
# ---------------------------------------------------------------------------


class TestMemoryRetrieveTenantFilter:
    def test_retrieve_filters_by_tenant_id(self, memory: ContinuumMemory) -> None:
        memory.add(id="r-1", content="alpha entry", importance=0.9, tenant_id="tenant-alpha")
        memory.add(id="r-2", content="beta entry", importance=0.9, tenant_id="tenant-beta")
        memory.add(id="r-3", content="alpha second", importance=0.9, tenant_id="tenant-alpha")

        results = memory.retrieve(tenant_id="tenant-alpha", limit=100)
        ids = {e.id for e in results}
        assert "r-1" in ids
        assert "r-3" in ids
        assert "r-2" not in ids


# ---------------------------------------------------------------------------
# 6. Cross-tenant isolation
# ---------------------------------------------------------------------------


class TestCrossTenantIsolation:
    def test_tenant_a_cannot_see_tenant_b_data(self, memory: ContinuumMemory) -> None:
        """Entries created under tenant-A are invisible to tenant-B queries."""
        memory.add(id="iso-a1", content="secret alpha", importance=0.8, tenant_id="tenant-A")
        memory.add(id="iso-b1", content="secret beta", importance=0.8, tenant_id="tenant-B")

        alpha_results = memory.retrieve(tenant_id="tenant-A", limit=100)
        beta_results = memory.retrieve(tenant_id="tenant-B", limit=100)

        alpha_ids = {e.id for e in alpha_results}
        beta_ids = {e.id for e in beta_results}

        assert "iso-a1" in alpha_ids
        assert "iso-b1" not in alpha_ids
        assert "iso-b1" in beta_ids
        assert "iso-a1" not in beta_ids

    def test_get_enforces_cross_tenant_boundary(self, memory: ContinuumMemory) -> None:
        memory.add(id="iso-x", content="x data", tenant_id="tenant-X")
        assert memory.get("iso-x", tenant_id="tenant-X") is not None
        assert memory.get("iso-x", tenant_id="tenant-Y") is None


# ---------------------------------------------------------------------------
# 7. PermissionChecker grants memory:manage to admin context
# ---------------------------------------------------------------------------


class TestPermissionCheckerGrants:
    def test_admin_granted_memory_manage(self, checker: PermissionChecker) -> None:
        ctx = _make_context(user_id="admin-user", roles={"admin"})
        decision = checker.check_permission(ctx, "memory.manage")
        assert decision.allowed is True

    def test_owner_granted_memory_manage(self, checker: PermissionChecker) -> None:
        ctx = _make_context(user_id="owner-user", roles={"owner"})
        decision = checker.check_permission(ctx, "memory.manage")
        assert decision.allowed is True


# ---------------------------------------------------------------------------
# 8. PermissionChecker denies memory:manage to viewer context
# ---------------------------------------------------------------------------


class TestPermissionCheckerDenies:
    def test_viewer_denied_memory_manage(self, checker: PermissionChecker) -> None:
        ctx = _make_context(user_id="viewer-user", roles={"viewer"})
        decision = checker.check_permission(ctx, "memory.manage")
        assert decision.allowed is False

    def test_member_denied_memory_manage(self, checker: PermissionChecker) -> None:
        ctx = _make_context(user_id="member-user", roles={"member"})
        decision = checker.check_permission(ctx, "memory.manage")
        assert decision.allowed is False

    def test_analyst_denied_memory_manage(self, checker: PermissionChecker) -> None:
        ctx = _make_context(user_id="analyst-user", roles={"analyst"})
        decision = checker.check_permission(ctx, "memory.manage")
        assert decision.allowed is False


# ---------------------------------------------------------------------------
# 9. Full pipeline: create entries -> RBAC check -> tenant filtering
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_end_to_end_pipeline(self, memory: ContinuumMemory, checker: PermissionChecker) -> None:
        """
        Simulate:
        1. Admin user adds memories for two tenants.
        2. Verify admin passes RBAC check for memory.manage.
        3. Verify viewer fails RBAC check for memory.manage.
        4. Verify tenant-scoped retrieval returns correct subset.
        """
        # Step 1: Admin adds entries for two tenants
        admin_ctx = _make_context(user_id="admin-1", roles={"admin"})
        viewer_ctx = _make_context(user_id="viewer-1", roles={"viewer"})

        memory.add(id="pipe-a1", content="alpha insight", importance=0.9, tenant_id="t-alpha")
        memory.add(id="pipe-a2", content="alpha pattern", importance=0.8, tenant_id="t-alpha")
        memory.add(id="pipe-b1", content="beta insight", importance=0.9, tenant_id="t-beta")

        # Step 2: Admin passes RBAC check
        admin_decision = checker.check_permission(admin_ctx, "memory.manage")
        assert admin_decision.allowed is True

        # Step 3: Viewer fails RBAC check
        viewer_decision = checker.check_permission(viewer_ctx, "memory.manage")
        assert viewer_decision.allowed is False

        # Step 4: Tenant-scoped retrieval
        alpha_entries = memory.retrieve(tenant_id="t-alpha", limit=100)
        beta_entries = memory.retrieve(tenant_id="t-beta", limit=100)

        alpha_ids = {e.id for e in alpha_entries}
        beta_ids = {e.id for e in beta_entries}

        assert alpha_ids == {"pipe-a1", "pipe-a2"}
        assert beta_ids == {"pipe-b1"}


# ---------------------------------------------------------------------------
# 10. Backward compatibility: operations without tenant_id still work
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_add_without_tenant_id_works(self, memory: ContinuumMemory) -> None:
        entry = memory.add(id="compat-1", content="no tenant", importance=0.5)
        assert entry is not None
        assert entry.id == "compat-1"
        assert "tenant_id" not in entry.metadata

    def test_get_without_tenant_id_returns_any_entry(self, memory: ContinuumMemory) -> None:
        memory.add(id="compat-2", content="tenanted", tenant_id="tenant-z")
        result = memory.get("compat-2")
        assert result is not None
        assert result.id == "compat-2"

    def test_retrieve_without_tenant_id_returns_all(self, memory: ContinuumMemory) -> None:
        memory.add(id="compat-a", content="alpha", importance=0.7, tenant_id="tenant-a")
        memory.add(id="compat-b", content="beta", importance=0.7, tenant_id="tenant-b")
        memory.add(id="compat-c", content="unscoped", importance=0.7)

        all_entries = memory.retrieve(limit=100)
        ids = {e.id for e in all_entries}
        assert "compat-a" in ids
        assert "compat-b" in ids
        assert "compat-c" in ids
