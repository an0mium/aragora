"""
Tests for memory:manage RBAC permission and tenant_id isolation (Phase T).

Covers:
- Permission definition existence and correctness
- Role assignment (admin has memory:manage, non-admin does not)
- Tenant isolation in ContinuumMemory add/get/retrieve
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Permission definition tests
# ---------------------------------------------------------------------------


class TestMemoryManagePermission:
    """Tests that the memory:manage permission is properly defined."""

    def test_perm_memory_manage_exists(self):
        """PERM_MEMORY_MANAGE should be importable from the permissions package."""
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE is not None

    def test_perm_memory_manage_key(self):
        """PERM_MEMORY_MANAGE key should be 'memory.manage'."""
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE.key == "memory.manage"

    def test_perm_memory_manage_in_system_permissions(self):
        """memory:manage should appear in SYSTEM_PERMISSIONS registry."""
        from aragora.rbac.defaults.registry import SYSTEM_PERMISSIONS

        # Check both dot and colon formats (registry adds colon aliases)
        assert "memory.manage" in SYSTEM_PERMISSIONS or "memory:manage" in SYSTEM_PERMISSIONS

    def test_perm_memory_manage_resource_type(self):
        """PERM_MEMORY_MANAGE should be on the MEMORY resource."""
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE
        from aragora.rbac.models import ResourceType

        assert PERM_MEMORY_MANAGE.resource == ResourceType.MEMORY

    def test_perm_memory_manage_action(self):
        """PERM_MEMORY_MANAGE should use the MANAGE action."""
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE
        from aragora.rbac.models import Action

        assert PERM_MEMORY_MANAGE.action == Action.MANAGE


# ---------------------------------------------------------------------------
# Role assignment tests
# ---------------------------------------------------------------------------


class TestMemoryManageRoles:
    """Tests that memory:manage is assigned to the correct roles."""

    def test_admin_role_has_memory_manage(self):
        """The admin role should include the memory:manage permission."""
        from aragora.rbac.defaults.roles import ROLE_ADMIN
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE.key in ROLE_ADMIN.permissions

    def test_owner_role_has_memory_manage(self):
        """The owner role (all permissions) should include memory:manage."""
        from aragora.rbac.defaults.roles import ROLE_OWNER
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE.key in ROLE_OWNER.permissions

    def test_viewer_role_lacks_memory_manage(self):
        """The viewer role should NOT have memory:manage."""
        from aragora.rbac.defaults.roles import ROLE_VIEWER
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE.key not in ROLE_VIEWER.permissions

    def test_member_role_lacks_memory_manage(self):
        """The member role should NOT have memory:manage."""
        from aragora.rbac.defaults.roles import ROLE_MEMBER
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE.key not in ROLE_MEMBER.permissions

    def test_analyst_role_lacks_memory_manage(self):
        """The analyst role should NOT have memory:manage."""
        from aragora.rbac.defaults.roles import ROLE_ANALYST
        from aragora.rbac.defaults.permissions import PERM_MEMORY_MANAGE

        assert PERM_MEMORY_MANAGE.key not in ROLE_ANALYST.permissions


# ---------------------------------------------------------------------------
# Tenant isolation tests
# ---------------------------------------------------------------------------


class TestTenantIsolation:
    """Tests for tenant_id filtering in ContinuumMemory."""

    @pytest.fixture()
    def memory(self, tmp_path: Path):
        """Create a ContinuumMemory instance with a temporary database."""
        from aragora.memory.continuum.core import ContinuumMemory

        db_path = tmp_path / "test_continuum.db"
        return ContinuumMemory(db_path=db_path)

    def test_add_with_tenant_id_stores_in_metadata(self, memory):
        """Adding a memory with tenant_id should store it in metadata."""
        entry = memory.add(
            id="t1_mem1",
            content="Tenant 1 secret",
            tenant_id="tenant_1",
        )
        assert entry.metadata.get("tenant_id") == "tenant_1"

    def test_get_with_matching_tenant_returns_entry(self, memory):
        """get() with matching tenant_id should return the entry."""
        memory.add(id="t1_mem2", content="visible", tenant_id="tenant_1")
        result = memory.get("t1_mem2", tenant_id="tenant_1")
        assert result is not None
        assert result.id == "t1_mem2"

    def test_get_with_wrong_tenant_returns_none(self, memory):
        """get() with a different tenant_id should return None."""
        memory.add(id="t1_mem3", content="hidden", tenant_id="tenant_1")
        result = memory.get("t1_mem3", tenant_id="tenant_2")
        assert result is None

    def test_get_without_tenant_returns_any(self, memory):
        """get() without tenant_id returns the entry regardless (backward compat)."""
        memory.add(id="t1_mem4", content="anything", tenant_id="tenant_1")
        result = memory.get("t1_mem4")
        assert result is not None

    def test_retrieve_filters_by_tenant(self, memory):
        """retrieve() with tenant_id should only return matching entries."""
        memory.add(id="iso_a1", content="alpha data", importance=0.9, tenant_id="alpha")
        memory.add(id="iso_b1", content="beta data", importance=0.9, tenant_id="beta")
        memory.add(id="iso_a2", content="more alpha data", importance=0.9, tenant_id="alpha")

        results = memory.retrieve(limit=100, tenant_id="alpha")
        ids = {e.id for e in results}
        assert "iso_a1" in ids
        assert "iso_a2" in ids
        assert "iso_b1" not in ids

    def test_retrieve_without_tenant_returns_all(self, memory):
        """retrieve() without tenant_id returns all entries (backward compat)."""
        memory.add(id="all_a", content="a data", importance=0.9, tenant_id="alpha")
        memory.add(id="all_b", content="b data", importance=0.9, tenant_id="beta")

        results = memory.retrieve(limit=100)
        ids = {e.id for e in results}
        assert "all_a" in ids
        assert "all_b" in ids

    def test_tenant_a_cannot_see_tenant_b_data(self, memory):
        """Full isolation: tenant A's retrieve must never include tenant B entries."""
        for i in range(5):
            memory.add(id=f"secA_{i}", content=f"secret A {i}", importance=0.8, tenant_id="A")
            memory.add(id=f"secB_{i}", content=f"secret B {i}", importance=0.8, tenant_id="B")

        results_a = memory.retrieve(limit=100, tenant_id="A")
        results_b = memory.retrieve(limit=100, tenant_id="B")

        a_ids = {e.id for e in results_a}
        b_ids = {e.id for e in results_b}

        # No overlap
        assert a_ids.isdisjoint(b_ids)
        # Correct counts
        assert len(a_ids) == 5
        assert len(b_ids) == 5
        # All A entries belong to tenant A
        for entry in results_a:
            assert entry.metadata.get("tenant_id") == "A"
        for entry in results_b:
            assert entry.metadata.get("tenant_id") == "B"
