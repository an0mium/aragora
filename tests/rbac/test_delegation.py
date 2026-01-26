"""
Tests for RBAC Permission Delegation.
"""

import pytest
from datetime import datetime, timedelta, timezone

from aragora.rbac.delegation import (
    DelegationStatus,
    DelegationConstraint,
    PermissionDelegation,
    DelegationManager,
    get_delegation_manager,
    set_delegation_manager,
    delegate_permission,
    check_delegated_permission,
    revoke_delegation,
)
from aragora.rbac.models import ResourceType


class TestPermissionDelegation:
    """Tests for PermissionDelegation dataclass."""

    def test_create_delegation(self):
        """Test creating a delegation."""
        delegation = PermissionDelegation.create(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
            reason="Covering vacation",
        )

        assert delegation.id is not None
        assert delegation.delegator_id == "manager-123"
        assert delegation.delegatee_id == "assistant-456"
        assert delegation.permission_id == "debates:create"
        assert delegation.org_id == "org-789"
        assert delegation.reason == "Covering vacation"
        assert delegation.status == DelegationStatus.ACTIVE

    def test_delegation_with_expiration(self):
        """Test delegation with expiration time."""
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        delegation = PermissionDelegation.create(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            expires_at=expires,
        )

        assert delegation.expires_at == expires
        assert not delegation.is_expired
        assert delegation.is_valid

    def test_expired_delegation(self):
        """Test that expired delegations are invalid."""
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        delegation = PermissionDelegation.create(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            expires_at=expired_time,
        )

        assert delegation.is_expired
        assert not delegation.is_valid

    def test_delegation_matches(self):
        """Test delegation matching logic."""
        delegation = PermissionDelegation.create(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
        )

        # Should match
        assert delegation.matches(
            user_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
        )

        # Wrong user
        assert not delegation.matches(
            user_id="other-user",
            permission_id="debates:create",
            org_id="org-789",
        )

        # Wrong permission
        assert not delegation.matches(
            user_id="assistant-456",
            permission_id="debates:delete",
            org_id="org-789",
        )

    def test_wildcard_delegation(self):
        """Test wildcard permission delegation."""
        delegation = PermissionDelegation.create(
            delegator_id="admin-123",
            delegatee_id="manager-456",
            permission_id="debates.*",
        )

        # Should match specific actions under wildcard
        assert delegation.matches(
            user_id="manager-456",
            permission_id="debates:create",
        )
        assert delegation.matches(
            user_id="manager-456",
            permission_id="debates:read",
        )
        assert delegation.matches(
            user_id="manager-456",
            permission_id="debates:delete",
        )

        # Should not match different resource
        assert not delegation.matches(
            user_id="manager-456",
            permission_id="agents:create",
        )

    def test_resource_scoped_delegation(self):
        """Test delegation with resource scope."""
        delegation = PermissionDelegation.create(
            delegator_id="owner-123",
            delegatee_id="collaborator-456",
            permission_id="debates:update",
            resource_type=ResourceType.DEBATE,
            resource_ids={"debate-1", "debate-2"},
        )

        # Should match allowed resources
        assert delegation.matches(
            user_id="collaborator-456",
            permission_id="debates:update",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        # Should not match other resources
        assert not delegation.matches(
            user_id="collaborator-456",
            permission_id="debates:update",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-3",
        )

    def test_revoke_delegation(self):
        """Test revoking a delegation."""
        delegation = PermissionDelegation.create(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
        )

        assert delegation.is_valid

        delegation.revoke(revoked_by="admin-789")

        assert delegation.status == DelegationStatus.REVOKED
        assert delegation.revoked_by == "admin-789"
        assert delegation.revoked_at is not None
        assert not delegation.is_valid

    def test_delegation_serialization(self):
        """Test delegation to/from dict."""
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        delegation = PermissionDelegation.create(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
            workspace_id="workspace-1",
            expires_at=expires,
            reason="Vacation coverage",
            can_redelegate=True,
        )

        data = delegation.to_dict()
        restored = PermissionDelegation.from_dict(data)

        assert restored.id == delegation.id
        assert restored.delegator_id == delegation.delegator_id
        assert restored.delegatee_id == delegation.delegatee_id
        assert restored.permission_id == delegation.permission_id
        assert restored.org_id == delegation.org_id
        assert restored.workspace_id == delegation.workspace_id
        assert restored.reason == delegation.reason
        assert restored.can_redelegate == delegation.can_redelegate


class TestDelegationManager:
    """Tests for DelegationManager."""

    def setup_method(self):
        """Reset delegation manager before each test."""
        set_delegation_manager(None)

    def test_delegate_permission(self):
        """Test delegating a permission."""
        manager = DelegationManager()

        delegation = manager.delegate(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
            reason="Project collaboration",
        )

        assert delegation.id is not None
        assert delegation.is_valid

        # Check that it can be found
        found = manager.check_delegation(
            user_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
        )
        assert found is not None
        assert found.id == delegation.id

    def test_delegate_with_delegator_permissions_check(self):
        """Test that delegation validates delegator has the permission."""
        manager = DelegationManager()

        # Should succeed when delegator has permission
        delegation = manager.delegate(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            delegator_permissions={"debates:create", "debates:read"},
        )
        assert delegation is not None

        # Should fail when delegator lacks permission
        with pytest.raises(ValueError, match="does not have permission"):
            manager.delegate(
                delegator_id="manager-123",
                delegatee_id="assistant-456",
                permission_id="admin.system_config",
                delegator_permissions={"debates:create", "debates:read"},
            )

    def test_delegation_chain_depth_limit(self):
        """Test that delegation chain depth is enforced."""
        manager = DelegationManager(max_chain_depth=2)

        # First delegation
        d1 = manager.delegate(
            delegator_id="owner-1",
            delegatee_id="manager-2",
            permission_id="debates:create",
            can_redelegate=True,
        )

        # Second delegation (chain depth 1)
        d2 = manager.delegate(
            delegator_id="manager-2",
            delegatee_id="assistant-3",
            permission_id="debates:create",
            can_redelegate=True,
            parent_delegation=d1,
        )

        # Third delegation should fail (chain depth 2 = max)
        with pytest.raises(ValueError, match="Maximum delegation chain depth"):
            manager.delegate(
                delegator_id="assistant-3",
                delegatee_id="intern-4",
                permission_id="debates:create",
                parent_delegation=d2,
            )

    def test_no_redelegate_flag(self):
        """Test that can_redelegate=False prevents redelegation."""
        manager = DelegationManager()

        d1 = manager.delegate(
            delegator_id="manager-1",
            delegatee_id="assistant-2",
            permission_id="debates:create",
            can_redelegate=False,
        )

        with pytest.raises(ValueError, match="does not allow redelegation"):
            manager.delegate(
                delegator_id="assistant-2",
                delegatee_id="intern-3",
                permission_id="debates:create",
                parent_delegation=d1,
            )

    def test_revoke_delegation(self):
        """Test revoking a delegation."""
        manager = DelegationManager()

        delegation = manager.delegate(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
        )

        # Should have access before revocation
        assert manager.has_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:create",
        )

        # Revoke
        result = manager.revoke(delegation.id, revoked_by="manager-123")
        assert result is True

        # Should not have access after revocation
        assert not manager.has_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:create",
        )

    def test_revoke_cascades_to_children(self):
        """Test that revoking a delegation also revokes child delegations."""
        manager = DelegationManager(max_chain_depth=3)

        # Create chain: owner -> manager -> assistant
        d1 = manager.delegate(
            delegator_id="owner-1",
            delegatee_id="manager-2",
            permission_id="debates:create",
            can_redelegate=True,
        )

        d2 = manager.delegate(
            delegator_id="manager-2",
            delegatee_id="assistant-3",
            permission_id="debates:create",
            parent_delegation=d1,
        )

        # Both should have access
        assert manager.has_delegated_permission("manager-2", "debates:create")
        assert manager.has_delegated_permission("assistant-3", "debates:create")

        # Revoke the parent delegation
        manager.revoke(d1.id, revoked_by="owner-1")

        # Both should lose access
        assert not manager.has_delegated_permission("manager-2", "debates:create")
        assert not manager.has_delegated_permission("assistant-3", "debates:create")

    def test_list_delegations_by_delegator(self):
        """Test listing delegations made by a user."""
        manager = DelegationManager()

        manager.delegate("manager-1", "user-a", "debates:create")
        manager.delegate("manager-1", "user-b", "debates:read")
        manager.delegate("manager-2", "user-c", "debates:create")

        delegations = manager.list_delegations_by_delegator("manager-1")
        assert len(delegations) == 2
        assert all(d.delegator_id == "manager-1" for d in delegations)

    def test_list_delegations_for_delegatee(self):
        """Test listing delegations granted to a user."""
        manager = DelegationManager()

        manager.delegate("manager-1", "assistant-1", "debates:create")
        manager.delegate("manager-2", "assistant-1", "debates:read")
        manager.delegate("manager-1", "assistant-2", "debates:create")

        delegations = manager.list_delegations_for_delegatee("assistant-1")
        assert len(delegations) == 2
        assert all(d.delegatee_id == "assistant-1" for d in delegations)

    def test_cleanup_expired(self):
        """Test cleaning up expired delegations."""
        manager = DelegationManager()

        # Create an expired delegation
        expired = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.delegate(
            delegator_id="manager-1",
            delegatee_id="assistant-1",
            permission_id="debates:create",
            expires_at=expired,
        )

        # Create a valid delegation
        manager.delegate(
            delegator_id="manager-2",
            delegatee_id="assistant-2",
            permission_id="debates:create",
        )

        count = manager.cleanup_expired()
        assert count == 1

        stats = manager.get_stats()
        assert stats["total_delegations"] == 1

    def test_stats(self):
        """Test delegation statistics."""
        manager = DelegationManager(max_chain_depth=3)

        manager.delegate("manager-1", "user-a", "debates:create")
        manager.delegate("manager-1", "user-b", "debates:read")
        d3 = manager.delegate("manager-2", "user-c", "debates:create")
        manager.revoke(d3.id, "manager-2")

        stats = manager.get_stats()
        assert stats["total_delegations"] == 3
        assert stats["active_delegations"] == 2
        assert stats["revoked_delegations"] == 1
        assert stats["max_chain_depth"] == 3


class TestGlobalDelegationFunctions:
    """Tests for global delegation convenience functions."""

    def setup_method(self):
        """Reset delegation manager before each test."""
        set_delegation_manager(None)

    def test_delegate_permission_function(self):
        """Test the delegate_permission convenience function."""
        delegation = delegate_permission(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
        )

        assert delegation is not None
        assert delegation.delegator_id == "manager-123"

    def test_check_delegated_permission_function(self):
        """Test the check_delegated_permission convenience function."""
        delegate_permission(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
        )

        assert check_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
        )

        assert not check_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:delete",
            org_id="org-789",
        )

    def test_revoke_delegation_function(self):
        """Test the revoke_delegation convenience function."""
        delegation = delegate_permission(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
        )

        result = revoke_delegation(delegation.id, revoked_by="manager-123")
        assert result is True

        assert not check_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:create",
        )


class TestWorkspaceScopedDelegations:
    """Tests for workspace-scoped delegations."""

    def setup_method(self):
        """Reset delegation manager before each test."""
        set_delegation_manager(None)

    def test_workspace_scoped_delegation(self):
        """Test delegation scoped to a workspace."""
        manager = DelegationManager()

        delegation = manager.delegate(
            delegator_id="manager-123",
            delegatee_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
            workspace_id="workspace-1",
        )

        # Should match with correct workspace
        assert manager.has_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
            workspace_id="workspace-1",
        )

        # Should not match with different workspace
        assert not manager.has_delegated_permission(
            user_id="assistant-456",
            permission_id="debates:create",
            org_id="org-789",
            workspace_id="workspace-2",
        )
