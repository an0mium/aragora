"""
Tests for multi-organization membership management.

Tests cover:
- Enum values (MembershipRole, MembershipStatus)
- Role permissions mapping
- OrganizationMembership dataclass methods
- MultiOrgManager CRUD operations
- Owner protection (can't remove/demote last owner)
- Context switching
- Permission checks
- Cross-org statistics
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from aragora.billing.multi_org import (
    MembershipRole,
    MembershipStatus,
    OrganizationMembership,
    OrgContext,
    MultiOrgManager,
    ROLE_PERMISSIONS,
    get_multi_org_manager,
)
from aragora.billing.models import Organization


# =============================================================================
# Enum Tests
# =============================================================================


class TestMembershipRole:
    """Tests for MembershipRole enum."""

    def test_owner_value(self):
        """Owner role has correct value."""
        assert MembershipRole.OWNER.value == "owner"

    def test_admin_value(self):
        """Admin role has correct value."""
        assert MembershipRole.ADMIN.value == "admin"

    def test_member_value(self):
        """Member role has correct value."""
        assert MembershipRole.MEMBER.value == "member"

    def test_viewer_value(self):
        """Viewer role has correct value."""
        assert MembershipRole.VIEWER.value == "viewer"

    def test_billing_value(self):
        """Billing role has correct value."""
        assert MembershipRole.BILLING.value == "billing"

    def test_all_roles_defined(self):
        """All expected roles are defined."""
        roles = [r.value for r in MembershipRole]
        assert set(roles) == {"owner", "admin", "member", "viewer", "billing"}


class TestMembershipStatus:
    """Tests for MembershipStatus enum."""

    def test_active_value(self):
        """Active status has correct value."""
        assert MembershipStatus.ACTIVE.value == "active"

    def test_pending_value(self):
        """Pending status has correct value."""
        assert MembershipStatus.PENDING.value == "pending"

    def test_suspended_value(self):
        """Suspended status has correct value."""
        assert MembershipStatus.SUSPENDED.value == "suspended"

    def test_expired_value(self):
        """Expired status has correct value."""
        assert MembershipStatus.EXPIRED.value == "expired"


# =============================================================================
# Role Permissions Tests
# =============================================================================


class TestRolePermissions:
    """Tests for ROLE_PERMISSIONS mapping."""

    def test_owner_has_all_permissions(self):
        """Owner has all permissions."""
        owner_perms = ROLE_PERMISSIONS[MembershipRole.OWNER]
        assert "org.delete" in owner_perms
        assert "org.update" in owner_perms
        assert "org.settings" in owner_perms
        assert "org.billing" in owner_perms
        assert "members.invite" in owner_perms
        assert "members.remove" in owner_perms
        assert "members.promote" in owner_perms
        assert "members.demote" in owner_perms
        assert "debates.create" in owner_perms
        assert "debates.view" in owner_perms
        assert "debates.delete" in owner_perms
        assert "api.access" in owner_perms
        assert "audit.view" in owner_perms

    def test_admin_permissions_subset_of_owner(self):
        """Admin permissions are a subset of owner (except owner-only)."""
        admin_perms = ROLE_PERMISSIONS[MembershipRole.ADMIN]
        owner_perms = ROLE_PERMISSIONS[MembershipRole.OWNER]
        # Admin should not have org.delete or promote/demote
        assert "org.delete" not in admin_perms
        assert "members.promote" not in admin_perms
        assert "members.demote" not in admin_perms
        # But should have these
        assert "org.update" in admin_perms
        assert "members.invite" in admin_perms

    def test_member_has_limited_permissions(self):
        """Member has limited permissions."""
        member_perms = ROLE_PERMISSIONS[MembershipRole.MEMBER]
        assert "debates.create" in member_perms
        assert "debates.view" in member_perms
        assert "api.access" in member_perms
        # Should not have admin permissions
        assert "members.invite" not in member_perms
        assert "org.settings" not in member_perms

    def test_viewer_has_minimal_permissions(self):
        """Viewer has read-only permissions."""
        viewer_perms = ROLE_PERMISSIONS[MembershipRole.VIEWER]
        assert viewer_perms == {"debates.view"}

    def test_billing_has_specific_permissions(self):
        """Billing role has billing-specific permissions."""
        billing_perms = ROLE_PERMISSIONS[MembershipRole.BILLING]
        assert "org.billing" in billing_perms
        assert "debates.view" in billing_perms
        # Should not have other admin permissions
        assert "members.invite" not in billing_perms


# =============================================================================
# OrganizationMembership Tests
# =============================================================================


class TestOrganizationMembership:
    """Tests for OrganizationMembership dataclass."""

    def test_default_values(self):
        """Membership has correct default values."""
        membership = OrganizationMembership()
        assert membership.user_id == ""
        assert membership.org_id == ""
        assert membership.role == MembershipRole.MEMBER
        assert membership.status == MembershipStatus.ACTIVE
        assert membership.is_primary is False
        assert membership.invited_by is None
        assert membership.invitation_token is None
        assert membership.custom_permissions == []
        assert membership.id  # Should have generated ID

    def test_permissions_property_returns_role_permissions(self):
        """Permissions property returns role-based permissions."""
        membership = OrganizationMembership(role=MembershipRole.OWNER)
        assert membership.permissions == ROLE_PERMISSIONS[MembershipRole.OWNER]

    def test_permissions_property_includes_custom(self):
        """Permissions property includes custom permissions."""
        membership = OrganizationMembership(
            role=MembershipRole.MEMBER,
            custom_permissions=["custom.permission", "another.permission"],
        )
        perms = membership.permissions
        # Should have base member permissions plus custom
        assert "debates.view" in perms
        assert "custom.permission" in perms
        assert "another.permission" in perms

    def test_has_permission_returns_true_for_role_permission(self):
        """has_permission returns True for role permissions."""
        membership = OrganizationMembership(role=MembershipRole.ADMIN)
        assert membership.has_permission("members.invite") is True

    def test_has_permission_returns_false_when_inactive(self):
        """has_permission returns False when membership is not active."""
        membership = OrganizationMembership(
            role=MembershipRole.OWNER,
            status=MembershipStatus.SUSPENDED,
        )
        assert membership.has_permission("org.delete") is False

    def test_has_permission_returns_false_for_missing_permission(self):
        """has_permission returns False for permissions not in role."""
        membership = OrganizationMembership(role=MembershipRole.VIEWER)
        assert membership.has_permission("debates.create") is False

    def test_can_manage_members_owner(self):
        """Owner can manage members."""
        membership = OrganizationMembership(role=MembershipRole.OWNER)
        assert membership.can_manage_members() is True

    def test_can_manage_members_admin(self):
        """Admin can manage members."""
        membership = OrganizationMembership(role=MembershipRole.ADMIN)
        assert membership.can_manage_members() is True

    def test_can_manage_members_member(self):
        """Regular member cannot manage members."""
        membership = OrganizationMembership(role=MembershipRole.MEMBER)
        assert membership.can_manage_members() is False

    def test_can_manage_billing_owner(self):
        """Owner can manage billing."""
        membership = OrganizationMembership(role=MembershipRole.OWNER)
        assert membership.can_manage_billing() is True

    def test_can_manage_billing_billing_role(self):
        """Billing role can manage billing."""
        membership = OrganizationMembership(role=MembershipRole.BILLING)
        assert membership.can_manage_billing() is True

    def test_can_manage_billing_member(self):
        """Regular member cannot manage billing."""
        membership = OrganizationMembership(role=MembershipRole.MEMBER)
        assert membership.can_manage_billing() is False

    def test_is_admin_or_owner_owner(self):
        """Owner is admin or owner."""
        membership = OrganizationMembership(role=MembershipRole.OWNER)
        assert membership.is_admin_or_owner() is True

    def test_is_admin_or_owner_admin(self):
        """Admin is admin or owner."""
        membership = OrganizationMembership(role=MembershipRole.ADMIN)
        assert membership.is_admin_or_owner() is True

    def test_is_admin_or_owner_member(self):
        """Member is not admin or owner."""
        membership = OrganizationMembership(role=MembershipRole.MEMBER)
        assert membership.is_admin_or_owner() is False

    def test_to_dict_serialization(self):
        """to_dict serializes membership correctly."""
        now = datetime.now(timezone.utc)
        membership = OrganizationMembership(
            id="mem123",
            user_id="user456",
            org_id="org789",
            role=MembershipRole.ADMIN,
            status=MembershipStatus.ACTIVE,
            is_primary=True,
            joined_at=now,
            updated_at=now,
            invited_by="inviter123",
            display_name="Test User",
            department="Engineering",
            title="Developer",
        )
        data = membership.to_dict()
        assert data["id"] == "mem123"
        assert data["user_id"] == "user456"
        assert data["org_id"] == "org789"
        assert data["role"] == "admin"
        assert data["status"] == "active"
        assert data["is_primary"] is True
        assert data["invited_by"] == "inviter123"
        assert "members.invite" in data["permissions"]
        assert data["display_name"] == "Test User"
        assert data["department"] == "Engineering"
        assert data["title"] == "Developer"

    def test_from_dict_deserialization(self):
        """from_dict creates membership from dictionary."""
        data = {
            "id": "mem123",
            "user_id": "user456",
            "org_id": "org789",
            "role": "admin",
            "status": "active",
            "is_primary": True,
            "joined_at": "2024-01-15T10:00:00+00:00",
            "updated_at": "2024-01-15T12:00:00+00:00",
            "invited_by": "inviter123",
            "custom_permissions": ["custom.perm"],
            "display_name": "Test User",
            "department": "Engineering",
            "title": "Developer",
        }
        membership = OrganizationMembership.from_dict(data)
        assert membership.id == "mem123"
        assert membership.user_id == "user456"
        assert membership.org_id == "org789"
        assert membership.role == MembershipRole.ADMIN
        assert membership.status == MembershipStatus.ACTIVE
        assert membership.is_primary is True
        assert membership.invited_by == "inviter123"
        assert "custom.perm" in membership.custom_permissions
        assert membership.display_name == "Test User"

    def test_from_dict_with_defaults(self):
        """from_dict uses defaults for missing fields."""
        membership = OrganizationMembership.from_dict({})
        assert membership.role == MembershipRole.MEMBER
        assert membership.status == MembershipStatus.ACTIVE
        assert membership.is_primary is False


# =============================================================================
# OrgContext Tests
# =============================================================================


class TestOrgContext:
    """Tests for OrgContext dataclass."""

    def test_context_construction(self):
        """OrgContext can be constructed with required fields."""
        membership = OrganizationMembership(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.ADMIN,
        )
        org = Organization(id="org456", name="Test Org")
        context = OrgContext(
            user_id="user123",
            org_id="org456",
            membership=membership,
            organization=org,
        )
        assert context.user_id == "user123"
        assert context.org_id == "org456"
        assert context.membership == membership
        assert context.organization == org
        assert context.switch_count == 0

    def test_context_with_switch_count(self):
        """OrgContext tracks switch count."""
        membership = OrganizationMembership()
        org = Organization(id="org1", name="Org 1")
        context = OrgContext(
            user_id="user1",
            org_id="org1",
            membership=membership,
            organization=org,
            switch_count=5,
        )
        assert context.switch_count == 5


# =============================================================================
# MultiOrgManager Tests
# =============================================================================


@pytest.fixture
def manager() -> MultiOrgManager:
    """Create a fresh in-memory manager for each test."""
    return MultiOrgManager(db_path=None)  # Uses :memory:


@pytest.fixture
def file_manager(tmp_path: Path) -> MultiOrgManager:
    """Create a file-based manager for persistence tests."""
    db_path = tmp_path / "multi_org.db"
    return MultiOrgManager(db_path=str(db_path))


class TestMultiOrgManagerInit:
    """Tests for MultiOrgManager initialization."""

    def test_init_in_memory(self):
        """Manager initializes with in-memory database."""
        manager = MultiOrgManager(db_path=None)
        assert manager.db_path == ":memory:"
        assert manager._persistent_conn is not None

    def test_init_with_file_path(self, tmp_path: Path):
        """Manager initializes with file-based database."""
        db_path = tmp_path / "test.db"
        manager = MultiOrgManager(db_path=str(db_path))
        assert manager.db_path == str(db_path)
        assert manager._persistent_conn is None
        assert db_path.exists()

    def test_init_max_orgs_per_user(self):
        """Manager respects max_orgs_per_user setting."""
        manager = MultiOrgManager(max_orgs_per_user=5)
        assert manager.max_orgs_per_user == 5


class TestMultiOrgManagerAddMember:
    """Tests for add_member method."""

    @pytest.mark.asyncio
    async def test_add_member_success(self, manager: MultiOrgManager):
        """Successfully add a member to an organization."""
        membership = await manager.add_member(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.ADMIN,
            invited_by="owner789",
        )
        assert membership.user_id == "user123"
        assert membership.org_id == "org456"
        assert membership.role == MembershipRole.ADMIN
        assert membership.invited_by == "owner789"
        assert membership.status == MembershipStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_add_member_first_becomes_primary(self, manager: MultiOrgManager):
        """First organization for user becomes primary."""
        membership = await manager.add_member(
            user_id="user123",
            org_id="org456",
        )
        assert membership.is_primary is True

    @pytest.mark.asyncio
    async def test_add_member_second_not_primary_by_default(self, manager: MultiOrgManager):
        """Second organization is not primary by default."""
        await manager.add_member(user_id="user123", org_id="org1")
        membership2 = await manager.add_member(user_id="user123", org_id="org2")
        assert membership2.is_primary is False

    @pytest.mark.asyncio
    async def test_add_member_explicit_primary_unsets_previous(self, manager: MultiOrgManager):
        """Setting explicit primary unsets previous primary."""
        m1 = await manager.add_member(user_id="user123", org_id="org1")
        assert m1.is_primary is True

        m2 = await manager.add_member(user_id="user123", org_id="org2", is_primary=True)
        assert m2.is_primary is True

        # Verify first is no longer primary
        m1_updated = await manager.get_membership("user123", "org1")
        assert m1_updated.is_primary is False

    @pytest.mark.asyncio
    async def test_add_member_duplicate_raises_error(self, manager: MultiOrgManager):
        """Adding duplicate membership raises ValueError."""
        await manager.add_member(user_id="user123", org_id="org456")
        with pytest.raises(ValueError, match="already member"):
            await manager.add_member(user_id="user123", org_id="org456")

    @pytest.mark.asyncio
    async def test_add_member_exceeds_limit_raises_error(self, manager: MultiOrgManager):
        """Exceeding org limit raises ValueError."""
        manager.max_orgs_per_user = 2
        await manager.add_member(user_id="user123", org_id="org1")
        await manager.add_member(user_id="user123", org_id="org2")
        with pytest.raises(ValueError, match="already in 2 organizations"):
            await manager.add_member(user_id="user123", org_id="org3")

    @pytest.mark.asyncio
    async def test_add_member_with_pending_status(self, manager: MultiOrgManager):
        """Add member with pending invitation status."""
        membership = await manager.add_member(
            user_id="user123",
            org_id="org456",
            status=MembershipStatus.PENDING,
            invitation_token="token123",
        )
        assert membership.status == MembershipStatus.PENDING
        assert membership.invitation_token == "token123"

    @pytest.mark.asyncio
    async def test_add_member_with_metadata(self, manager: MultiOrgManager):
        """Add member with display name, department, title."""
        membership = await manager.add_member(
            user_id="user123",
            org_id="org456",
            display_name="John Doe",
            department="Engineering",
            title="Senior Developer",
        )
        assert membership.display_name == "John Doe"
        assert membership.department == "Engineering"
        assert membership.title == "Senior Developer"


class TestMultiOrgManagerRemoveMember:
    """Tests for remove_member method."""

    @pytest.mark.asyncio
    async def test_remove_member_success(self, manager: MultiOrgManager):
        """Successfully remove a member."""
        await manager.add_member(user_id="user123", org_id="org456")
        result = await manager.remove_member(user_id="user123", org_id="org456")
        assert result is True

        # Verify membership no longer exists
        membership = await manager.get_membership("user123", "org456")
        assert membership is None

    @pytest.mark.asyncio
    async def test_remove_member_not_found(self, manager: MultiOrgManager):
        """Removing non-existent member returns False."""
        result = await manager.remove_member(user_id="user123", org_id="org456")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_last_owner_raises_error(self, manager: MultiOrgManager):
        """Cannot remove the last owner of an organization."""
        await manager.add_member(
            user_id="owner123",
            org_id="org456",
            role=MembershipRole.OWNER,
        )
        with pytest.raises(ValueError, match="Cannot remove the last owner"):
            await manager.remove_member(user_id="owner123", org_id="org456")

    @pytest.mark.asyncio
    async def test_remove_owner_with_other_owners_allowed(self, manager: MultiOrgManager):
        """Can remove owner if there are other owners."""
        await manager.add_member(
            user_id="owner1",
            org_id="org456",
            role=MembershipRole.OWNER,
        )
        await manager.add_member(
            user_id="owner2",
            org_id="org456",
            role=MembershipRole.OWNER,
        )
        result = await manager.remove_member(user_id="owner1", org_id="org456")
        assert result is True

    @pytest.mark.asyncio
    async def test_remove_member_clears_context(self, manager: MultiOrgManager):
        """Removing member from current org clears context."""
        await manager.add_member(user_id="user123", org_id="org456")
        await manager.switch_context(user_id="user123", org_id="org456")

        # Verify context exists
        context = await manager.get_current_context("user123")
        assert context is not None

        # Remove member
        await manager.remove_member(user_id="user123", org_id="org456")

        # Context should be cleared
        context = await manager.get_current_context("user123")
        assert context is None


class TestMultiOrgManagerUpdateRole:
    """Tests for update_role method."""

    @pytest.mark.asyncio
    async def test_update_role_success(self, manager: MultiOrgManager):
        """Successfully update a member's role."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.MEMBER,
        )
        membership = await manager.update_role(
            user_id="user123",
            org_id="org456",
            new_role=MembershipRole.ADMIN,
        )
        assert membership.role == MembershipRole.ADMIN

    @pytest.mark.asyncio
    async def test_update_role_not_found_raises_error(self, manager: MultiOrgManager):
        """Updating role for non-member raises ValueError."""
        with pytest.raises(ValueError, match="No membership found"):
            await manager.update_role(
                user_id="user123",
                org_id="org456",
                new_role=MembershipRole.ADMIN,
            )

    @pytest.mark.asyncio
    async def test_demote_last_owner_raises_error(self, manager: MultiOrgManager):
        """Cannot demote the last owner."""
        await manager.add_member(
            user_id="owner123",
            org_id="org456",
            role=MembershipRole.OWNER,
        )
        with pytest.raises(ValueError, match="Cannot demote the last owner"):
            await manager.update_role(
                user_id="owner123",
                org_id="org456",
                new_role=MembershipRole.ADMIN,
            )

    @pytest.mark.asyncio
    async def test_demote_owner_with_other_owners_allowed(self, manager: MultiOrgManager):
        """Can demote owner if there are other owners."""
        await manager.add_member(
            user_id="owner1",
            org_id="org456",
            role=MembershipRole.OWNER,
        )
        await manager.add_member(
            user_id="owner2",
            org_id="org456",
            role=MembershipRole.OWNER,
        )
        membership = await manager.update_role(
            user_id="owner1",
            org_id="org456",
            new_role=MembershipRole.ADMIN,
        )
        assert membership.role == MembershipRole.ADMIN

    @pytest.mark.asyncio
    async def test_update_role_updates_timestamp(self, manager: MultiOrgManager):
        """Updating role updates the updated_at timestamp."""
        membership = await manager.add_member(
            user_id="user123",
            org_id="org456",
        )
        original_updated_at = membership.updated_at

        updated = await manager.update_role(
            user_id="user123",
            org_id="org456",
            new_role=MembershipRole.ADMIN,
        )
        assert updated.updated_at > original_updated_at


class TestMultiOrgManagerGetMembership:
    """Tests for get_membership method."""

    @pytest.mark.asyncio
    async def test_get_membership_found(self, manager: MultiOrgManager):
        """Get existing membership."""
        await manager.add_member(user_id="user123", org_id="org456")
        membership = await manager.get_membership("user123", "org456")
        assert membership is not None
        assert membership.user_id == "user123"
        assert membership.org_id == "org456"

    @pytest.mark.asyncio
    async def test_get_membership_not_found(self, manager: MultiOrgManager):
        """Get non-existent membership returns None."""
        membership = await manager.get_membership("user123", "org456")
        assert membership is None


class TestMultiOrgManagerGetUserMemberships:
    """Tests for get_user_memberships method."""

    @pytest.mark.asyncio
    async def test_get_user_memberships(self, manager: MultiOrgManager):
        """Get all memberships for a user."""
        await manager.add_member(user_id="user123", org_id="org1")
        await manager.add_member(user_id="user123", org_id="org2")
        await manager.add_member(user_id="user123", org_id="org3")

        memberships = await manager.get_user_memberships("user123")
        assert len(memberships) == 3

    @pytest.mark.asyncio
    async def test_get_user_memberships_excludes_inactive(self, manager: MultiOrgManager):
        """Default query excludes inactive memberships."""
        await manager.add_member(user_id="user123", org_id="org1")
        await manager.add_member(
            user_id="user123",
            org_id="org2",
            status=MembershipStatus.SUSPENDED,
        )

        memberships = await manager.get_user_memberships("user123")
        assert len(memberships) == 1
        assert memberships[0].org_id == "org1"

    @pytest.mark.asyncio
    async def test_get_user_memberships_include_pending(self, manager: MultiOrgManager):
        """Can include pending memberships."""
        await manager.add_member(user_id="user123", org_id="org1")
        await manager.add_member(
            user_id="user123",
            org_id="org2",
            status=MembershipStatus.PENDING,
        )

        memberships = await manager.get_user_memberships("user123", include_pending=True)
        assert len(memberships) == 2

    @pytest.mark.asyncio
    async def test_get_user_memberships_ordered_by_primary(self, manager: MultiOrgManager):
        """Memberships are ordered with primary first."""
        await manager.add_member(user_id="user123", org_id="org1")  # becomes primary
        await manager.add_member(user_id="user123", org_id="org2")
        await manager.add_member(user_id="user123", org_id="org3", is_primary=True)

        memberships = await manager.get_user_memberships("user123")
        assert memberships[0].org_id == "org3"  # New primary first


class TestMultiOrgManagerGetOrgMembers:
    """Tests for get_org_members method."""

    @pytest.mark.asyncio
    async def test_get_org_members_all(self, manager: MultiOrgManager):
        """Get all members of an organization."""
        await manager.add_member(user_id="user1", org_id="org456", role=MembershipRole.OWNER)
        await manager.add_member(user_id="user2", org_id="org456", role=MembershipRole.ADMIN)
        await manager.add_member(user_id="user3", org_id="org456", role=MembershipRole.MEMBER)

        members = await manager.get_org_members("org456")
        assert len(members) == 3

    @pytest.mark.asyncio
    async def test_get_org_members_filtered_by_role(self, manager: MultiOrgManager):
        """Get organization members filtered by role."""
        await manager.add_member(user_id="user1", org_id="org456", role=MembershipRole.ADMIN)
        await manager.add_member(user_id="user2", org_id="org456", role=MembershipRole.ADMIN)
        await manager.add_member(user_id="user3", org_id="org456", role=MembershipRole.MEMBER)

        admins = await manager.get_org_members("org456", role_filter=MembershipRole.ADMIN)
        assert len(admins) == 2
        assert all(m.role == MembershipRole.ADMIN for m in admins)

    @pytest.mark.asyncio
    async def test_get_org_members_excludes_inactive(self, manager: MultiOrgManager):
        """Get org members excludes inactive."""
        await manager.add_member(user_id="user1", org_id="org456")
        await manager.add_member(
            user_id="user2",
            org_id="org456",
            status=MembershipStatus.SUSPENDED,
        )

        members = await manager.get_org_members("org456")
        assert len(members) == 1


class TestMultiOrgManagerPrimaryOrg:
    """Tests for set_primary_org and get_primary_org methods."""

    @pytest.mark.asyncio
    async def test_set_primary_org(self, manager: MultiOrgManager):
        """Set a user's primary organization."""
        await manager.add_member(user_id="user123", org_id="org1")
        await manager.add_member(user_id="user123", org_id="org2")

        membership = await manager.set_primary_org("user123", "org2")
        assert membership.is_primary is True

        # Verify org1 is no longer primary
        org1 = await manager.get_membership("user123", "org1")
        assert org1.is_primary is False

    @pytest.mark.asyncio
    async def test_set_primary_org_not_member_raises_error(self, manager: MultiOrgManager):
        """Setting primary for non-member raises ValueError."""
        with pytest.raises(ValueError, match="No membership found"):
            await manager.set_primary_org("user123", "org456")

    @pytest.mark.asyncio
    async def test_set_primary_org_inactive_raises_error(self, manager: MultiOrgManager):
        """Cannot set inactive membership as primary."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            status=MembershipStatus.SUSPENDED,
        )
        with pytest.raises(ValueError, match="Cannot set inactive membership as primary"):
            await manager.set_primary_org("user123", "org456")

    @pytest.mark.asyncio
    async def test_get_primary_org(self, manager: MultiOrgManager):
        """Get user's primary organization."""
        await manager.add_member(user_id="user123", org_id="org456")
        primary = await manager.get_primary_org("user123")
        assert primary is not None
        assert primary.org_id == "org456"
        assert primary.is_primary is True

    @pytest.mark.asyncio
    async def test_get_primary_org_none(self, manager: MultiOrgManager):
        """Get primary org returns None if no memberships."""
        primary = await manager.get_primary_org("user123")
        assert primary is None


class TestMultiOrgManagerContextSwitching:
    """Tests for switch_context and get_current_context methods."""

    @pytest.mark.asyncio
    async def test_switch_context_success(self, manager: MultiOrgManager):
        """Successfully switch organization context."""
        await manager.add_member(user_id="user123", org_id="org456")
        context = await manager.switch_context(user_id="user123", org_id="org456")

        assert context.user_id == "user123"
        assert context.org_id == "org456"
        assert context.membership is not None
        assert context.organization is not None

    @pytest.mark.asyncio
    async def test_switch_context_not_member_raises_error(self, manager: MultiOrgManager):
        """Switching to non-member org raises ValueError."""
        with pytest.raises(ValueError, match="not member of org"):
            await manager.switch_context(user_id="user123", org_id="org456")

    @pytest.mark.asyncio
    async def test_switch_context_inactive_raises_error(self, manager: MultiOrgManager):
        """Switching to inactive membership raises ValueError."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            status=MembershipStatus.SUSPENDED,
        )
        with pytest.raises(ValueError, match="not active"):
            await manager.switch_context(user_id="user123", org_id="org456")

    @pytest.mark.asyncio
    async def test_switch_context_increments_count(self, manager: MultiOrgManager):
        """Switching context increments switch count."""
        await manager.add_member(user_id="user123", org_id="org1")
        await manager.add_member(user_id="user123", org_id="org2")

        ctx1 = await manager.switch_context(user_id="user123", org_id="org1")
        assert ctx1.switch_count == 0

        ctx2 = await manager.switch_context(user_id="user123", org_id="org2")
        assert ctx2.switch_count == 1

        ctx3 = await manager.switch_context(user_id="user123", org_id="org1")
        assert ctx3.switch_count == 2

    @pytest.mark.asyncio
    async def test_switch_context_with_organization(self, manager: MultiOrgManager):
        """Switch context with provided Organization object."""
        await manager.add_member(user_id="user123", org_id="org456")
        org = Organization(id="org456", name="Test Organization")

        context = await manager.switch_context(
            user_id="user123",
            org_id="org456",
            organization=org,
        )
        assert context.organization.name == "Test Organization"

    @pytest.mark.asyncio
    async def test_get_current_context(self, manager: MultiOrgManager):
        """Get user's current context."""
        await manager.add_member(user_id="user123", org_id="org456")
        await manager.switch_context(user_id="user123", org_id="org456")

        context = await manager.get_current_context("user123")
        assert context is not None
        assert context.org_id == "org456"

    @pytest.mark.asyncio
    async def test_get_current_context_none(self, manager: MultiOrgManager):
        """Get current context returns None if not set."""
        context = await manager.get_current_context("user123")
        assert context is None


class TestMultiOrgManagerPermissions:
    """Tests for check_permission and get_user_permissions methods."""

    @pytest.mark.asyncio
    async def test_check_permission_has_permission(self, manager: MultiOrgManager):
        """Check permission returns True when user has permission."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.ADMIN,
        )
        has_perm = await manager.check_permission("user123", "org456", "members.invite")
        assert has_perm is True

    @pytest.mark.asyncio
    async def test_check_permission_no_permission(self, manager: MultiOrgManager):
        """Check permission returns False when user lacks permission."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.VIEWER,
        )
        has_perm = await manager.check_permission("user123", "org456", "debates.create")
        assert has_perm is False

    @pytest.mark.asyncio
    async def test_check_permission_not_member(self, manager: MultiOrgManager):
        """Check permission returns False for non-member."""
        has_perm = await manager.check_permission("user123", "org456", "debates.view")
        assert has_perm is False

    @pytest.mark.asyncio
    async def test_get_user_permissions(self, manager: MultiOrgManager):
        """Get all permissions for user in org."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.ADMIN,
        )
        perms = await manager.get_user_permissions("user123", "org456")
        assert "members.invite" in perms
        assert "debates.create" in perms

    @pytest.mark.asyncio
    async def test_get_user_permissions_not_member(self, manager: MultiOrgManager):
        """Get permissions returns empty set for non-member."""
        perms = await manager.get_user_permissions("user123", "org456")
        assert perms == set()


class TestMultiOrgManagerInvitations:
    """Tests for accept_invitation method."""

    @pytest.mark.asyncio
    async def test_accept_invitation_success(self, manager: MultiOrgManager):
        """Successfully accept an invitation."""
        # Create pending invitation
        await manager.add_member(
            user_id="placeholder",
            org_id="org456",
            status=MembershipStatus.PENDING,
            invitation_token="valid_token_123",
        )

        # Accept invitation
        membership = await manager.accept_invitation(
            invitation_token="valid_token_123",
            user_id="actual_user_456",
        )

        assert membership.status == MembershipStatus.ACTIVE
        assert membership.user_id == "actual_user_456"
        assert membership.invitation_token is None

    @pytest.mark.asyncio
    async def test_accept_invitation_invalid_token(self, manager: MultiOrgManager):
        """Accepting with invalid token raises ValueError."""
        with pytest.raises(ValueError, match="Invalid or expired invitation token"):
            await manager.accept_invitation(
                invitation_token="invalid_token",
                user_id="user123",
            )

    @pytest.mark.asyncio
    async def test_accept_invitation_already_active(self, manager: MultiOrgManager):
        """Cannot accept invitation that's already active."""
        await manager.add_member(
            user_id="user123",
            org_id="org456",
            status=MembershipStatus.ACTIVE,
            invitation_token="token123",
        )
        # Token won't be found because status is not pending
        with pytest.raises(ValueError, match="Invalid or expired invitation token"):
            await manager.accept_invitation(
                invitation_token="token123",
                user_id="user123",
            )


class TestMultiOrgManagerStatistics:
    """Tests for get_member_count and get_cross_org_stats methods."""

    @pytest.mark.asyncio
    async def test_get_member_count(self, manager: MultiOrgManager):
        """Get member counts by role."""
        await manager.add_member(user_id="user1", org_id="org456", role=MembershipRole.OWNER)
        await manager.add_member(user_id="user2", org_id="org456", role=MembershipRole.ADMIN)
        await manager.add_member(user_id="user3", org_id="org456", role=MembershipRole.ADMIN)
        await manager.add_member(user_id="user4", org_id="org456", role=MembershipRole.MEMBER)

        counts = await manager.get_member_count("org456")
        assert counts["total"] == 4
        assert counts["owner"] == 1
        assert counts["admin"] == 2
        assert counts["member"] == 1

    @pytest.mark.asyncio
    async def test_get_member_count_excludes_inactive(self, manager: MultiOrgManager):
        """Member count excludes inactive members."""
        await manager.add_member(user_id="user1", org_id="org456")
        await manager.add_member(
            user_id="user2",
            org_id="org456",
            status=MembershipStatus.SUSPENDED,
        )

        counts = await manager.get_member_count("org456")
        assert counts["total"] == 1

    @pytest.mark.asyncio
    async def test_get_cross_org_stats(self, manager: MultiOrgManager):
        """Get cross-organization statistics for user."""
        await manager.add_member(user_id="user123", org_id="org1", role=MembershipRole.OWNER)
        await manager.add_member(user_id="user123", org_id="org2", role=MembershipRole.ADMIN)
        await manager.add_member(user_id="user123", org_id="org3", role=MembershipRole.MEMBER)

        stats = await manager.get_cross_org_stats("user123")
        assert stats["total_organizations"] == 3
        assert stats["roles"]["owner"] == 1
        assert stats["roles"]["admin"] == 1
        assert stats["roles"]["member"] == 1
        assert stats["primary_org_id"] == "org1"  # First is primary
        assert len(stats["organizations"]) == 3

    @pytest.mark.asyncio
    async def test_get_cross_org_stats_no_memberships(self, manager: MultiOrgManager):
        """Cross-org stats for user with no memberships."""
        stats = await manager.get_cross_org_stats("user123")
        assert stats["total_organizations"] == 0
        assert stats["roles"] == {}
        assert stats["primary_org_id"] is None


class TestMultiOrgManagerPersistence:
    """Tests for file-based persistence."""

    @pytest.mark.asyncio
    async def test_file_persistence(self, tmp_path: Path):
        """Data persists across manager instances."""
        db_path = str(tmp_path / "test_persist.db")

        # Create membership with first manager
        manager1 = MultiOrgManager(db_path=db_path)
        await manager1.add_member(user_id="user123", org_id="org456")

        # Create new manager with same db path
        manager2 = MultiOrgManager(db_path=db_path)
        membership = await manager2.get_membership("user123", "org456")

        assert membership is not None
        assert membership.user_id == "user123"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetMultiOrgManager:
    """Tests for get_multi_org_manager factory function."""

    def test_returns_manager_instance(self):
        """Factory returns a MultiOrgManager instance."""
        # Reset global for test isolation
        import aragora.billing.multi_org as mod

        mod._manager = None

        manager = get_multi_org_manager()
        assert isinstance(manager, MultiOrgManager)

    def test_returns_same_instance(self):
        """Factory returns singleton instance."""
        import aragora.billing.multi_org as mod

        mod._manager = None

        manager1 = get_multi_org_manager()
        manager2 = get_multi_org_manager()
        assert manager1 is manager2
