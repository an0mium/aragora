"""
Tests for Organization Management Handlers.

Tests cover:
- Organization access control (role hierarchy)
- Member management (invite, remove, role updates)
- Invitation workflow (create, accept, revoke)
- Permission checks for all operations
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import OrganizationInvitation
from aragora.server.handlers.organizations import (
    OrganizationsHandler,
    ROLE_HIERARCHY,
)

# Common path for patching jwt_auth
JWT_AUTH_PATH = "aragora.billing.jwt_auth.extract_user_from_request"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def org_handler():
    """Create OrganizationsHandler with mock context."""
    handler = OrganizationsHandler({})  # Pass context as positional arg
    # Invitations are now stored in user_store (persistent SQLite)
    return handler


@pytest.fixture
def mock_user_store():
    """Create mock user store."""
    store = MagicMock()
    return store


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler for request context."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {}
    return handler


@pytest.fixture
def owner_user():
    """Create an owner user."""
    user = MagicMock()
    user.id = "owner-123"
    user.email = "owner@example.com"
    user.org_id = "org-123"
    user.role = "owner"
    user.name = "Owner User"
    user.is_active = True
    user.created_at = datetime.utcnow()
    user.last_login_at = datetime.utcnow()
    return user


@pytest.fixture
def admin_user():
    """Create an admin user."""
    user = MagicMock()
    user.id = "admin-123"
    user.email = "admin@example.com"
    user.org_id = "org-123"
    user.role = "admin"
    user.name = "Admin User"
    user.is_active = True
    user.created_at = datetime.utcnow()
    user.last_login_at = datetime.utcnow()
    return user


@pytest.fixture
def member_user():
    """Create a member user."""
    user = MagicMock()
    user.id = "member-123"
    user.email = "member@example.com"
    user.org_id = "org-123"
    user.role = "member"
    user.name = "Member User"
    user.is_active = True
    user.created_at = datetime.utcnow()
    user.last_login_at = datetime.utcnow()
    return user


@pytest.fixture
def sample_org():
    """Create a sample organization."""
    org = MagicMock()
    org.id = "org-123"
    org.name = "Test Organization"
    org.slug = "test-org"
    org.tier = MagicMock(value="pro")
    org.owner_id = "owner-123"
    org.debates_used_this_month = 10
    org.limits = MagicMock(debates_per_month=100, users_per_org=10)
    org.settings = {}
    org.created_at = datetime.utcnow()
    return org


# =============================================================================
# Role Hierarchy Tests
# =============================================================================


class TestRoleHierarchy:
    """Tests for role hierarchy."""

    def test_owner_highest_level(self):
        """Test owner has highest permission level."""
        assert ROLE_HIERARCHY["owner"] > ROLE_HIERARCHY["admin"]
        assert ROLE_HIERARCHY["owner"] > ROLE_HIERARCHY["member"]

    def test_admin_above_member(self):
        """Test admin is above member."""
        assert ROLE_HIERARCHY["admin"] > ROLE_HIERARCHY["member"]

    def test_member_lowest_level(self):
        """Test member has lowest permission level."""
        assert ROLE_HIERARCHY["member"] == 1


# =============================================================================
# Route Matching Tests
# =============================================================================


class TestOrganizationsRouting:
    """Tests for route matching."""

    def test_can_handle_org_detail(self, org_handler):
        """Test handler matches org detail route."""
        assert org_handler.can_handle("/api/org/org-123") is True

    def test_can_handle_members(self, org_handler):
        """Test handler matches members route."""
        assert org_handler.can_handle("/api/org/org-123/members") is True

    def test_can_handle_invite(self, org_handler):
        """Test handler matches invite route."""
        assert org_handler.can_handle("/api/org/org-123/invite") is True

    def test_can_handle_invitations(self, org_handler):
        """Test handler matches invitations list route."""
        assert org_handler.can_handle("/api/org/org-123/invitations") is True

    def test_can_handle_invitation_detail(self, org_handler):
        """Test handler matches invitation detail route."""
        assert org_handler.can_handle("/api/org/org-123/invitations/inv-456") is True

    def test_can_handle_member_detail(self, org_handler):
        """Test handler matches member detail route."""
        assert org_handler.can_handle("/api/org/org-123/members/user-456") is True

    def test_can_handle_role_update(self, org_handler):
        """Test handler matches role update route."""
        assert org_handler.can_handle("/api/org/org-123/members/user-456/role") is True

    def test_can_handle_pending_invitations(self, org_handler):
        """Test handler matches pending invitations route."""
        assert org_handler.can_handle("/api/invitations/pending") is True

    def test_can_handle_accept_invitation(self, org_handler):
        """Test handler matches accept invitation route."""
        assert org_handler.can_handle("/api/invitations/token-abc/accept") is True

    def test_cannot_handle_unknown(self, org_handler):
        """Test handler rejects unknown routes."""
        assert org_handler.can_handle("/api/users") is False
        assert org_handler.can_handle("/api/org") is False


# =============================================================================
# Access Control Tests
# =============================================================================


class TestAccessControl:
    """Tests for organization access control."""

    def test_check_org_access_not_authenticated(self, org_handler):
        """Test access denied when not authenticated."""
        has_access, err = org_handler._check_org_access(None, "org-123")
        assert has_access is False
        assert "Authentication" in err

    def test_check_org_access_wrong_org(self, org_handler, member_user):
        """Test access denied for wrong organization."""
        member_user.org_id = "other-org"
        has_access, err = org_handler._check_org_access(member_user, "org-123")
        assert has_access is False
        assert "member" in err.lower()

    def test_check_org_access_member_allowed(self, org_handler, member_user):
        """Test member can access with member role."""
        has_access, err = org_handler._check_org_access(member_user, "org-123", min_role="member")
        assert has_access is True
        assert err == ""

    def test_check_org_access_member_denied_admin(self, org_handler, member_user):
        """Test member denied for admin operations."""
        has_access, err = org_handler._check_org_access(member_user, "org-123", min_role="admin")
        assert has_access is False
        assert "admin" in err.lower()

    def test_check_org_access_admin_allowed(self, org_handler, admin_user):
        """Test admin can access admin operations."""
        has_access, err = org_handler._check_org_access(admin_user, "org-123", min_role="admin")
        assert has_access is True

    def test_check_org_access_admin_denied_owner(self, org_handler, admin_user):
        """Test admin denied for owner-only operations."""
        has_access, err = org_handler._check_org_access(admin_user, "org-123", min_role="owner")
        assert has_access is False
        assert "owner" in err.lower()

    def test_check_org_access_owner_allowed(self, org_handler, owner_user):
        """Test owner has full access."""
        has_access, err = org_handler._check_org_access(owner_user, "org-123", min_role="owner")
        assert has_access is True


# =============================================================================
# Get Organization Tests
# =============================================================================


class TestGetOrganization:
    """Tests for get organization endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_get_org_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user, sample_org
    ):
        """Test successful organization retrieval."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = [member_user]

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "GET"

        result = org_handler.handle("/api/org/org-123", {}, mock_http_handler, method="GET")

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["organization"]["id"] == "org-123"
        assert body["organization"]["name"] == "Test Organization"

    @patch(JWT_AUTH_PATH)
    def test_get_org_not_member(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test access denied when not a member."""
        member_user.org_id = "other-org"
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user

        org_handler.ctx = {"user_store": mock_user_store}

        result = org_handler.handle("/api/org/org-123", {}, mock_http_handler, method="GET")

        assert result.status_code == 403


# =============================================================================
# Update Organization Tests
# =============================================================================


class TestUpdateOrganization:
    """Tests for update organization endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_update_org_requires_admin(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test update requires admin role."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(return_value={"name": "New Name"})
        mock_http_handler.command = "PUT"

        result = org_handler.handle("/api/org/org-123", {}, mock_http_handler, method="PUT")

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_update_org_admin_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, sample_org
    ):
        """Test admin can update organization."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user
        mock_user_store.update_organization.return_value = True
        mock_user_store.get_organization_by_id.return_value = sample_org

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(return_value={"name": "New Name"})
        mock_http_handler.command = "PUT"

        result = org_handler.handle("/api/org/org-123", {}, mock_http_handler, method="PUT")

        assert result.status_code == 200
        mock_user_store.update_organization.assert_called_once()

    @patch(JWT_AUTH_PATH)
    def test_update_org_name_too_short(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user
    ):
        """Test name validation - too short."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(return_value={"name": "A"})
        mock_http_handler.command = "PUT"

        result = org_handler.handle("/api/org/org-123", {}, mock_http_handler, method="PUT")

        assert result.status_code == 400


# =============================================================================
# List Members Tests
# =============================================================================


class TestListMembers:
    """Tests for list members endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_list_members_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user, admin_user
    ):
        """Test successful member listing."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user
        mock_user_store.get_org_members.return_value = [member_user, admin_user]

        org_handler.ctx = {"user_store": mock_user_store}

        result = org_handler.handle("/api/org/org-123/members", {}, mock_http_handler, method="GET")

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["count"] == 2
        assert len(body["members"]) == 2


# =============================================================================
# Invite Member Tests
# =============================================================================


class TestInviteMember:
    """Tests for invite member endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_invite_requires_admin(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test invite requires admin role."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(
            return_value={"email": "new@example.com", "role": "member"}
        )
        mock_http_handler.command = "POST"

        result = org_handler.handle("/api/org/org-123/invite", {}, mock_http_handler, method="POST")

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_invite_existing_user_to_org(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, sample_org
    ):
        """Test inviting an existing user without an org."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = [admin_user]

        # Existing user without an org
        existing_user = MagicMock()
        existing_user.id = "new-user-123"
        existing_user.org_id = None  # Not in any org
        mock_user_store.get_user_by_email.return_value = existing_user
        mock_user_store.add_user_to_org.return_value = True

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(
            return_value={"email": "existing@example.com", "role": "member"}
        )
        mock_http_handler.command = "POST"

        result = org_handler.handle("/api/org/org-123/invite", {}, mock_http_handler, method="POST")

        assert result.status_code == 200
        mock_user_store.add_user_to_org.assert_called_once()

    @patch(JWT_AUTH_PATH)
    def test_invite_user_already_in_org(
        self,
        mock_extract,
        org_handler,
        mock_http_handler,
        mock_user_store,
        admin_user,
        sample_org,
        member_user,
    ):
        """Test cannot invite user already in the org."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = [admin_user]
        mock_user_store.get_user_by_email.return_value = member_user

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(
            return_value={"email": "member@example.com", "role": "member"}
        )
        mock_http_handler.command = "POST"

        result = org_handler.handle("/api/org/org-123/invite", {}, mock_http_handler, method="POST")

        assert result.status_code == 400
        assert b"already a member" in result.body

    @patch(JWT_AUTH_PATH)
    def test_invite_creates_invitation(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, sample_org
    ):
        """Test creating invitation for new user."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = [admin_user]
        mock_user_store.get_user_by_email.return_value = None  # User doesn't exist
        mock_user_store.get_invitation_by_email.return_value = None  # No existing invitation
        mock_user_store.create_invitation.return_value = True

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(
            return_value={"email": "new@example.com", "role": "member"}
        )
        mock_http_handler.command = "POST"

        result = org_handler.handle("/api/org/org-123/invite", {}, mock_http_handler, method="POST")

        assert result.status_code == 201
        import json

        body = json.loads(result.body)
        assert "invitation_id" in body
        assert body["message"] == "Invitation sent to new@example.com"
        mock_user_store.create_invitation.assert_called_once()

    @patch(JWT_AUTH_PATH)
    def test_invite_invalid_role(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, sample_org
    ):
        """Test invitation rejects invalid roles."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = [admin_user]

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(
            return_value={"email": "new@example.com", "role": "owner"}  # Can't invite as owner
        )
        mock_http_handler.command = "POST"

        result = org_handler.handle("/api/org/org-123/invite", {}, mock_http_handler, method="POST")

        assert result.status_code == 400

    @patch(JWT_AUTH_PATH)
    def test_invite_org_member_limit(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, sample_org
    ):
        """Test invitation fails when org at member limit."""
        sample_org.limits.users_per_org = 1  # Only 1 user allowed
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = [admin_user]  # Already at limit

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(
            return_value={"email": "new@example.com", "role": "member"}
        )
        mock_http_handler.command = "POST"

        result = org_handler.handle("/api/org/org-123/invite", {}, mock_http_handler, method="POST")

        assert result.status_code == 403
        assert b"limit" in result.body.lower()


# =============================================================================
# Remove Member Tests
# =============================================================================


class TestRemoveMember:
    """Tests for remove member endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_remove_member_requires_admin(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test remove requires admin role."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "DELETE"

        result = org_handler.handle(
            "/api/org/org-123/members/user-456", {}, mock_http_handler, method="DELETE"
        )

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_cannot_remove_owner(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, owner_user
    ):
        """Test cannot remove the owner."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            admin_user if uid == admin_user.id else owner_user
        )

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "DELETE"

        result = org_handler.handle(
            "/api/org/org-123/members/owner-123", {}, mock_http_handler, method="DELETE"
        )

        assert result.status_code == 403
        assert b"owner" in result.body.lower()

    @patch(JWT_AUTH_PATH)
    def test_cannot_remove_self(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user
    ):
        """Test cannot remove yourself."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "DELETE"

        result = org_handler.handle(
            f"/api/org/org-123/members/{admin_user.id}", {}, mock_http_handler, method="DELETE"
        )

        assert result.status_code == 400
        assert b"yourself" in result.body.lower()

    @patch(JWT_AUTH_PATH)
    def test_only_owner_can_remove_admin(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user
    ):
        """Test only owner can remove admin members."""
        # Another admin trying to remove this admin
        other_admin = MagicMock()
        other_admin.id = "other-admin"
        other_admin.org_id = "org-123"
        other_admin.role = "admin"

        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=other_admin.id)
        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            other_admin if uid == other_admin.id else admin_user
        )

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "DELETE"

        result = org_handler.handle(
            f"/api/org/org-123/members/{admin_user.id}", {}, mock_http_handler, method="DELETE"
        )

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_remove_member_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user, member_user
    ):
        """Test successful member removal."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            admin_user if uid == admin_user.id else member_user
        )
        mock_user_store.remove_user_from_org.return_value = True

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "DELETE"

        result = org_handler.handle(
            f"/api/org/org-123/members/{member_user.id}", {}, mock_http_handler, method="DELETE"
        )

        assert result.status_code == 200
        mock_user_store.remove_user_from_org.assert_called_once()


# =============================================================================
# Update Member Role Tests
# =============================================================================


class TestUpdateMemberRole:
    """Tests for update member role endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_update_role_requires_owner(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user
    ):
        """Test role update requires owner role."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(return_value={"role": "admin"})
        mock_http_handler.command = "PUT"

        result = org_handler.handle(
            "/api/org/org-123/members/user-456/role", {}, mock_http_handler, method="PUT"
        )

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_cannot_change_owner_role(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, owner_user
    ):
        """Test cannot change the owner's role."""
        other_owner = MagicMock()
        other_owner.id = "owner-456"
        other_owner.org_id = "org-123"
        other_owner.role = "owner"

        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=owner_user.id)
        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            owner_user if uid == owner_user.id else other_owner
        )

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(return_value={"role": "admin"})
        mock_http_handler.command = "PUT"

        result = org_handler.handle(
            "/api/org/org-123/members/owner-456/role", {}, mock_http_handler, method="PUT"
        )

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_update_role_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, owner_user, member_user
    ):
        """Test successful role update."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=owner_user.id)
        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            owner_user if uid == owner_user.id else member_user
        )
        mock_user_store.update_user.return_value = True

        org_handler.ctx = {"user_store": mock_user_store}
        org_handler.read_json_body = MagicMock(return_value={"role": "admin"})
        mock_http_handler.command = "PUT"

        result = org_handler.handle(
            f"/api/org/org-123/members/{member_user.id}/role", {}, mock_http_handler, method="PUT"
        )

        assert result.status_code == 200
        mock_user_store.update_user.assert_called_with(member_user.id, role="admin")


# =============================================================================
# Invitation Management Tests
# =============================================================================


class TestInvitationManagement:
    """Tests for invitation list and revoke."""

    @patch(JWT_AUTH_PATH)
    def test_list_invitations_requires_admin(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test listing invitations requires admin."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user

        org_handler.ctx = {"user_store": mock_user_store}

        result = org_handler.handle(
            "/api/org/org-123/invitations", {}, mock_http_handler, method="GET"
        )

        assert result.status_code == 403

    @patch(JWT_AUTH_PATH)
    def test_list_invitations_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user
    ):
        """Test successful invitation listing."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user

        # Create a test invitation via mock user_store
        invitation = OrganizationInvitation(
            org_id="org-123",
            email="invitee@example.com",
            role="member",
            invited_by=admin_user.id,
        )
        mock_user_store.get_invitations_for_org.return_value = [invitation]
        mock_user_store.cleanup_expired_invitations.return_value = 0

        org_handler.ctx = {"user_store": mock_user_store}

        result = org_handler.handle(
            "/api/org/org-123/invitations", {}, mock_http_handler, method="GET"
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["count"] == 1

    @patch(JWT_AUTH_PATH)
    def test_revoke_invitation_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, admin_user
    ):
        """Test successful invitation revocation."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=admin_user.id)
        mock_user_store.get_user_by_id.return_value = admin_user

        invitation = OrganizationInvitation(
            org_id="org-123",
            email="invitee@example.com",
            role="member",
            invited_by=admin_user.id,
        )
        mock_user_store.get_invitation_by_id.return_value = invitation
        mock_user_store.update_invitation_status.return_value = True

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "DELETE"

        result = org_handler.handle(
            f"/api/org/org-123/invitations/{invitation.id}", {}, mock_http_handler, method="DELETE"
        )

        assert result.status_code == 200
        mock_user_store.update_invitation_status.assert_called_once_with(invitation.id, "revoked")


# =============================================================================
# Accept Invitation Tests
# =============================================================================


class TestAcceptInvitation:
    """Tests for accepting invitations."""

    @patch(JWT_AUTH_PATH)
    def test_accept_requires_auth(self, mock_extract, org_handler, mock_http_handler):
        """Test accept requires authentication."""
        mock_extract.return_value = MagicMock(is_authenticated=False)
        org_handler.ctx = {"user_store": MagicMock()}
        mock_http_handler.command = "POST"

        result = org_handler.handle(
            "/api/invitations/token-abc/accept", {}, mock_http_handler, method="POST"
        )

        assert result.status_code == 401

    @patch(JWT_AUTH_PATH)
    def test_accept_wrong_email(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test cannot accept invitation for different email."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user
        member_user.org_id = None  # User has no org

        invitation = OrganizationInvitation(
            org_id="org-123",
            email="different@example.com",  # Different email
            role="member",
            invited_by="admin-123",
        )
        mock_user_store.get_invitation_by_token.return_value = invitation

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "POST"

        result = org_handler.handle(
            f"/api/invitations/{invitation.token}/accept", {}, mock_http_handler, method="POST"
        )

        assert result.status_code == 403
        assert b"different email" in result.body.lower()

    @patch(JWT_AUTH_PATH)
    def test_accept_already_in_org(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user
    ):
        """Test cannot accept if already in an org."""
        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user

        invitation = OrganizationInvitation(
            org_id="org-456",  # Different org
            email=member_user.email,
            role="member",
            invited_by="admin-123",
        )
        mock_user_store.get_invitation_by_token.return_value = invitation

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "POST"

        result = org_handler.handle(
            f"/api/invitations/{invitation.token}/accept", {}, mock_http_handler, method="POST"
        )

        assert result.status_code == 400
        assert b"already a member" in result.body.lower()

    @patch(JWT_AUTH_PATH)
    def test_accept_invitation_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user, sample_org
    ):
        """Test successful invitation acceptance."""
        member_user.org_id = None  # User has no org

        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user
        mock_user_store.get_organization_by_id.return_value = sample_org
        mock_user_store.get_org_members.return_value = []
        mock_user_store.add_user_to_org.return_value = True
        mock_user_store.update_invitation_status.return_value = True

        invitation = OrganizationInvitation(
            org_id="org-123",
            email=member_user.email,
            role="member",
            invited_by="admin-123",
        )
        mock_user_store.get_invitation_by_token.return_value = invitation

        org_handler.ctx = {"user_store": mock_user_store}
        mock_http_handler.command = "POST"

        result = org_handler.handle(
            f"/api/invitations/{invitation.token}/accept", {}, mock_http_handler, method="POST"
        )

        assert result.status_code == 200
        mock_user_store.add_user_to_org.assert_called_once()
        mock_user_store.update_invitation_status.assert_called_once()


# =============================================================================
# Pending Invitations Tests
# =============================================================================


class TestPendingInvitations:
    """Tests for user's pending invitations endpoint."""

    @patch(JWT_AUTH_PATH)
    def test_get_pending_requires_auth(self, mock_extract, org_handler, mock_http_handler):
        """Test getting pending invitations requires auth."""
        mock_extract.return_value = MagicMock(is_authenticated=False)
        org_handler.ctx = {"user_store": MagicMock()}

        result = org_handler.handle("/api/invitations/pending", {}, mock_http_handler, method="GET")

        assert result.status_code == 401

    @patch(JWT_AUTH_PATH)
    def test_get_pending_success(
        self, mock_extract, org_handler, mock_http_handler, mock_user_store, member_user, sample_org
    ):
        """Test getting pending invitations."""
        member_user.org_id = None

        mock_extract.return_value = MagicMock(is_authenticated=True, user_id=member_user.id)
        mock_user_store.get_user_by_id.return_value = member_user
        mock_user_store.get_organization_by_id.return_value = sample_org

        # Create pending invitation via mock user_store
        invitation = OrganizationInvitation(
            org_id="org-123",
            email=member_user.email,
            role="member",
            invited_by="admin-123",
        )
        mock_user_store.get_pending_invitations_by_email.return_value = [invitation]

        org_handler.ctx = {"user_store": mock_user_store}

        result = org_handler.handle("/api/invitations/pending", {}, mock_http_handler, method="GET")

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["invitations"][0]["org_name"] == "Test Organization"
