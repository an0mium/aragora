"""
Tests for aragora.server.handlers.organizations - Organization management handler.

Tests cover:
- can_handle() route pattern matching
- handle() route dispatching
- Rate limiting
- _get_organization() - Get org details
- _update_organization() - Update org settings
- _list_members() - List org members
- _invite_member() - Invite users
- _remove_member() - Remove members
- _update_member_role() - Change member roles
- _list_invitations() - List pending invitations
- _revoke_invitation() - Revoke invitations
- _get_pending_invitations() - User's pending invitations
- _accept_invitation() - Accept invitation
- Role hierarchy and authorization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    org_id: Optional[str] = "org-123"
    role: str = "member"
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None


@dataclass
class MockOrganization:
    """Mock organization for testing."""

    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: Any = None  # Mock tier
    owner_id: str = "owner-123"
    debates_used_this_month: int = 0
    limits: Any = None  # Mock limits
    settings: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.tier is None:
            self.tier = MagicMock()
            self.tier.value = "pro"
        if self.limits is None:
            self.limits = MagicMock()
            self.limits.debates_per_month = 100
            self.limits.users_per_org = 10


@dataclass
class MockInvitation:
    """Mock invitation for testing."""

    id: str = "inv-123"
    org_id: str = "org-123"
    email: str = "invite@example.com"
    role: str = "member"
    token: str = "abc123token"
    invited_by: str = "admin-123"
    status: str = "pending"
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=7)
    )
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_pending(self) -> bool:
        return self.status == "pending" and self.expires_at > datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "org_id": self.org_id,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat(),
        }


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict = None,
        client_address: tuple = None,
        body: bytes = None,
        method: str = "GET",
    ):
        self.headers = headers or {}
        self.client_address = client_address or ("127.0.0.1", 12345)
        self._body = body or b""
        self.command = method
        self.rfile = MagicMock()
        self.rfile.read.return_value = self._body


class MockAuthContext:
    """Mock auth context for testing."""

    def __init__(self, is_authenticated: bool = True, user_id: str = "user-123"):
        self.is_authenticated = is_authenticated
        self.user_id = user_id


def create_org_handler():
    """Create an OrganizationsHandler with empty context."""
    from aragora.server.handlers.organizations import OrganizationsHandler

    return OrganizationsHandler({})


def get_body(result) -> dict:
    """Extract body as dict from HandlerResult."""
    if hasattr(result, "body"):
        return json.loads(result.body.decode("utf-8"))
    return result


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result


# ===========================================================================
# Test can_handle() Route Pattern Matching
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route pattern matching."""

    def test_handles_org_detail(self):
        """Should handle /api/org/{org_id}."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123") is True
        assert handler.can_handle("/api/v1/org/my_org") is True

    def test_handles_members_list(self):
        """Should handle /api/org/{org_id}/members."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123/members") is True

    def test_handles_invite(self):
        """Should handle /api/org/{org_id}/invite."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123/invite") is True

    def test_handles_invitations_list(self):
        """Should handle /api/org/{org_id}/invitations."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123/invitations") is True

    def test_handles_invitation_revoke(self):
        """Should handle /api/org/{org_id}/invitations/{id}."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123/invitations/inv-456") is True

    def test_handles_member_remove(self):
        """Should handle /api/org/{org_id}/members/{user_id}."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123/members/user-456") is True

    def test_handles_member_role(self):
        """Should handle /api/org/{org_id}/members/{user_id}/role."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/org/org-123/members/user-456/role") is True

    def test_handles_pending_invitations(self):
        """Should handle /api/invitations/pending."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/invitations/pending") is True

    def test_handles_accept_invitation(self):
        """Should handle /api/invitations/{token}/accept."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/invitations/abc123/accept") is True

    def test_rejects_unknown_routes(self):
        """Should reject unknown routes."""
        handler = create_org_handler()
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/org") is False  # No org_id
        assert handler.can_handle("/api/v1/organizations") is False


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting in handle()."""

    def test_rate_limit_exceeded_returns_429(self):
        """Should return 429 when rate limit exceeded."""
        from aragora.server.handlers.organizations import _org_limiter

        handler = create_org_handler()
        mock_http = MockHandler(client_address=("192.168.1.100", 12345))

        with patch.object(_org_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/org/org-123", {}, mock_http, "GET")

        assert get_status(result) == 429
        assert "Rate limit" in get_body(result)["error"]


# ===========================================================================
# Test Role Hierarchy
# ===========================================================================


class TestRoleHierarchy:
    """Tests for role hierarchy constants."""

    def test_role_hierarchy_order(self):
        """Role hierarchy should be member < admin < owner."""
        from aragora.server.handlers.organizations import ROLE_HIERARCHY

        assert ROLE_HIERARCHY["member"] < ROLE_HIERARCHY["admin"]
        assert ROLE_HIERARCHY["admin"] < ROLE_HIERARCHY["owner"]


# ===========================================================================
# Test _check_org_access()
# ===========================================================================


class TestCheckOrgAccess:
    """Tests for _check_org_access() authorization."""

    def test_no_user_fails(self):
        """Should fail when no user provided."""
        handler = create_org_handler()
        has_access, err = handler._check_org_access(None, "org-123")

        assert has_access is False
        assert "Authentication required" in err

    def test_wrong_org_fails(self):
        """Should fail when user is in different org."""
        handler = create_org_handler()
        user = MockUser(org_id="other-org")

        has_access, err = handler._check_org_access(user, "org-123")

        assert has_access is False
        assert "Not a member" in err

    def test_member_access_for_member_role(self):
        """Member should have access when member role required."""
        handler = create_org_handler()
        user = MockUser(role="member", org_id="org-123")

        has_access, err = handler._check_org_access(user, "org-123", min_role="member")

        assert has_access is True

    def test_member_no_admin_access(self):
        """Member should not have access when admin role required."""
        handler = create_org_handler()
        user = MockUser(role="member", org_id="org-123")

        has_access, err = handler._check_org_access(user, "org-123", min_role="admin")

        assert has_access is False
        assert "Requires admin" in err

    def test_admin_has_admin_access(self):
        """Admin should have access when admin role required."""
        handler = create_org_handler()
        user = MockUser(role="admin", org_id="org-123")

        has_access, err = handler._check_org_access(user, "org-123", min_role="admin")

        assert has_access is True

    def test_owner_has_all_access(self):
        """Owner should have access to all roles."""
        handler = create_org_handler()
        user = MockUser(role="owner", org_id="org-123")

        has_access_member, _ = handler._check_org_access(user, "org-123", min_role="member")
        has_access_admin, _ = handler._check_org_access(user, "org-123", min_role="admin")
        has_access_owner, _ = handler._check_org_access(user, "org-123", min_role="owner")

        assert has_access_member is True
        assert has_access_admin is True
        assert has_access_owner is True


# ===========================================================================
# Test _get_organization()
# ===========================================================================


class TestGetOrganization:
    """Tests for _get_organization()."""

    @pytest.mark.no_auto_auth
    def test_unauthenticated_returns_401(self):
        """Should return 401 when not authenticated."""
        handler = create_org_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_get_current_user", return_value=(None, None)):
            result = handler._get_organization(mock_http, "org-123")

        assert get_status(result) == 401

    def test_wrong_org_returns_403(self):
        """Should return 403 when accessing different org."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(org_id="other-org")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._get_organization(mock_http, "org-123")

        assert get_status(result) == 403

    def test_org_not_found_returns_404(self):
        """Should return 404 when org doesn't exist."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(org_id="org-123")
        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = None

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._get_organization(mock_http, "org-123")

        assert get_status(result) == 404

    def test_returns_org_details(self):
        """Should return organization details."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(org_id="org-123")
        org = MockOrganization()
        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = org
        mock_user_store.get_org_members.return_value = [user]

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._get_organization(mock_http, "org-123")

        body = get_body(result)
        assert get_status(result) == 200
        assert body["organization"]["id"] == "org-123"
        assert body["organization"]["name"] == "Test Org"
        assert body["organization"]["member_count"] == 1


# ===========================================================================
# Test _update_organization()
# ===========================================================================


class TestUpdateOrganization:
    """Tests for _update_organization()."""

    def test_member_cannot_update(self):
        """Member should not be able to update org."""
        handler = create_org_handler()
        mock_http = MockHandler(body=b'{"name": "New Name"}', method="PUT")
        user = MockUser(role="member", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._update_organization(mock_http, "org-123")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_admin_can_update(self):
        """Admin should be able to update org."""
        handler = create_org_handler()
        mock_http = MockHandler(body=b'{"name": "New Name"}', method="PUT")
        user = MockUser(role="admin", org_id="org-123")
        org = MockOrganization(name="New Name")
        mock_user_store = MagicMock()
        mock_user_store.update_organization.return_value = True
        mock_user_store.get_organization_by_id.return_value = org

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "read_json_body", return_value={"name": "New Name"}),
        ):
            result = handler._update_organization(mock_http, "org-123")

        body = get_body(result)
        assert get_status(result) == 200
        assert "message" in body

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_validates_name_length(self):
        """Should validate name length."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")
        user = MockUser(role="admin", org_id="org-123")

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "read_json_body", return_value={"name": "X"}),  # Too short
        ):
            result = handler._update_organization(mock_http, "org-123")

        assert get_status(result) == 400
        assert "at least 2 characters" in get_body(result)["error"]


# ===========================================================================
# Test _list_members()
# ===========================================================================


class TestListMembers:
    """Tests for _list_members()."""

    def test_returns_member_list(self):
        """Should return list of organization members."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(org_id="org-123")
        members = [
            MockUser(id="user-1", email="a@test.com", role="owner"),
            MockUser(id="user-2", email="b@test.com", role="admin"),
            MockUser(id="user-3", email="c@test.com", role="member"),
        ]
        mock_user_store = MagicMock()
        mock_user_store.get_org_members.return_value = members

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._list_members(mock_http, "org-123")

        body = get_body(result)
        assert get_status(result) == 200
        assert body["count"] == 3
        assert len(body["members"]) == 3


# ===========================================================================
# Test _invite_member()
# ===========================================================================


class TestInviteMember:
    """Tests for _invite_member()."""

    def test_member_cannot_invite(self):
        """Member should not be able to invite."""
        handler = create_org_handler()
        mock_http = MockHandler(method="POST")
        user = MockUser(role="member", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._invite_member(mock_http, "org-123")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_invalid_role_rejected(self):
        """Should reject invalid role."""
        handler = create_org_handler()
        mock_http = MockHandler(method="POST")
        user = MockUser(role="admin", org_id="org-123")
        org = MockOrganization()
        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = org
        mock_user_store.get_org_members.return_value = [user]

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(
                handler,
                "read_json_body",
                return_value={"email": "new@test.com", "role": "superadmin"},
            ),
            patch("aragora.server.handlers.organizations.validate_against_schema") as mock_validate,
        ):
            mock_validate.return_value = MagicMock(is_valid=True)
            result = handler._invite_member(mock_http, "org-123")

        assert get_status(result) == 400
        assert "Invalid role" in get_body(result)["error"]


# ===========================================================================
# Test _remove_member()
# ===========================================================================


class TestRemoveMember:
    """Tests for _remove_member()."""

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_cannot_remove_owner(self):
        """Should not allow removing owner."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        admin = MockUser(id="admin-1", role="admin", org_id="org-123")
        owner = MockUser(id="owner-1", role="owner", org_id="org-123")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = owner

        with (
            patch.object(handler, "_get_current_user", return_value=(admin, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._remove_member(mock_http, "org-123", "owner-1")

        assert get_status(result) == 403
        assert "Cannot remove the organization owner" in get_body(result)["error"]

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_cannot_remove_self(self):
        """Should not allow removing self."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        admin = MockUser(id="admin-1", role="admin", org_id="org-123")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = admin

        with (
            patch.object(
                handler,
                "_get_current_user",
                return_value=(admin, MockAuthContext(user_id="admin-1")),
            ),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._remove_member(mock_http, "org-123", "admin-1")

        assert get_status(result) == 400
        assert "Cannot remove yourself" in get_body(result)["error"]

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_admin_cannot_remove_other_admin(self):
        """Admin should not be able to remove other admins."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        admin1 = MockUser(id="admin-1", role="admin", org_id="org-123")
        admin2 = MockUser(id="admin-2", role="admin", org_id="org-123")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = admin2

        with (
            patch.object(handler, "_get_current_user", return_value=(admin1, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._remove_member(mock_http, "org-123", "admin-2")

        assert get_status(result) == 403
        assert "Only the owner can remove admin" in get_body(result)["error"]


# ===========================================================================
# Test _update_member_role()
# ===========================================================================


class TestUpdateMemberRole:
    """Tests for _update_member_role()."""

    def test_only_owner_can_update_roles(self):
        """Only owner should be able to update roles."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")
        admin = MockUser(role="admin", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(admin, MockAuthContext())):
            result = handler._update_member_role(mock_http, "org-123", "user-456")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_cannot_change_owner_role(self):
        """Should not allow changing owner's role."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")
        owner = MockUser(id="owner-1", role="owner", org_id="org-123")
        target_owner = MockUser(id="owner-1", role="owner", org_id="org-123")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = target_owner

        with (
            patch.object(handler, "_get_current_user", return_value=(owner, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "read_json_body", return_value={"role": "admin"}),
        ):
            result = handler._update_member_role(mock_http, "org-123", "owner-1")

        assert get_status(result) == 403
        assert "Cannot change the owner's role" in get_body(result)["error"]

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_owner_can_promote_to_admin(self):
        """Owner should be able to promote member to admin."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")
        owner = MockUser(id="owner-1", role="owner", org_id="org-123")
        member = MockUser(id="member-1", role="member", org_id="org-123")
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = member
        mock_user_store.update_user.return_value = True

        with (
            patch.object(handler, "_get_current_user", return_value=(owner, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "read_json_body", return_value={"role": "admin"}),
        ):
            result = handler._update_member_role(mock_http, "org-123", "member-1")

        body = get_body(result)
        assert get_status(result) == 200
        assert body["role"] == "admin"


# ===========================================================================
# Test Invitation Endpoints
# ===========================================================================


class TestListInvitations:
    """Tests for _list_invitations()."""

    def test_returns_invitation_list(self):
        """Should return list of invitations."""
        handler = create_org_handler()
        mock_http = MockHandler()
        admin = MockUser(role="admin", org_id="org-123")
        invitations = [
            MockInvitation(id="inv-1", status="pending"),
            MockInvitation(id="inv-2", status="accepted"),
        ]
        mock_user_store = MagicMock()
        mock_user_store.cleanup_expired_invitations.return_value = None

        with (
            patch.object(handler, "_get_current_user", return_value=(admin, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
            patch.object(handler, "_get_invitations_for_org", return_value=invitations),
        ):
            result = handler._list_invitations(mock_http, "org-123")

        body = get_body(result)
        assert get_status(result) == 200
        assert body["count"] == 2
        assert body["pending_count"] == 1


class TestRevokeInvitation:
    """Tests for _revoke_invitation()."""

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_revokes_pending_invitation(self):
        """Should revoke pending invitation."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        admin = MockUser(role="admin", org_id="org-123")
        invitation = MockInvitation(status="pending")
        mock_user_store = MagicMock()
        mock_user_store.get_invitation_by_id.return_value = invitation
        mock_user_store.update_invitation_status.return_value = True

        with (
            patch.object(handler, "_get_current_user", return_value=(admin, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._revoke_invitation(mock_http, "org-123", "inv-123")

        body = get_body(result)
        assert get_status(result) == 200
        assert "revoked" in body["message"].lower()

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_cannot_revoke_accepted_invitation(self):
        """Should not revoke already accepted invitation."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        admin = MockUser(role="admin", org_id="org-123")
        invitation = MockInvitation(status="accepted")
        mock_user_store = MagicMock()
        mock_user_store.get_invitation_by_id.return_value = invitation

        with (
            patch.object(handler, "_get_current_user", return_value=(admin, MockAuthContext())),
            patch.object(handler, "_get_user_store", return_value=mock_user_store),
        ):
            result = handler._revoke_invitation(mock_http, "org-123", "inv-123")

        assert get_status(result) == 400


class TestAcceptInvitation:
    """Tests for _accept_invitation()."""

    def test_wrong_email_rejected(self):
        """Should reject if invitation email doesn't match user."""
        handler = create_org_handler()
        mock_http = MockHandler(method="POST")
        user = MockUser(email="different@test.com")
        invitation = MockInvitation(email="invite@example.com")

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_invitation_by_token", return_value=invitation),
        ):
            result = handler._accept_invitation(mock_http, "abc123token")

        assert get_status(result) == 403
        assert "different email" in get_body(result)["error"]

    def test_user_already_in_org_rejected(self):
        """Should reject if user is already in an organization."""
        handler = create_org_handler()
        mock_http = MockHandler(method="POST")
        user = MockUser(email="invite@example.com", org_id="other-org")
        invitation = MockInvitation(email="invite@example.com")

        with (
            patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())),
            patch.object(handler, "_get_invitation_by_token", return_value=invitation),
        ):
            result = handler._accept_invitation(mock_http, "abc123token")

        assert get_status(result) == 400
        assert "already a member" in get_body(result)["error"]


# ===========================================================================
# Test Route Dispatching
# ===========================================================================


class TestRouteDispatching:
    """Tests for handle() route dispatching."""

    def test_get_org_routes_correctly(self):
        """GET /api/org/{id} should route to _get_organization."""
        handler = create_org_handler()
        mock_http = MockHandler()

        with patch.object(handler, "_get_organization") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/org/org-123", {}, mock_http, "GET")

            mock_method.assert_called_once()

    def test_put_org_routes_correctly(self):
        """PUT /api/org/{id} should route to _update_organization."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")

        with patch.object(handler, "_update_organization") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            handler.handle("/api/v1/org/org-123", {}, mock_http, "PUT")

            mock_method.assert_called_once()

    def test_unsupported_method_returns_405(self):
        """Unsupported method should return 405."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PATCH")

        result = handler.handle("/api/v1/org/org-123", {}, mock_http, "PATCH")

        assert get_status(result) == 405

    def test_returns_none_for_unknown_route(self):
        """Should return None for unhandled routes."""
        handler = create_org_handler()
        mock_http = MockHandler()

        result = handler.handle("/api/v1/unknown/route", {}, mock_http, "GET")

        assert result is None


# ===========================================================================
# RBAC Tests
# ===========================================================================


@dataclass
class MockPermissionDecision:
    """Mock RBAC permission decision."""

    allowed: bool = True
    reason: str = "Allowed by test"


def mock_check_permission_allowed(*args, **kwargs):
    """Mock check_permission that always allows."""
    return MockPermissionDecision(allowed=True)


def mock_check_permission_denied(*args, **kwargs):
    """Mock check_permission that always denies."""
    return MockPermissionDecision(allowed=False, reason="Permission denied by test")


class TestOrganizationsRBAC:
    """Tests for RBAC permission checks in OrganizationsHandler."""

    def test_rbac_helper_methods_exist(self):
        """Handler should have RBAC helper methods."""
        handler = create_org_handler()
        assert hasattr(handler, "_check_rbac_permission")
        assert hasattr(handler, "_get_auth_context")

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", False)
    def test_permission_check_without_rbac(self):
        """Permission check should pass when RBAC not available."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(role="admin", org_id="org-123")

        result = handler._check_rbac_permission(mock_http, "organizations.update", user)
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_allowed)
    def test_permission_check_allowed(self):
        """Permission check should pass when RBAC allows."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(role="admin", org_id="org-123")

        result = handler._check_rbac_permission(mock_http, "organizations.update", user)
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_denied)
    def test_permission_check_denied(self):
        """Permission check should return error when RBAC denies."""
        handler = create_org_handler()
        mock_http = MockHandler()
        user = MockUser(role="viewer", org_id="org-123")

        result = handler._check_rbac_permission(mock_http, "organizations.update", user)
        assert result is not None
        assert get_status(result) == 403
        body = get_body(result)
        # Error is wrapped as {"error": {"error": "...", "reason": "..."}}
        error_data = body.get("error", {})
        if isinstance(error_data, dict):
            assert "Permission denied" in error_data.get("error", "")
        else:
            assert "Permission denied" in str(error_data)

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_denied)
    def test_update_organization_rbac_denied(self):
        """Update organization should deny when RBAC denies."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")
        user = MockUser(role="admin", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._update_organization(mock_http, "org-123")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_denied)
    def test_invite_member_rbac_denied(self):
        """Invite member should deny when RBAC denies."""
        handler = create_org_handler()
        mock_http = MockHandler(method="POST")
        user = MockUser(role="admin", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._invite_member(mock_http, "org-123")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_denied)
    def test_remove_member_rbac_denied(self):
        """Remove member should deny when RBAC denies."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        user = MockUser(role="admin", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._remove_member(mock_http, "org-123", "user-456")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_denied)
    def test_update_member_role_rbac_denied(self):
        """Update member role should deny when RBAC denies."""
        handler = create_org_handler()
        mock_http = MockHandler(method="PUT")
        user = MockUser(role="owner", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._update_member_role(mock_http, "org-123", "user-456")

        assert get_status(result) == 403

    @patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.organizations.check_permission", mock_check_permission_denied)
    def test_revoke_invitation_rbac_denied(self):
        """Revoke invitation should deny when RBAC denies."""
        handler = create_org_handler()
        mock_http = MockHandler(method="DELETE")
        user = MockUser(role="admin", org_id="org-123")

        with patch.object(handler, "_get_current_user", return_value=(user, MockAuthContext())):
            result = handler._revoke_invitation(mock_http, "org-123", "inv-123")

        assert get_status(result) == 403

    def test_get_auth_context_returns_context(self):
        """_get_auth_context should return AuthorizationContext."""
        handler = create_org_handler()
        user = MockUser(role="admin", org_id="org-123")

        with patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True):
            ctx = handler._get_auth_context(MockHandler(), user)

            if ctx is not None:  # Only check if RBAC is available
                assert ctx.user_id == user.id
                assert "admin" in ctx.roles
                assert ctx.org_id == user.org_id

    def test_get_auth_context_returns_none_without_user(self):
        """_get_auth_context should return None when no user."""
        handler = create_org_handler()

        with (
            patch("aragora.server.handlers.organizations.RBAC_AVAILABLE", True),
            patch.object(handler, "_get_current_user", return_value=(None, None)),
        ):
            ctx = handler._get_auth_context(MockHandler())
            assert ctx is None
